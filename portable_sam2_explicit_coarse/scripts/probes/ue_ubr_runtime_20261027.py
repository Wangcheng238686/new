"""Fresh-foundation G/UBR runtime; no changes to frozen previous runners.

This is integration infrastructure, not authorization for a full training run.
The inherited update schedule is the original joint-training schedule.
"""
from copy import deepcopy
from pathlib import Path
import sys

WORK = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(WORK), str(WORK / 'scripts/probes')]
import ue_candidate_detail_ddp_20261026 as previous
from data.ue_candidate_evidence import load_candidate_geometry
from data.ue_instance_targets import center_targets
from data.ue_joint_inputs import JointUAVInputs
from rsprompter.uav.compact_joint import CompactJoint
from rsprompter.uav.candidate_detail_interaction import install as install_g
from rsprompter.uav.ubr_detector_integration import (
    disable_udpr_config, install_ubr, PredictionPool)
import numpy as np
import torch
from torch import nn

base = previous.base
core = base.core


class GWithUBR(nn.Module):
    def __init__(self, joint, controller, enabled):
        super().__init__()
        self.joint = joint
        self.controller = controller  # not nn.Module: parameters belong to satellite
        self.ubr_enabled = enabled
        self.pool_callback = None

    def forward(self, sample, geometry, rgb, index, anchor, instance_target, center_loss_fn):
        joint, satellite = self.joint, self.joint.satellite
        joint.prepare_uav(geometry, rgb, index, anchor)
        features = satellite.extract_feat(sample['inputs'])
        with core.C.cached_extract(satellite, features):
            losses = satellite.loss(sample['inputs'], sample['data_samples'])
        direct, _ = center_loss_fn(joint.uav.outputs, instance_target)
        losses = dict(losses, loss_uav_instances=direct)
        if self.ubr_enabled:
            meta = sample['data_samples'][0].metainfo
            prepared = self.controller.prepare(features, meta)
            pool = self.pool_callback(features, geometry, prepared)
            gt = sample['data_samples'][0].gt_instances
            masks = gt.masks.to_tensor(dtype=torch.float32, device=geometry.device)
            losses.update(self.controller.training_loss(prepared, pool, gt.bboxes, masks, geometry))
        return losses


class Runtime(base.Runtime):
    def __init__(self, content='correct', variant='detail', enabled=True):
        if content not in ('correct', 'constant') or variant != 'detail':
            raise ValueError('first P1 runtime supports G with correct/constant RGB only')
        self.root = base.ROOT
        self.arm = {'correct': 'rgbd', 'constant': 'constant_rgb'}[content]
        self.content = content
        initialization = self.root / 'common_init.pth'
        import json
        receipt = json.loads((self.root / 'initialization.json').read_text())
        assert receipt['common_init_sha256'] == core.C.file_sha(initialization)
        assert receipt['source_hashes'] == base.sources()
        cp = torch.load(initialization, map_location='cpu', weights_only=False)
        assert cp['task_weights_loaded'] is False
        assert cp['foundation_sha256'] == core.C.file_sha(base.FOUNDATION)
        core.C.seed(base.PROTOCOL['seed'])
        config = disable_udpr_config(cp['model_config'])
        satellite = core.C._register_and_build(config)
        old_state = cp['satellite']
        retired = [k for k in old_state if k.startswith('roi_head.mask_head.decoder_tail_refiner.')]
        kept = {k: v for k, v in old_state.items() if k not in retired}
        # Strict equality for every retained parameter/buffer: no broad strict=False.
        satellite.load_state_dict(kept, strict=True)
        assert satellite.roi_head.mask_head.decoder_tail_refiner is None
        assert not satellite.roi_head.mask_head.decoder_tail_enabled
        self.joint = CompactJoint(satellite, base.FOUNDATION, source_seed=2045)
        self.joint.uav.load_state_dict(cp['uav'], strict=True)
        self.joint.cuda()
        install_g(self.joint.uav, variant, satellite)
        # UBR capacity must not perturb subsequent data/dropout random streams.
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(2070)
            self.controller = install_ubr(satellite)
        self.satellite = satellite
        self.satellite.rpn_head.test_cfg.nms_pre = base.PROTOCOL['RPN_pre']
        self.network = GWithUBR(self.joint, self.controller, enabled)
        self.network.pool_callback = self.select_pool
        self.bare = self.network
        self.uav_inputs = JointUAVInputs(core.previous.DATA,
            '/data/wangcheng/cache/ue_v1_native_geometry',
            '/data/wangcheng/cache/ue_v1_s2db_geometry_v2', base.PLANE)
        self.generator = torch.Generator().manual_seed(44)
        self.optimizer = torch.optim.AdamW(
            [p for p in self.network.parameters() if p.requires_grad],
            lr=base.PROTOCOL['lr'], weight_decay=base.PROTOCOL['weight_decay'])
        self.scaler = torch.amp.GradScaler('cuda', init_scale=base.PROTOCOL['amp_init_scale'])
        self.initialization_audit = dict(common_init=str(initialization),
            common_init_sha256=receipt['common_init_sha256'],
            task_weights_loaded=False, retired_state_keys=retired,
            retained_state_strict=True, udpr_never_instantiated=True,
            model_config=config, UBR_seed=2070)
        core.C.seed(base.PROTOCOL['seed'])

    def payload(self, ident):
        value = super().payload(ident)
        self.joint.uav.candidate_geometry = load_candidate_geometry(
            core.previous.DATA, ident, value['index'], value['anchor'])
        return value

    @torch.no_grad()
    def select_pool(self, features, geometry, prepared=None):
        """Same bbox/score policy as old inference_outputs, WITHOUT mask decode.

        Current scope is the 512-square, unscaled v1 dataset. No hidden
        application of reference context NMS. Final selected order is the ID.
        """
        if prepared is None:
            raise ValueError('pass prepared image so extraPE is applied only once')
        if tuple(prepared.meta['ori_shape'][:2]) != (512, 512):
            raise ValueError('P1 runtime currently supports original 512-square data')
        if not np.array_equal(np.asarray(prepared.meta['scale_factor']), np.ones(2)):
            raise ValueError('P1 runtime requires scale_factor=(1,1)')
        head, cfg = self.satellite.roi_head, self.satellite.roi_head.test_cfg
        xx, meta = prepared.roi_features, prepared.meta
        with self.controller._prediction_context_mode():
            cls, reg = self.satellite.rpn_head(features[0])
            props = self.satellite.rpn_head.predict_by_feat(
                cls, reg, batch_img_metas=[meta], rescale=False)
            sat = head.predict_bbox(xx, [meta], props, cfg, rescale=False)[0]
            rois = torch.cat((props[0].bboxes.new_zeros((len(props[0]), 1)), props[0].bboxes), 1)
            br = head._bbox_forward(xx, rois)
            raw = head.bbox_head._predict_by_feat_single(
                rois, br['cls_score'], br['bbox_pred'], meta, False, None)
            boxes, confidence = core.consumer.hypotheses(
                raw.bboxes, raw.scores[:, 0], sat.bboxes, cfg)
            prefix = torch.cat((sat.bboxes, boxes))
            prefix_scores = torch.cat((sat.scores, cfg.score_thr * confidence.clamp(0, 1)))
            scores = torch.as_tensor(core.consumer.trusted_score(prefix_scores.cpu().numpy()),
                                     device=prefix.device, dtype=torch.float64)
            sources = ('sat',) * len(sat) + ('sat_supplement',) * len(boxes)
            if self.joint.uav.enabled:
                boxes, confidence = core.pilot.decode(self.joint.uav.outputs, geometry[:, 4:5])
                boxes, confidence = core.consumer.hypotheses(boxes, confidence, prefix, cfg)
                scores = torch.cat((scores, (.25 * confidence.clamp(0, 1)).to(scores)))
                prefix = torch.cat((prefix, boxes))
                sources += ('uav',) * len(boxes)
        assert len(prefix) <= 250
        return PredictionPool(prefix.detach(), scores.detach(),
            torch.arange(len(prefix), device=prefix.device, dtype=torch.long), sources)

    def objective_with_ubr(self, sample, ident):
        boxes = sample['data_samples'][0].gt_instances.bboxes.detach().cpu().numpy()
        losses = self.network(sample, **self.payload(ident),
            instance_target=center_targets(boxes), center_loss_fn=core.pilot.loss_fn)
        return sum(v for k, value in losses.items() if 'loss' in k
                   for v in (value if isinstance(value, (tuple, list)) else [value])
                   if torch.is_tensor(v) and v.requires_grad)


def build_runtime(content='correct', variant='detail', enabled=True):
    return Runtime(content, variant, enabled)
