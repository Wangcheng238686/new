"""Independent S/U/T/C construction and objective; no training launched on import.

Old ABC numerical sources are deliberately untouched. This module is an
engineering API, not a replacement for their frozen training/evaluation runner.
A new four-rank acceptance receipt is required before a SUTC training wrapper.
"""
import json
from pathlib import Path
import sys

WORK = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(WORK.parent / 'sam2'), str(WORK), str(WORK / 'scripts/probes')]
import torch
import ue_ubr_native_runtime_20261027 as legacy
from data.ue_instance_targets import center_targets
from data.ue_view_subset_inputs import SubsetUAVInputs
from rsprompter.uav.local_adapter_probe import YAWS
from rsprompter.uav.view_subset import canonical_views, PRESETS, configure_upstream_views
from rsprompter.uav.native_sutc import SUTCSwitches, TailP0Evidence, SatelliteTailJoint
from rsprompter.uav.ubr_native_integration import native_model_config

base, core = legacy.base, legacy.core
CONTRACT = WORK / 'configs/ablations/ue_native_sutc_contract_20261028.json'


class Runtime:
    """Explicit switches cover inputs, forward losses AND inference candidates.

    Caller may wrap ``network`` in DDP; objective always calls that wrapper.
    No inherited old runner is allowed to unconditionally prepare UAV targets.
    """
    def __init__(self, arm, *, uav_view_ids=YAWS, subset_cache_root=None):
        self.view_ids = canonical_views(uav_view_ids)
        if arm != 'S' and self.view_ids != YAWS and subset_cache_root is None:
            raise ValueError('Reduced-view training requires a rebuilt subset cache root')
        # Validate full cache BEFORE allocating a model/GPU. S has no UAV cache.
        subset_inputs = (SubsetUAVInputs(core.previous.DATA, subset_cache_root, self.view_ids)
                         if arm != 'S' and subset_cache_root is not None else None)
        self.switches = SUTCSwitches.for_arm(arm)
        self.matrix_arm = arm
        self.root = base.ROOT
        if self.switches.upstream:
            old = legacy.build_runtime('C_correct' if self.switches.tail_evidence else 'B_zero')
            self.__dict__.update(old.__dict__)
        else:
            path = self.root / 'common_init.pth'
            receipt = json.loads((self.root / 'initialization.json').read_text())
            if receipt['source_hashes'] != base.sources():
                raise RuntimeError('Common initialization numerical source identity changed')
            if receipt['common_init_sha256'] != core.C.file_sha(path):
                raise RuntimeError('Common initialization digest mismatch')
            cp = torch.load(path, map_location='cpu', weights_only=False)
            if cp['task_weights_loaded'] is not False or cp['foundation_sha256'] != core.C.file_sha(base.FOUNDATION):
                raise RuntimeError('Only shared generic-foundation initialization is allowed')
            core.C.seed(base.PROTOCOL['seed'])
            config = native_model_config(cp['model_config'])
            satellite = core.C._register_and_build(config)
            retired = [k for k in cp['satellite'] if k.startswith('roi_head.mask_head.decoder_tail_refiner.')]
            satellite.load_state_dict({k: v for k, v in cp['satellite'].items() if k not in retired}, strict=True)
            provider = TailP0Evidence(base.FOUNDATION, cp['uav'], view_ids=self.view_ids) if self.switches.tail_evidence else None
            self.joint = SatelliteTailJoint(satellite, provider).cuda()
            self.satellite = satellite
            self.network = self.bare = self.joint
            self.shim = satellite.roi_head.mask_head.mask_decoder
            satellite.rpn_head.test_cfg.nms_pre = base.PROTOCOL['RPN_pre']
            # S never even constructs the UAV input loader.
            self.uav_inputs = None
            if self.switches.tail_evidence:
                self.uav_inputs = legacy.JointUAVInputs(core.previous.DATA,
                    '/data/wangcheng/cache/ue_v1_native_geometry',
                    '/data/wangcheng/cache/ue_v1_s2db_geometry_v2', base.PLANE)
            self.optimizer = torch.optim.AdamW([p for p in self.network.parameters() if p.requires_grad],
                lr=base.PROTOCOL['lr'], weight_decay=base.PROTOCOL['weight_decay'])
            self.scaler = torch.amp.GradScaler('cuda', init_scale=base.PROTOCOL['amp_init_scale'])
            self.initialization_audit = dict(common_init_sha256=receipt['common_init_sha256'],
                foundation_sha256=cp['foundation_sha256'], task_weights_loaded=False,
                udpr_never_instantiated=True, retired_tail_state_keys=retired,
                retained_state_strict=True, main_mask_size=list(config['train_cfg']['rcnn']['mask_size']),
                head_seed=2080, source_subset='rgb_encoder.*,rgb_project.*,rgb_mean,rgb_std' if provider is not None else 'none')
        self.arm = arm
        if subset_inputs is not None:
            self.uav_inputs = subset_inputs
        if self.switches.upstream and self.view_ids != YAWS:
            configure_upstream_views(self.joint.uav, self.view_ids)
        self.generator = torch.Generator().manual_seed(base.PROTOCOL['seed'])
        self.initialization_audit.update(matrix_arm=arm, upstream=self.switches.upstream,
            tail_evidence=self.switches.tail_evidence, UBR_present=True,
            UAV_instance_loss=self.switches.upstream, UAV_proposals=self.switches.upstream,
            uav_view_ids=list(self.view_ids) if arm != 'S' else [],
            uav_view_count=len(self.view_ids) if arm != 'S' else 0,
            subset_cache_manifest_sha256=subset_inputs.complete['manifest_sha256'] if subset_inputs else None,
            view_count_denominator=5, source_slots='canonical five slots; encode selected only')
        self._assert_wiring()
        core.C.seed(base.PROTOCOL['seed'])

    def _assert_wiring(self):
        head = self.satellite.roi_head.mask_head
        if head.decoder_tail_enabled or head.decoder_tail_refiner is not None:
            raise RuntimeError('UDPR must not be constructed in SUTC')
        if not hasattr(head, 'native_refiner') or self.shim.zero_evidence != (not self.switches.tail_evidence):
            raise RuntimeError('UBR evidence port does not match arm')
        if self.joint.uav.enabled != self.switches.upstream:
            raise RuntimeError('UAV proposal switch does not match upstream switch')
        if self.arm == 'S' and (self.uav_inputs is not None or any(n.startswith('provider.') or n.startswith('uav.') for n, _ in self.network.named_parameters())):
            raise RuntimeError('S must have no UAV loader or UAV parameters')

    def payload(self, ident):
        if self.arm == 'S':
            return {}
        if self.switches.upstream:
            value = base.Runtime.payload(self, ident)
            self.joint.uav.candidate_geometry = legacy.load_candidate_geometry(
                core.previous.DATA, ident, value['index'], value['anchor'])
            return value
        value = self.uav_inputs.load(ident, geometry_mode='measured', rgb_mode='correct')
        if value.pop('provenance')['measured_depth'] is not True:
            raise RuntimeError('T requires measured correspondence')
        return value

    def losses(self, sample, ident):
        payload = self.payload(ident)
        if self.switches.upstream:
            boxes = sample['data_samples'][0].gt_instances.bboxes.detach().cpu().numpy()
            losses = self.network(sample, **payload, instance_target=center_targets(boxes),
                                  center_loss_fn=core.pilot.loss_fn)
        else:
            losses = self.network(sample, **payload)
            if any('uav' in name.lower() or 'decoder_tail' in name.lower() or 'ubr' in name.lower() for name in losses):
                raise RuntimeError('S/T must use only original satellite task losses')
        return losses

    def objective(self, sample, ident):
        terms = []
        for name, value in self.losses(sample, ident).items():
            if 'loss' not in name:
                continue
            for item in value if isinstance(value, (list, tuple)) else [value]:
                if torch.is_tensor(item) and item.requires_grad:
                    terms.append(item)
        if not terms:
            raise RuntimeError('No differentiable task loss')
        return sum(terms)

    def inference(self, sample, ident):
        """Original formal consumer; caller owns eval(), FP32 and inference_mode."""
        if self.bare.training or torch.is_autocast_enabled():
            raise RuntimeError('Evaluation requires eval mode and FP32')
        payload = self.payload(ident)
        self.joint.prepare_uav(**payload)
        features = self.satellite.extract_feat(sample['inputs'])
        return core.inference_outputs(self.satellite, self.joint.uav, sample, features, payload.get('geometry'))

    def clear(self):
        self.joint.clear()

    def set_epoch(self, epoch):
        for module in self.bare.modules():
            setter = getattr(module, 'set_current_epoch', None)
            if callable(setter):
                setter(epoch)


def build_runtime(arm, *, uav_view_ids=None, view_count=None, subset_cache_root=None):
    if uav_view_ids is not None and view_count is not None:
        raise ValueError('Specify view IDs or a preset count, not both')
    if view_count is not None and view_count not in PRESETS:
        raise ValueError('Preset view count must be 1, 3 or 5')
    views = PRESETS[view_count] if view_count is not None else (YAWS if uav_view_ids is None else uav_view_ids)
    return Runtime(arm, uav_view_ids=views, subset_cache_root=subset_cache_root)


if __name__ == '__main__':
    # CPU-only contract inspection; never silently launch training/CUDA models.
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=list('SUTC'), default='C')
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument('--view-count', type=int, choices=sorted(PRESETS))
    choice.add_argument('--view-ids', type=int, nargs='+')
    parser.add_argument('--subset-cache-root', type=Path)
    args = parser.parse_args()
    views = canonical_views(PRESETS[args.view_count] if args.view_count else (YAWS if args.view_ids is None else args.view_ids))
    print(json.dumps(dict(contract=json.loads(CONTRACT.read_text()), arm=args.arm,
        resolved_uav_view_ids=list(views) if args.arm != 'S' else [],
        subset_cache_root=str(args.subset_cache_root) if args.subset_cache_root else None,
        action='inspect_only_no_model_no_training'), indent=2))
