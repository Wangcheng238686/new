"""Joint-model conditional evaluation (prereg part 2 closing read).

Same consumption interface as the conditional runner, but the prefix is the
JOINT model's own strong-SAT-style detections (pre2000 + conservative RoI
supplement) instead of the frozen reference records, and the geometry
candidates come from the jointly-trained head. Full test, FP32, maxDets250.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
from mmcv.ops import batched_nms
from mmengine.structures import InstanceData
from pycocotools import mask as mu
import ue_detection_attrition_probe_20261023 as d
import ue_geometry_proposal_pilot_20261024 as pilot
import ue_geometry_mechanism_probe as g

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'analysis'))
from ue_geometry_conditional_complement_20261024 import trusted_score

_L = Path('/data2/wangcheng/remote-lthpc')
pilot.OUT = _L / 'checkpoint/ue_geometry_proposal_pilot_local_20261024'
pilot.CACHE = _L / 'cache/ue_v1_native_geometry'
pilot.CONSTANTS = _L / 'checkpoint/ue_geometry_mechanism_20261022/common_constants.json'
d.DATA = _L / 'dataset/CitySample_SatUAV_TrainMinimal_v1_20261021'
d.BASE = _L / 'checkpoint/ue_minimal_single_20261021_EXP-0017/checkpoints/best_model_epoch73.pth'
import ue_s2db_probe as _s2
_s2.DATA = d.DATA
_s2.BASE = d.BASE
g.DATA = d.DATA
OUT = _L / 'checkpoint/ue_joint_eval_20261024'
JOINT = Path(os.environ.get('JOINT_WEIGHTS', str(_L / 'checkpoint/ue_joint_geo_sat_20261024/last.pth')))


def arrays(row):
    x = copy.deepcopy(row)
    x['bboxes'] = np.asarray(x['bboxes'], np.float32).reshape(-1, 4)
    x['labels'] = np.asarray(x['labels'], np.int64)
    x['scores'] = np.asarray(x['scores'], np.float64)
    return x


def merge(base, extra):
    x = copy.deepcopy(base)
    for key in ('bboxes', 'labels', 'scores'):
        x[key] = np.concatenate((x[key], extra[key]))
    x['rles'] += extra['rles']
    x['mask_count'] += extra['mask_count']
    x['mask_fill_sum'] += extra['mask_fill_sum']
    return x


def hypotheses(boxes, confidence, original, cfg):
    valid = torch.isfinite(boxes).all(1) & (boxes[:, 2:] > boxes[:, :2]).all(1) & torch.isfinite(confidence) & (confidence > 0)
    if len(boxes) and len(original):
        from mmdet.structures.bbox import bbox_overlaps
        valid &= bbox_overlaps(boxes, original).max(1).values <= cfg.nms.iou_threshold
    boxes, confidence = boxes[valid], confidence[valid]
    if len(boxes):
        dets, _ = batched_nms(boxes, confidence, torch.zeros(len(boxes), device='cuda', dtype=torch.long), cfg.nms)
        dets = dets[:max(0, 250 - len(original))]
        boxes, confidence = dets[:, :4], dets[:, 4]
    return boxes, confidence


def masks(head, xx, f, meta, boxes, scores):
    if not len(boxes):
        return dict(bboxes=np.zeros((0, 4), np.float32), labels=np.zeros(0, np.int64), scores=np.zeros(0, np.float32), rles=[], mask_count=0, mask_fill_sum=0.)
    inst = InstanceData(bboxes=boxes, scores=scores, labels=torch.zeros(len(boxes), device='cuda', dtype=torch.long))
    pred = head.predict_mask(xx, [meta], [inst], rescale=True, image_embeddings=f[1], image_positional_embeddings=f[2], high_res_features=f[3])[0]
    return d.common.gt_arrays(pred)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--content', choices=['geometry', 'jointonly'], default='geometry')
    p.add_argument('--split', default='test')
    a = p.parse_args()
    out = OUT / a.split
    out.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_per_process_memory_fraction(.6)
    d.common.seed(44)
    model, _snap = d.build()
    cp = torch.load(JOINT, map_location='cpu', weights_only=False)
    assert cp['epoch'] == int(os.environ.get('JOINT_EPOCHS', '10'))
    model.load_state_dict(cp['model'], strict=True)
    model.cuda().eval()
    for prm in model.parameters():
        prm.requires_grad_(False)
    model.rpn_head.test_cfg.nms_pre = 2000
    head = model.roi_head
    cfg = head.test_cfg
    net = pilot.Net().cuda()
    net.load_state_dict(cp['net'], strict=True)
    net.eval()
    means = torch.tensor(json.loads(pilot.CONSTANTS.read_text())['mean_fields'], device='cuda')[None, :, None, None]
    ds = d.common.MinimalSatelliteDataset(d.DATA, a.split, augment=False)
    rows = []
    metas = []
    with torch.inference_mode():
        for i in range(len(ds)):
            batch = d.common.rtmdet_collate_fn([ds[i]])
            sample = d.common.processed(model, batch)
            f = model.extract_feat(sample['inputs'])
            meta = sample['data_samples'][0].metainfo
            xx = head._inject_extra_pe(f[0])
            cls, reg = model.rpn_head(f[0])
            props = model.rpn_head.predict_by_feat(cls, reg, batch_img_metas=[meta], rescale=False)
            sat = head.predict_bbox(xx, [meta], props, cfg, rescale=False)[0]
            boxes = props[0].bboxes
            rois = torch.cat((boxes.new_zeros((len(boxes), 1)), boxes), 1)
            br = head._bbox_forward(xx, rois)
            raw = head.bbox_head._predict_by_feat_single(rois, br['cls_score'], br['bbox_pred'], meta, False, None)
            sb, sc = hypotheses(raw.bboxes, raw.scores[:, 0], sat.bboxes, cfg)
            sat_det = masks(head, xx, f, meta, sat.bboxes, sat.scores)
            prefix = merge(dict(bboxes=sat.bboxes.cpu().numpy(), labels=sat.labels.cpu().numpy(),
                                scores=sat.scores.cpu().numpy().astype(np.float64), rles=sat_det['rles'],
                                mask_count=sat_det['mask_count'], mask_fill_sum=sat_det['mask_fill_sum']),
                           masks(head, xx, f, meta, sb, cfg.score_thr * sc.clamp(0, 1)))
            folder = d.common.point_folder(ds, i, d.DATA)
            with np.load(pilot.CACHE / f'{folder.name}.npz') as z:
                x = torch.from_numpy(np.concatenate((z['fields'].astype(np.float32), z['support'][None].astype(np.float32)))).cuda()[None]
            gb, gc = pilot.decode(net(x), x[:, 4:5])
            gb, gc = hypotheses(gb, gc, sat.bboxes, cfg)
            extra = masks(head, xx, f, meta, gb, cfg.score_thr * gc.clamp(0, 1))
            trusted = torch.tensor(prefix['bboxes'], device='cuda', dtype=torch.float32)
            from mmdet.structures.bbox import bbox_overlaps
            keep = bbox_overlaps(gb, trusted).max(1).values <= cfg.nms.iou_threshold if len(gb) and len(trusted) else torch.ones(len(gb), device='cuda', dtype=torch.bool)
            ids = torch.where(keep)[0][:max(0, 250 - len(trusted))].cpu().tolist()
            combined = copy.deepcopy(prefix)
            combined['scores'] = np.asarray(trusted_score(prefix['scores']), np.float64)
            added = dict(bboxes=extra['bboxes'][ids], labels=extra['labels'][ids],
                         scores=extra['scores'][ids].astype(np.float64) / .05 * .25,
                         rles=[extra['rles'][j] for j in ids], mask_count=len(ids),
                         mask_fill_sum=float(np.sum(mu.area([extra['rles'][j] for j in ids])) / 512 ** 2) if ids else 0.)
            combined = merge(combined, added)
            assert len(combined['scores']) <= 250
            gt = d.common.gt_arrays(sample['data_samples'][0].gt_instances)
            rows.append(dict(image_id=ds.samples[i]['image_id'], gt=d.common.serialized(gt), dt=d.common.serialized(combined)))
            metas.append(batch['img_metas'][0])
            if (i + 1) % 25 == 0:
                print('EVAL', i + 1, flush=True)
    cg, cd = d.common.build_coco_gt_and_dt([arrays(r['gt']) if 'scores' in r['gt'] else {**r['gt'], 'bboxes': np.asarray(r['gt']['bboxes'], np.float32), 'labels': np.asarray(r['gt']['labels'], np.int64)} for r in rows],
                                           [arrays(r['dt']) for r in rows], metas, score_key='scores')
    metrics = {}
    for kind in ('segm', 'bbox'):
        metrics.update(d.common.run_coco_eval(cg, cd, kind, max_dets=[1, 10, 250]))
    d.common.dump(out / 'combined_metrics.json', metrics)
    d.common.dump(out / 'records.json', rows)
    print('METRICS joint', metrics['segm/mAP'], metrics['bbox/mAP'], flush=True)


if __name__ == '__main__':
    main()
