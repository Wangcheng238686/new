#!/usr/bin/env python3
"""R0-new error decomposition on the frozen base's TEST predictions
(WARM-BA-H precondition gate, zero-GPU: reads base_disabled records.json)."""
import json
import numpy as np
from pathlib import Path
from pycocotools import mask as maskUtils
from scipy import ndimage

R = json.load(open('/data2/wangcheng/remote-lthpc/evals/base_disabled/disabled/records.json'))
RD = 3

def to_rle(e):
    if isinstance(e, dict):
        return {'size': e['size'], 'counts': e['counts'].encode() if isinstance(e['counts'], str) else e['counts']}
    return e

def main():
    agg = dict(scenes=0, gt=0, pred=0, miss=0, false=0, split_pairs=0, split_members=0,
               merge_preds=0, banded=0, good=0, miss_s=0, miss_m=0, miss_l=0,
               gt_s=0, gt_m=0, gt_l=0, split_soft=0)
    per_scene = []
    for rec in R:
        gt_rles = [to_rle(e) for e in rec['gt']['rles']]
        dt_rles = [to_rle(e) for e in rec['dt']['rles']]
        ng, nd = len(gt_rles), len(dt_rles)
        agg['scenes'] += 1
        agg['gt'] += ng
        agg['pred'] += nd
        if ng and nd:
            ious = maskUtils.iou(dt_rles, gt_rles, [0] * ng)
            best_gt = ious.argmax(1)
            best_iou = ious.max(1)
        else:
            ious = np.zeros((nd, max(ng, 1)))
            best_gt, best_iou = np.zeros(0, int), np.zeros(0)
        gt_areas = np.array([maskUtils.area(g) for g in gt_rles]) if ng else np.zeros(0)
        for gi in range(ng):
            a = gt_areas[gi]
            agg['gt_s' if a < 1024 else 'gt_m' if a < 9216 else 'gt_l'] += 1
        matched_gt = set(best_gt[best_iou >= 0.5].tolist())
        miss_idx = [g for g in range(ng) if g not in matched_gt]
        agg['miss'] += len(miss_idx)
        for g in miss_idx:
            a = gt_areas[g]
            agg['miss_s' if a < 1024 else 'miss_m' if a < 9216 else 'miss_l'] += 1
        agg['false'] += int((best_iou < 0.5).sum())
        agg['merge_preds'] += int(sum(1 for p in range(nd) if (ious[p] >= 0.5).sum() >= 2))
        agg['banded'] += int(((best_iou >= 0.5) & (best_iou < 0.75)).sum())
        agg['good'] += int((best_iou >= 0.75).sum())
        P = [maskUtils.decode(d).astype(bool) for d in dt_rles]
        sm = set()
        soft = 0
        for x in range(nd):
            for y in range(x + 1, nd):
                if best_iou[x] >= 0.3 and best_iou[y] >= 0.3 and best_gt[x] == best_gt[y]:
                    adj = (ndimage.binary_dilation(P[x], iterations=RD) & ndimage.binary_dilation(P[y], iterations=RD)).any()
                    if adj:
                        if best_iou[x] >= 0.5 and best_iou[y] >= 0.5:
                            sm.update((x, y))
                            agg['split_pairs'] += 1
                        else:
                            soft += 1
        agg['split_members'] += len(sm)
        agg['split_soft'] += soft
        per_scene.append(dict(image_id=rec['image_id'], gt=ng, pred=nd, miss=len(miss_idx),
                              false=int((best_iou < 0.5).sum()), split=len(sm),
                              merge=int(sum(1 for p in range(nd) if (ious[p] >= 0.5).sum() >= 2))))
        if (len(per_scene)) % 100 == 0:
            print(len(per_scene), flush=True)
    total_err = agg['miss'] + agg['false'] + agg['split_members'] + agg['merge_preds']
    out = dict(agg, total_error_mass=total_err,
               interface_frac_of_preds=agg['split_members'] / max(agg['pred'], 1),
               interface_frac_of_errors=(agg['split_members'] + agg['merge_preds']) / max(total_err, 1),
               miss_frac_of_gt=agg['miss'] / max(agg['gt'], 1))
    out['gate'] = {'rule': 'interface members >=3% preds AND >=10% error mass',
                   'pass': bool(out['interface_frac_of_preds'] >= 0.03 and out['interface_frac_of_errors'] >= 0.10)}
    print(json.dumps(out, indent=2))
    Path('/data2/wangcheng/remote-lthpc/r0new_test.json').write_text(json.dumps(out, indent=2))

if __name__ == '__main__':
    main()
