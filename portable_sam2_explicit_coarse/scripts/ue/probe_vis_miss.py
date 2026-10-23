#!/usr/bin/env python3
"""VIS-MISS: UAV geometric visibility of MISSED GT instances on test.
Direction gate for the recall-rescue arm."""
import json
import numpy as np
from pathlib import Path
from pycocotools import mask as maskUtils

R = json.load(open('/data2/wangcheng/remote-lthpc/evals/base_disabled/disabled/records.json'))
CACHE = Path('/data2/wangcheng/remote-lthpc/ue_v1_local_adapter_geometry')

def to_rle(e):
    if isinstance(e, dict):
        return {'size': e['size'], 'counts': e['counts'].encode() if isinstance(e['counts'], str) else e['counts']}
    return e

agg = dict(gt=0, miss=0, miss_s=0, miss_m=0, miss_l=0,
           miss_s_v50=0, miss_s_v70=0, miss_s_v90=0, miss_m_v50=0,
           matched_s_v50=0, matched_s=0, miss_cov_sum=0.0)
covs_miss_small = []
covs_all = []
for rec in R:
    pid = f"{rec['image_id']:06d}"
    npz = CACHE / f'{pid}.npz'
    if not npz.exists():
        continue
    with np.load(npz) as a:
        index = a['index']
    support = (index >= 0).any(1).reshape(128, 128).astype(np.uint8)
    sup512 = np.kron(support, np.ones((4, 4), np.uint8)).astype(bool)
    gt_rles = [to_rle(e) for e in rec['gt']['rles']]
    dt_rles = [to_rle(e) for e in rec['dt']['rles']]
    ng, nd = len(gt_rles), len(dt_rles)
    if ng and nd:
        ious = maskUtils.iou(dt_rles, gt_rles, [0] * ng)
        if ious.shape[1] < ng:            # pycocotools collapses single-GT dims
            ious = ious.reshape(nd, ng) if ious.size == nd * ng else ious
        best_iou = ious.max(0) if ious.size else np.zeros(ng)
    elif ng:
        best_iou = np.zeros(ng)
    else:
        continue
    for g in range(ng):
        m = maskUtils.decode(gt_rles[g]).astype(bool)
        cov = float(sup512[m].mean())
        covs_all.append(cov)
        area = maskUtils.area(gt_rles[g])
        agg['gt'] += 1
        if best_iou[g] >= 0.5:
            if area < 1024:
                agg['matched_s'] += 1
                agg['matched_s_v50'] += int(cov >= 0.5)
            continue
        agg['miss'] += 1
        agg['miss_cov_sum'] += cov
        if area < 1024:
            agg['miss_s'] += 1
            covs_miss_small.append(cov)
            agg['miss_s_v50'] += int(cov >= 0.5)
            agg['miss_s_v70'] += int(cov >= 0.7)
            agg['miss_s_v90'] += int(cov >= 0.9)
        elif area < 9216:
            agg['miss_m'] += 1
            agg['miss_m_v50'] += int(cov >= 0.5)
        else:
            agg['miss_l'] += 1
out = dict(agg,
           miss_s_frac_v50=agg['miss_s_v50'] / max(agg['miss_s'], 1),
           miss_s_frac_v70=agg['miss_s_v70'] / max(agg['miss_s'], 1),
           miss_s_frac_v90=agg['miss_s_v90'] / max(agg['miss_s'], 1),
           matched_s_frac_v50=agg['matched_s_v50'] / max(agg['matched_s'], 1),
           miss_mean_cov=agg['miss_cov_sum'] / max(agg['miss'], 1),
           miss_small_cov_median=float(np.median(covs_miss_small)) if covs_miss_small else None,
           all_cov_median=float(np.median(covs_all)))
print(json.dumps(out, indent=2))
Path('/data2/wangcheng/remote-lthpc/vis_miss_test.json').write_text(json.dumps(out, indent=2))
