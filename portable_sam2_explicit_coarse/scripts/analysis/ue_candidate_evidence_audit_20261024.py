"""Read-only coverage audit of SAT budget recovery, using existing test outputs.

No model fitting, threshold selection, cache mutation or new test inference.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from pycocotools import mask as mu

ROOT = Path('/data/wangcheng/checkpoint/ue_rpn_geometry_20261023')
OUT = Path('/data/wangcheng/checkpoint/ue_candidate_evidence_20261024')
CACHE = Path('/data/wangcheng/cache/ue_v1_native_geometry')


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    paths = [ROOT/'test/baseline/records.json', ROOT/'test/budget/records.json']
    old, new = [json.loads(p.read_text()) for p in paths]
    assert len(old) == len(new) == 459
    rows = []
    cache_hashes = {}
    for a, b in zip(old, new):
        assert a['image_id'] == b['image_id'] and a['gt'] == b['gt']
        ident = a['image_id']
        p = CACHE/f'{ident:06d}.npz'
        cache_hashes[p.name] = sha(p)
        with np.load(p) as z:
            support = z['support'].astype(bool)
            fields = z['fields'].astype(np.float32)
        assert support.shape == (512, 512) and fields.shape == (4, 512, 512)
        aa, bb = a['stages'], b['stages']
        masks = mu.decode(a['gt']['rles']).astype(bool)
        if masks.ndim == 2:
            masks = masks[..., None]
        assert masks.shape[2] == len(a['gt']['bboxes'])
        for j, box in enumerate(a['gt']['bboxes']):
            mask = masks[..., j]
            area = int(mask.sum())
            obs = mask & support
            x1, y1, x2, y2 = box
            x, y = int(np.floor((x1+x2)/2)), int(np.floor((y1+y2)/2))
            center_valid = 0 <= x < 512 and 0 <= y < 512
            row = dict(image_id=ident, gt_index=j, density=len(a['gt']['bboxes']),
                       mask_area=area, mask_supported_pixels=int(obs.sum()),
                       mask_support_fraction=float(obs.sum()/max(area, 1)),
                       bbox_center_support=bool(center_valid and support[y, x]),
                       supported_mean_views=float(fields[3][obs].mean()*5) if obs.any() else None,
                       supported_height_span_m=float(np.ptp(fields[0][obs])*100) if obs.any() else None)
            for stage in aa:
                row[stage+'_old'] = aa[stage]['best_iou'][j]
                row[stage+'_new'] = bb[stage]['best_iou'][j]
            rows.append(row)
        print('IMAGE', ident, flush=True)
    groups = {
        'all': lambda r: True,
        'topk_lost_original': lambda r: r['rpn_decoded_all_old'] >= .5 > r['rpn_topk_old'],
        'topk_recovered_budget': lambda r: r['rpn_topk_old'] < .5 <= r['rpn_topk_new'],
        'final_recovered_budget': lambda r: r['final_cap_old'] < .5 <= r['final_cap_new'],
        'final_lost_budget': lambda r: r['final_cap_new'] < .5 <= r['final_cap_old'],
        'final_still_missing': lambda r: r['final_cap_new'] < .5,
    }
    summary = {}
    for name, predicate in groups.items():
        selected = [r for r in rows if predicate(r)]
        n = len(selected)
        summary[name] = dict(gt=n,
            small_mask_count=sum(r['mask_area'] < 1024 for r in selected),
            density_gt100_count=sum(r['density'] > 100 for r in selected),
            any_mask_support_count=sum(r['mask_supported_pixels'] > 0 for r in selected),
            at_least_half_mask_support_count=sum(r['mask_support_fraction'] >= .5 for r in selected),
            bbox_center_supported_count=sum(r['bbox_center_support'] for r in selected),
            mean_instance_support_fraction=float(np.mean([r['mask_support_fraction'] for r in selected])) if n else None,
            pixel_weighted_support_fraction=sum(r['mask_supported_pixels'] for r in selected)/max(1,sum(r['mask_area'] for r in selected)),
            final_old_covered=sum(r['final_cap_old'] >= .5 for r in selected),
            final_new_covered=sum(r['final_cap_new'] >= .5 for r in selected))
    report = dict(complete=True, images=459, instances=len(rows), groups=summary,
        sources={str(p): sha(p) for p in paths}, cache_hashes=cache_hashes,
        script_sha256=sha(Path(__file__)), optimizer_steps=0,
        thresholds=dict(box_iou=.5, small_mask_area=1024, dense_instances=100, adequate_mask_support=.5),
        scope='Per-GT any-box coverage, not one-to-one matching or AP decomposition. '
              'GT mask support is projected observed-depth coverage, not certified surface correspondence. '
              'BBox center is a diagnostic proxy, not the actual best candidate anchor center. '
              'Previously exposed v1 test; exploratory diagnosis only.')
    (OUT/'instances.json').write_text(json.dumps(rows))
    (OUT/'summary.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
