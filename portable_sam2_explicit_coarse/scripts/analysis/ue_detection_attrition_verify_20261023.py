"""Independent CPU identity/count audit after the detector stage probe finishes."""
import hashlib
import json
import os
from pathlib import Path
import time
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
from pycocotools import mask as mu

OUT=Path('/data/wangcheng/checkpoint/ue_detection_attrition_20261023')

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    while not (OUT/'complete.json').exists():
        p=OUT/'state.json'
        if p.exists() and read(p).get('stage')=='failed':raise RuntimeError('Main probe failed')
        time.sleep(10)
    plan=read(OUT/'plan.json');done=read(OUT/'complete.json');rows=read(OUT/'records.json');summary=read(OUT/'summary.json')
    assert done['complete'] and not done['gate_only'] and len(rows)==done['images']==459
    assert sha(OUT/'records.json')==done['records_sha256']
    assert all(sha(p)==h for p,h in plan['source_hashes'].items())
    assert sha(plan['checkpoint'])==plan['checkpoint_sha256']
    assert sha(plan['reference'])==plan['reference_sha256']
    ref=read(Path(plan['reference']));threshold_differences={str(t):0 for t in plan['thresholds']}
    ng=nd=0;final_covered={str(t):0 for t in plan['thresholds']}
    for row,b in zip(rows,ref):
        assert row['image_id']==b['image_id'] and row['final_boxes']==b['dt']['bboxes'] and row['final_scores']==b['dt']['scores']
        assert all(x==0 for x in b['dt']['labels'])
        assert row['gt_count']==len(b['gt']['bboxes'])
        assert row['mask_area']==mu.area(b['gt']['rles']).astype(int).tolist()
        ng+=row['gt_count'];nd+=len(row['final_boxes'])
        boxes=np.array(row['final_boxes'],dtype=float).reshape(-1,4);gt=np.array(b['gt']['bboxes'],dtype=float).reshape(-1,4)
        boxes[:,2:]-=boxes[:,:2];gt[:,2:]-=gt[:,:2]
        iou=mu.iou(boxes.tolist(),gt.tolist(),[0]*len(gt))
        best=iou.max(0) if len(boxes) else np.zeros(len(gt))
        for t in plan['thresholds']:
            got=np.array(row['stages']['final_cap']['best_iou'])>=t
            threshold_differences[str(t)]+=int((got!=(best>=t)).sum())
            final_covered[str(t)]+=int(got.sum())
        for s in plan['stages']:
            x=row['stages'][s]
            assert len(x['best_iou'])==row['gt_count'] and np.isfinite(x['best_iou']).all()
            for t in plan['thresholds']:
                any_count=int((np.array(x['best_iou'])>=t).sum())
                assert 0<=x['one_to_one_capacity'][str(t)]<=min(x['candidates'],any_count)
    assert ng==44988 and nd==40932
    for t in plan['thresholds']:
        group=summary['groups']['all'][str(t)]
        assert group['gt']==ng and group['any_candidate_covered']['final_cap']==final_covered[str(t)]
        assert sum(group['final_uncovered_last_transition'].values())==ng-final_covered[str(t)]
        for s in plan['stages']:
            assert summary['one_to_one_capacity'][str(t)][s]==sum(r['stages'][s]['one_to_one_capacity'][str(t)] for r in rows)
    result=dict(passed=True,images=459,gt=ng,final_detections=nd,baseline_boxes_scores_labels_exact=True,
        gt_mask_areas_exact=True,source_and_checkpoint_sha_exact=True,count_invariants=True,
        final_coverage_fp32_vs_independent_coco_float64_threshold_disagreements=threshold_differences,
        note='Any FP32/float64 near-threshold differences are reported, not silently rewritten; capacities are not COCO score-order matching.',
        records_sha256=sha(OUT/'records.json'),summary_sha256=sha(OUT/'summary.json'),
        verification_script_sha256=sha(__file__),optimizer_steps=0)
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
