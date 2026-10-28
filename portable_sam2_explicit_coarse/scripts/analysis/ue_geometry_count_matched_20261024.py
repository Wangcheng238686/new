"""GT-free equal extra-count control using fixed seed44 test predictions."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import copy
import json
from pathlib import Path
import sys
import numpy as np
from pycocotools import mask as mu

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'probes'))
from ue_v1_adapter import build_coco_gt_and_dt,run_coco_eval

SRC=Path('/data/wangcheng/checkpoint/ue_geometry_conditional_online_20261024')
OUT=Path('/data/wangcheng/checkpoint/ue_geometry_count_matched_20261024')


def arrays(r):
    x=dict(r);x['bboxes']=np.asarray(x['bboxes'],np.float32).reshape(-1,4);x['labels']=np.asarray(x['labels'],np.int64)
    if 'scores' in x:x['scores']=np.asarray(x['scores'],np.float64)
    return x


def main():
    OUT.mkdir(exist_ok=True)
    prefix=json.loads((SRC/'geometry/test/prefix_records.json').read_text())
    g,k=[json.loads((SRC/arm/'test/combined_records.json').read_text()) for arm in ['geometry','constant']]
    quotas=[]
    for b,a,c in zip(prefix,g,k):
        assert b['image_id']==a['image_id']==c['image_id'] and b['gt']==a['gt']==c['gt']
        n=len(b['dt']['scores']);quotas.append((n,min(len(a['dt']['scores'])-n,len(c['dt']['scores'])-n)))
    metrics={}
    for arm,source in [('geometry',g),('constant',k)]:
        rows=[]
        for r,(n,q) in zip(source,quotas):
            row=copy.deepcopy(r);dt=row['dt']
            for key in ('bboxes','labels','scores','rles'):dt[key]=dt[key][:n+q]
            dt['mask_count']=n+q;dt['mask_fill_sum']=float(np.sum(mu.area(dt['rles']))/512**2)
            row['count_match']=dict(prefix=n,extra=q);rows.append(row)
        (OUT/f'{arm}_records.json').write_text(json.dumps(rows))
        cg,cd=build_coco_gt_and_dt([arrays(r['gt']) for r in rows],[arrays(r['dt']) for r in rows],[dict(img_shape=(512,512)) for _ in rows],score_key='scores')
        m={}
        for kind in ('segm','bbox'):m.update(run_coco_eval(cg,cd,kind,max_dets=[1,10,250]))
        (OUT/f'{arm}_metrics.json').write_text(json.dumps(m,indent=2));metrics[arm]=m
        print('METRICS',arm,m['segm/mAP'],m['bbox/mAP'],flush=True)
    (OUT/'summary.json').write_text(json.dumps(dict(complete=True,images=459,equal_extra_count_per_image=True,
        total_extra_count=sum(q for _,q in quotas),metrics=metrics,
        scope='Post-hoc budget robustness diagnostic, not a newly selected inference policy. '
              'Per-image min(G,K) quota uses predictions only, no GT; keep own-score highest extras. '
              'All original strongSAT prefix masks/ranks retained. No refitting or new model inference.'),indent=2))


if __name__=='__main__':main()
