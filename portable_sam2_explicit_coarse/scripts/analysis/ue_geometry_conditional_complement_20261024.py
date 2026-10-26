"""Cached-mask complement on the stronger SAT score-band prefix; val only.

No new fitting or inference. Existing geometry hypotheses are a cached subset.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from pycocotools import mask as mu
from ue_geometry_proposal_complement_20261024 import overlaps

CODE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(CODE/'scripts/probes'))
from ue_v1_adapter import build_coco_gt_and_dt,run_coco_eval

SRC=Path('/data/wangcheng/checkpoint/ue_geometry_score_band_20261024')
OUT=Path('/data/wangcheng/checkpoint/ue_geometry_conditional_complement_20261024')


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def arrays(d):
    x=dict(d);x['bboxes']=np.asarray(x['bboxes'],np.float32).reshape(-1,4)
    x['labels']=np.asarray(x['labels'],np.int64)
    if 'scores' in x:x['scores']=np.asarray(x['scores'],np.float64)
    return x


def trusted_score(s):
    # Strictly increasing for every positive float32 score, numerically stable
    # even for tiny rejected-RoI probabilities; new hypotheses stay below .25.
    x=np.asarray(s,np.float64);out=np.full_like(x,.5);pos=x>0
    out[pos]=.5+.5/(1-np.log(x[pos]))
    return out.tolist()


def main():
    OUT.mkdir(exist_ok=True)
    source=SRC/'satellite/val/records.json';base=json.loads(source.read_text())
    assert len(base)==459
    prefix=[]
    for b in base:
        row=dict(b);row['dt']=dict(b['dt']);row['dt']['scores']=trusted_score(b['dt']['scores']);prefix.append(row)
    old=np.concatenate([np.asarray(b['dt']['scores']) for b in base]);new=np.concatenate([np.asarray(b['dt']['scores']) for b in prefix])
    assert np.array_equal(np.argsort(-old,kind='stable'),np.argsort(-new,kind='stable')),'Prefix rank changed'
    (OUT/'prefix_records.json').write_text(json.dumps(prefix))
    protocol=dict(source_sha256=sha(Path(__file__)),source_records_sha256=sha(source),split='val459',
        baseline='SAT pre2000/post1000 + own rejected RoI supplement, complete trusted prefix',
        score='prefix .5+.5/(1-log(s)), positive s; zero maps .5; extras .25 * own head_confidence',
        admission='Existing cached G/K extras only; reject IoU>0.5 against all trusted SAT detections; fill remaining final250; no new masks or labels for admission',
        caveat='Cached subset may omit lower confidence geometry candidates previously capped; not a full new inference pipeline. AP nondecrease is design property, not content proof.')
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    results={}
    for arm in ['prefix','geometry','constant']:
        records=prefix
        if arm!='prefix':
            candidates=json.loads((SRC/arm/'val/records.json').read_text());records=[]
            for b,c in zip(prefix,candidates):
                assert b['image_id']==c['image_id'] and b['gt']==c['gt']
                start=c['admission']['original'];n=len(b['dt']['scores'])
                extras=np.asarray(c['dt']['bboxes'][start:],float).reshape(-1,4)
                sat=np.asarray(b['dt']['bboxes'],float).reshape(-1,4)
                eligible=overlaps(extras,sat).max(1)<=.5 if len(sat) else np.ones(len(extras),bool)
                ids=np.flatnonzero(eligible)[:max(0,250-n)]+start
                dt=dict(b['dt'])
                for k in ('bboxes','labels','rles'):dt[k]=list(dt[k])+[c['dt'][k][j] for j in ids]
                dt['scores']=list(dt['scores'])+[float(c['dt']['scores'][j])/.05*.25 for j in ids]
                dt['mask_count']=len(dt['scores']);dt['mask_fill_sum']=float(np.sum(mu.area(dt['rles']))/512**2)
                assert len(dt['scores'])<=250 and all(s<=.25+1e-7 for s in dt['scores'][n:])
                records.append(dict(image_id=b['image_id'],gt=b['gt'],dt=dt,admission=dict(prefix=n,cached_extras=len(extras),accepted=len(ids))))
        path=OUT/f'{arm}_records.json';path.write_text(json.dumps(records))
        cg,cd=build_coco_gt_and_dt([arrays(r['gt']) for r in records],[arrays(r['dt']) for r in records],[dict(img_shape=(512,512)) for _ in records],score_key='scores')
        metrics={}
        for kind in ('segm','bbox'):metrics.update(run_coco_eval(cg,cd,kind,max_dets=[1,10,250]))
        (OUT/f'{arm}_metrics.json').write_text(json.dumps(metrics,indent=2));results[arm]=metrics
        if arm=='prefix':
            expected=json.loads((SRC/'satellite/val/metrics.json').read_text())
            for key in ['segm/mAP','bbox/mAP']:assert abs(metrics[key]-expected[key])<1e-12,('Prefix AP changed',key)
        else:
            for key in ['segm/mAP','bbox/mAP']:assert metrics[key]>=results['prefix'][key]-1e-12
        print('METRICS',arm,metrics['segm/mAP'],metrics['bbox/mAP'],flush=True)
    (OUT/'summary.json').write_text(json.dumps(dict(complete=True,metrics=results,images=459,protocol=protocol),indent=2))


if __name__=='__main__':main()
