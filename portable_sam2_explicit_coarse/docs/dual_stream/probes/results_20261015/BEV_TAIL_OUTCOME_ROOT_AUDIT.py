import json,hashlib
from pathlib import Path
import numpy as np
R=Path('/tmp/ue_uav_experiments/20261014_ad_seed42/final_analysis/full_fixed_box_audit');ids=[1218,950,306,1216,948,505]
a={}
for arm in ['dual_correct','single_best']:
 p=R/(arm+'.json');d=json.loads(p.read_text());assert d['weights']=='ema' and d['strict'] and not d['training'] and not d['optimizer_created']
 a[arm]={r['image_id']:r for r in d['records'] if r['box_type']=='gt' and r['image_id'] in ids}
 assert set(a[arm])==set(ids)
rows=[]
for iid in ids:
 d=a['dual_correct'][iid];s=a['single_best'][iid]
 assert d['boxes_sha256']==s['boxes_sha256'] and d['boxes_xyxy']==s['boxes_xyxy']
 di=np.asarray(d['mask_iou']);si=np.asarray(s['mask_iou']);assert len(di)==len(si)==len(d['masks'])==len(s['masks'])
 rows.append(dict(image_id=iid,n=len(di),dual=float(di.mean()),single=float(si.mean()),degradation_single_minus_dual=float((si-di).mean())))
out=dict(validation='PASS',scope='Independent existing fixed-GT-box outcome identities and means only; no spatial/tail relation',per_image=rows,n=sum(x['n'] for x in rows),image_equal_degradation=float(np.mean([x['degradation_single_minus_dual'] for x in rows])))
Path('/tmp/ue_bev_tail_association/root_outcome_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
