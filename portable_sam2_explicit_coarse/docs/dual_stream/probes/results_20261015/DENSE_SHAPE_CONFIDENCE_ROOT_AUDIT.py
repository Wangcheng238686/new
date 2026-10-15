from pathlib import Path
from statistics import fmean
from collections import defaultdict
import json,numpy as np
p=Path('/tmp/ue_uav_experiments/20261014_ad_seed42/final_analysis/dense_shape_confidence');m=json.loads((p/'full/run_manifest.json').read_text());s=json.loads((p/'summary.json').read_text());names=['baseline','pred_binary_pm4','gt_binary_pm4'];r={k:json.loads((p/'full'/(k+'.json')).read_text()) for k in names};b=r['baseline'];assert m['complete'] and len(b)==4496
ix=defaultdict(list);bg=defaultdict(list)
for j,z in enumerate(b):ix[z['image_id']].append(j);bg[z['global_id']].append(j)
ids=m['image_ids'];draw=np.random.default_rng(1780).integers(0,153,(1000,153));out={}
for key,lhs,rhs,primary in [('pred_binary_pm4_minus_baseline','pred_binary_pm4','baseline',True),('gt_binary_pm4_minus_pred_binary_pm4','gt_binary_pm4','pred_binary_pm4',True),('secondary','gt_binary_pm4','baseline',False)]:
 d=[x['gt_iou']-y['gt_iou'] for x,y in zip(r[lhs],r[rhs])];z=s['primary_contrasts'][key] if primary else s['secondary_gt_minus_baseline'];mean=fmean(d);assert abs(mean-z['mean'])<1e-12
 sums=[sum(d[j] for j in ix[i]) for i in ids];counts=[len(ix[i]) for i in ids];boot=[sum(sums[k] for k in a)/sum(counts[k] for k in a) for a in draw];ci=np.quantile(boot,[.0125,.9875] if primary else [.025,.975]);assert np.allclose(ci,z['image_paired_interval'],rtol=0,atol=1e-12);be=fmean(fmean(d[j] for j in v) for v in bg.values());assert abs(be-z['building_equal_mean'])<1e-12
 if primary:assert z['gate']['status']==('PASS' if mean>=.005 and ci[0]>0 else 'FAIL')
 out[key]={'mean':mean,'interval':ci.tolist(),'building_equal_mean':be}
for n,rr in r.items():
 assert len(rr)==4496
 for x,y in zip(rr,b):
  assert (x['image_id'],x['gt_slot'],x['global_id'])==(y['image_id'],y['gt_slot'],y['global_id'])
  assert x['actual_sparse_sha256']==y['actual_sparse_sha256']
  if x['dense_support_cells']==0:assert x['gt_iou']==y['gt_iou'] and x['paste_xor_pixels']==0
 assert abs(fmean(x['gt_iou'] for x in rr)-s['groups'][n]['all']['final_iou'])<1e-12
assert sum(x['dense_support_cells']==0 for x in b)==197
print(json.dumps({'verdict':'PASS','contrasts':out,'sourcepoints_original':True,'support_zero_exact':197},indent=2))
