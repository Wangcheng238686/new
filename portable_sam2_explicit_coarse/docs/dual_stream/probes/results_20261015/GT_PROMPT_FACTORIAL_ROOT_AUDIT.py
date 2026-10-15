"""Root independently reaggregates raw four-cell results; no summary imports."""
from pathlib import Path
from collections import defaultdict
from statistics import fmean
import json
import numpy as np
p=Path('/tmp/ue_uav_experiments/20261014_ad_seed42/final_analysis/gt_prompt_factorial')
s=json.loads((p/'summary.json').read_text());m=json.loads((p/'full/run_manifest.json').read_text());names=['baseline','gt_point_only','gt_dense_only','joint_gt_prompt'];rows={k:json.loads((p/'full'/(k+'.json')).read_text()) for k in names}
assert m['complete'] and all(len(v)==4496 for v in rows.values())
b=rows['baseline'];ids=m['image_ids'];byimg=defaultdict(list);bybuilding=defaultdict(list)
for j,x in enumerate(b):
 byimg[x['image_id']].append(j);bybuilding[x['global_id']].append(j)
 for v in rows.values():assert (v[j]['image_id'],v[j]['gt_slot'],v[j]['global_id'])==(x['image_id'],x['gt_slot'],x['global_id'])
assert len(byimg)==153 and len(bybuilding)==237
rng=np.random.default_rng(1770);draws=rng.integers(0,153,(1000,153))
sel={'all':lambda r:True,'small':lambda r:r['bbox_area']<1024,'medium':lambda r:1024<=r['bbox_area']<9216,'large':lambda r:r['bbox_area']>=9216,'P2':lambda r:r['roi_level']==0,'nonP2':lambda r:r['roi_level']!=0,'dense_support_zero':lambda r:r['dense_support_cells']==0,'dense_support_nonzero':lambda r:r['dense_support_cells']>0}
for mode,v in rows.items():
 for group,test in sel.items():
  rr=[r for r in v if test(r)];z=s['groups'][mode][group];assert len(rr)==z['n'] and abs(fmean(r['gt_iou'] for r in rr)-z['final_iou'])<1e-12
  assert sum(r['paste_xor_pixels']>0 for r in rr)==z['paste_changed_rois']
  for i,slot in enumerate(['p1','p2','n1','n2']):assert sum(r['point_valid'][i] for r in rr)==z['point_slots'][slot]['valid'] and sum(r['point_correct'][i] for r in rr)==z['point_slots'][slot]['correct']
coeffs={'gt_point_only_minus_baseline':[-1,1,0,0],'gt_dense_only_minus_baseline':[-1,0,1,0],'joint_minus_baseline':[-1,0,0,1],'interaction_GG_minus_GP_minus_PG_plus_PP':[1,-1,-1,1],'points_given_GT_dense_GG_minus_PG':[0,0,-1,1],'dense_given_GT_points_GG_minus_GP':[0,-1,0,1]};out={}
for key,c in coeffs.items():
 delta=[sum(c[k]*rows[name][j]['gt_iou'] for k,name in enumerate(names)) for j in range(4496)];primary=key in s['primary_contrasts'];z=(s['primary_contrasts'] if primary else s['secondary_descriptive'])[key];mean=fmean(delta);assert abs(mean-z['mean'])<1e-12
 sums=[sum(delta[j] for j in byimg[i]) for i in ids];counts=[len(byimg[i]) for i in ids];boot=[sum(sums[k] for k in draw)/sum(counts[k] for k in draw) for draw in draws];ci=np.quantile(boot,[.0125,.9875] if primary else [.025,.975]);assert np.allclose(ci,z['image_paired_interval'],rtol=0,atol=1e-12)
 be=fmean(fmean(delta[j] for j in jj) for jj in bybuilding.values());assert abs(be-z['building_equal_weight_mean'])<1e-12
 for group,test in sel.items():
  dd=[d for d,r in zip(delta,b) if test(r)];g=z['by_group'][group];assert len(dd)==g['n'] and abs(fmean(dd)-g['mean'])<1e-12
  assert sum(d>.002 for d in dd)==g['improved_gt002'] and sum(d<-.002 for d in dd)==g['worse_gt002']
 if primary:assert z['gate']['passed']==bool(mean>=.005 and ci[0]>0)
 out[key]={'mean':mean,'interval':ci.tolist(),'building_equal_mean':be}
zero=[j for j,r in enumerate(b) if r['dense_support_cells']==0];assert len(zero)==197
for j in zero:
 assert rows['gt_dense_only'][j]['paste_xor_pixels']==0 and rows['gt_dense_only'][j]['gt_iou']==b[j]['gt_iou']
 assert rows['joint_gt_prompt'][j]['gt_iou']==rows['gt_point_only'][j]['gt_iou']
 assert rows['joint_gt_prompt'][j]['actual_dense_sha256']==rows['gt_point_only'][j]['actual_dense_sha256']
 assert rows['joint_gt_prompt'][j]['actual_sparse_sha256']==rows['gt_point_only'][j]['actual_sparse_sha256']
print(json.dumps({'verdict':'PASS','contrasts':out,'support_zero_exact_rois':len(zero)},indent=2))
