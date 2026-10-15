"""Summarize frozen 2x2 route patching; paired instance mean via image sum/count."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
R=Path('/tmp/ue_uav_experiments/20261014_ad_seed42/final_analysis');O=R/'fusion_route_audit'
def read(p):return json.loads(Path(p).read_text())
names=['CC','CD','DC','DD'];donors=['donor_a','donor_b'];ids=read(R/'rgb_content_audit/donor_maps.json')['image_ids'];stage=read(R/'full_stage_audit/dual_correct.json')['instances'];byimage=defaultdict(list)
for x in stage:byimage[x['image_id']].append(x)
identity=[]
for iid in ids:identity.extend(sorted(byimage[iid],key=lambda x:x['gt_slot']))
assert len(identity)==4496
counts=np.array([len(byimage[i]) for i in ids]);starts=np.r_[0,np.cumsum(counts)[:-1]];areas=np.array([x['bbox_area'] for x in identity]);buildings=sorted(set(x['global_id'] for x in identity));building_indices={g:np.array([i for i,x in enumerate(identity) if x['global_id']==g]) for g in buildings}
rng=np.random.default_rng(1710);draw=rng.integers(0,153,size=(1000,153));denominator=counts[draw].sum(axis=1)
coefficients=dict(sam_when_fpn_correct=[1,-1,0,0],sam_when_fpn_donor=[0,0,1,-1],fpn_when_sam_correct=[1,0,-1,0],fpn_when_sam_donor=[0,1,0,-1],interaction=[1,-1,-1,1])
result=dict(scope='Frozen development subset validation:153 images,4496 appearances,237 buildings; not independent test/full future dataset',primary='GT common-box mask IoU',bootstrap=dict(seed=1710,resamples=1000,method='paired image sums/counts; exploratory, spatial/model-selection/seed uncertainty not covered'),donors={})
for donor in donors:
 m=read(O/donor/'run_manifest.json');assert m['complete'] and m['image_ids']==ids and m['ground_truth_instances']==4496 and m['parameter_versions_unchanged'];checks=read(O/donor/'invariance_checks.json');assert len(checks)==153 and all(all(x[k] for k in ['nonrgb_geometry_unchanged','pe_high_identical','unfused_fpn_identical','cc_exact','dd_exact','feature_hashes_unchanged']) for x in checks)
 values=[];hashes={}
 for name in names:
  rows=read(O/donor/(name+'_fixed_box.json'));assert [x['image_id'] for x in rows]==ids;chunks=[]
  for row in rows:
   iid=row['image_id'];assert len(row['mask_iou'])==len(byimage[iid]);chunks.extend(row['mask_iou'])
   if name=='CC':hashes[iid]=row['boxes_sha256']
   else:assert hashes[iid]==row['boxes_sha256']
  values.append(chunks)
 v=np.array(values);assert v.shape==(4,4496)
 means=v.mean(axis=1);effects={}
 for effect,coef in coefficients.items():
  delta=np.array(coef)@v;perimage=np.add.reduceat(delta,starts);boot=perimage[draw].sum(axis=1)/denominator
  assert abs(perimage.sum()/counts.sum()-delta.mean())<1e-12
  effects[effect]=dict(mean=float(delta.mean()),percentile95=np.quantile(boot,[.025,.975]).tolist(),building_equal_weight_mean=float(np.mean([delta[ix].mean() for ix in building_indices.values()])),positive_gt05=int((delta>.05).sum()),negative_gt05=int((delta<-.05).sum()),by_size={size:dict(n=int(sel.sum()),mean=float(delta[sel].mean())) for size,sel in [('small',areas<1024),('medium',(areas>=1024)&(areas<9216)),('large',areas>=9216)]})
 dt={n:read(O/donor/(n+'_dt_records.json')) for n in names}
 for a,b in [('CC','CD'),('DC','DD')]:
  assert [{k:x[k] for k in ('image_id','category_id','bbox','score')} for x in dt[a]]==[{k:x[k] for k in ('image_id','category_id','bbox','score')} for x in dt[b]]
 metrics=read(O/donor/'metrics.json')
 result['donors'][donor]=dict(mask_iou_means=dict(zip(names,means.tolist())),metrics=metrics,effects=effects)
fpn=[result['donors'][d]['effects'][e]['mean'] for d in donors for e in ['fpn_when_sam_correct','fpn_when_sam_donor']];sam=[result['donors'][d]['effects'][e]['mean'] for d in donors for e in ['sam_when_fpn_correct','sam_when_fpn_donor']];interaction=[result['donors'][d]['effects']['interaction']['mean'] for d in donors]
result['preregistered_gates']=dict(mainly_sam=all(x>=.005 for x in sam) and all(abs(x)<=.002 for x in fpn),mainly_fpn=all(x>=.005 for x in fpn) and all(abs(x)<=.002 for x in sam),fpn_uniform_negative=all(x<=-.005 for x in fpn),sam_uniform_negative=all(x<=-.005 for x in sam),interaction_sensitive=all(abs(x)>=.005 for x in interaction) and interaction[0]*interaction[1]>0,limitation='Conditional intervention sensitivity, not additive net contribution or best trained route')
(O/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
