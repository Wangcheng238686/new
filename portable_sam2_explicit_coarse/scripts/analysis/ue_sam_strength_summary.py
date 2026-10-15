"""Primary half-strength paired comparison; never select best scale post hoc."""
import contextlib
import copy
import io
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
R=Path('/tmp/ue_uav_experiments/20261014_ad_seed42/final_analysis');O=R/'sam_strength_audit';F=O/'full'
def read(p):return json.loads(Path(p).read_text())
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2))
names=['s0','s05','s1'];manifest=read(F/'run_manifest.json');assert manifest['complete'] and manifest['processed_images']==153 and manifest['ground_truth_instances']==4496 and manifest['parameter_versions_unchanged'];ids=manifest['image_ids'];checks=read(F/'invariance_checks.json');assert len(checks)==153
for x in checks:
 assert x['hook_calls']==1 and all(x[k] for k in ['geometry_rgb_unchanged','hook_removed','feature_hashes_unchanged','bbox_scores_exact','original_s1_exact'])
quiet=io.StringIO();gt=read(R/'dual_correct/gt_records.json');images=[dict(id=i,height=512,width=512) for i in ids]
for i,x in enumerate(gt):x.update(id=i+1,area=x['bbox'][2]*x['bbox'][3])
metrics=read(F/'metrics.json');evaluators={};caches={};points={};records={}
for name in names:
 dt=read(F/(name+'_dt_records.json'));records[name]=dt
 if name=='s1':assert dt==read(R/'rgb_content_audit/correct_replay/dt_records.json')
 else:assert [{k:x[k] for k in ['image_id','category_id','bbox','score']} for x in dt]==[{k:x[k] for k in ['image_id','category_id','bbox','score']} for x in read(F/'s1_dt_records.json')]
 with contextlib.redirect_stdout(quiet):
  c=COCO();c.dataset=dict(images=images,annotations=copy.deepcopy(gt),categories=[dict(id=1,name='building')]);c.createIndex();d=c.loadRes(copy.deepcopy(dt));e=COCOeval(c,d,'segm');e.params.imgIds=ids;e.params.maxDets=[100];e.params.areaRng=[[0,1e10]];e.params.areaRngLbl=['all'];e.evaluate();e.accumulate()
 p=e.eval['precision'];points[name]=float(p[p>=0].mean());assert abs(points[name]-metrics[name]['segm/mAP'])<1e-12;evaluators[name]=e;caches[name]=e.evalImgs
 print('point verified',name,points[name],flush=True)
def aggregate(name,index):
 e=evaluators[name];e.evalImgs=[caches[name][i] for i in index];e.params.imgIds=list(range(len(index)));e._paramsEval=copy.deepcopy(e.params)
 with contextlib.redirect_stdout(quiet):e.accumulate()
 p=e.eval['precision'];return float(p[p>=0].mean())
probe=[0,0,2,5,9,9,11,15]
for name in names:
 assert abs(aggregate(name,list(range(153)))-points[name])<1e-12
 g=[];d=[]
 for fresh,i in enumerate(probe):
  for x in gt:
   if x['image_id']==ids[i]:v=copy.deepcopy(x);v.update(image_id=fresh,id=len(g)+1);g.append(v)
  for x in records[name]:
   if x['image_id']==ids[i]:v=copy.deepcopy(x);v.update(image_id=fresh);d.append(v)
 with contextlib.redirect_stdout(quiet):
  c=COCO();c.dataset=dict(images=[dict(id=i,height=512,width=512) for i in range(len(probe))],annotations=g,categories=[dict(id=1,name='building')]);c.createIndex();d=c.loadRes(d);e=COCOeval(c,d,'segm');e.params.maxDets=[100];e.params.areaRng=[[0,1e10]];e.params.areaRngLbl=['all'];e.evaluate();e.accumulate()
 p=e.eval['precision'];assert abs(float(p[p>=0].mean())-aggregate(name,probe))<1e-12
rng=np.random.default_rng(1720);samples=[]
for b in range(500):
 index=rng.integers(0,153,153).tolist();samples.append({n:aggregate(n,index) for n in names})
 if (b+1)%100==0:print('AP bootstrap',b+1,flush=True)
result=dict(scope='Frozen development subset validation only;153 images/4496 appearances/237 buildings',metrics=metrics,primary_comparison='s05 minus s1',sam_alpha=manifest['sam_alpha'],ap_bootstrap=dict(seed=1720,resamples=500,full_point_exact=True,duplicate_image_direct_exact=True,limitations='Exploratory image resampling, ignores spatial dependence/selection/training-seed variance',comparisons={}),mask_fixed_box={},mechanism=dict(residual_norm_ratio_mean=float(np.mean([x['residual_norm_ratio'] for x in checks]))))
for n in ['s0','s05']:
 d=np.array([x[n]-x['s1'] for x in samples]);result['ap_bootstrap']['comparisons'][n+'_minus_s1']=dict(delta=points[n]-points['s1'],percentile95=np.quantile(d,[.025,.975]).tolist())
stage=read(R/'full_stage_audit/dual_correct.json')['instances'];byimage=defaultdict(list)
for x in stage:byimage[x['image_id']].append(x)
identity=[]
for i in ids:identity.extend(sorted(byimage[i],key=lambda x:x['gt_slot']))
counts=np.array([len(byimage[i]) for i in ids]);starts=np.r_[0,np.cumsum(counts)[:-1]];areas=np.array([x['bbox_area'] for x in identity]);buildings=sorted(set(x['global_id'] for x in identity));ix={g:np.array([i for i,x in enumerate(identity) if x['global_id']==g]) for g in buildings};assert len(buildings)==237
values={};hashes={}
for n in names:
 rows=read(F/(n+'_fixed_box.json'));assert [x['image_id'] for x in rows]==ids;parts=[]
 for x in rows:
  iid=x['image_id'];assert len(x['mask_iou'])==len(byimage[iid]);parts.extend(x['mask_iou'])
  if n=='s0':hashes[iid]=x['boxes_sha256']
  else:assert hashes[iid]==x['boxes_sha256']
 values[n]=np.array(parts);assert len(values[n])==4496
result['mask_fixed_box']['means']={n:float(v.mean()) for n,v in values.items()};result['mask_fixed_box']['comparisons']={};rng=np.random.default_rng(1721);draw=rng.integers(0,153,(1000,153));denom=counts[draw].sum(axis=1)
for n in ['s0','s05']:
 delta=values[n]-values['s1'];sums=np.add.reduceat(delta,starts);assert abs(sums.sum()/4496-delta.mean())<1e-12;boot=sums[draw].sum(axis=1)/denom
 result['mask_fixed_box']['comparisons'][n+'_minus_s1']=dict(mean=float(delta.mean()),percentile95=np.quantile(boot,[.025,.975]).tolist(),building_equal_weight_mean=float(np.mean([delta[v].mean() for v in ix.values()])),positive_gt05=int((delta>.05).sum()),negative_gt05=int((delta<-.05).sum()),by_size={size:dict(n=int(sel.sum()),mean=float(delta[sel].mean())) for size,sel in [('small',areas<1024),('medium',(areas>=1024)&(areas<9216)),('large',areas>=9216)]})
result['mask_fixed_box']['bootstrap']=dict(seed=1721,resamples=1000,method='paired image sum/count')
a=result['ap_bootstrap']['comparisons'];m=result['mask_fixed_box']['comparisons'];bbox_equal=all(all(v==metrics['s1'][k] for k,v in metrics[n].items() if k.startswith('bbox/')) for n in names)
result['preregistered_gates']=dict(support_overstrong=a['s05_minus_s1']['delta']>=.005 and a['s05_minus_s1']['percentile95'][0]>0 and m['s05_minus_s1']['mean']>=.002 and bbox_equal,half_near_tie=abs(a['s05_minus_s1']['delta'])<=.002 and abs(m['s05_minus_s1']['mean'])<=.002,original_better_than_both_reductions=all(a[n+'_minus_s1']['delta']<=-.005 and m[n+'_minus_s1']['mean']<=-.002 for n in ['s0','s05']),bbox_exact=bbox_equal)
dump(O/'summary.json',result);dump(O/'bootstrap_samples.json',samples);print(json.dumps(result,indent=2),flush=True)
