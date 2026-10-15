"""Paired content sensitivity, primary AP bootstrap plus fixed-box decomposition."""
import contextlib
import copy
import io
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
R=Path('/tmp/ue_uav_experiments/20261014_ad_seed42/final_analysis');O=R/'rgb_content_audit'
def read(p):return json.loads(Path(p).read_text())
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2))
names=['correct_replay','donor_a','donor_b'];ids=read(O/'donor_maps.json')['image_ids'];quiet=io.StringIO()
gt=read(R/'dual_correct/gt_records.json')
for i,x in enumerate(gt):x.update(id=i+1,area=x['bbox'][2]*x['bbox'][3])
images=[dict(id=i,height=512,width=512) for i in ids];evaluators={};caches={};points={};metrics={}
invariant_references={}
for name in names:
 manifest=read(O/name/'run_manifest.json');assert manifest['complete'] and manifest['image_ids']==ids and manifest['ground_truth_instances']==4496
 checks=read(O/name/'invariance_checks.json');assert len(checks)==153 and all(x['geometry_unchanged'] for x in checks)
 if name=='correct_replay':
  assert all(x['full_prediction_exact'] for x in checks)
  invariant_references={x['image_id']:(x['geometry_hashes'],x['satellite_sha256'],x['target_sha256'],x['original_rgb_sha256']) for x in checks}
 else:
  for x in checks:assert invariant_references[x['image_id']]==(x['geometry_hashes'],x['satellite_sha256'],x['target_sha256'],x['original_rgb_sha256'])
 assert manifest['parameter_versions_unchanged'] and 'git_dirty' in manifest
 metrics[name]=read(O/name/'metrics.json')
 for kind in ['segm','bbox']:
  with contextlib.redirect_stdout(quiet):
   c=COCO();c.dataset=dict(images=images,annotations=copy.deepcopy(gt),categories=[dict(id=1,name='building')]);c.createIndex()
   dt=c.loadRes(copy.deepcopy(read(O/name/'dt_records.json')));e=COCOeval(c,dt,kind);e.params.imgIds=ids;e.params.maxDets=[100];e.params.areaRng=[[0,1e10]];e.params.areaRngLbl=['all'];e.evaluate();e.accumulate()
  p=e.eval['precision'];v=float(p[p>=0].mean());assert abs(v-metrics[name][kind+'/mAP'])<1e-12
  key=(name,kind);evaluators[key]=e;caches[key]=e.evalImgs;points[key]=v
 print('point checks',name,flush=True)
def aggregate(key,indices):
 e=evaluators[key];e.evalImgs=[caches[key][i] for i in indices];e.params.imgIds=list(range(len(indices)));e._paramsEval=copy.deepcopy(e.params)
 with contextlib.redirect_stdout(quiet):e.accumulate()
 p=e.eval['precision'];return float(p[p>=0].mean())
for key in points:assert abs(aggregate(key,list(range(len(ids))))-points[key])<1e-12
# Check duplicate-image cached aggregation against a conventional COCO recomputation.
probe=[0,0,2,5,9,9,11,15]
for name in names:
 originaldt=read(O/name/'dt_records.json');g=[];d=[]
 for fresh,i in enumerate(probe):
  for x in gt:
   if x['image_id']==ids[i]:v=copy.deepcopy(x);v.update(image_id=fresh,id=len(g)+1);g.append(v)
  for x in originaldt:
   if x['image_id']==ids[i]:v=copy.deepcopy(x);v.update(image_id=fresh);d.append(v)
 for kind in ['segm','bbox']:
  with contextlib.redirect_stdout(quiet):
   c=COCO();c.dataset=dict(images=[dict(id=i,height=512,width=512) for i in range(len(probe))],annotations=copy.deepcopy(g),categories=[dict(id=1,name='building')]);c.createIndex();dt=c.loadRes(copy.deepcopy(d));e=COCOeval(c,dt,kind);e.params.maxDets=[100];e.params.areaRng=[[0,1e10]];e.params.areaRngLbl=['all'];e.evaluate();e.accumulate()
  p=e.eval['precision'];assert abs(float(p[p>=0].mean())-aggregate((name,kind),probe))<1e-12
rng=np.random.default_rng(1703);samples=[]
for b in range(500):
 indices=rng.integers(0,len(ids),len(ids)).tolist();samples.append({name:{k:aggregate((name,k),indices) for k in ['segm','bbox']} for name in names})
 if (b+1)%50==0:print('bootstrap',b+1,flush=True)
summary=dict(scope='Current frozen development subset validation: 153 images,4496 repeated building appearances,237 global buildings; not test/full future dataset',metrics=metrics,bootstrap=dict(seed=1703,resamples=500,method='paired image percentile; exploratory, ignores spatial correlation/model selection/seed variance',full_point_exact=True,duplicate_image_direct_exact=True,comparisons={}),fixed_box={})
for donor in ['donor_a','donor_b']:
 comparison={}
 for kind in ['segm','bbox']:
  v=np.array([x['correct_replay'][kind]-x[donor][kind] for x in samples]);comparison[kind]=dict(correct_minus_donor=points['correct_replay',kind]-points[donor,kind],percentile95=np.quantile(v,[.025,.975]).tolist())
 summary['bootstrap']['comparisons'][donor]=comparison
stage=read(R/'full_stage_audit/dual_correct.json')['instances'];byimage=defaultdict(list)
for x in stage:byimage[x['image_id']].append(x)
identity=[]
for iid in ids:identity.extend(sorted(byimage[iid],key=lambda x:x['gt_slot']))
assert len(identity)==4496
fixed={}
for name in names:
 fixed[name]={(x['image_id'],x['box_type']):x for x in read(O/name/'fixed_box.json')}
for protocol in ['gt','single_pred','dual_pred']:
 values={}
 for name in names:
  chunks=[]
  for iid in ids:
   x=fixed[name][iid,protocol];assert x['boxes_sha256']==fixed['correct_replay'][iid,protocol]['boxes_sha256'];assert len(x['mask_iou'])==len(byimage[iid]);chunks.extend(x['mask_iou'])
  values[name]=np.array(chunks)
 result=dict(means={name:float(v.mean()) for name,v in values.items()},correct_minus_donor={},by_size={})
 for donor in ['donor_a','donor_b']:
  delta=values['correct_replay']-values[donor];group=defaultdict(list)
  for x,v in zip(identity,delta):group[x['global_id']].append(float(v))
  result['correct_minus_donor'][donor]=dict(mean=float(delta.mean()),median=float(np.median(delta)),positive_gt05=int((delta>.05).sum()),negative_gt05=int((delta<-.05).sum()),building_equal_weight_mean=float(np.mean([np.mean(v) for v in group.values()])))
 areas=np.array([x['bbox_area'] for x in identity])
 for size,sel in [('small',areas<32**2),('medium',(areas>=32**2)&(areas<96**2)),('large',areas>=96**2)]:
  result['by_size'][size]=dict(n=int(sel.sum()),means={name:float(v[sel].mean()) for name,v in values.items()},correct_minus_donor={name:float((values['correct_replay']-values[name])[sel].mean()) for name in ['donor_a','donor_b']})
 summary['fixed_box'][protocol]=result
ap_gate=all(summary['bootstrap']['comparisons'][d]['segm']['correct_minus_donor']>=.005 and summary['bootstrap']['comparisons'][d]['segm']['percentile95'][0]>0 for d in ['donor_a','donor_b'])
mask_gate=all(summary['fixed_box'][p]['correct_minus_donor'][d]['mean']>=.005 for d in ['donor_a','donor_b'] for p in ['gt','single_pred','dual_pred'])
weak_gate=all(abs(summary['bootstrap']['comparisons'][d]['segm']['correct_minus_donor'])<=.002 for d in ['donor_a','donor_b']) and all(abs(summary['fixed_box'][p]['correct_minus_donor'][d]['mean'])<=.002 for d in ['donor_a','donor_b'] for p in ['gt','single_pred','dual_pred'])
summary['preregistered_gates']=dict(stable_paired_rgb_content_sensitivity=ap_gate,fixed_box_mask_sensitivity=mask_gate,weak_content_sensitivity=weak_gate,interpretation='Input sensitivity under RGB-depth inconsistency; not net benefit or proof of useful novel information')
dump(O/'summary.json',summary);dump(O/'bootstrap_samples.json',samples)
print(json.dumps(summary,indent=2),flush=True)
