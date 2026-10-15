"""Prespecified quarter-strength P2 transfer: conditional AP and fixed-GT masks."""
import contextlib
import copy
import io
import json
from pathlib import Path
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
R=Path('/tmp/ue_uav_experiments/20261014_ad_seed42/final_analysis');O=R/'p2_transfer_audit';D=O/'full'
def read(p):return json.loads(Path(p).read_text())
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2))
m=read(D/'run_manifest.json');assert m['complete'] and m['processed_images']==153 and m['ground_truth_instances']==4496 and m['parameter_versions_unchanged'];ids=m['image_ids'];names=m['variants'];checks=read(D/'invariance_checks.json');assert len(checks)==153 and [x['image_id'] for x in checks]==ids
for x in checks:
 assert x['bev_hook_calls']==x['p3_hook_calls']==2 and all(x[k] for k in ['hooks_removed','geometry_rgb_restored','feature_hashes_unchanged','only_p2_changed','bbox_scores_exact','baseline_exact','nonp2_gt_masks_exact'])
metrics=read(D/'metrics.json');reference=read(R/'rgb_content_audit/correct_replay/dt_records.json');records={n:read(D/(n+'_dt_records.json')) for n in names};assert records['baseline']==reference
for n in names:
 assert [{k:x[k] for k in ['image_id','category_id','bbox','score']} for x in records[n]]==[{k:x[k] for k in ['image_id','category_id','bbox','score']} for x in reference]
quiet=io.StringIO();gt=read(R/'dual_correct/gt_records.json');images=[dict(id=i,height=512,width=512) for i in ids]
for i,x in enumerate(gt):x.update(id=i+1,area=x['bbox'][2]*x['bbox'][3])
points={};evaluators={};caches={}
for n in names:
 with contextlib.redirect_stdout(quiet):
  c=COCO();c.dataset=dict(images=images,annotations=copy.deepcopy(gt),categories=[dict(id=1,name='building')]);c.createIndex();dt=c.loadRes(copy.deepcopy(records[n]));e=COCOeval(c,dt,'segm');e.params.imgIds=ids;e.params.maxDets=[100];e.params.areaRng=[[0,1e10]];e.params.areaRngLbl=['all'];e.evaluate();e.accumulate()
 p=e.eval['precision'];points[n]=float(p[p>=0].mean());assert abs(points[n]-metrics[n]['segm/mAP'])<1e-12;evaluators[n]=e;caches[n]=e.evalImgs
 print('conditional AP point',n,points[n],flush=True)
def aggregate(n,index):
 e=evaluators[n];e.evalImgs=[caches[n][i] for i in index];e.params.imgIds=list(range(len(index)));e._paramsEval=copy.deepcopy(e.params)
 with contextlib.redirect_stdout(quiet):e.accumulate()
 p=e.eval['precision'];return float(p[p>=0].mean())
probe=[0,0,2,5,9,9,11,15]
for n in names:
 assert abs(aggregate(n,list(range(153)))-points[n])<1e-12;gg=[];dd=[]
 for fresh,j in enumerate(probe):
  for x in gt:
   if x['image_id']==ids[j]:v=copy.deepcopy(x);v.update(image_id=fresh,id=len(gg)+1);gg.append(v)
  for x in records[n]:
   if x['image_id']==ids[j]:v=copy.deepcopy(x);v.update(image_id=fresh);dd.append(v)
 with contextlib.redirect_stdout(quiet):
  c=COCO();c.dataset=dict(images=[dict(id=i,height=512,width=512) for i in range(len(probe))],annotations=gg,categories=[dict(id=1,name='building')]);c.createIndex();dt=c.loadRes(dd);e=COCOeval(c,dt,'segm');e.params.maxDets=[100];e.params.areaRng=[[0,1e10]];e.params.areaRngLbl=['all'];e.evaluate();e.accumulate()
 p=e.eval['precision'];assert abs(float(p[p>=0].mean())-aggregate(n,probe))<1e-12
rng=np.random.default_rng(1741);samples=[]
for b in range(500):
 index=rng.integers(0,153,153).tolist();samples.append({n:aggregate(n,index) for n in names})
 if (b+1)%100==0:print('AP bootstrap',b+1,flush=True)
identity=read(D/'gt_identities.json');assert len(identity)==4496;counts=np.array([sum(x['image_id']==i for x in identity) for i in ids]);assert counts.sum()==4496;starts=np.r_[0,np.cumsum(counts)[:-1]];assert [identity[s]['image_id'] for s in starts]==ids
buildings=sorted(set(x['global_id'] for x in identity));assert len(buildings)==237;ix={g:np.array([i for i,x in enumerate(identity) if x['global_id']==g]) for g in buildings};areas=np.array([x['bbox_area'] for x in identity]);levels=np.array([x['roi_level'] for x in identity]);assert np.bincount(levels,minlength=4).tolist()==[3657,839,0,0]
values={};bands={};hashes={}
for n in names:
 rows=read(D/(n+'_fixed_box.json'));assert [x['image_id'] for x in rows]==ids;vv=[];bb=[];ll=[]
 for row in rows:
  iid=row['image_id'];assert len(row['mask_iou'])==len(row['boundary_band_iou'])==len(row['roi_levels'])==counts[ids.index(iid)];vv.extend(row['mask_iou']);bb.extend(row['boundary_band_iou']);ll.extend(row['roi_levels'])
  if n=='baseline':hashes[iid]=row['boxes_sha256']
  else:assert hashes[iid]==row['boxes_sha256']
 values[n]=np.array(vv);bands[n]=np.array(bb);assert np.array_equal(np.array(ll),levels)
 if n!='baseline':assert np.array_equal(values[n][levels!=0],values['baseline'][levels!=0]) and np.array_equal(bands[n][levels!=0],bands['baseline'][levels!=0])
contrasts=[(n,'baseline') for n in names if n!='baseline']+[('native_correct_q','native_donor_q'),('native_correct_q','upsample_correct_q')]
rng=np.random.default_rng(1740);draw=rng.integers(0,153,(1000,153));denom=counts[draw].sum(axis=1);effects={}
for a,b in contrasts:
 d=values[a]-values[b];sums=np.add.reduceat(d,starts);assert abs(sums.sum()/4496-d.mean())<1e-12;boot=sums[draw].sum(axis=1)/denom;ap=np.array([v[a]-v[b] for v in samples]);key=a+'_minus_'+b
 effects[key]=dict(gt_iou_delta=float(d.mean()),gt_iou_percentile95=np.quantile(boot,[.025,.975]).tolist(),building_equal_weight_mean=float(np.mean([d[v].mean() for v in ix.values()])),conditional_ap_delta=points[a]-points[b],conditional_ap_percentile95=np.quantile(ap,[.025,.975]).tolist(),boundary_band_iou_delta=float((bands[a]-bands[b]).mean()),positive_gt05=int((d>.05).sum()),negative_gt05=int((d<-.05).sum()),by_size={n:dict(n=int(sel.sum()),mean=float(d[sel].mean())) for n,sel in [('small',areas<1024),('medium',(areas>=1024)&(areas<9216)),('large',areas>=9216)]},by_roi_level={str(level):dict(n=int((levels==level).sum()),mean=float(d[levels==level].mean())) for level in np.unique(levels)})
def gate(key,ap_min,iou_min):
 e=effects[key];return bool(e['conditional_ap_delta']>=ap_min and e['gt_iou_delta']>=iou_min and e['gt_iou_percentile95'][0]>0)
e=effects['native_correct_q_minus_baseline'];main=gate('native_correct_q_minus_baseline',.005,.002) and e['conditional_ap_percentile95'][0]>0
mechanism={}
for n in names:
 v=np.array([x['residual_norm_ratio'][n] for x in checks]);w=np.array([x['supported_rms_ratio'][n] for x in checks]);assert np.isfinite(v).all() and np.isfinite(w).all();mechanism[n]=dict(norm_ratio_mean=float(v.mean()),norm_ratio_max=float(v.max()),supported_rms_ratio_mean=float(w.mean()))
result=dict(scope='Frozen development subset validation only,153images/4496 appearances/237buildings',protocol=m['protocol'],selection=m['primary'],metrics=metrics,gt_mask_iou_means={n:float(v.mean()) for n,v in values.items()},boundary_band_iou_means={n:float(v.mean()) for n,v in bands.items()},effects=effects,roi_counts=dict(P2=3657,P3=839,P4=0,P5=0),bootstrap=dict(iou_seed=1740,iou_resamples=1000,conditional_ap_seed=1741,conditional_ap_resamples=500,full_point_exact=True,duplicate_image_direct_exact=True,limitation='Exploratory paired images; ignores spatial/model-selection/seed variance; fixed boxes condition away detection changes'),mechanism=dict(p3_alpha=m['p3_alpha'],residuals=mechanism),preregistered_gates=dict(main_transfer_support=bool(main),paired_rgb_support=gate('native_correct_q_minus_native_donor_q',.002,.001),native_grid_support=gate('native_correct_q_minus_upsample_correct_q',.002,.001),run_full_modified_detection_next=bool(main)),limits='Frozen P3-fuser cross-scale transfer, not trained P2. Negative result does not reject P2 training. donor RGB-D mismatch; upsampling comparison not pure resolution ablation.')
dump(O/'summary.json',result);dump(O/'bootstrap_samples.json',samples);print(json.dumps(result,indent=2),flush=True)
