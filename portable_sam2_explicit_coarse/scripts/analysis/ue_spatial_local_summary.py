"""Paired translation controls plus descriptive local geometry/quality strata."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
R=Path('/tmp/ue_uav_experiments/20261014_ad_seed42/final_analysis');O=R/'spatial_local_audit';F=O/'full'
def read(p):return json.loads(Path(p).read_text())
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2))
manifest=read(F/'run_manifest.json');assert manifest['complete'] and manifest['processed_images']==153 and manifest['ground_truth_instances']==4496 and manifest['parameter_versions_unchanged'];ids=manifest['image_ids'];names=manifest['variants'];directions=names[1:5]
checks=read(F/'invariance_checks.json');assert len(checks)==153 and [x['image_id'] for x in checks]==ids
for x in checks:assert all(x[k] for k in ['identity_exact','bbox_scores_exact','geometry_rgb_unchanged','feature_hashes_unchanged']) and x['hook_calls']==1
for x in manifest['synthetic_checks'].values():assert x
metrics=read(F/'metrics.json');records={n:read(F/(n+'_dt_records.json')) for n in names};reference=read(R/'rgb_content_audit/correct_replay/dt_records.json');assert records['identity']==reference
for n in names:assert [{k:x[k] for k in ['image_id','category_id','bbox','score']} for x in records[n]]==[{k:x[k] for k in ['image_id','category_id','bbox','score']} for x in reference]
geo=read(F/'local_geometry.json');assert len(geo)==4496;byimage=defaultdict(list)
for x in geo:byimage[x['image_id']].append(x)
identity=[]
for i in ids:
 rows=byimage[i];assert [x['gt_slot'] for x in rows]==list(range(len(rows)));identity.extend(rows)
assert identity==geo
counts=np.array([len(byimage[i]) for i in ids]);starts=np.r_[0,np.cumsum(counts)[:-1]];areas=np.array([x['bbox_area'] for x in geo]);buildings=sorted(set(x['global_id'] for x in geo));assert len(buildings)==237;ix={g:np.array([i for i,x in enumerate(geo) if x['global_id']==g]) for g in buildings}
values={};local={};hashes={}
for n in names:
 rows=read(F/(n+'_fixed_box.json'));assert [x['image_id'] for x in rows]==ids;parts=[];localparts=[]
 for row in rows:
  iid=row['image_id'];assert len(row['mask_iou'])==len(byimage[iid])==len(row['local_metrics']);parts.extend(row['mask_iou']);localparts.extend(row['local_metrics'])
  if n=='identity':hashes[iid]=row['boxes_sha256']
  else:assert hashes[iid]==row['boxes_sha256']
 values[n]=np.array(parts);local[n]=localparts
oldgt={x['image_id']:x for x in read(R/'full_fixed_box_audit/dual_correct.json')['records'] if x['box_type']=='gt'}
for row in read(F/'identity_fixed_box.json'):assert row['mask_iou']==oldgt[row['image_id']]['mask_iou'] and row['boxes_sha256']==oldgt[row['image_id']]['boxes_sha256']
contrasts=[(n,b) for b in ['identity','smooth_symmetric'] for n in directions];delta=np.array([values[n]-values[b] for n,b in contrasts]);point=delta.mean(axis=1);sums=np.add.reduceat(delta,starts,axis=1);assert np.allclose(sums.sum(axis=1)/4496,point,atol=1e-14,rtol=0);rng=np.random.default_rng(1730);draw=rng.integers(0,153,(1000,153));denom=counts[draw].sum(axis=1);boot=np.stack([v[draw].sum(axis=1)/denom for v in sums]);radius=float(np.quantile(np.max(np.abs(boot-point[:,None]),axis=0),.95))
effects={}
for k,(n,b) in enumerate(contrasts):
 d=delta[k];effects[n+'_minus_'+b]=dict(mean=float(point[k]),simultaneous95=[float(point[k]-radius),float(point[k]+radius)],building_equal_weight_mean=float(np.mean([d[v].mean() for v in ix.values()])),positive_gt05=int((d>.05).sum()),negative_gt05=int((d<-.05).sum()))
smooth=values['smooth_symmetric']-values['identity'];ss=np.add.reduceat(smooth,starts);sb=ss[draw].sum(axis=1)/denom
effects['smooth_symmetric_minus_identity']=dict(mean=float(smooth.mean()),percentile95=np.quantile(sb,[.025,.975]).tolist(),building_equal_weight_mean=float(np.mean([smooth[v].mean() for v in ix.values()])))
localmeans={}
for n in names:
 localmeans[n]={}
 for key in ['boundary_band_iou','inner_fn_rate','far_fp_per_gt_area']:
  v=[x[key] for x in local[n] if x[key] is not None];localmeans[n][key]=dict(mean=float(np.mean(v)),n=len(v))
# Join prior half-strength probe by the same image/GT order, never select matches anew.
strength={n:{x['image_id']:x for x in read(R/'sam_strength_audit/full'/(n+'_fixed_box.json'))} for n in ['s05','s1']};half=[]
for iid in ids:
 assert strength['s1'][iid]['boxes_sha256']==hashes[iid];half.extend(np.array(strength['s05'][iid]['mask_iou'])-np.array(strength['s1'][iid]['mask_iou']))
half=np.array(half);assert len(half)==4496
support=np.array([x['fine_support'] for x in geo]);clipped=np.array([x['clipped'] for x in geo]);baseline_boundary=np.array([x['boundary_band_iou'] for x in local['identity']])
sizebins=[('small',areas<1024),('medium',(areas>=1024)&(areas<9216)),('large',areas>=9216)]
supportbins=[('low_0_to_025',support<.25),('mid_025_to_075',(support>=.25)&(support<.75)),('high_075_to_1',support>=.75)]
def stratum(sel):
 n=int(sel.sum())
 if not n:return dict(n=0)
 return dict(n=n,identity_iou=float(values['identity'][sel].mean()),identity_boundary_band_iou=float(baseline_boundary[sel].mean()),half_minus_identity_iou=float(half[sel].mean()),mean_fine_support=float(support[sel].mean()),mean_weak_coarse_fraction=float(np.mean([x['weak_coarse_fraction'] for x,b in zip(geo,sel) if b])),shift_iou_deltas={v:float((values[v]-values['identity'])[sel].mean()) for v in names if v!='identity'})
strata=dict(by_support={n:stratum(s) for n,s in supportbins},by_size={n:stratum(s) for n,s in sizebins},by_clipping={'clipped':stratum(clipped),'not_clipped':stratum(~clipped)},support_within_size={sz:{n:stratum(ss&sel) for n,sel in supportbins} for sz,ss in sizebins})
correlations={}
for key in ['fine_support','fine_quality_supported','sam_coverage_gt','sam_confidence_gt','weak_coarse_fraction','boundary_fine_support','bbox_area']:
 valid=np.array([x[key] is not None for x in geo]);v=np.array([float(x[key]) if x[key] is not None else np.nan for x in geo]);correlations[key]={}
 for name,target in [('half_minus_identity_iou',half),('identity_boundary_band_iou',baseline_boundary)]:
  rho=spearmanr(v[valid],target[valid]).statistic if len(np.unique(v[valid]))>1 else np.nan
  correlations[key][name]=dict(n=int(valid.sum()),spearman_rho=float(rho) if np.isfinite(rho) else None)
gates={}
for n in directions:
 a=effects[n+'_minus_identity'];b=effects[n+'_minus_smooth_symmetric'];ap=metrics[n]['segm/mAP']-metrics['identity']['segm/mAP'];gates[n]=bool(ap>=.005 and a['mean']>=.002 and a['simultaneous95'][0]>0 and b['mean']>=.001 and b['simultaneous95'][0]>0)
smoothing=bool(metrics['smooth_symmetric']['segm/mAP']-metrics['identity']['segm/mAP']>=.005 and effects['smooth_symmetric_minus_identity']['mean']>=.002)
bbox_equal=all(all(v==metrics['identity'][k] for k,v in metrics[n].items() if k.startswith('bbox/')) for n in names);assert bbox_equal
camera=dict(sampled_depth_pixels=sum(x['camera_checks']['sampled_depth_pixels'] for x in checks),max_same_camera_uv_error_px=max(x['camera_checks']['max_same_camera_uv_error_px'] for x in checks),max_same_camera_z_error_m=max(x['camera_checks']['max_same_camera_z_error_m'] for x in checks),max_sat_world_local_uv_error_px=max(x['camera_checks']['max_sat_world_local_uv_error_px'] for x in checks))
result=dict(scope='153 frozen development validation images,4496 repeated building appearances,237 buildings; not test or full future dataset',metrics=metrics,gt_mask_iou_means={n:float(v.mean()) for n,v in values.items()},effects=effects,bootstrap=dict(seed=1730,resamples=1000,simultaneous_contrasts=8,common_max_abs_radius95=radius,limitations='Exploratory image sum/count bootstrap; ignores spatial/model-selection/training-seed uncertainty'),local_metric_means=localmeans,local_quality_strata=strata,descriptive_correlations=correlations,camera_algebra_checks=camera,preregistered_gates=dict(directional_support=gates,symmetric_smoothing_support=smoothing,bbox_exact=bbox_equal),limits='Translations move learned effective residual including gate pattern, mask to original support and interpolate; cannot establish camera calibration truth. Local associations are observational; GT bands not deployable gates.')
dump(O/'summary.json',result);dump(O/'per_gt_joined.json',[dict(**g,half_minus_identity_iou=float(h),identity_iou=float(i),identity_boundary_band_iou=float(b)) for g,h,i,b in zip(geo,half,values['identity'],baseline_boundary)]);print(json.dumps(result,indent=2))
