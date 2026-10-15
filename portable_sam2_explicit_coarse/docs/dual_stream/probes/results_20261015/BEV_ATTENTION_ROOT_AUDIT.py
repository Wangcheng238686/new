"""Independent root summary check; source query NPZ only, no model code imports."""
import json
from pathlib import Path
import numpy as np
R=Path('/tmp/ue_uav_experiments/20261014_ad_seed42/final_analysis/bev_attention_audit');m=json.loads((R/'full/run_manifest.json').read_text());s=json.loads((R/'summary.json').read_text());assert m['complete'] and m['image_ids']==[1218,950,306,1216,948,505];out={}
for layer in [0,1]:
 aa=[]
 for iid in m['image_ids']:
  with np.load(R/'full'/f'attention_{iid}_layer{layer}.npz') as z:aa.append({k:z[k] for k in z.files})
 a={k:np.concatenate([x[k] for x in aa]) for k in aa[0]};has=a['candidate_count']>0;pr=a['projected_mask'].astype(bool);hv=a['height_valid'].astype(bool);obs=a['observation_support']>0
 masks={'has_candidate':has,'has_candidate_observed_height_valid':has&hv&obs,'has_candidate_height_invalid':has&~hv,'has_candidate_observed_height_invalid':has&obs&~hv,'final_projected':pr}
 for group,mask in masks.items():
  target=s['pooled_query'][str(layer)][group];assert int(mask.sum())==target['n'] and int(a['candidate_count'][mask].sum())==target['candidate_votes']
  modes={'query_equal':np.ones(mask.sum())}
  if group=='final_projected':modes.update(source_raw_mass=a['project_source_mass'][mask],source_mass_quality=a['project_source_mass_quality'][mask])
  for mode,w in modes.items():
   for metric in ['height_mass','xycore_mass','joint_mass','entropy_normalized','weighted_signed_z_m','weighted_abs_dz_m','weighted_xy_offset_norm_m','weighted_abs_xy_distance_m']:
    v=a[metric][mask];actual=float(np.dot(v,w)/w.sum());expected=target['aggregation'][mode][metric]['mean'];assert abs(actual-expected)<1e-12,(layer,group,mode,metric)
   for prefix in ['height','xycore','joint']:
    v=a[prefix+'_mass'][mask]-a[prefix+'_candidate_fraction'][mask];actual=float(np.dot(v,w)/w.sum());assert abs(actual-target['aggregation'][mode][prefix+'_mass_minus_uniform']['mean'])<1e-12
  for prefix in ['height','xycore','joint']:
   actual=a[prefix+'_candidate_count'][mask].sum()/a['candidate_count'][mask].sum();assert abs(actual-target['candidate_vote_equal_fractions'][prefix])<1e-12
 out[str(layer)]={k:s['pooled_query'][str(layer)]['final_projected']['aggregation'][k]['height_mass']['mean'] for k in ['query_equal','source_raw_mass','source_mass_quality']}
 # Per-image-equal means must give equal image influence despite different projected counts.
 for group in masks:
  for metric in ['height_mass','xycore_mass','joint_mass','entropy_normalized','weighted_abs_dz_m']:
   vals=[]
   for x in aa:
    hh=x['has_candidate'].astype(bool);pp=x['projected_mask'].astype(bool);vv=x['height_valid'].astype(bool);oo=x['observation_support']>0
    mm={'has_candidate':hh,'has_candidate_observed_height_valid':hh&vv&oo,'has_candidate_height_invalid':hh&~vv,'has_candidate_observed_height_invalid':hh&oo&~vv,'final_projected':pp}[group]
    if mm.any():vals.append(float(x[metric][mm].astype(np.float64).mean()))
   assert abs(float(np.mean(vals))-s['per_image_equal'][str(layer)][group][metric]['mean'])<1e-12
print(json.dumps({'verdict':'PASS','projected_height_mass_by_weighting':out,'all_pooled_metric_means_and_uniform_differences_reproduced':True,'image_equal_means_reproduced':True},indent=2))
