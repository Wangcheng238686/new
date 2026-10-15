"""Aggregate read-only stage changes; quality deltas stay separate from magnitude."""
import json
from pathlib import Path
import numpy as np
R=Path('/tmp/ue_uav_experiments/20261014_ad_seed42/final_analysis');O=R/'p2_prompt_trace';D=O/'full'
def read(p):return json.loads(Path(p).read_text())
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2))
m=read(D/'run_manifest.json');assert m['complete'] and m['processed_images']==153 and m['ground_truth_instances']==4496 and m['parameter_versions_unchanged'];ids=m['image_ids'];names=m['variants'];records={n:read(D/(n+'.json')) for n in names};base=records['baseline'];assert len(base)==4496
identity=[(x['image_id'],x['gt_slot'],x['global_id'],x['bbox_area'],x['roi_level']) for x in base];assert len(set((x[0],x[1]) for x in identity))==4496
for n in names:assert [(x['image_id'],x['gt_slot'],x['global_id'],x['bbox_area'],x['roi_level']) for x in records[n]]==identity
checks=read(D/'invariance_checks.json');assert [x['image_id'] for x in checks]==ids
for x in checks:assert x['bev_hook_calls']==2 and all(x[k] for k in ['endpoint_iou_exact','nonp2_roi_coarse_paste_exact','geometry_rgb_restored','features_unchanged','observer_hooks_removed'])
levels=np.array([x['roi_level'] for x in base]);areas=np.array([x['bbox_area'] for x in base]);support=np.array([x['dense_support_cells'] for x in base]);counts=np.array([sum(x['image_id']==i for x in base) for i in ids]);starts=np.r_[0,np.cumsum(counts)[:-1]];assert np.bincount(levels,minlength=4).tolist()==[3657,839,0,0];buildings=sorted(set(x['global_id'] for x in base));assert len(buildings)==237;ix={g:np.array([i for i,x in enumerate(base) if x['global_id']==g]) for g in buildings}
for n in names:
 prior={x['image_id']:x for x in read(R/'p2_transfer_audit/full'/(n+'_fixed_box.json'))}
 for x in records[n]:assert x['gt_iou']==prior[x['image_id']]['mask_iou'][x['gt_slot']]
 for x in records[n]:
  assert len(x['point_valid'])==len(x['point_correct'])==4 and 0<=x['coarse_iou']<=1 and np.isfinite(x['coarse_bce']) and x['dense_support_cells']>=0
  if x['dense_support_cells']==0:assert x['dense_no_mask_delta_norm']==0
  for v in x['stages'].values():assert np.isfinite(v['rms']) and np.isfinite(v['relative_l2']) and np.isfinite(v['mae'])
rows_groups=dict(all=np.ones(4496,dtype=bool),P2=levels==0,nonP2=levels!=0,small=areas<1024,medium=(areas>=1024)&(areas<9216),large=areas>=9216,dense_support_zero=support==0,dense_support_nonzero=support>0)
def mean(v,sel):return float(np.mean(np.asarray(v)[sel])) if sel.any() else None

def aggregate(name,sel):
 rows=records[name];n=int(sel.sum());r=[x for x,b in zip(rows,sel) if b]
 if not n:return dict(n=0)
 stage={}
 for key in rows[0]['stages']:
  stage[key]=dict(mean_rms=float(np.mean([x['stages'][key]['rms'] for x in r])),mean_relative_l2=float(np.mean([x['stages'][key]['relative_l2'] for x in r])),strict_changed_count=sum(x['stages'][key]['strict_changed'] for x in r),rms_gt1e6_count=sum(x['stages'][key]['changed_gt1e6'] for x in r))
 valid=np.array([x['point_valid'] for x in r],dtype=bool);correct=np.array([x['point_correct'] for x in r],dtype=bool);slots={}
 for j,slot in enumerate(['p1','p2','n1','n2']):
  count=int(valid[:,j].sum());slots[slot]=dict(valid=count,correct=int(correct[:,j].sum()),accuracy=float(correct[:,j].sum()/count) if count else None)
 attenuation=[x['dense_final_to_pe_delta_ratio'] for x in r if x['dense_final_to_pe_delta_ratio'] is not None]
 result=dict(n=n,gt_iou=float(np.mean([x['gt_iou'] for x in r])),coarse_iou=float(np.mean([x['coarse_iou'] for x in r])),coarse_bce=float(np.mean([x['coarse_bce'] for x in r])),point_slots=slots,point_changed_roi_count=sum(x['point_changed_count']>0 for x in r),point_changed_slots_total=sum(x['point_changed_count'] for x in r),mean_coarse_prob_mae=float(np.mean([x['coarse_prob_mae'] for x in r])),coarse_binary_changed_roi_count=sum(x['coarse_binary_flip_fraction']>0 for x in r),mean_coarse_binary_flip_fraction=float(np.mean([x['coarse_binary_flip_fraction'] for x in r])),sam_raw_binary_changed_roi_count=sum(x['sam_raw_binary_flip_fraction']>0 for x in r),tail_binary_changed_roi_count=sum(x['tail_final_binary_flip_fraction']>0 for x in r),paste_changed_roi_count=sum(x['paste_xor_pixels']>0 for x in r),mean_paste_xor_pixels=float(np.mean([x['paste_xor_pixels'] for x in r])),dense_support_zero_count=sum(x['dense_support_cells']==0 for x in r),sparse_token_count_change_rois=sum(x['sparse_token_count_changed'] for x in r),dense_final_to_pe_delta_ratio=dict(n=len(attenuation),mean=float(np.mean(attenuation)) if attenuation else None,max=float(np.max(attenuation)) if attenuation else None),stage_changes=stage)
 result['joint_counts']=dict(coarse_numeric_changes_points_same=sum(x['stages']['coarse']['changed_gt1e6'] and x['point_changed_count']==0 for x in r),pe_dense_changes_final_dense_same=sum(x['stages']['pe_dense']['changed_gt1e6'] and not x['stages']['dense']['changed_gt1e6'] for x in r),dense_changes_paste_same=sum(x['stages']['dense']['changed_gt1e6'] and x['paste_xor_pixels']==0 for x in r),sparse_dense_both_same_paste_changed=sum(not x['stages']['sparse']['strict_changed'] and not x['stages']['dense']['strict_changed'] and x['paste_xor_pixels']>0 for x in r))
 return result
summary=dict(scope='Current frozen development subset only:153 val images,4496 appearances,237 buildings; no AP rerun/training',manifest_fields={k:m[k] for k in ['mask_thr_binary','mask_logit_threshold','dense_shape_alpha','dense_detach_input','coarse_loss_weight','shape_fusion_type','explicit_prompt_mode','mask_roi_chunk_size']},groups={name:{group:aggregate(name,sel) for group,sel in rows_groups.items()} for name in names},support_zero_by_size={group:dict(n=int(sel.sum()),support_zero=int(((support==0)&sel).sum()),fraction=float(((support==0)&sel).sum()/max(sel.sum(),1))) for group,sel in rows_groups.items() if group in ['all','small','medium','large']},comparisons={})
rng=np.random.default_rng(1750);draw=rng.integers(0,153,(1000,153));denom=counts[draw].sum(axis=1)
for n in names[1:]:
 effects={}
 for key in ['coarse_iou','gt_iou']:
  delta=np.array([x[key]-y[key] for x,y in zip(records[n],base)]);sums=np.add.reduceat(delta,starts);assert abs(sums.sum()/4496-delta.mean())<1e-12;boot=sums[draw].sum(axis=1)/denom
  effects[key]=dict(mean=float(delta.mean()),percentile95=np.quantile(boot,[.025,.975]).tolist(),building_equal_weight_mean=float(np.mean([delta[v].mean() for v in ix.values()])),improved_gt002=int((delta>.002).sum()),worse_gt002=int((delta<-.002).sum()),by_group={g:dict(n=int(sel.sum()),mean=mean(delta,sel)) for g,sel in rows_groups.items()})
 summary['comparisons'][n+'_minus_baseline']=effects
# Direct content contrast remains descriptive; not a new intervention or gain gate.
a=records['native_correct_q'];b=records['native_donor_q'];summary['correct_minus_donor']=dict(coarse_iou_mean=float(np.mean([x['coarse_iou']-y['coarse_iou'] for x,y in zip(a,b)])),final_iou_mean=float(np.mean([x['gt_iou']-y['gt_iou'] for x,y in zip(a,b)])))
summary['bootstrap']=dict(seed=1750,resamples=1000,method='paired image sum/count; exploratory, ignores spatial/selection/seed uncertainty')
summary['interpretation_limits']='Read-only responses are not causal isolation of dense/sparse/tail. Cross-scale P3 fuser unadapted; changes reaching decoder need not be useful. GT64 targets only diagnostics; final loss detach/no_grad static path is not training-gradient magnitude measurement.'
dump(O/'summary.json',summary);print(json.dumps(summary,indent=2))
