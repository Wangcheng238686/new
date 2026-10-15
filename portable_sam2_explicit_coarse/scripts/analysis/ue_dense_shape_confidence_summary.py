"""CPU audit and preregistered paired-image dense source contrasts."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path('/tmp/ue_uav_experiments/20261014_ad_seed42/final_analysis');NAMES=['baseline','pred_binary_pm4','gt_binary_pm4']
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def ident(r):return tuple(r[k] for k in ('image_id','gt_slot','global_id','bbox_area','roi_level'))
def main():
 p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,default=ROOT/'dense_shape_confidence/full');a=p.parse_args();d=a.run_dir;m=read(d/'run_manifest.json');cs=read(d/'invariance_checks.json');rows={k:read(d/(k+'.json')) for k in NAMES};b=rows['baseline'];n=len(b);ids=m['image_ids'];complete=m['complete'];assert m['variants']==NAMES and m['actual_point_source']=='P';assert m['strict'] and m['weights']=='ema' and not m['load_report']['missing_keys'] and not m['load_report']['unexpected_keys'];assert m['training'] is False and m['optimizer_created'] is False and m['parameter_versions_unchanged'] is True;assert m['amplitude']==4 and m['coarse_binary_threshold']==0 and m['mask_thr_binary']==.4 and abs(m['mask_logit_threshold']-np.log(.4/.6))<1e-12;assert m['dense_detach_input'] is True and m['dense_shape_alpha']==.32814547419548035 and m['mask_roi_chunk_size']==32
 for k,path in [('checkpoint_sha256',Path(m['checkpoint'])),('script_sha256',Path('/tmp/ue_dense_shape_confidence_probe.py')),('helper_script_sha256',Path('/tmp/ue_rgb_content_probe.py')),('oracle_helper_sha256',Path('/tmp/ue_gt_prompt_oracle_probe.py')),('trace_helper_sha256',Path('/tmp/ue_p2_prompt_trace.py')),('manifest_sha256',ROOT.parent/'dataset/manifest.json'),('preregistration_sha256',d.parent/'PREREGISTRATION.md')]:assert m[k]==sha(path),k
 assert len(ids)==len(set(ids))==m['processed_images']==len(cs) and n==m['ground_truth_instances'];assert complete==(len(ids)==153)
 if complete:assert n==4496 and len(set(r['global_id'] for r in b))==237
 assert [c['image_id'] for c in cs]==ids
 hm=read(ROOT/'gt_prompt_factorial/full/run_manifest.json');assert m['architecture_contract']==hm['architecture_contract'];oldcoarse={(r['image_id'],r['gt_slot']):r for r in read(ROOT/'gt_prompt_oracle/full/baseline.json')}
 hist={name:{(r['image_id'],r['gt_slot']):r for r in read(ROOT/'gt_prompt_factorial/full'/(old+'.json'))} for name,old in [('baseline','baseline'),('gt_binary_pm4','gt_dense_only')]}
 for name,rr in rows.items():
  assert [ident(r) for r in rr]==[ident(r) for r in b]
  for r,z in zip(rr,b):
   authority=oldcoarse[(r['image_id'],r['gt_slot'])];assert r['dense_support_cells']==authority['dense_support_cells'] and r['gt_shape_fg_pixels']==authority['coarse_target_fg_pixels'] and r['gt_shape_fg_pixels']>0
   if name!='gt_binary_pm4':assert r['coarse_binary_iou']==authority['coarse_iou']
   if name=='baseline':assert r['paste_xor_pixels']==0
   assert r['actual_sparse_sha256']==r['baseline_sparse_sha256']==z['actual_sparse_sha256'];assert r['sparse_token_count']==z['sparse_token_count'] and r['sparse_token_count'] in (2,6);assert r['dense_support_cells']==z['dense_support_cells'];assert r['pred_shape_fg_pixels']==z['pred_shape_fg_pixels'] and r['gt_shape_fg_pixels']==z['gt_shape_fg_pixels'];assert all(len(r[k])==64 for k in ('actual_sparse_sha256','actual_dense_sha256'))
   assert 0<=r['gt_iou']<=1 and np.isfinite(r['gt_iou']) and 0<=r['coarse_binary_iou']<=1 and r['paste_xor_pixels']>=0
   assert r['source_shape_fg_pixels']==(r['gt_shape_fg_pixels'] if name=='gt_binary_pm4' else r['pred_shape_fg_pixels'])
   if name=='pred_binary_pm4':assert r['coarse_binary_iou']==z['coarse_binary_iou']
   if name=='gt_binary_pm4':assert r['coarse_binary_iou']==1
   if name in hist:
    h=hist[name][(r['image_id'],r['gt_slot'])];assert ident(r)==ident(h) and r['gt_iou']==h['gt_iou'] and r['actual_sparse_sha256']==h['actual_sparse_sha256'] and r['actual_dense_sha256']==h['actual_dense_sha256']
   if r['dense_support_cells']==0:assert r['dense_no_mask_delta_norm']==0 and r['paste_xor_pixels']==0 and r['gt_iou']==z['gt_iou'] and r['actual_dense_sha256']==z['actual_dense_sha256']
 counts=np.array([sum(r['image_id']==iid for r in b) for iid in ids]);assert (counts>0).all();assert [r['image_id'] for r in b]==[iid for iid,c in zip(ids,counts) for _ in range(c)]
 for c,count in zip(cs,counts):
  assert all(k in (2,6) for k in c['actual_sparse_token_counts']);sz=[min(32,int(count)-j) for j in range(0,int(count),32)];assert c['chunk_sizes']==sz and len(c['actual_sparse_token_counts'])==len(sz);ix=[r for r in b if r['image_id']==c['image_id']];assert [r['gt_slot'] for r in ix]==list(range(int(count)));start=0
  for size,tokens in zip(sz,c['actual_sparse_token_counts']):
   assert all(r['sparse_token_count']==tokens for r in ix[start:start+size]);start+=size
  bounds=np.r_[0,np.cumsum(sz)];assert c['target_ranges']==[[int(x),int(y)] for x,y in zip(bounds[:-1],bounds[1:])];assert c['support0_count']==sum(r['image_id']==c['image_id'] and r['dense_support_cells']==0 for r in b)
  for k in ('historical_endpoints_exact','actual_sparse_exact','actual_visual_values_exact','features_unchanged','geometry_rgb_unchanged','hooks_removed','original_coarse_exact','support_zero_masks_exact','binary_shape_preserved_exact','pm4_values_exact'):assert c[k] is True,k
  assert c['sham_pixel_exact']==(None if complete else {k:True for k in NAMES})
 area=np.array([r['bbox_area'] for r in b]);level=np.array([r['roi_level'] for r in b]);support=np.array([r['dense_support_cells'] for r in b]);groups={'all':np.ones(n,bool),'small':area<1024,'medium':(area>=1024)&(area<9216),'large':area>=9216,'P2':level==0,'nonP2':level!=0,'dense_support_zero':support==0,'dense_support_nonzero':support>0};vals={k:np.array([r['gt_iou'] for r in rr]) for k,rr in rows.items()};buildings=sorted(set(r['global_id'] for r in b));bix={g:np.array([j for j,r in enumerate(b) if r['global_id']==g]) for g in buildings};rng=np.random.default_rng(1780);draw=rng.integers(0,len(ids),(1000,len(ids)));starts=np.r_[0,np.cumsum(counts)[:-1]];denom=counts[draw].sum(1)
 def compare(delta,primary):
  sums=np.add.reduceat(delta,starts);q=[.0125,.9875] if primary else [.025,.975];ci=np.quantile(sums[draw].sum(1)/denom,q).tolist();mean=float(delta.mean());v=dict(mean=mean,image_paired_interval=ci,interval_level=.975 if primary else .95,building_equal_mean=float(np.mean([delta[ix].mean() for ix in bix.values()])),by_group={g:dict(n=int(s.sum()),mean=float(delta[s].mean()) if s.any() else None,improved_gt002=int((delta[s]>.002).sum()),worse_gt002=int((delta[s]<-.002).sum())) for g,s in groups.items()})
  if primary:v['gate']=dict(mean_minimum=.005,ci_lower_strictly_positive=True,status=('PASS' if mean>=.005 and ci[0]>0 else 'FAIL') if complete else 'NOT_EVALUATED_SMOKE')
  return v
 primary={'pred_binary_pm4_minus_baseline':compare(vals['pred_binary_pm4']-vals['baseline'],True),'gt_binary_pm4_minus_pred_binary_pm4':compare(vals['gt_binary_pm4']-vals['pred_binary_pm4'],True)}
 aggregate={name:{g:dict(n=int(s.sum()),final_iou=float(vals[name][s].mean()) if s.any() else None,coarse_binary_iou=float(np.mean([r['coarse_binary_iou'] for r,yes in zip(rr,s) if yes])) if s.any() else None,paste_changed_rois=sum(r['paste_xor_pixels']>0 for r,yes in zip(rr,s) if yes),dense_changed_rois=sum(r['actual_dense_sha256']!=z['actual_dense_sha256'] for r,z,yes in zip(rr,b,s) if yes)) for g,s in groups.items()} for name,rr in rows.items()}
 result=dict(validation='PASS',scope=dict(images=len(ids),appearances=n,unique_buildings=len(buildings),complete=complete,current_development_subset=True),primary_contrasts=primary,secondary_gt_minus_baseline=compare(vals['gt_binary_pm4']-vals['baseline'],False),groups=aggregate,invariants=dict(multi_chunk_images=sum(len(c['chunk_sizes'])>1 for c in cs),chunks=sum(len(c['chunk_sizes']) for c in cs),support_zero=int((support==0).sum()),all_actual_points_original=True,all_historical_endpoints_exact=True),bootstrap=dict(seed=1780,resamples=1000,primary_quantiles=[.0125,.9875],secondary_quantiles=[.025,.975],method='paired-image sum/count; two primary Bonferroni marginal 97.5% intervals; exploratory spatial/seed/checkpoint uncertainties uncovered'),source_sha256={str(p):sha(p) for p in [d/(k+'.json') for k in NAMES]+[d/'run_manifest.json',d/'invariance_checks.json',Path(__file__)]},limits='Frozen GT-box conditional dense diagnostic, original sparse always fixed. Prediction +/-4 contrast changes binarization/amplitude/within-shape distribution, not pure temperature. GT-pred equal-amplitude compares shape conditional on original potentially inconsistent points. Not UAV attainability, AP, deployable oracle or detach-training benefit.')
 output=d.parent/('summary.json' if complete else 'smoke_summary.json');output.write_text(json.dumps(result,indent=2));print(json.dumps(dict(primary=primary,secondary=result['secondary_gt_minus_baseline']),indent=2));print('SAVED',output)
if __name__=='__main__':main()
