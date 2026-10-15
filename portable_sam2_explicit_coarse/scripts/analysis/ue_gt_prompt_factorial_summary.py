"""Independent CPU audit of frozen actual-decoder prompt 2x2 exchange.
Two preregistered primary contrasts; Bonferroni simultaneous marginal intervals.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
ROOT=Path('/tmp/ue_uav_experiments/20261014_ad_seed42/final_analysis')
NAMES=['baseline','gt_point_only','gt_dense_only','joint_gt_prompt']
SOURCES={'baseline':('P','P'),'gt_point_only':('G','P'),'gt_dense_only':('P','G'),'joint_gt_prompt':('G','G')}
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def require(b,msg):
 if not b:raise AssertionError(msg)
def ident(r):return tuple(r[k] for k in ('image_id','gt_slot','global_id','bbox_area','roi_level'))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run-dir',type=Path,default=ROOT/'gt_prompt_factorial/full');ap.add_argument('--output',type=Path);args=ap.parse_args();d=args.run_dir
 m=read(d/'run_manifest.json');checks=read(d/'invariance_checks.json');require(m['variants']==NAMES,'Four fixed cells/order');require(m['source_map']=={k:list(v) for k,v in SOURCES.items()},'Manifest source map')
 require(m['training'] is False and m['optimizer_created'] is False and m['parameter_versions_unchanged'] is True,'Frozen run')
 require(m['strict'] and not m['load_report']['missing_keys'] and not m['load_report']['unexpected_keys'] and m['weights']=='ema','Strict EMA load')
 require(m['actual_prompt_contract']==dict(sparse_last_dimension=256,sparse_tokens=[2,6],dense_shape=[256,32,32]),'Actual prompt shape contract')
 require(abs(m['mask_logit_threshold']-np.log(.4/.6))<1e-12,'Actual mask logit threshold')
 require(m['oracle_logit_amplitude']==4 and m['mask_thr_binary']==.4,'Fixed GT amplitude/deployment threshold')
 require(m['mask_roi_chunk_size']==32,'Preregistered production chunk32')
 require(m['dense_shape_alpha']==.32814547419548035 and m['dense_detach_input'] is True,'Original shape dense contract')
 for key,path in [('checkpoint_sha256',Path(m['checkpoint'])),('script_sha256',Path('/tmp/ue_gt_prompt_factorial_probe.py')),('helper_script_sha256',Path('/tmp/ue_rgb_content_probe.py')),('trace_helper_sha256',Path('/tmp/ue_p2_prompt_trace.py')),('oracle_helper_sha256',Path('/tmp/ue_gt_prompt_oracle_probe.py')),('preregistration_sha256',d.parent/'PREREGISTRATION.md'),('manifest_sha256',ROOT.parent/'dataset/manifest.json')]:require(m[key]==sha(path),'SHA '+key)
 ids=m['image_ids'];require(len(ids)==len(set(ids))==m['processed_images'],'Unique image count');complete=m['complete'];require(complete==(len(ids)==153),'Scope completion');require([c['image_id'] for c in checks]==ids,'Checks order')
 rows={k:read(d/(k+'.json')) for k in NAMES};b=rows['baseline'];n=len(b);require(n==m['ground_truth_instances'],'GT count')
 identity=[ident(x) for x in b];require(len(set((x[0],x[1]) for x in identity))==n,'GT unique identities')
 for name in NAMES:require([ident(x) for x in rows[name]]==identity,'Paired GT order '+name)
 historical={name:{(x['image_id'],x['gt_slot']):x for x in read(ROOT/'gt_prompt_oracle/full'/(name+'.json'))} for name in ('baseline','joint_gt_prompt')}
 for i,x in enumerate(b):
  for name in ('baseline','joint_gt_prompt'):
   y=rows[name][i];z=historical[name][(x['image_id'],x['gt_slot'])];require(ident(y)==ident(z) and y['gt_iou']==z['gt_iou'],'Exact historical diagonal '+name)
   for key in ('point_valid','point_correct','point_labels','point_coords','point_local_yx','sparse_token_count','dense_support_cells','coarse_target_fg_pixels'):require(y[key]==z[key],'Historical diagonal source diagnostics '+key)
  for name in NAMES:
   r=rows[name][i];ps,ds=SOURCES[name];require((r['actual_point_source'],r['actual_dense_source'])==(ps,ds),'Exact intended prompt sources')
   sourcepoint=rows['baseline' if ps=='P' else 'joint_gt_prompt'][i];sourcedense=rows['baseline' if ds=='P' else 'joint_gt_prompt'][i]
   for kind,source in [('sparse',sourcepoint),('dense',sourcedense)]:
    key='actual_'+kind+'_sha256';value=r[key];require(isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value),'Valid actual tensor SHA')
    require(value==r['source_'+kind+'_sha256']==source[key],'Actual decoder input equals intended live source '+kind)
   require(r['upstream_coarse_source']==('G' if name=='joint_gt_prompt' else 'P'),'Upstream coarse diagnostic does not claim cross GT')
   require(r['dense_no_mask_delta_norm']==sourcedense['dense_no_mask_delta_norm'],'Actual selected dense norm')
   require(r['coarse_target_fg_pixels']==x['coarse_target_fg_pixels'] and 0<=r['coarse_target_fg_pixels']<=4096,'Fixed GT64 target')
   require(r['sparse_token_count'] in (2,6) and r['sparse_token_count']==sourcepoint['sparse_token_count'],'Actual source token count')
   for key in ('point_valid','point_correct','point_labels','point_coords','point_local_yx'):require(r[key]==sourcepoint[key],'Actual source point diagnostics '+key)
   require(len(r['point_valid'])==len(r['point_correct'])==len(r['point_labels'])==len(r['point_coords'])==len(r['point_local_yx'])==4,'Four point slots')
   require(all(a in (-1,0,1) and bool(q)==(a>=0) for a,q in zip(r['point_labels'],r['point_valid'])),'Point validity')
   require(all(not correct or valid for correct,valid in zip(r['point_correct'],r['point_valid'])),'Point correct only valid')
   require(all(len(p)==2 and all(np.isfinite(z) for z in p) for p in r['point_coords']),'Finite point coordinates')
   require(all(len(p)==2 and all(isinstance(z,int) and 0<=z<64 for z in p) for p in r['point_local_yx']),'Valid GT64 point coordinates')
   if r['dense_support_cells']==0:require(r['dense_no_mask_delta_norm']==0,'Actual no-support dense base')
   require(np.isfinite(r['gt_iou']) and 0<=r['gt_iou']<=1,'Valid IoU');require(r['dense_support_cells']==x['dense_support_cells'],'Fixed support');require(r['paste_xor_pixels']>=0,'XOR legal')
   if name=='baseline':require(r['paste_xor_pixels']==0,'Baseline self XOR')
   if name=='gt_dense_only' and r['dense_support_cells']==0:require(r['paste_xor_pixels']==0 and r['gt_iou']==x['gt_iou'],'Dense-only no-support exact baseline')
 counts=np.array([sum(x['image_id']==iid for x in b) for iid in ids]);require((counts>0).all(),'Nonempty images');require([x['image_id'] for x in b]==[iid for iid,c in zip(ids,counts) for _ in range(c)],'Contiguous image rows')
 for c,count in zip(checks,counts):
  chunk=m['mask_roi_chunk_size'] or int(count);expected=[min(chunk,int(count)-i) for i in range(0,int(count),chunk)]
  require(c['chunk_sizes']==expected,'Production chunk slices')
  for key in ('diagonal_endpoints_exact','source_point_metadata_exact','source_prompt_hashes_unchanged','source_box_tokens_exact','actual_source_values_exact','actual_visual_args_unchanged','features_unchanged','geometry_rgb_unchanged','hooks_removed','dense_only_support0_paste_exact'):require(c[key] is True,'Explicit invariant '+key)
  require(c['gt_slots_in_order']==list(range(int(count))),'Actual GT order')
  bounds=np.r_[0,np.cumsum(expected)].tolist();require(c['gt_source_target_ranges']==[[a,b] for a,b in zip(bounds[:-1],bounds[1:])],'Actual GT target slices')
  require(c['support0_count']==sum(r['image_id']==c['image_id'] and r['dense_support_cells']==0 for r in b),'Support0 image count')
  require(c['diagonal_exchange_sham_pixel_exact']==(None if complete else {'baseline':True,'joint_gt_prompt':True}),'Scope-dependent diagonal exchange sham')
 if complete:require(n==4496 and len(set(x['global_id'] for x in b))==237,'Full frozen subset')
 area=np.array([x['bbox_area'] for x in b]);level=np.array([x['roi_level'] for x in b]);support=np.array([x['dense_support_cells'] for x in b]);groups={'all':np.ones(n,dtype=bool),'small':area<1024,'medium':(area>=1024)&(area<9216),'large':area>=9216,'P2':level==0,'nonP2':level!=0,'dense_support_zero':support==0,'dense_support_nonzero':support>0}
 values={k:np.array([r['gt_iou'] for r in v]) for k,v in rows.items()};rng=np.random.default_rng(1770);draw=rng.integers(0,len(ids),(1000,len(ids)));starts=np.r_[0,np.cumsum(counts)[:-1]];denom=counts[draw].sum(1);buildings=sorted(set(x['global_id'] for x in b));building_indices={g:np.array([j for j,x in enumerate(b) if x['global_id']==g]) for g in buildings}
 def contrast(delta,primary=False):
  sums=np.add.reduceat(delta,starts);require(abs(sums.sum()/n-delta.mean())<1e-12,'Image sum/count means');boot=sums[draw].sum(1)/denom;quantiles=[.0125,.9875] if primary else [.025,.975];ci=np.quantile(boot,quantiles).tolist();mean=float(delta.mean())
  result=dict(mean=mean,image_paired_interval=ci,interval_level=.975 if primary else .95,quantiles=quantiles,building_equal_weight_mean=float(np.mean([delta[ix].mean() for ix in building_indices.values()])),by_group={g:dict(n=int(s.sum()),mean=float(delta[s].mean()) if s.any() else None,improved_gt002=int((delta[s]>.002).sum()),worse_gt002=int((delta[s]<-.002).sum())) for g,s in groups.items()})
  if primary:result['gate']=dict(mean_minimum=.005,ci_lower_strictly_positive=True,status=('PASS' if mean>=.005 and ci[0]>0 else 'FAIL') if complete else 'NOT_EVALUATED_SMOKE',passed=bool(mean>=.005 and ci[0]>0) if complete else None)
  return result
 primary={name+'_minus_baseline':contrast(values[name]-values['baseline'],True) for name in ('gt_point_only','gt_dense_only')}
 secondary={'joint_minus_baseline':contrast(values['joint_gt_prompt']-values['baseline']),'interaction_GG_minus_GP_minus_PG_plus_PP':contrast(values['joint_gt_prompt']-values['gt_point_only']-values['gt_dense_only']+values['baseline']),'points_given_GT_dense_GG_minus_PG':contrast(values['joint_gt_prompt']-values['gt_dense_only']),'dense_given_GT_points_GG_minus_GP':contrast(values['joint_gt_prompt']-values['gt_point_only'])}
 def aggregate(rr,sel):
  a=[r for r,s in zip(rr,sel) if s]
  if not a:return dict(n=0)
  slots={}
  for j,slot in enumerate(('p1','p2','n1','n2')):
   valid=sum(r['point_valid'][j] for r in a);correct=sum(r['point_correct'][j] for r in a);slots[slot]=dict(valid=valid,correct=correct,accuracy=correct/valid if valid else None)
  return dict(n=len(a),point_slots=slots,actual_sparse_token_counts={str(k):sum(r['sparse_token_count']==k for r in a) for k in (2,6)},final_iou=float(np.mean([r['gt_iou'] for r in a])),paste_changed_rois=sum(r['paste_xor_pixels']>0 for r in a),mean_paste_xor_pixels=float(np.mean([r['paste_xor_pixels'] for r in a])),dense_support_zero_count=sum(r['dense_support_cells']==0 for r in a))
 result=dict(scope=dict(images=len(ids),appearances=n,unique_buildings=len(buildings),complete=complete,current_development_subset=True),validation='PASS',invariant_counts=dict(images_checked=len(checks),multi_chunk_images=sum(len(c['chunk_sizes'])>1 for c in checks),total_chunks=sum(len(c['chunk_sizes']) for c in checks),support_zero_rois=int((support==0).sum())),primary_contrasts=primary,secondary_descriptive=secondary,groups={name:{g:aggregate(rr,s) for g,s in groups.items()} for name,rr in rows.items()},bootstrap=dict(seed=1770,resamples=1000,method='Paired image sum/count; two primary contrasts each marginal 97.5% CI via Bonferroni, secondary intervals descriptive 95%. Does not cover spatial, seed or selected-checkpoint uncertainty.'),source_files_sha256={str(p):sha(p) for p in [d/(k+'.json') for k in NAMES]+[d/'run_manifest.json',d/'invariance_checks.json',Path(__file__)]},interpretation_limits='Actual frozen decoder sparse/dense exchange diagnostic with GT boxes and fixed GT64 +/-4 sources. GT changes shape, confidence and distributions. Conditional modality effects and interaction are not attainable UAV gains, AP changes, mathematical upper bounds or detach-training benefits. Sparse includes fixed box tokens and actual token layout. For interaction, by_group improved_gt002/worse_gt002 are only contrast-sign threshold counts, not instance mask-quality improvement/degradation.')
 output=args.output or d.parent/('summary.json' if complete else 'smoke_summary.json');output.write_text(json.dumps(result,indent=2));print(json.dumps(dict(primary=primary,secondary=secondary),indent=2));print('SAVED',output)
if __name__=='__main__':main()
