"""CPU-only descriptive audit of actual CVT attention; no labels or fitting."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
R=Path('/tmp/ue_uav_experiments/20261014_ad_seed42/final_analysis');IDS=[1218,950,306,1216,948,505]
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def check(v,msg):
 if not v:raise AssertionError(msg)
def stats(v,w=None):
 v=np.asarray(v,dtype=float);check(np.isfinite(v).all(),'Finite included statistics')
 if not len(v):return dict(n=0,mean=None,quantiles=None,weight_sum=0)
 if w is None:return dict(n=len(v),mean=float(v.mean()),quantiles=dict(zip(['p0','p10','p25','p50','p75','p90','p99','p100'],np.quantile(v,[0,.1,.25,.5,.75,.9,.99,1]).tolist())),weight_sum=float(len(v)))
 w=np.asarray(w,dtype=float);check((w>=0).all() and np.isfinite(w).all(),'Valid nonnegative weights');total=w.sum()
 if not total:return dict(n=len(v),mean=None,quantiles=None,weight_sum=0)
 order=np.argsort(v,kind='stable');cum=np.cumsum(w[order]);q=[v[order][min(np.searchsorted(cum,p*total,side='left'),len(v)-1)] for p in [0,.1,.25,.5,.75,.9,.99,1]]
 return dict(n=len(v),mean=float(np.dot(v,w)/total),quantiles=dict(zip(['p0','p10','p25','p50','p75','p90','p99','p100'],map(float,q))),weight_sum=float(total))
def groups(a):
 has=a['has_candidate'].astype(bool);obs=a['observation_support']>0;hv=a['height_valid'].astype(bool);pr=a['projected_mask'].astype(bool)
 return {'has_candidate':has,'has_candidate_height_valid':has&hv,'has_candidate_observed_height_valid':has&obs&hv,'has_candidate_height_invalid':has&~hv,'has_candidate_observed_height_invalid':has&obs&~hv,'final_projected':pr}
def metrics(a):
 d={k:a[k] for k in ('height_mass','xycore_mass','joint_mass','entropy_normalized','entropy_headmean_distribution','weighted_signed_z_m','weighted_abs_dz_m','weighted_xy_offset_norm_m','weighted_abs_xy_distance_m')}
 d['view_dominance_headmean']=a['view_mass'].max(1)
 for key in ('height','xycore','joint'):
  d['uniform_'+key+'_mass']=a[key+'_candidate_fraction'];d[key+'_mass_minus_uniform']=a[key+'_mass']-a[key+'_candidate_fraction']
 for key in ('uniform_weighted_signed_z_m','uniform_weighted_abs_dz_m','uniform_weighted_abs_xy_distance_m','uniform_weighted_xy_offset_norm_m','uniform_entropy_normalized'):
  if key in a:d[key]=a[key]
 if 'uniform_view_mass' in a:d['uniform_view_dominance']=a['uniform_view_mass'].max(1)
 return d
def aggregate(a):
 output={};has=a['has_candidate'].astype(bool)
 for name,sel in groups(a).items():
  modes={'query_equal':None}
  if name=='final_projected':modes.update(source_raw_mass=a['project_source_mass'][sel],source_mass_quality=a['project_source_mass_quality'][sel])
  output[name]=dict(n=int(sel.sum()),candidate_votes=int(a['candidate_count'][sel].sum()),aggregation={mode:{k:stats(v[sel],weights) for k,v in metrics(a).items()} for mode,weights in modes.items()},candidate_vote_equal_fractions={k:float(a[k+'_candidate_count'][sel].sum()/a['candidate_count'][sel].sum()) if a['candidate_count'][sel].sum() else None for k in ('height','xycore','joint')})
 output['denominators']=dict(all_grid_queries=len(has),no_candidate=int((~has).sum()),has_candidate=int(has.sum()),observation_support_positive=int((a['observation_support']>0).sum()),height_valid=int(a['height_valid'].sum()),final_projected=int(a['projected_mask'].sum()),raw_mass_sum=float(a['project_source_mass'].sum()),mass_quality_sum=float(a['project_source_mass_quality'].sum()))
 return output

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run-dir',type=Path,default=R/'bev_attention_audit/full');args=ap.parse_args();d=args.run_dir;m=read(d/'run_manifest.json');complete=m['complete'];ids=IDS if complete else IDS[:1]
 check(m['image_ids']==ids and [r['image_id'] for r in m['records']]==ids,'Fixed image order');check(complete or m['smoke_complete'],'Completed scoped run');check(m['training'] is False and m['optimizer_created'] is False and m['parameter_versions_unchanged'],'Frozen parameters')
 check(m['weights']=='ema' and m['strict'] and not m['load_report']['missing_keys'] and not m['load_report']['unexpected_keys'],'Strict EMA')
 check(sha(m['checkpoint'])==m['checkpoint_sha256'],'Checkpoint SHA');check(sha(d.parent/'PREREGISTRATION.md')==m['preregistration_sha256'],'Prereg SHA')
 for path,h in m['source_sha256'].items():check(sha(path)==h,'Source SHA '+path)
 check(m['manifest_sha256']=='495841b2174242c87661e6b40dba2ddcca13d64b13798116efc8c933c679ca59','Dataset identity')
 check(sha(R.parent/'dataset/manifest.json')==m['manifest_sha256'],'Dataset manifest entity SHA')
 prior=read(R/'p2_prompt_trace/full/run_manifest.json');check(m['architecture_contract']==prior['architecture_contract'],'Frozen historical architecture')
 cfg=m['geometry'];check(cfg==read('/tmp/ue_bev_geometry_quality/summary.json')['log_config'],'Historical six-image geometry identity');check(cfg['side_m']/cfg['grid_size']==4 and cfg['height_bin_m']==2,'Fixed XY core half2m/height2m')
 arrays={};per_image={}
 for rec in m['records']:
  iid=rec['image_id'];check(sha(rec['geometry_cache']['path'])==rec['geometry_cache']['sha256'] and rec['geometry_cache_arrays_exact'],'Historical geometry cache identity');check(rec['hooks_removed'] and rec['payload_unchanged'],'Inputs/hooks unchanged');check(rec['hookon_off_features_exact'] is (None if complete else True),'Smoke exact feature test scope');check([x['layer'] for x in rec['layers']]==[0,1],'Two actual layers');per_image[str(iid)]={}
  for info in rec['layers']:
   layer=info['layer'];p=d/f'attention_{iid}_layer{layer}.npz';sp=d/f'sdpa_sample_{iid}_layer{layer}.npz';check(sha(p)==info['npz_sha256'] and sha(sp)==info['sample_npz_sha256'],'Raw artifact SHA');check(info['input_hashes_unchanged'] and info['heads']==8,'Layer invariants')
   check(info['reconstruction_dtype']=='float32','Actual FP32 no-autocast')
   a=dict(np.load(p));n=cfg['grid_size']**2;check(np.array_equal(a['query_id'],np.arange(n)) and info['queries']==n,'All unique queries');check(all(len(x)==n for x in a.values()),'Aligned query arrays')
   has=a['candidate_count']>0;check(np.array_equal(has,a['has_candidate']),'Has candidate denominator');pr=a['projected_mask'].astype(bool);check(np.all(has[pr]) and np.all(a['height_valid'][pr]) and np.all(a['observation_support'][pr]>0),'Projected valid observed candidates')
   check(info['projected_unique_queries']==int(pr.sum()),'Projected count');check(np.array_equal(pr,a['project_source_vote_count']>0),'Projection votes');check(np.all(a['project_source_mass'][pr]>0) and np.all(a['project_source_mass_quality'][pr]>0),'Positive projection weights');check(np.all(a['project_source_mass_quality']<=a['project_source_mass']+1e-5),'Quality bound')
   attention_fields=[k for k in metrics(a) if k in a]+['view_mass','uniform_view_mass','entropy_per_head','weighted_signed_xy_m','uniform_weighted_signed_xy_m']
   for key,v in a.items():
    if key in attention_fields or (np.issubdtype(v.dtype,np.floating) and np.isnan(v).any()):
     check(np.isfinite(v[has]).all() and np.isnan(v[~has]).all(),'Finite candidate / NaN no-candidate '+key)
    else:check(np.isfinite(v).all(),'Finite non-attention '+key)
   for k in ('height','xycore','joint'):
    check(np.all(a[k+'_candidate_count']<=a['candidate_count']) and np.all(a[k+'_candidate_count']>=0),'Candidate subset');check(np.allclose(a[k+'_candidate_fraction'],a[k+'_candidate_count']/np.maximum(a['candidate_count'],1),atol=1e-12),'Uniform candidate fraction');check(np.all((a[k+'_mass'][has]>=-1e-6)&(a[k+'_mass'][has]<=1+1e-6)),'Mass range');check(np.isnan(a[k+'_mass'][~has]).all(),'No-candidate not fake mass');check(np.allclose(a['uniform_'+k+'_mass'][has],a[k+'_candidate_fraction'][has],atol=2e-6,rtol=0),'Uniform explicit mass matches fraction')
   check(np.allclose(a['view_mass'][has].sum(1),1,atol=2e-6,rtol=0),'View mass sum');check(np.all(a['joint_mass'][has]<=np.minimum(a['height_mass'][has],a['xycore_mass'][has])+2e-6),'Joint subset mass');check(np.allclose(np.linalg.norm(a['weighted_signed_xy_m'][has],axis=1),a['weighted_xy_offset_norm_m'][has],atol=2e-5,rtol=1e-6),'Signed XY vector norm')
   check(np.all(a['weighted_abs_dz_m'][has]+1e-5+1e-6*np.abs(a['weighted_signed_z_m'][has])>=np.abs(a['weighted_signed_z_m'][has])),'AbsZ vs signedZ');check(np.all(a['weighted_abs_xy_distance_m'][has]+1e-5+1e-6*a['weighted_xy_offset_norm_m'][has]>=a['weighted_xy_offset_norm_m'][has]),'Distance vs offset')
   check(a['view_mass'].shape==(n,5) and a['entropy_per_head'].shape==(n,8) and a['reference_xyz'].shape==(n,3),'Array shape contract')
   check(np.allclose(a['entropy_normalized'][has],a['entropy_per_head'][has].mean(1),atol=2e-6,rtol=0),'Entropy means actual heads')
   check(np.all((a['entropy_normalized'][has]>=-1e-6)&(a['entropy_normalized'][has]<=1+1e-6)),'Normalized entropy range')
   check(np.all(a['entropy_headmean_distribution'][has]+2e-6>=a['entropy_normalized'][has]),'Entropy concavity')
   check(info['sdpa_max_abs']<=1e-5 or info['sdpa_max_relative_l2']<=1e-5,'Preregistered SDPA reconstruction gate')
   sm=dict(np.load(sp));check(len(sm['query_id'])==info['sample_queries']==sum(info['sample_group_counts'].values()),'Numerical sample denominator');check(sm['weights_heads'].shape[1]==8,'Numerical sample heads')
   check(np.allclose(sm['attend_actual'],sm['attend_manual'],atol=1e-5,rtol=1e-5) or np.linalg.norm(sm['attend_actual']-sm['attend_manual'])/max(np.linalg.norm(sm['attend_actual']),1e-6)<=1e-5,'Saved numerical sample reconstruction')
   arrays.setdefault(layer,[]).append(a);per_image[str(iid)][str(layer)]=aggregate(a)
 pooled={str(k):aggregate({field:np.concatenate([a[field] for a in all_a]) for field in all_a[0]}) for k,all_a in arrays.items()}
 image_equal={}
 for layer in arrays:
  image_equal[str(layer)]={}
  for group in groups(arrays[layer][0]):
   image_equal[str(layer)][group]={key:stats([per_image[str(iid)][str(layer)][group]['aggregation']['query_equal'][key]['mean'] for iid in ids if per_image[str(iid)][str(layer)][group]['n']]) for key in metrics(arrays[layer][0])}
 result=dict(validation='PASS',per_image_equal=image_equal,scope=dict(images=ids,layers=[0,1],all_queries_per_image_per_layer=cfg['grid_size']**2,complete=complete,no_GT=True,no_training=True),per_image=per_image,pooled_query=pooled,definitions=dict(uniform='Equal valid candidates within each query, then same query weighting. Candidate-vote fractions separately weight queries by candidate_count.',source_raw_mass='Per-source sum(project_mass), before per-satellite-target normalization.',source_mass_quality='Per-source sum(project_mass*project_quality), before per-satellite-target normalization; not actual normalized downstream contribution.',entropy_normalized='Mean per-head H/log(candidate_count), count1=0. entropy_headmean_distribution is separate entropy of head-mean distribution.',view_dominance_headmean='Largest five-view mass after averaging candidate weights over heads.',invalidheight='Reference height invalid; offsets/masses only describe model coordinate inputs, not reliable physical height truth.',quantiles='Unweighted linear interpolation; weighted discrete inverse-CDF with stable sort.'),interpretation_limits='Descriptive fixed six current EMA images. Attention compatibility is geometric consistency, not semantic correctness or utility. Rows without any candidate are excluded from all quality summaries; no comparison to E72 raw training .508. No GT, fit, intervention, AP, success gate or causal mechanism claim.',source_files_sha256={str(d/'run_manifest.json'):sha(d/'run_manifest.json'),str(Path(__file__)):sha(__file__)})
 out=d.parent/('summary.json' if complete else 'smoke_summary.json');out.write_text(json.dumps(result,indent=2));print('PASS SAVED',out)
if __name__=='__main__':main()
