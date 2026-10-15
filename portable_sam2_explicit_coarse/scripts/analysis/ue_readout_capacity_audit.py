import json,hashlib
from pathlib import Path
import numpy as np
O=Path('/tmp/ue_uav_experiments/20261014_ad_seed42/final_analysis/uav_complementarity')
OUT=Path('/tmp/ue_readout_capacity_audit.json')
meta=json.loads((O/'features/run_manifest.json').read_text());models=json.loads((O/'ridge_models.json').read_text());out={'scope':'Read-only cached-feature covariance and existing ridge prediction audit. No new supervised fit, no selection, no network forward.','source_manifest_sha256':hashlib.sha256((O/'features/run_manifest.json').read_bytes()).hexdigest(),'splits':{}}
def covariance(a):
 a=a.reshape(-1,a.shape[-1]).astype('float64');m=a.mean(0);return (a.T@a)/len(a)-np.outer(m,m)
def rankstats(c):
 ev=np.linalg.eigvalsh(c).clip(0);s=ev.sum();return {'trace':float(s),'participation_ratio':float(s*s/max((ev*ev).sum(),1e-30)),'top1_fraction':float(ev[-1]/max(s,1e-30)),'top4_fraction':float(ev[-4:].sum()/max(s,1e-30)),'eigen_min':float(ev[0]),'eigen_max':float(ev[-1])}
for split in ['train','validation']:
 records=[r for r in meta['records'] if r['split']==split];arr=[]
 for r in records:
  p=Path(r['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'];f=np.load(p);arr.append({k:f[k] for k in f.files});f.close()
 d={k:np.concatenate([f[k] for f in arr]) for k in arr[0]};del arr
 res={'n_rois':len(d['prob']),'n_images':len(records),'n_buildings':len(np.unique(d['global_id'])),'features':{},'existing_models':{}}
 for key in ['sat32','sat_extra32','uav32','shift32','noise32']:
  a=d[key].astype('float64');cov=covariance(a);within=((a-a.mean(1,keepdims=True))**2).mean((0,1));tot=np.diag(cov);mean=a.mean((0,1));std=np.sqrt(tot)
  res['features'][key]={'std_min':float(std.min()),'std_median':float(np.median(std)),'std_max':float(std.max()),'within_roi_variance_fraction':float(within.sum()/tot.sum()),'raw_covariance':rankstats(cov),'correlation':rankstats(cov/np.maximum(np.outer(std,std),1e-24)),'mean':mean.tolist(),'std':std.tolist()}
  del a
 y=(d['target']-d['prob']).astype('float64');res['residual']={'mean':float(y.mean()),'mse':float((y*y).mean()),'variance':float(y.var()),'bias_fraction_mse':float(y.mean()**2/(y*y).mean()),'within_roi_fraction_variance':float(((y-y.mean(1,keepdims=True))**2).mean()/y.var()),'partial_cell_fraction':float(((d['target']>0)&(d['target']<1)).mean()),'empty_cell_fraction':float((d['target']==0).mean()),'full_cell_fraction':float((d['target']==1).mean())}
 preds={}
 for cell,extra in [('SG',None),('S64','sat_extra32'),('SU','uav32'),('SX','shift32'),('SN','noise32')]:
  m=models[cell];x=np.concatenate([d['sat32'],d['geometry'],d[extra] if extra else np.zeros_like(d['uav32'])],-1).astype('float64');z=(x-m['mean'])/m['std'];b=np.array(m['beta']);pred=(d['prob']+z@b+m['intercept']).clip(0,1);preds[cell]=pred;cov=covariance(z);ev=np.linalg.eigvalsh(cov).clip(0);active=int((np.diag(cov)>1e-12).sum());p={}
  for name,sl in [('satellite',slice(0,32)),('geometry',slice(32,39)),('extra',slice(39,71))]:
   contribution=z[...,sl]@b[sl];p[name]={'beta_l2':float(np.linalg.norm(b[sl])),'contribution_rms':float(np.sqrt((contribution**2).mean())),'contribution_mean':float(contribution.mean())}
  res['existing_models'][cell]={'mse':float(((pred-d['target'])**2).mean()),'prediction_delta_rms':float(np.sqrt(((pred-d['prob'])**2).mean())),'correlation_covariance':rankstats(cov),'effective_dof_lambda01':float((ev/(ev+.01)).sum()),'active_features':active,'ridge_retention_quantiles':np.quantile(ev[ev>1e-12]/(ev[ev>1e-12]+.01),[0,.1,.5,.9,1]).tolist(),'block_contributions':p}
  if cell=='SU':
   # Covariance-only linear redundancy, unsupervised; no residual labels or new predictor.
   a=cov[:39,:39];cross=cov[:39,39:];u=cov[39:,39:];schur=u-cross.T@np.linalg.pinv(a,rcond=1e-10)@cross
   res['uav_covariance_remaining_after_shared']={'trace_fraction':float(np.trace(schur)/np.trace(u)),'rankstats':rankstats(schur)}
  del z,x
 if split=='validation':
  saved=np.load(O/'val_predictions.npz');res['saved_prediction_max_abs_difference']={k:float(np.abs(preds[k]-saved[k]).max()) for k in preds};res['error_categories']={}
  for name,mask in [('empty',d['target']==0),('partial',(d['target']>0)&(d['target']<1)),('full',d['target']==1)]:
   r={'n_cells':int(mask.sum())}
   for c in preds:r[c+'_mse']=float(((preds[c]-d['target'])**2)[mask].mean())
   r['SU_minus_SG_reduction']=r['SG_mse']-r['SU_mse'];r['SU_minus_SX_reduction']=r['SX_mse']-r['SU_mse'];res['error_categories'][name]=r
 out['splits'][split]=res
 del d
tr=out['splits']['train'];va=out['splits']['validation'];out['train_val_feature_drift']={}
for k in tr['features']:
 t=tr['features'][k];v=va['features'][k];delta=(np.array(v['mean'])-t['mean'])/t['std'];ratio=np.array(v['std'])/t['std'];out['train_val_feature_drift'][k]={'mean_shift_train_std_rms':float(np.sqrt((delta*delta).mean())),'std_ratio_min_median_max':np.quantile(ratio,[0,.5,1]).tolist()}
OUT.write_text(json.dumps(out,indent=2));print(str(OUT));print(json.dumps({s:{'residual':out['splits'][s]['residual'],'uav_remaining':out['splits'][s]['uav_covariance_remaining_after_shared'],'ranks':{k:{'within':v['within_roi_variance_fraction'],'raw_rank':v['raw_covariance']['participation_ratio'],'corr_rank':v['correlation']['participation_ratio']} for k,v in out['splits'][s]['features'].items()},'SU':out['splits'][s]['existing_models']['SU']} for s in out['splits']},indent=2));print(json.dumps(out['train_val_feature_drift'],indent=2))
