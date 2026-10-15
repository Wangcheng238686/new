"""Independent root reconstruction from feature files, no author summary imports."""
from pathlib import Path
from collections import defaultdict
import json,hashlib,numpy as np
p=Path('/tmp/ue_uav_experiments/20261014_ad_seed42/final_analysis/uav_complementarity');m=json.loads((p/'features/run_manifest.json').read_text());s=json.loads((p/'summary.json').read_text());models=json.loads((p/'ridge_models.json').read_text());assert m['complete'] and m['processed_images']==281
D={}
for split in ['train','validation']:
 d=defaultdict(list)
 for row in m['records']:
  if row['split']!=split:continue
  assert hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()==row['sha256']
  with np.load(row['path']) as f:
   for k in f.files:d[k].append(f[k])
 D[split]={k:np.concatenate(v) for k,v in d.items()}
t=D['train'];v=D['validation'];assert not set(t['global_id'])&set(v['global_id']);assert len(np.unique(t['image_id']))==128 and len(np.unique(v['image_id']))==153
pred={};byimage=[np.where(v['image_id']==i)[0] for i in np.unique(v['image_id'])];counts=np.array([len(ix) for ix in byimage]);buildings=[np.where(v['global_id']==g)[0] for g in np.unique(v['global_id'])];target=(t['target']-t['prob']).reshape(-1).astype(np.float64);intercept=target.mean()
for cell,key in [('SG',None),('S64','sat_extra32'),('SU','uav32'),('SX','shift32'),('SN','noise32')]:
 extra=np.zeros_like(t['uav32']) if key is None else t[key];x=np.concatenate([t['sat32'],t['geometry'],extra],axis=2).reshape(-1,71).astype(np.float64);mean=x.mean(axis=0);std=np.maximum(x.std(axis=0),1e-6);z=(x-mean)/std
 beta=np.linalg.solve((z.T@z)/len(z)+np.eye(71)*.01,z.T@(target-intercept)/len(z));fit=models[cell];assert np.allclose(mean,fit['mean'],atol=1e-12,rtol=0) and np.allclose(std,fit['std'],atol=1e-12,rtol=0);assert np.allclose(beta,fit['beta'],atol=1e-10,rtol=0) and intercept==fit['intercept']
 xx=np.concatenate([v['sat32'],v['geometry'],np.zeros_like(v['uav32']) if key is None else v[key]],axis=2).astype(np.float64);pp=np.clip(v['prob']+(xx-mean)/std@beta+intercept,0,1);br=((pp-v['target'])**2).mean(axis=1);tp=pp>=.5;tt=v['target']>=.5;io=(tp&tt).sum(1)/np.maximum((tp|tt).sum(1),1);a=s['metrics'][cell]['all'];assert abs(br.mean()-a['brier'])<1e-12 and abs(io.mean()-a['binary8_iou'])<1e-12
 assert abs(np.mean([br[ix].mean() for ix in buildings])-a['building_equal_brier'])<1e-12;pred[cell]=(br,io)
 del x,z,xx,pp
rng=np.random.default_rng(1790);draw=rng.integers(0,153,(1000,153));out={}
for control in ['SG','SX','S64','SN']:
 delta=pred[control][0]-pred['SU'][0];key='SU_vs_'+control;z=s['contrasts'][key]['all'];assert abs(delta.mean()-z['brier_reduction'])<1e-12
 if control in ['SG','SX']:
  sums=np.array([delta[ix].sum() for ix in byimage]);bs=sums[draw].sum(1)/counts[draw].sum(1);ci=np.quantile(bs,[.0125,.9875]);assert np.allclose(ci,z['bonferroni_two_primary_CI95'],atol=1e-12,rtol=0);assert s['gates'][control]==bool(delta.mean()>=.001 and ci[0]>0)
  out[key]={'brier_reduction':float(delta.mean()),'interval':ci.tolist()}
 else:out[key]={'brier_reduction':float(delta.mean())}
print(json.dumps({'verdict':'PASS','all_five_train_only_ridge_coefficients_reproduced':True,'train_rois':len(t['prob']),'val_rois':len(v['prob']),'comparisons':out,'overall_gate':s['overall_gate']},indent=2))
