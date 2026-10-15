from pathlib import Path
from collections import defaultdict
import json,numpy as np
O=Path('/tmp/ue_uav_experiments/20261014_ad_seed42/final_analysis/uav_complementarity');m=json.loads((O/'features/run_manifest.json').read_text());model=json.loads((O/'ridge_models.json').read_text())['SU'];readout=json.loads(Path('/tmp/ue_readout_capacity_audit.json').read_text());strata=json.loads(Path('/tmp/ue_bev_error_strata_summary.json').read_text());sets={};report={}
for split in ['train','validation']:
 data=defaultdict(list);totals={k:dict(n=0,sm=np.zeros(32),sq=np.zeros((32,32)),within=0.) for k in ['sat32','uav32']};zn=0;zs=np.zeros(71);zz=np.zeros((71,71))
 for r in m['records']:
  if r['split']!=split:continue
  with np.load(r['path']) as f:
   for k in ['image_id','global_id','gt_slot','geometry','prob','target','boxes']:data[k].append(f[k])
   for k in ['sat32','uav32']:
    a=f[k].astype(np.float64);flat=a.reshape(-1,32);t=totals[k];t['n']+=len(flat);t['sm']+=flat.sum(0);t['sq']+=flat.T@flat;t['within']+=np.square(a-a.mean(1,keepdims=True)).sum()
   if split=='train':
    x=np.concatenate([f['sat32'],f['geometry'],f['uav32']],-1).reshape(-1,71).astype(np.float64);z=(x-model['mean'])/model['std'];zn+=len(z);zs+=z.sum(0);zz+=z.T@z
 d={k:np.concatenate(v) for k,v in data.items()};sets[split]=d;out={}
 for k,t in totals.items():
  cov=t['sq']/t['n']-np.outer(t['sm']/t['n'],t['sm']/t['n']);within=t['within']/t['n']/np.trace(cov);corr=cov/np.sqrt(np.outer(np.diag(cov),np.diag(cov)));ev=np.maximum(np.linalg.eigvalsh(corr),0);pr=ev.sum()**2/(ev@ev);a=readout['splits'][split]['features'][k];assert abs(within-a['within_roi_variance_fraction'])<1e-9 and abs(pr-a['correlation']['participation_ratio'])<1e-8;out[k]={'within_roi_fraction':float(within),'correlation_rank':float(pr)}
 if split=='train':
  cov=zz/zn-np.outer(zs/zn,zs/zn);ev=np.maximum(np.linalg.eigvalsh(cov),0);ret=ev/(ev+.01);dof=ret.sum();a=readout['splits']['train']['existing_models']['SU'];assert abs(dof-a['effective_dof_lambda01'])<1e-8;assert np.allclose(np.quantile(ret,[0,.1,.5,.9,1]),a['ridge_retention_quantiles'],atol=1e-8,rtol=0);out['ridge_dof']=float(dof)
 report[split]=out
tr=sets['train'];va=sets['validation'];cov={k:d['geometry'][:,:,-1].mean(1) for k,d in sets.items()};err={k:np.square(d['prob']-d['target']).mean(1) for k,d in sets.items()};cuts={'coverage':np.quantile(cov['train'],[1/3,2/3]),'coarse_error':np.quantile(err['train'],[1/3,2/3])}
for k,v in cuts.items():assert np.allclose(v,strata['train_cuts'][k],rtol=0,atol=1e-12)
pred=np.load(O/'val_predictions.npz');n=len(va['prob']);assert n==4496 and np.array_equal(pred['truth'],va['target']);oracle={}
R=O.parent
for name in ['baseline','gt_dense_only']:oracle[name]={(r['image_id'],r['gt_slot']):r for r in json.loads((R/'gt_prompt_factorial/full'/(name+'.json')).read_text())}
h=np.array([oracle['gt_dense_only'][(int(i),int(j))]['gt_iou']-oracle['baseline'][(int(i),int(j))]['gt_iou'] for i,j in zip(va['image_id'],va['gt_slot'])]);groups={'all':np.ones(n,bool),'coverage_zero':cov['validation']==0,'coarse_error_high':np.searchsorted(cuts['coarse_error'],err['validation'],side='right')==2,'oracle_nonpositive':h<=0,'oracle_large':h>.02}
res={}
for c in ['SG','SX','S64']:
 loss=np.square(pred[c]-va['target'])-np.square(pred['SU']-va['target']);delta=loss.mean(1);res[c]={}
 for k,mask in groups.items():
  z=strata['contrasts'][c][k];mean=float(delta[mask].mean());assert int(mask.sum())==z['n'] and abs(mean-z['mean'])<1e-12;res[c][k]={'n':int(mask.sum()),'mean':mean}
 for k,mask in [('pure_background',va['target']<=1e-6),('pure_foreground',va['target']>=1-1e-6),('mixed_occupancy',(va['target']>1e-6)&(va['target']<1-1e-6))]:
  cnt=mask.sum(1);ok=cnt>0;roi=(loss*mask).sum(1)/np.maximum(cnt,1);z=strata['regions'][k][c];assert int(mask.sum())==z['n_cells'] and int(ok.sum())==z['n_rois'];assert abs(float(loss[mask].mean())-z['cell_equal_mean'])<1e-12 and abs(float(roi[ok].mean())-z['roi_equal_mean'])<1e-12
report.update(verdict='PASS',strata=res,train_quantile_cuts={k:v.tolist() for k,v in cuts.items()});print(json.dumps(report,indent=2))
