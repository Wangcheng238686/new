import json,hashlib
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
O=Path('/tmp/ue_uav_experiments/20261014_ad_seed42/final_analysis/uav_complementarity')
D=O.parent
meta=json.loads((O/'features/run_manifest.json').read_text())
def load(split):
 fs=[]
 for r in meta['records']:
  if r['split']!=split:continue
  assert hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['sha256']
  with np.load(r['path']) as f:fs.append({k:f[k] for k in f.files})
 return {k:np.concatenate([f[k] for f in fs]) for k in fs[0]}
tr=load('train');va=load('validation');pred=np.load(O/'val_predictions.npz');n=len(va['prob']);assert n==4496
for k,v in [('baseline','prob'),('truth','target'),('image_id','image_id'),('global_id','global_id')]:assert np.array_equal(pred[k],va[v])
gid=va['global_id'];ids=va['image_id'];area=np.prod(va['boxes'][:,2:]-va['boxes'][:,:2],1)
def bm(x,g=gid):return float(np.mean([x[g==i].mean() for i in np.unique(g)]))
def summary(x,m):
 z=x[m];g=gid[m]
 return dict(n=int(m.sum()),buildings=int(len(np.unique(g))),mean=float(z.mean()) if len(z) else None,building_equal_mean=bm(z,g) if len(z) else None,median=float(np.median(z)) if len(z) else None,positive_fraction=float((z>0).mean()) if len(z) else None,gt_001_fraction=float((z>.001).mean()) if len(z) else None,lt_minus001_fraction=float((z<-.001).mean()) if len(z) else None)
coverage={s:d['geometry'][:,:,-1].mean(1) for s,d in [('train',tr),('val',va)]}
err={s:((d['prob']-d['target'])**2).mean(1) for s,d in [('train',tr),('val',va)]}
cuts={k:np.quantile(v['train'],[1/3,2/3]).tolist() for k,v in [('coverage',coverage),('coarse_error',err)]}
masks={'all':np.ones(n,bool),'small':area<1024,'medium':(area>=1024)&(area<9216),'large':area>=9216,'coverage_zero':coverage['val']==0,'coverage_positive':coverage['val']>0}
for name,v in [('coverage',coverage),('coarse_error',err)]:
 c=np.array(cuts[name]);cl=np.searchsorted(c,v['val'],side='right')
 for j,lab in enumerate(['low','mid','high']):masks[name+'_'+lab]=cl==j
for s in ['small','medium','large']:
 for cov in ['zero','positive','low','mid','high']:masks[s+'_coverage_'+cov]=masks[s]&masks['coverage_'+cov]
base=json.loads((D/'gt_prompt_factorial/full/baseline.json').read_text());oracle=json.loads((D/'gt_prompt_factorial/full/gt_dense_only.json').read_text())
b={(x['image_id'],x['gt_slot']):x for x in base};o={(x['image_id'],x['gt_slot']):x for x in oracle}
head=[];ds=[]
for i,(im,sl,g) in enumerate(zip(ids,va['gt_slot'],gid)):
 k=(int(im),int(sl));assert int(b[k]['global_id'])==int(g)==int(o[k]['global_id']);head.append(o[k]['gt_iou']-b[k]['gt_iou']);ds.append(b[k]['dense_support_cells'])
head=np.array(head);ds=np.array(ds)
masks.update(oracle_nonpositive=head<=0,oracle_tiny=(head>0)&(head<=.002),oracle_moderate=(head>.002)&(head<=.02),oracle_large=head>.02,dense_support_zero=ds==0)
br={c:(pred[c]-va['target'])**2 for c in ['SG','SX','S64','SN','SU']}
delta={c:br[c]-br['SU'] for c in ['SG','SX','S64','SN']}
contrasts={c:{k:summary(d.mean(1),m) for k,m in masks.items()} for c,d in delta.items()}
regions={'pure_background':va['target']<=1e-6,'pure_foreground':va['target']>=1-1e-6,'mixed_occupancy':(va['target']>1e-6)&(va['target']<1-1e-6)}
reg={}
for r,rm in regions.items():
 reg[r]={}
 for c,d in delta.items():
  cnt=rm.sum(1);ok=cnt>0;v=(d*rm).sum(1)/np.maximum(cnt,1)
  reg[r][c]={'n_cells':int(rm.sum()),'n_rois':int(ok.sum()),'cell_equal_mean':float(d[rm].mean()),'roi_equal_mean':float(v[ok].mean()),'building_equal_roi_mean':bm(v[ok],gid[ok])}
vars={}
for name,k in [('sat_variation','sat32'),('uav_variation','uav32')]:
 scale=tr[k].std(axis=(0,1)).clip(1e-6);v=va[k];vars[name]=np.sqrt(np.mean(((v-v.mean(1,keepdims=True))/scale)**2,axis=(1,2)))
scale=tr['uav32'].std(axis=(0,1)).clip(1e-6);vars['aligned_shift_distance']=np.sqrt(np.mean(((va['uav32']-va['shift32'])/scale)**2,axis=(1,2)))
vars.update(coverage=coverage['val'],coarse_error=err['val'],oracle_headroom=head)
cor={}
for name,v in vars.items():
 cor[name]={'descriptor_mean':float(v.mean()),'descriptor_quantiles':np.quantile(v,[0,.25,.5,.75,1]).tolist()}
 for c in ['SG','SX','S64']:
  d=delta[c].mean(1);rho=float(spearmanr(v,d).statistic);gs=np.unique(gid);gv=np.array([v[gid==g].mean() for g in gs]);gd=np.array([d[gid==g].mean() for g in gs]);cor[name][c]={'rho_appearances':rho,'rho_building_means':float(spearmanr(gv,gd).statistic)}
# fixed diagnostic ordering, not representative examples
ok=(head>.02)&(delta['SG'].mean(1)<=0);idx=np.where(ok)[0];idx=idx[np.argsort(-head[idx],kind='stable')][:10]
cases=[dict(image_id=int(ids[i]),gt_slot=int(va['gt_slot'][i]),global_id=int(gid[i]),area=float(area[i]),coverage=float(coverage['val'][i]),baseline8_brier=float(err['val'][i]),su_vs_sg=float(delta['SG'].mean(1)[i]),su_vs_sx=float(delta['SX'].mean(1)[i]),gt_dense_iou_headroom=float(head[i])) for i in idx]
output=dict(scope='Exploratory existing frozen readout stratification; no fitting, GPU, AP or new success gate',n=n,n_buildings=len(np.unique(gid)),n_images=len(np.unique(ids)),train_cuts=cuts,strata_counts={k:int(v.sum()) for k,v in masks.items()},contrasts=contrasts,regions=reg,descriptive_correlations=cor,oracle_iou_headroom={k:summary(head,m) for k,m in masks.items()},baseline_error_by_stratum={k:summary(err['val'],m) for k,m in masks.items()},coverage_by_stratum={k:summary(coverage['val'],m) for k,m in masks.items()},train_val_descriptors={key:{sp:dict(mean=float(v[sp].mean()),n=len(v[sp])) for sp in ['train','val']} for key,v in [('coverage',coverage),('coarse_error',err)]},diagnostic_cases=cases,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),prereg_sha256=hashlib.sha256(Path('/tmp/ue_bev_error_strata_prereg.md').read_bytes()).hexdigest())
sources={'cache_manifest':O/'features/run_manifest.json','val_predictions':O/'val_predictions.npz','ridge_models':O/'ridge_models.json','oracle_baseline':D/'gt_prompt_factorial/full/baseline.json','oracle_gt_dense':D/'gt_prompt_factorial/full/gt_dense_only.json','script':Path(__file__),'prereg':Path('/tmp/ue_bev_error_strata_prereg.md')}
output['source_sha256']={k:dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for k,p in sources.items()}
Path('/tmp/ue_bev_error_strata_summary.json').write_text(json.dumps(output,indent=2));print(json.dumps({k:output[k] for k in ['n','n_buildings','train_cuts','strata_counts','regions','descriptive_correlations']},indent=2))
