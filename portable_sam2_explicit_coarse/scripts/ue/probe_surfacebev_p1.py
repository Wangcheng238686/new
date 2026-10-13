#!/usr/bin/env python3
"""Recoverable P1-V2 frozen readout; test is opened only after val selection."""
from __future__ import annotations
import argparse, hashlib, json, os, sys
from pathlib import Path
import cv2, numpy as np, torch, torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from data.ue_raw_pair_dataset import UERawPairDataset
from inference.infer_from_checkpoint import _load_checkpoint,_load_model_state,_register_and_build,_resolve_model_config,_resolve_sam2_repo,_select_state_dict,_snapshot
from rsprompter.surfacebev import rasterize_surfacebev
from probe_surfacebev_p1_h0 import cameras,make_uav_encoder,uav_descriptors,sha_file,model_input,digest_predictions
from ue.features import build_baseline, observe_satellite, implementation_fingerprints
from ue.contract import verify_identity, guard_cache
SEED=20261012;ARMS=('SAT','correct','sham90');CS=(.01,.1,1.,10.);THRS=np.arange(.1,1,.1);VERSION='UE-WHU-P1-v1'
def args():
 p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--phase',choices=('harvest','evaluate'),required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--baseline-contract',type=Path,required=True);p.add_argument('--h0',type=Path,required=True);p.add_argument('--experiment-id',required=True);p.add_argument('--root',type=Path,default=ROOT/'data/ue_raw_hero_uav_pairset_20261011');p.add_argument('--sam2-ckpt',default=os.environ.get('SAM2_CKPT'));p.add_argument('--max-train',type=int,default=0);p.add_argument('--max-val',type=int,default=0);p.add_argument('--allow-test',action='store_true');return p.parse_args()
def boundary(m):m=m.astype(np.uint8);return cv2.dilate(m,np.ones((5,5),np.uint8))-cv2.erode(m,np.ones((5,5),np.uint8))>0
def p3mask(m):return F.max_pool2d(torch.from_numpy(m.astype(np.float32))[None,None],8,8)[0,0].numpy().astype(np.uint8).reshape(-1)
def f1(p,y):t=((p>0)&(y>0)).sum();return float(2*t/max(2*t+(p>0).sum()+(y>0).sum()-2*t,1))
def boot(x):r=np.random.default_rng(SEED);d=x[r.integers(0,len(x),(10000,len(x)))].mean(1);return {'mean':float(x.mean()),'ci95':[float(v) for v in np.quantile(d,[.025,.975])]}
def atomic_json(p,d):
 q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(d,indent=2));os.replace(q,p)
def atomic_npz(p,**d):
 q=p.with_suffix(p.suffix+'.tmp')
 with open(q,'wb') as f:np.savez_compressed(f,**d)
 os.replace(q,p)
def preflight(a):
 if not a.h0.is_file():raise RuntimeError('H0 artifact missing')
 h=json.loads(a.h0.read_text())
 checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
 identity=verify_identity(a.checkpoint,checkpoint,a.baseline_contract)
 if h.get('baseline_identity')!=identity:raise RuntimeError('H0 baseline identity mismatch')
 if h.get('implementation_fingerprints')!=implementation_fingerprints():raise RuntimeError('H0 implementation fingerprint mismatch')
 smoke=a.phase=='harvest' and bool(a.max_train or a.max_val)
 if h.get('verdict')!='H0_PASS' and not (smoke and h.get('verdict')=='SMOKE_ONLY'):raise RuntimeError('H0 not passed')
 if identity['validation_only'] and (not smoke or a.allow_test):raise RuntimeError('temporary baseline is smoke only')
 mark=torch.arange(128*128).reshape(128,128);pack=mark[None,:,:,None].repeat(12,1,1,10).permute(1,2,0,3).reshape(-1,120)
 if not torch.equal(pack[:,0],mark.reshape(-1)) or not torch.equal(pack[:,110],mark.reshape(-1)):raise RuntimeError('P3/SurfaceBEV packing failed')
 return h
def make_contract(a,h):
 return {'version':VERSION,'seed':SEED,'baseline_identity':h['baseline_identity'],
         'h0_sha256':sha_file(a.h0),'split':'surfbev_b0_split_v2.json',
         'height_edges_cm':h['height_edges_cm'],'arms':list(ARMS),'dim':64,
         'per_class_cap':256,'experiment_id':a.experiment_id,
         'max_train':a.max_train,'max_val':a.max_val,
         'implementation_fingerprints':implementation_fingerprints()}
def cache_contract(cache,a,h):
 guard_cache(cache/'contract.json',make_contract(a,h))
@torch.inference_mode()
def pred_p3(model,item,dev):
 features,outputs,digest=observe_satellite(model,item,dev)
 p3=features.pyramids[1][0].float()
 x=getattr(outputs[0].pred_instances,'masks')
 x=x.to_ndarray() if hasattr(x,'to_ndarray') else (x.masks if hasattr(x,'masks') else x)
 return p3.detach().cpu().numpy(),(x.detach().cpu().numpy() if isinstance(x,torch.Tensor) else np.asarray(x)).astype(bool),digest
def build(a,dev):
 model,_,_,_=build_baseline(a.checkpoint,a.baseline_contract,a.sam2_ckpt,dev,
                           allow_validation=a.phase=='harvest' and bool(a.max_train or a.max_val))
 return model,make_uav_encoder(a,dev)
def projections(dev):
 r=np.random.default_rng(SEED);return {'sat':r.normal(size=(512,64)).astype('float32')/8,'desc':torch.tensor(r.normal(size=(256,4)).astype('float32')/16,device=dev),'comb':r.normal(size=(184,64)).astype('float32')/np.sqrt(184)}
def feature(p3,surf,arm,p):
 s=torch.from_numpy(p3).float().permute(1,2,0).reshape(-1,512).numpy()@p['sat']
 if arm=='SAT':return s.astype('float32')
 q=torch.einsum('bchw,cd->bdhw',surf[:,:256],p['desc']).permute(0,2,3,1);q=torch.cat((q,surf[:,256:].permute(0,2,3,1)),3).permute(1,2,0,3).reshape(-1,120).cpu().numpy()
 return(np.c_[s,q]@p['comb']).astype('float32')
def pool(x,masks):
 o=[]
 for m in masks:
  q=p3mask(m).astype(bool);o.append(x[q].mean(0) if q.any() else np.zeros(x.shape[1]))
 return np.asarray(o,np.float32)
def iiou(pred,gt):
 if not len(pred):return np.zeros(0,np.float32)
 if not len(gt):return np.zeros(len(pred),np.float32)
 a=pred.reshape(len(pred),-1).astype(np.int32);b=gt.reshape(len(gt),-1).astype(np.int32);i=a@b.T;d=a.sum(1)[:,None]+b.sum(1)[None]-i;return(i/np.maximum(d,1)).max(1).astype(np.float32)
def pack_scene(path,split,i,item,p3,pred,sc,ss,p):
 x={a:feature(p3,sc if a!='sham90' else ss,a,p) for a in ARMS};gt=np.any(item['gt_masks'].astype(bool),0) if len(item['gt_masks']) else np.zeros((1024,1024),bool);yb=p3mask(boundary(gt));pe=np.any(pred,0) if len(pred) else np.zeros_like(gt);ye=p3mask(boundary(pe)^boundary(gt));q=iiou(pred,item['gt_masks']);d={'scene_id':np.array(item['img_metas']['scene_id']),'yb':yb,'ye':ye,'pred':pred,'qi':q}
 if split=='train':
  for tid,(name,y) in enumerate((('boundary',yb),('error',ye))):
   pos=np.flatnonzero(y);neg=np.flatnonzero(~y.astype(bool));n=min(256,len(pos),len(neg))
   if not n:raise RuntimeError('class missing')
   r=np.random.default_rng(np.random.SeedSequence([SEED,i,tid]));sel=np.r_[r.choice(pos,n,False),r.choice(neg,n,False)];d['y_'+name]=y[sel]
   for a in ARMS:d[name+'_'+a]=x[a][sel]
  d['y_iou']=q
  for a in ARMS:d['iou_'+a]=pool(x[a],pred)
 else:
  for a in ARMS:d['x_'+a]=x[a]
 atomic_npz(path,**d)
def run_harvest(a,out,h):
 if a.allow_test:raise RuntimeError('harvest must not allow test')
 cache=out/'cache';cache.mkdir(parents=True,exist_ok=True);cache_contract(cache,a,h);dev=torch.device(a.device);m,e=build(a,dev);ed=torch.tensor(h['height_edges_cm'],device=dev);p=projections(dev);ds={s:UERawPairDataset(a.root,s,True,'surfbev_b0_split_v2.json',load_targets=True) for s in ('train','val')};tele=cache/'telemetry_train_val.jsonl'
 for s,lim in (('train',a.max_train),('val',a.max_val)):
  n=len(ds[s]) if not lim else min(lim,len(ds[s]));d=cache/s;d.mkdir(exist_ok=True)
  for i in range(n):
   f=d/f'{i:04d}.npz'
   if f.exists():print(s,i+1,'/',n,'cached',flush=True);continue
   it=ds[s][i];p3,pred,ph=pred_p3(m,it,dev);desc=uav_descriptors(e,it['drone_images'].to(dev));uk,uc,sk,sw=cameras(it,dev);sc,ct=rasterize_surfacebev(desc,it['devicez_rgb'].to(dev),uk,uc,sk,sw,ed,(128,128));ss,st=rasterize_surfacebev(desc,it['devicez_rgb'].to(dev),uk,uc,sk,sw,ed,(128,128),True)
   if ct['input_points']!=st['input_points'] or not(torch.isfinite(sc).all() and torch.isfinite(ss).all()):raise RuntimeError('surface contract')
   for z in(ct,st):
    if z['zbuffer_points']<=0 or z['covered_cells']<=0 or not np.isfinite(z['zbuffer_residual_mean']):raise RuntimeError('SurfaceBEV telemetry contract')
   pack_scene(f,s,i,it,p3,pred,sc,ss,p)
   with open(tele,'a') as q:q.write(json.dumps({'split':s,'scene_id':it['img_metas']['scene_id'],'prediction_hash':ph,'descriptor_shape':list(desc.shape),'input_equal':True,'correct':ct,'sham90':st})+'\n')
   print(s,i+1,'/',n,flush=True)
 if a.max_train or a.max_val:
  atomic_json(out/'smoke_summary.json',{'status':'SMOKE_ONLY','baseline_identity':h['baseline_identity'],'train_used':min(a.max_train,len(ds['train'])),'val_used':min(a.max_val,len(ds['val'])),'test_read':False,'evaluation_run':False})
  return
 atomic_json(cache/'train_val_complete.json',{'train':len(ds['train']),'val':len(ds['val'])})
def trainxy(cache,target,arm):
 xs=[];ys=[]
 for f in sorted((cache/'train').glob('*.npz')):
  z=np.load(f);xs.append(z[target+'_'+arm]);ys.append(z['y_'+target])
 return np.concatenate(xs),np.concatenate(ys)
def fit(cache,C):
 m={}
 for target in ('boundary','error'):
  m[target]={}
  for arm in ARMS:
   x,y=trainxy(cache,target,arm);s=StandardScaler().fit(x);c=LogisticRegression(C=C,class_weight='balanced',max_iter=2000,random_state=SEED).fit(s.transform(x),y);m[target][arm]=(s,c);del x,y
 m['iou']={}
 for arm in ARMS:
  x,y=trainxy(cache,'iou',arm);s=StandardScaler().fit(x);c=Ridge(alpha=1/C).fit(s.transform(x),y);m['iou'][arm]=(s,c);del x,y
 return m
def evaluate(files,m,t):
 rows=[]
 for f in files:
  z=np.load(f);r={'scene_id':str(z['scene_id'])}
  for a in ARMS:
   x=z['x_'+a];pb=m['boundary'][a][1].predict_proba(m['boundary'][a][0].transform(x))[:,1];pe=m['error'][a][1].predict_proba(m['error'][a][0].transform(x))[:,1];pi=m['iou'][a][1].predict(m['iou'][a][0].transform(pool(x,z['pred']))) if len(z['pred']) else np.zeros(0)
   r[a]={'f1':f1(pb>=t,z['yb']),'auc':float(roc_auc_score(z['ye'],pe)) if len(np.unique(z['ye']))==2 else .5,'rho':float(spearmanr(pi,z['qi']).statistic) if len(z['qi'])>1 and np.std(z['qi'])>0 else 0.}
  rows.append(r)
 return rows
def harvest_test(a,cache,h,sel):
 lock=cache/'test_opened_once.json'
 if not lock.exists():atomic_json(lock,{'selected':sel,'reason':'val selection frozen before test'})
 elif json.loads(lock.read_text()).get('selected')!=sel:raise RuntimeError('test lock selection mismatch')
 dev=torch.device(a.device);m,e=build(a,dev);ed=torch.tensor(h['height_edges_cm'],device=dev);p=projections(dev);ds=UERawPairDataset(a.root,'test',True,'surfbev_b0_split_v2.json',load_targets=True);d=cache/'test';d.mkdir(exist_ok=True);tele=cache/'telemetry_test.jsonl'
 for i in range(len(ds)):
  f=d/f'{i:04d}.npz'
  if f.exists():print('test',i+1,'/',len(ds),'cached',flush=True);continue
  it=ds[i];p3,pred,ph=pred_p3(m,it,dev);desc=uav_descriptors(e,it['drone_images'].to(dev));uk,uc,sk,sw=cameras(it,dev);sc,ct=rasterize_surfacebev(desc,it['devicez_rgb'].to(dev),uk,uc,sk,sw,ed,(128,128));ss,st=rasterize_surfacebev(desc,it['devicez_rgb'].to(dev),uk,uc,sk,sw,ed,(128,128),True)
  if ct['input_points']!=st['input_points'] or not(torch.isfinite(sc).all() and torch.isfinite(ss).all()):raise RuntimeError('surface contract')
  for z in(ct,st):
   if z['zbuffer_points']<=0 or z['covered_cells']<=0 or not np.isfinite(z['zbuffer_residual_mean']):raise RuntimeError('SurfaceBEV telemetry contract')
  pack_scene(f,'test',i,it,p3,pred,sc,ss,p)
  with open(tele,'a') as q:q.write(json.dumps({'split':'test','scene_id':it['img_metas']['scene_id'],'prediction_hash':ph,'descriptor_shape':list(desc.shape),'input_equal':True,'correct':ct,'sham90':st})+'\n')
  print('test',i+1,'/',len(ds),flush=True)
def run_eval(a,out,h):
 if not a.allow_test:raise RuntimeError('evaluation requires --allow-test')
 cache=out/'cache';cache_contract(cache,a,h)
 if json.loads((cache/'train_val_complete.json').read_text())!={'train':694,'val':100}:raise RuntimeError('train/val cache incomplete')
 vf=sorted((cache/'val').glob('*.npz'));best=None
 for C in CS:
  m=fit(cache,C)
  for t in THRS:
   v=evaluate(vf,m,t);cand=(np.mean([x['correct']['f1'] for x in v]),-C,-t,C,t,m)
   if best is None or cand[:3]>best[:3]:best=cand
 _,_,_,C,t,m=best;sel={'C':C,'threshold':t};harvest_test(a,cache,h,sel);tf=sorted((cache/'test').glob('*.npz'))
 if len(tf)!=151:raise RuntimeError('test cache incomplete')
 rows=evaluate(tf,m,t);met={a:{q:np.array([x[a][q] for x in rows]) for q in ('f1','auc','rho')} for a in ARMS};ds={}
 for q in ('f1','auc','rho'):
  for b in ('SAT','sham90'):ds['correct_minus_'+b+'_'+q]=boot(met['correct'][q]-met[b][q])
 passed=all(v['ci95'][0]>0 for v in ds.values());summary={'experiment_id':a.experiment_id,'baseline_identity':h['baseline_identity'],'status':'complete','decision':'P1_PASS' if passed else 'P1_FAIL','selected':sel,'n':{'train':694,'val':100,'test':151},'arms':{a:{q:float(met[a][q].mean()) for q in met[a]} for a in ARMS},'bootstrap':ds,'invariants':{'test_opened_once':True,'parity_and_surface_contract':True,'same_raw_input':True,'train_val_test_processed':[694,100,151],'cache_contract':json.loads((cache/'contract.json').read_text())}}
 atomic_json(out/'summary.json',summary);(out/'per_scene.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows));print(json.dumps(summary,indent=2))
def main():
 a=args();
 if a.max_train<0 or a.max_val<0 or (bool(a.max_train)!=bool(a.max_val)):raise ValueError('smoke requires positive limits for both train and val')
 h=preflight(a);out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True);
 if any(out.iterdir()) and not (out/'cache/contract.json').is_file():raise RuntimeError('output directory contains historical/unidentified artifacts')
 run_harvest(a,out,h) if a.phase=='harvest' else run_eval(a,out,h)
if __name__=='__main__':main()
