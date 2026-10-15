"""One prespecified train-only ridge fit per cell; validation evaluation only."""
import hashlib,json,subprocess
from pathlib import Path
import numpy as np
O=Path('/tmp/ue_uav_experiments/20261014_ad_seed42/final_analysis/uav_complementarity')
CELLS=['SG','S64','SU','SX','SN'];EXTRA={'S64':'sat_extra32','SU':'uav32','SX':'shift32','SN':'noise32'}
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def key(i):return hashlib.sha256(('uav-complementarity-v1:'+str(i)).encode()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2))
def load(split):
 rows=[z for z in read(O/'features/run_manifest.json')['records'] if z['split']==split];files=[];seen=set()
 for z in rows:
  assert sha(z['path'])==z['sha256'];f=np.load(z['path']);n=z['n_rois'];assert set(f['image_id'])=={z['image_id']} and np.array_equal(f['gt_slot'],np.arange(n))
  for k in f.files:assert len(f[k])==n and np.isfinite(f[k]).all(),k
  for k in ['sat32','sat_extra32','uav32','shift32','noise32']:assert f[k].shape==(n,64,32)
  assert f['geometry'].shape==(n,64,7) and f['prob'].shape==f['target'].shape==(n,64)
  assert ((f['prob']>=0)&(f['prob']<=1)).all() and ((f['target']>=0)&(f['target']<=1)).all()
  for slot in f['gt_slot']:
   identity=(z['image_id'],int(slot));assert identity not in seen;seen.add(identity)
  files.append(f)
 return {k:np.concatenate([f[k] for f in files]) for k in files[0].files}
def design(d,cell):
 extra=d[EXTRA[cell]] if cell in EXTRA else np.zeros_like(d['uav32']);return np.concatenate([d['sat32'],d['geometry'],extra],-1).astype(np.float64)
def roi_stats(pred,target):
 brier=((pred-target)**2).mean(1);p=pred>=.5;t=target>=.5;iou=(p&t).sum(1)/np.maximum((p|t).sum(1),1);return brier,iou

def agg(v,mask):return float(v[mask].mean()) if mask.any() else None
def bootstrap(v,ids,alpha=.025):
 uid=np.unique(ids);s=np.array([v[ids==i].sum() for i in uid]);n=np.array([(ids==i).sum() for i in uid]);rng=np.random.default_rng(1790);a=rng.integers(len(uid),size=(1000,len(uid)));b=s[a].sum(1)/n[a].sum(1);return np.quantile(b,[alpha,1-alpha]).tolist()
def buildingmean(v,gid):return float(np.mean([v[gid==g].mean() for g in np.unique(gid)]))
def main():
 meta=read(O/'features/run_manifest.json');assert meta['complete'] and meta['processed_images']==281
 assert meta['weights']=='ema' and meta['training'] is False and meta['optimizer'] is False
 expectedrun=read('/tmp/ue_uav_experiments/20261014_ad_seed42/final_analysis/dual_correct/run_manifest.json');assert meta['architecture_contract']==expectedrun['architecture_contract']
 assert meta['strict_load']['missing_keys']==[] and meta['strict_load']['unexpected_keys']==[]
 code="import torch,numpy as np,hashlib,json; out=[]\nfor c,seed in [(512,1811),(512,1812),(256,1813)]:\n g=torch.Generator(device='cpu').manual_seed(seed);a=(torch.randn(c,32,generator=g)/(c**.5)).numpy();out.append(dict(seed=seed,in_dim=c,out_dim=32,sha256=hashlib.sha256(str((a.dtype,a.shape)).encode()+a.tobytes()).hexdigest()))\nprint(json.dumps(out))"
 matrices=json.loads(subprocess.check_output(['/data/wangcheng/envs/cvt2/bin/python','-c',code],text=True));assert matrices==meta['projection_matrices']
 assert sha('/tmp/ue_complementarity_probe.py')==meta['script_sha256'] and sha(O/'PREREGISTRATION.md')==meta['protocol_sha256'] and sha(meta['helper_path'])==meta['helper_sha256'] and sha(meta['checkpoint'])==meta['checkpoint_sha256']
 man=read('/tmp/ue_uav_experiments/20261014_ad_seed42/dataset/manifest.json');assert sha('/tmp/ue_uav_experiments/20261014_ad_seed42/dataset/manifest.json')==meta['manifest_sha256']=='495841b2174242c87661e6b40dba2ddcca13d64b13798116efc8c933c679ca59'
 expected={'train':sorted(sorted(man['splits']['train'],key=key)[:128]),'validation':man['splits']['validation']};assert meta['selected_image_ids']==expected
 for split in expected:
  actual=[r['image_id'] for r in meta['records'] if r['split']==split];assert sorted(actual)==sorted(expected[split]) and len(set(actual))==len(actual)
 assert all(r['source_features_unchanged'] and r['payload_rgb_satellite_unchanged'] and r['hooks_removed'] for r in meta['records']) and meta['parameter_versions_unchanged']
 tr=load('train');va=load('validation');assert not(set(tr['global_id'])&set(va['global_id']));y=(tr['target']-tr['prob']).reshape(-1).astype(np.float64);models={};pred={};metrics={};fit={}
 masks={'all':np.ones(len(va['prob']),bool),'support_positive':va['geometry'][:,:,-1].mean(1)>0};area=np.prod(va['boxes'][:,2:]-va['boxes'][:,:2],1);masks.update(small=area<1024,medium=(area>=1024)&(area<9216),large=area>=9216)
 for cell in CELLS:
  x=design(tr,cell).reshape(-1,71);mu=x.mean(0);std=x.std(0).clip(1e-6);z=(x-mu)/std;ym=y.mean();beta=np.linalg.solve(z.T@z/len(z)+.01*np.eye(71),z.T@(y-ym)/len(z));pv=va['prob']+(((design(va,cell)-mu)/std)@beta+ym);pv=pv.clip(0,1);pt=(tr['prob']+((design(tr,cell)-mu)/std)@beta+ym).clip(0,1)
  br,io=roi_stats(pv,va['target']);tb,ti=roi_stats(pt,tr['target']);pred[cell]=pv;models[cell]=dict(mean=mu.tolist(),std=std.tolist(),beta=beta.tolist(),intercept=float(ym));metrics[cell]={group:dict(n=int(mask.sum()),brier=agg(br,mask),binary8_iou=agg(io,mask),building_equal_brier=buildingmean(br[mask],va['global_id'][mask]) if mask.any() else None) for group,mask in masks.items()};fit[cell]=dict(train_brier=float(tb.mean()),train_binary8_iou=float(ti.mean()),n_rows=len(y),n_rois=len(tr['prob']),feature_dim=71,active_input_dim=int((x.std(0)>1e-6).sum()))
 basebr,baseio=roi_stats(va['prob'],va['target']);trainbasebr,trainbaseio=roi_stats(tr['prob'],tr['target']);contrasts={}
 for control in ['SG','SX','S64','SN']:
  cb,ci=roi_stats(pred[control],va['target']);ab,ai=roi_stats(pred['SU'],va['target']);v=cb-ab;contrasts['SU_vs_'+control]={g:dict(n=int(mask.sum()),brier_reduction=agg(v,mask),brier_reduction_building_equal=buildingmean(v[mask],va['global_id'][mask]) if mask.any() else None,binary8_iou_delta=agg(ai-ci,mask)) for g,mask in masks.items()};
  if control in ['SG','SX']:contrasts['SU_vs_'+control]['all']['bonferroni_two_primary_CI95']=bootstrap(v,va['image_id'],.0125)
 gates={c:contrasts['SU_vs_'+c]['all']['brier_reduction']>=.001 and contrasts['SU_vs_'+c]['all']['bonferroni_two_primary_CI95'][0]>0 for c in ['SG','SX']}
 # Spatial audit from manifest only; exact visible building check above.
 man=read('/tmp/ue_uav_experiments/20261014_ad_seed42/dataset/manifest.json');by={z['index']:z for z in man['points']};tid=meta['selected_image_ids']['train'];vid=meta['selected_image_ids']['validation'];dist=[];overlap=0
 for i in vid:
  v=by[i];dist.append(min(np.linalg.norm(np.array(v['origin_m'][:2])-by[j]['origin_m'][:2]) for j in tid));a=v['footprint_xy_m']
  for j in tid:
   b=by[j]['footprint_xy_m'];overlap+=int(max(a[0],b[0])<min(a[2],b[2]) and max(a[1],b[1])<min(a[3],b[3]))
 summary=dict(scope='Frozen E72 dual-trained satellite/BEV residual readout, not single-model or final-mask gain',primary='Brier reduction SU vs SG and SX',n_train_images=len(tid),n_val_images=len(vid),n_train_rois=len(tr['prob']),n_val_rois=len(va['prob']),n_train_rows=len(y),lambda_fixed=.01,threshold=.5,models_capacity=fit,baseline=dict(brier=float(basebr.mean()),binary8_iou=float(baseio.mean()),train_brier=float(trainbasebr.mean()),train_binary8_iou=float(trainbaseio.mean())),metrics=metrics,contrasts=contrasts,gates=gates,overall_gate=all(gates.values()),spatial_audit=dict(train_visible_buildings=len(set(tr['global_id'])),val_visible_buildings=len(set(va['global_id'])),shared_visible_buildings=0,train_val_overlapping_satellite_AABB_pairs=overlap,val_nearest_selected_train_origin_m_min=float(min(dist)),val_nearest_selected_train_origin_m_median=float(np.median(dist))),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 dump(O/'ridge_models.json',models);dump(O/'summary.json',summary);np.savez_compressed(O/'val_predictions.npz',**pred,truth=va['target'],baseline=va['prob'],global_id=va['global_id'],image_id=va['image_id']);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
