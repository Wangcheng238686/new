"""Fixed cached-feature readout fit. Requires root/audit admission before GPU.
Train-only timing smoke does not load validation feature files.
"""
import argparse,hashlib,json,time,sys,os
from pathlib import Path
import numpy as np
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch
from torch import nn
O=Path('/tmp/ue_stage_shape_probe');NAMES=['SG','S64','C','CX','U','UX'];BLOCK={'S64':'sat_extra32','C':'cvt32','CX':'cvt_shift32','U':'uniform32','UX':'uniform_shift32'};SEED=1831
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2))
def ck(v,s):
 if not v:raise AssertionError(s)
def tensorhash(x):
 a=x.detach().cpu().contiguous().numpy();return hashlib.sha256(str((a.dtype,a.shape)).encode()+a.tobytes()).hexdigest()
def nchw(a,c):
 a=np.asarray(a)
 if c==1:return a.reshape(-1,1,16,16).astype(np.float32)
 ck(a.shape[-1]==c,'Last feature channels');return a.reshape(-1,16,16,c).transpose(0,3,1,2).copy().astype(np.float32)
def load_split(manifest,split):
 records=[r for r in manifest['records'] if r['split']==split];images=[];mom={};total=0
 for rec in records:
  ck(sha(rec['path'])==rec['sha256'],'Feature SHA');z=dict(np.load(rec['path'],allow_pickle=False));n=rec['n_rois'];ck(len(z['prob'])==n,'ROI count')
  arrays={k:nchw(z[k],32) for k in ['sat32','sat_extra32','cvt32','cvt_shift32','uniform32','uniform_shift32']};arrays['geometry']=nchw(z['geometry'],7);ck('common_coverage' in z and 'original_coverage' in z,'Common support diagnostic fields');arrays['prob']=nchw(z['prob'],1);arrays['target']=nchw(z['target'],1);arrays['boundary_weight']=nchw(z['boundary_weight'],1);arrays['original_coverage']=nchw(z['original_coverage'],1);arrays['common_coverage']=nchw(z['common_coverage'],1)
  for k,a in arrays.items():ck(np.isfinite(a).all() and len(a)==n,'Finite shape '+k)
  ck(((arrays['target']>=0)&(arrays['target']<=1)).all(),'Target range');ck(((arrays['prob']>=0)&(arrays['prob']<=1)).all(),'Probability range')
  for k,a in arrays.items():
   if k in ['prob','target','boundary_weight','original_coverage','common_coverage']:continue
   v=a.astype(np.float64);s=v.sum((0,2,3));ss=(v*v).sum((0,2,3));cnt=n*256
   if k not in mom:mom[k]=[s,ss,cnt]
   else:mom[k][0]+=s;mom[k][1]+=ss;mom[k][2]+=cnt
  images.append(dict(image_id=rec['image_id'],n=n,arrays=arrays,identity={k:z[k] for k in ['global_id','boxes','image_id','gt_slot']}));total+=n
 norm={k:dict(mean=(v[0]/v[2]).tolist(),std=np.sqrt(np.maximum(v[1]/v[2]-(v[0]/v[2])**2,0)).clip(1e-6).tolist(),n_cells=v[2]) for k,v in mom.items()}
 return images,norm,total
class Head(nn.Module):
 def __init__(self,arch):
  super().__init__();self.net=nn.Conv2d(71,1,1) if arch=='linear' else nn.Sequential(nn.Conv2d(71,32,3,padding=1),nn.GELU(),nn.Conv2d(32,1,1))
 def forward(self,x):return self.net(x)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--features',type=Path,default=O/'features/run_manifest.json');ap.add_argument('--timing-smoke',action='store_true');ap.add_argument('--device',default='cuda:1');args=ap.parse_args();torch.set_num_threads(2);torch.manual_seed(SEED);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True);device=torch.device(args.device)
 m=read(args.features);ck(m['parameter_versions_unchanged'] and m['weights']=='ema','Frozen feature extraction');ck(m['complete'] or args.timing_smoke,'Only scoped smoke may use partial extraction');ck(not m['strict_load']['missing_keys'] and not m['strict_load']['unexpected_keys'],'Strict source load');ck(sha(O/'PREREGISTRATION.md')==m['protocol_sha256'],'Frozen extraction protocol');out=O/('fit_smoke' if args.timing_smoke else 'fit');out.mkdir(exist_ok=True);started=time.monotonic()
 # Identity-only precheck: no validation feature/target arrays are read here.
 gids={'train':set(),'validation':set()}
 for rec in m['records']:
  ck(sha(rec['path'])==rec['sha256'],'Identity cache SHA')
  with np.load(rec['path'],allow_pickle=False) as z:
   gid=z['global_id'];iid=z['image_id'];slots=z['gt_slot'];boxes=z['boxes'];ck(len(gid)==rec['n_rois'] and np.all(iid==rec['image_id']) and np.array_equal(slots,np.arange(len(gid))),'Pre-fit identity/slot order');ck(boxes.shape==(len(gid),4),'Identity boxes')
   gids[rec['split']].update(map(int,gid))
 ck(not gids['train']&gids['validation'],'Pre-fit actual visible building isolation')
 for split in gids:ck(gids[split]==set(map(int,m['split_global_ids'][split])),'Declared/actual visible IDs')
 train,norm,nt=load_split(m,'train');ck(len(train)>0,'Training data');
 if not args.timing_smoke:
  ck(len(train)==128 and nt==12621,'Frozen training IDs/counts');ck([r['image_id'] for r in m['records'] if r['split']=='train']==m['selected_image_ids']['train'],'Train selected IDs');ck([r['image_id'] for r in m['records'] if r['split']=='validation']==m['selected_image_ids']['validation'] and len(m['selected_image_ids']['validation'])==153,'Declared validation IDs/count')
 dump(out/'normalization.json',norm);ck(not set(m['split_global_ids']['train'])&set(m['split_global_ids']['validation']),'Pre-fit declared visible building isolation')
 steps=32 if args.timing_smoke else 1000;sampler=np.random.default_rng(SEED);sample=[]
 for s in range(1000):
  im=sampler.integers(0,len(train),64);slot=np.array([sampler.integers(0,train[int(i)]['n']) for i in im]);sample.append(np.stack([im,slot],1))
 sample=np.stack(sample);np.save(out/'sample_sequence.npy',sample);weights={};history={};timings={};init_hashes={};trainfinal={};status=dict(complete=False,timing_smoke=args.timing_smoke,script_sha256=sha(__file__),feature_manifest_sha256=sha(args.features),protocol_sha256=sha(O/'PREREGISTRATION.md'),seed=SEED,steps=steps,batch_rois=64,sampling='image uniform then within-image ROI uniform, replacement',optimizer=dict(name='AdamW',lr=.001,weight_decay=.0001),selection='last fixed step; no validation selection',sample_sha256=sha(out/'sample_sequence.npy'),normalization_sha256=sha(out/'normalization.json'),normalization='train only per raw channel, population std floor1e-6',train_images=len(train),train_rois=nt,validation_targets_loaded_before_fits=False,models={},training_formal_model=False)
 dump(out/'fit_manifest.json',status)
 def batch_data(pairs):return {k:torch.from_numpy(np.stack([train[int(i)]['arrays'][k][int(j)] for i,j in pairs])).to(device) for k in train[0]['arrays']}
 tn={k:(torch.tensor(v['mean'],device=device).reshape(1,-1,1,1),torch.tensor(v['std'],device=device).reshape(1,-1,1,1)) for k,v in norm.items()}
 def inputs(batch,name):
  def normalized(k):a,b=tn[k];return (batch[k]-a)/b
  return torch.cat([normalized('sat32'),torch.zeros_like(batch['sat32']) if name=='SG' else normalized(BLOCK[name]),normalized('geometry')],1)
 # All heads share batches; each architecture's six cells start bit-identically.
 heads={};opts={}
 for arch in ['linear','nonlinear']:
  torch.manual_seed(SEED);template=Head(arch);state=template.state_dict();init_hashes[arch]={k:tensorhash(v) for k,v in state.items()}
  for name in NAMES:
   key=arch+'_'+name;head=Head(arch).to(device);head.load_state_dict(state);ck({k:tensorhash(v) for k,v in head.state_dict().items()}==init_hashes[arch],'Paired init exact');heads[key]=head;opts[key]=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.0001);history[key]=[]
 del template,state
 first=time.monotonic()
 for step in range(steps):
  data=batch_data(sample[step]);target=data['target']-data['prob']
  for key,head in heads.items():
   name=key.split('_',1)[1];x=inputs(data,name);pred=head(x);loss=(pred-target).square().mean();ck(torch.isfinite(loss).item(),'Finite train loss '+key);opts[key].zero_grad(set_to_none=True);loss.backward();ck(all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()),'Finite gradients');opts[key].step()
   if step==0 or (step+1)%100==0 or step+1==steps:history[key].append(dict(step=step+1,residual_mse=float(loss.detach())))
  if step%8==0:print('TRAINSTEP',step+1,'of',steps,flush=True)
  if time.monotonic()-first>900:raise RuntimeError('Fixed 15min fit operational cap; incomplete, no validation')
 torch.cuda.synchronize(device);fit_seconds=time.monotonic()-first
 status.update(fit_seconds=fit_seconds,projected_1000_step_seconds=fit_seconds*1000/steps,paired_initial_state_sha256=init_hashes,loss_history=history,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device),all_trainable_heads_finite=all(torch.isfinite(p).all().item() for h in heads.values() for p in h.parameters()))
 for key,h in heads.items():
  p=out/(key+'.pth');torch.save({k:v.detach().cpu() for k,v in h.state_dict().items()},p);status['models'][key]=dict(path=str(p),sha256=sha(p),parameters=sum(p.numel() for p in h.parameters()),last_step=steps)
 if args.timing_smoke:
  status.update(complete=True,validation_evaluated=False,elapsed_seconds=time.monotonic()-started);dump(out/'fit_manifest.json',status);print('DONE TRAIN-ONLY TIMING',fit_seconds,'projected',status['projected_1000_step_seconds'],flush=True);return
 # Only after every head's fixed last step, load val and evaluate once. Train last
 # output is also reported, using same frozen heads and train-only normalization.
 val,unused,nv=load_split(m,'validation');del unused;ck(len(val)==153 and nv==4496,'Frozen validation count');ck(not set(np.concatenate([z['identity']['global_id'] for z in train]))&set(np.concatenate([z['identity']['global_id'] for z in val])),'Actual building isolation');allpred={};identities={}
 with torch.inference_mode():
  for split,images in [('train',train),('validation',val)]:
   accum={k:[] for k in heads};targets=[];probs=[];bounds=[];original_cov=[];common_cov=[];gid=[];iid=[];slots=[];boxes=[]
   for image in images:
    for start in range(0,image['n'],64):
     stop=start+64;data={k:torch.from_numpy(v[start:stop]).to(device) for k,v in image['arrays'].items()}
     for key,h in heads.items():accum[key].append((data['prob']+h(inputs(data,key.split('_',1)[1]))).cpu().numpy()[:,0])
     targets.append(data['target'].cpu().numpy()[:,0]);probs.append(data['prob'].cpu().numpy()[:,0]);bounds.append(data['boundary_weight'].cpu().numpy()[:,0]);original_cov.append(data['original_coverage'].cpu().numpy()[:,0]);common_cov.append(data['common_coverage'].cpu().numpy()[:,0])
    for key,buf in [('global_id',gid),('image_id',iid),('gt_slot',slots),('boxes',boxes)]:buf.append(image['identity'][key])
   arrays={k:np.concatenate(v) for k,v in accum.items()};arrays.update(target=np.concatenate(targets),prob=np.concatenate(probs),boundary_weight=np.concatenate(bounds),original_coverage=np.concatenate(original_cov),common_coverage=np.concatenate(common_cov),global_id=np.concatenate(gid),image_id=np.concatenate(iid),gt_slot=np.concatenate(slots),boxes=np.concatenate(boxes));np.savez_compressed(out/(split+'_predictions.npz'),**arrays)
 status.update(complete=True,validation_evaluated=True,validation_rois=nv,validation_images=len(val),elapsed_seconds=time.monotonic()-started,prediction_sha256={s:sha(out/(s+'_predictions.npz')) for s in ['train','validation']});dump(out/'fit_manifest.json',status);print('DONE FIXED HEAD FIT',flush=True)
if __name__=='__main__':main()
