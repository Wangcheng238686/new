"""Reviewed native P1: six fixed-checkpoint assignment sensitivity conditions.

Separate from the historical five-mode spatial diagnostic. No RMS matching,
training, parameter tuning, or modification of an existing experiment source.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import fcntl
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts/probes'),str(ROOT)]
BASE=Path('/data/wangcheng/checkpoint/ue_minimal_single_20261021_EXP-0017/checkpoints/best_model_epoch73.pth')
ADAPTER=Path('/data/wangcheng/checkpoint/ue_v1_adapter_probe_20261021/native_geometry/last.pth')
DATA=Path('/data/wangcheng/dataset/CitySample_SatUAV_TrainMinimal_v1_20261021')
CACHE=Path('/data/wangcheng/cache/ue_v1_native_geometry')
OLD=Path('/data/wangcheng/checkpoint/ue_v1_adapter_probe_20261021')
MODES=['correct','zero_eval','mean_eval','permute_s1','permute_s2','permute_s3']
SEEDS={'permute_s1':20261023,'permute_s2':20261024,'permute_s3':20261025}
FROZEN='c6921c30416595fb994facd3d4d927408be9bdd636399fd669cfbe5968329536'
SOURCES=['scripts/probes/ue_native_assignment_probe.py','scripts/probes/ue_native_assignment_analysis.py',
 'scripts/probes/ue_v1_adapter.py','scripts/probes/ue_v1_adapter_analysis.py','scripts/probes/ue_s2db_analysis.py',
 'rsprompter/uav/native_geometry_probe.py','rsprompter/uav/local_adapter_probe.py','utils/coco_eval_utils.py']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(obj,indent=2)+'\n');t.replace(p)


def intervene(fields,support,mode,iid):
 import numpy as np
 assert fields.shape==(4,512,512) and support.shape==(512,512)
 assert np.isfinite(fields).all() and not np.any(fields[:,~support])
 valid=np.flatnonzero(support.ravel());out=fields.copy();mapping=None
 if mode=='zero_eval':out[:]=0
 elif mode=='mean_eval':
  if len(valid):out.reshape(4,-1)[:,valid]=fields.reshape(4,-1)[:,valid].mean(1,keepdims=True,dtype=np.float64).astype(np.float32)
 elif mode in SEEDS:
  rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([SEEDS[mode],int(iid)])))
  mapping=rng.permutation(len(valid));out.reshape(4,-1)[:,valid]=fields.reshape(4,-1)[:,valid[mapping]]
  inverse=np.argsort(mapping)
  assert np.array_equal(out.reshape(4,-1)[:,valid[inverse]],fields.reshape(4,-1)[:,valid])
 else:assert mode=='correct'
 assert np.isfinite(out).all() and not np.any(out[:,~support])
 return out,dict(mode=mode,seed=SEEDS.get(mode),image_id=int(iid),support_pixels=len(valid),
  support_sha256=hashlib.sha256(support.tobytes()).hexdigest(),
  mapping_sha256=hashlib.sha256(mapping.astype('<i8').tobytes()).hexdigest() if mapping is not None else None,
  original_fields_sha256=hashlib.sha256(fields.tobytes()).hexdigest(),
  altered_fields_sha256=hashlib.sha256(out.tobytes()).hexdigest(),joint_vector_multiset_exact=mode.startswith('permute') or mode=='correct')


def cpu_gate(out):
 import numpy as np
 support=np.zeros((512,512),bool);support[1:4,1:4]=True
 f=np.zeros((4,512,512),np.float32);f[:,support]=np.arange(36).reshape(4,9)
 mapping=[]
 for mode in MODES:
  a,ident=intervene(f,support,mode,1238);b,repeat=intervene(f,support,mode,1238)
  assert np.array_equal(a,b) and ident==repeat
  if mode.startswith('permute'):mapping.append(ident['mapping_sha256'])
  if mode=='correct':assert np.array_equal(a,f)
  if mode=='zero_eval':assert not a.any()
  if mode=='mean_eval':assert np.array_equal(a[:,support],np.repeat(f[:,support].mean(1,keepdims=True),9,axis=1))
 assert len(set(mapping))==3
 dump(Path(out)/'cpu_gate.json',dict(passed=True,modes=MODES,same_seed_replay=True,distinct_seed_mappings=True,
  joint_vectors_preserved_by_permutation=True,no_unsupported_values=True,mean_definition='per-channel float64 mean cast to float32',
  script_sha256=sha(__file__)))


def build(a):
 import ue_v1_adapter as c
 import torch
 from rsprompter.uav.native_geometry_probe import NativeGeometryAdapter
 c.seed(44);a.base=BASE;a.data_root=DATA;a.cache=CACHE;a.adapter=ADAPTER;a.source_mode='fine';a.route='highres';a.seed=44;a.appearance='correct'
 cp=torch.load(ADAPTER,map_location='cpu',weights_only=False)
 assert cp['epoch']==10 and cp['step']==2670 and cp['contract']['base_sha256']==sha(BASE)
 model,unused,_=c.make_model(a)
 for handle in unused.handles:handle.remove()
 del unused;torch.cuda.empty_cache()
 adapter=NativeGeometryAdapter().cuda();adapter.load_state_dict(cp['adapter'],strict=True);adapter.attach(model)
 model.eval();adapter.eval();return c,model,adapter


def load_fields(folder):
 import numpy as np
 p=CACHE/f'{folder.name}.npz';assert p.exists(),p
 with np.load(p) as data:
  assert str(data['version'])=='native512_height_dx_dy_viewcount_v1'
  return data['fields'].astype(np.float32),data['support'].astype(bool)


def replay_gate(a):
 import torch
 import numpy as np
 c,model,adapter=build(a);before=c.state_digest(model);assert before==FROZEN
 samples=read('/data/wangcheng/checkpoint/ue_native_source_audit_20261022/manifest.json')['samples']
 rows=[];original_encode=adapter.encode
 for sample in samples:
  for handle in adapter.handles:handle.remove()
  adapter.handles=[];adapter.encode=original_encode;adapter.attach(model)
  ds=c.MinimalSatelliteDataset(DATA,sample['split']);i=next(i for i,s in enumerate(ds.samples) if s['image_id']==sample['image_id'])
  p=c.processed(model,c.rtmdet_collate_fn([ds[i]]));folder=c.point_folder(ds,i,DATA)
  fields,support=load_fields(folder);changed,ident=intervene(fields,support,'correct',sample['image_id'])
  raw=(None,torch.from_numpy(support.astype(np.float32))[None,None].cuda(),torch.from_numpy(fields)[None].cuda())
  alt=(None,raw[1],torch.from_numpy(changed)[None].cuda());captured=[]
  h=model.roi_head.mask_head.register_forward_hook(lambda m,x,o:captured.append(o.mask_logits.detach().clone()))
  with torch.inference_mode():
   features=c.bind(model,adapter,p,raw)
   with c.cached_extract(model,features):original=model.predict(p['inputs'],p['data_samples'],rescale=True)[0].pred_instances
   old_logits=list(captured);captured.clear();install_measurement(model,adapter);features=c.bind(model,adapter,p,alt)
   with c.cached_extract(model,features):replay=model.predict(p['inputs'],p['data_samples'],rescale=True)[0].pred_instances
  h.remove()
  assert len(old_logits)==len(captured) and all(torch.equal(x,y) for x,y in zip(old_logits,captured))
  for k in ('masks','bboxes','scores','labels'):assert torch.equal(getattr(original,k),getattr(replay,k))
  rows.append(dict(**sample,raw_logits_exact=True,mask_box_score_label_exact=True));adapter.context=None
 assert c.state_digest(model)==before
 dump(a.out/'gpu_gate.json',dict(images=8,samples=rows,fp32=True,satellite_state_sha256=before,passed=True))


def install_measurement(model,adapter):
 import torch
 observations=[];calls=[];current={}
 original_encode=adapter.encode
 def encode(rgb,support,fields,high):
  original_encode(rgb,support,fields,high);calls.clear();current.clear()
  current.update(rois=0,elements=0,base_square_sum=0.,residual_square_sum=0.,injected_square_sum=0.)
 adapter.encode=encode
 def inject(module,args,base):
  if not adapter.enabled or adapter.context is None:return base
  support,fields,high=adapter.context;q=adapter.query(high)
  delta=torch.cat([adapter._chunk(q,fields,support,b,base.shape[-1]) for b in adapter.boxes.split(4)])
  applied=adapter.alpha*delta
  sums={k:float(v.double().square().sum()) for k,v in [('base_square_sum',base),('residual_square_sum',delta),('injected_square_sum',applied)]}
  row=dict(rois=len(base),elements=base.numel(),**sums)
  for k,v in row.items():current[k]+=v
  calls.append(row)
  adapter.stats=dict(alpha=float(adapter.alpha),support_fraction=float(support.mean()),
   source_delta_rms=float(delta.float().square().mean().sqrt()),applied_delta_rms=float(applied.float().square().mean().sqrt()),
   base_rms=float(base.float().square().mean().sqrt()),rois=len(base),support_grid=512)
  return base+applied
 # Replace only the new process's hook. Satellite module implementations unchanged.
 for handle in adapter.handles:handle.remove()
 head=model.roi_head.mask_head
 adapter.handles=[head.register_forward_pre_hook(adapter._boxes,with_kwargs=True),head.mask_decoder.decoder.conv_s0.register_forward_hook(inject)]
 return observations,calls,current


def evaluate(a):
 import numpy as np
 import torch
 c,model,adapter=build(a);before=c.state_digest(model);assert before==FROZEN
 observations,calls,current=install_measurement(model,adapter)
 input_ids=[]
 def payload(folder,cache,appearance='correct',donor=None):
  fields,support=load_fields(folder);alt,ident=intervene(fields,support,a.mode,int(folder.name));ident['native_cache_sha256']=sha(CACHE/f'{folder.name}.npz');input_ids.append(ident)
  return None,torch.from_numpy(support.astype(np.float32))[None,None].cuda(),torch.from_numpy(alt)[None].cuda()
 c.payload=payload;predict=model.predict
 def measured(*args,**kwargs):
  result=predict(*args,**kwargs);observations.append(dict(**current,calls=list(calls)));return result
 model.predict=measured;a.eval_split='test';a.max_images=0
 with torch.inference_mode():c.evaluate(a,model,adapter,c.MinimalSatelliteDataset(DATA,'test'),'correct',a.out)
 rows=read(a.out/'records.json');old=read(OLD/'test/native_geometry/correct/records.json')
 assert len(rows)==len(observations)==len(input_ids)==459
 for row,ref,obs,ident in zip(rows,old,observations,input_ids):
  assert row['image_id']==ref['image_id']==ident['image_id'] and row['gt']==ref['gt']
  for k in ('bboxes','scores','labels'):assert row['dt'][k]==ref['dt'][k]
  if a.mode=='correct':assert row['dt']==ref['dt']
  assert obs['rois']==len(row['dt']['scores']);obs['image_id']=row['image_id']
  obs['injected_to_base_rms_ratio']=(obs['injected_square_sum']/obs['base_square_sum'])**.5 if obs['base_square_sum'] else None
 assert c.state_digest(model)==before
 dump(a.out/'all_call_squares.json',observations);dump(a.out/'input_mapping_identity.json',input_ids)
 dump(a.out/'diagnostic_identity.json',dict(mode=a.mode,images=459,precision='fp32',base_sha256=sha(BASE),adapter_sha256=sha(ADAPTER),
  satellite_state_sha256=before,satellite_state_unchanged=True,detector_gt_exact=True,correct_archive_exact=a.mode=='correct',
  source_hashes={p:sha(ROOT/p) for p in SOURCES},source_cache_read_only=True))


def old_reserves_gpu1():
 state=read(OLD/'test/test_state.json')
 return any(j['stage']!='complete' for j in state['evaluations'])


def run(a):
 a.out.mkdir(parents=True,exist_ok=True);lock=(a.out/'owner.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 provenance=read('/data/wangcheng/checkpoint/ue_native_source_audit_20261022/complete_root_audit.json');assert provenance['completed']
 plan=dict(modes=MODES,seeds=SEEDS,primary_metric='MaskAP',primary_contrasts=['correct-mean_eval','correct-average(permutation APs)'],
  bootstrap=500,bootstrap_seed=20261021,confidence=[.95,.975],base_sha256=sha(BASE),adapter_sha256=sha(ADAPTER),
  source_hashes={p:sha(ROOT/p) for p in SOURCES},archive_hashes={str(p.relative_to(DATA)):sha(p) for p in [DATA/'dataset.json',DATA/'SHA256SUMS.txt']+[DATA/k/(s+'.json') for k in ('splits','annotations') for s in ('train','val','test')]},P2_audit_sha256=sha('/data/wangcheng/checkpoint/ue_native_source_audit_20261022/complete_root_audit.json'),
  protocol_sha256=sha(ROOT/'docs/dual_stream/UE_NEXT_PROBE_PLAN_20261022.md'),
  reused_historical_five_mode_results=False,rms_matching=False,registered_utc=time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime()),
  limitation='Assignment sensitivity only; OOD, frequency, query correlation and amplitude not identifiable separately')
 if (a.out/'plan.json').exists():
  saved=read(a.out/'plan.json');assert saved['source_hashes']==plan['source_hashes'];plan=saved
 else:
  dump(a.out/'plan.json',plan)
  for p in SOURCES:
   dest=a.out/'source'/p;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/p).read_bytes())
 cpu_gate(a.out)
 import numpy as np
 actual=[]
 for sample in read('/data/wangcheng/checkpoint/ue_native_source_audit_20261022/manifest.json')['samples']:
  fields,support=load_fields(DATA/'points'/f"{sample['image_id']:06d}")
  conditions=[]
  for mode in MODES:
   _,ident=intervene(fields,support,mode,sample['image_id']);conditions.append(ident)
  actual.append(dict(**sample,conditions=conditions))
 dump(a.out/'actual8_cpu_gate.json',dict(images=8,conditions_per_image=6,passed=True,samples=actual,script_sha256=sha(__file__)))
 state=dict(stage='queued_waiting_idle_gpu',owner_pid=os.getpid(),gpu=a.gpu,completed=[])
 dump(a.out/'state.json',state)
 # Wait for the old GPU1 reservation to end. Do not occupy a momentarily idle
 # device while that owner still has two tests waiting for their checkpoints.
 while True:
  assert all(sha(ROOT/p)==h for p,h in plan['source_hashes'].items())
  used=int(subprocess.check_output(['nvidia-smi',f'--id={a.gpu}','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
  if used<200 and (a.gpu!=1 or not old_reserves_gpu1()):break
  time.sleep(15)
 # Hold visible memory across serial child gaps so the existing idle-device
 # scheduler sees this device in use. No reservation while other work is live.
 os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
 env=dict(os.environ,PYTHONUNBUFFERED='1')
 guard_code="import torch,sys; x=torch.empty(256*1024*1024,dtype=torch.uint8,device='cuda'); print('READY',flush=True); sys.stdin.read()"
 guard=subprocess.Popen([sys.executable,'-c',guard_code],env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
 assert guard.stdout.readline().strip()=='READY'
 state['reservation_pid']=guard.pid;dump(a.out/'state.json',state)
 def child(name,cmd,artifact):
  if Path(artifact).exists():return
  with (a.out/(name+'.log')).open('ab') as log:
   proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   state.update(stage=name,child_pid=proc.pid);dump(a.out/'state.json',state);rc=proc.wait()
  if rc:state.update(stage='failed',failed_job=name,returncode=rc);dump(a.out/'state.json',state);raise RuntimeError(name)
  assert Path(artifact).exists();state['completed'].append(name);dump(a.out/'state.json',state)
 py=sys.executable
 child('gpu_gate',[py,__file__,'gate','--out',str(a.out)],a.out/'gpu_gate.json')
 for mode in MODES:
  child('eval_'+mode,[py,__file__,'eval','--mode',mode,'--out',str(a.out/mode)],a.out/mode/'diagnostic_identity.json')
 # Fixed8-image delta diagnostics only; not on test, never used to tune scale.
 child('delta8',[py,str(ROOT/'scripts/probes/ue_native_assignment_analysis.py'),'delta8','--root',str(a.out)],a.out/'delta8.json')
 guard.stdin.close();guard.wait(timeout=30);state.update(stage='cpu_analysis',reservation_pid=None);dump(a.out/'state.json',state)
 child('analysis',[py,str(ROOT/'scripts/probes/ue_native_assignment_analysis.py'),'report','--root',str(a.out)],a.out/'complete_audit_summary.json')
 state.update(stage='complete');dump(a.out/'state.json',state)


def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['cpu','gate','eval','run']);p.add_argument('--out',type=Path,required=True)
 p.add_argument('--mode',choices=MODES,default='correct');p.add_argument('--gpu',type=int,default=1);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 if a.action=='cpu':cpu_gate(a.out)
 elif a.action=='gate':replay_gate(a)
 elif a.action=='eval':evaluate(a)
 else:run(a)
if __name__=='__main__':
 try:main()
 except BaseException as exc:
  if '--out' in sys.argv:
   r=Path(sys.argv[sys.argv.index('--out')+1]);p=r/'state.json'
   if p.exists():
    state=read(p);state.update(stage='owner_failed',error=repr(exc));dump(p,state)
  raise
