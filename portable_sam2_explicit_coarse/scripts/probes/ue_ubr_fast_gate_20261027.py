"""Four-rank real-model numerical/efficiency gate. No AP or long training.

Tail benchmarks include gradients through the real source/G/satellite/decoder,
but OMIT original task losses; separate full-objective AMP updates measure the
actual training resource requirement. Profiled stage times overlap and include
synchronization overhead; speed ratios use separate uninstrumented executions.
"""
import argparse
from contextlib import contextmanager
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback
import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
WORK=Path(__file__).resolve().parents[2]; REPO=WORK.parent
sys.path[:0]=[str(WORK),str(WORK/'scripts/probes')]


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def tensor_sha(value): return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def write(path,value):
 path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');tmp.replace(path)
def rng(rt): return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(),loader=rt.generator.get_state())
def restore_rng(rt,s):
 random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch']);torch.cuda.set_rng_state(s['cuda']);rt.generator.set_state(s['loader'])
def all_reports(row,cpu):
 out=[None]*4;dist.all_gather_object(out,row,group=cpu);return out

def sources():
 out={str(Path(__file__).resolve().relative_to(REPO)):sha(__file__)}
 for module in tuple(sys.modules.values()):
  path=getattr(module,'__file__',None)
  if path:
   path=Path(path).resolve()
   if path.is_file() and path.suffix=='.py' and REPO in path.parents:out[str(path.relative_to(REPO))]=sha(path)
 return out


class Recorder:
 def __init__(self):self.phase='forward';self.times={};self.bands=[];self.contexts=[];self.patches=[]
 def wrap(self,obj,name,label,capture=None):
  original=getattr(obj,name)
  def traced(*args,**kwargs):
   torch.cuda.synchronize();begin=time.perf_counter();result=original(*args,**kwargs);torch.cuda.synchronize()
   key=self.phase+'/'+label;row=self.times.setdefault(key,dict(calls=0,seconds=0.));row['calls']+=1;row['seconds']+=time.perf_counter()-begin
   if capture is not None:capture(result,args)
   return result
  self.patches.append((obj,name,original));setattr(obj,name,traced)
 @contextmanager
 def install(self,old,fast):
  import rsprompter.uav.ubr_detector_integration as legacy
  import rsprompter.uav.ubr_detector_fast as optimized
  import rsprompter.uav.ubr_coordinate_adapter as bridge
  def record_sdt(result,args):
   if self.phase=='forward':
    for value in result:self.bands.append(tensor_sha(value.abs()<=8))
  def record_context(result,args):
   self.contexts.append(tensor_sha(result.retained_masks))
  self.wrap(old,'decode_native','native_decoder')
  self.wrap(old,'prediction_context','context_total',record_context)
  if fast is not old:self.wrap(fast,'prediction_context','context_total',record_context)
  self.wrap(old.refiner,'forward','UBR_convolution')
  self.wrap(legacy,'signed_distance_transform','SDT',record_sdt)
  self.wrap(optimized,'signed_distance_transform','SDT',record_sdt)
  self.wrap(legacy.PredictionContext,'neighbors','neighbors')
  self.wrap(optimized.FastPredictionContext,'neighbors','neighbors')
  original=bridge.paste_native_roi
  def paste(values,*args,**kwargs):
   torch.cuda.synchronize();begin=time.perf_counter();result=original(values,*args,**kwargs);torch.cuda.synchronize()
   label='feature_paste' if values.shape[1]==32 else 'probability_paste';key=self.phase+'/'+label
   row=self.times.setdefault(key,dict(calls=0,seconds=0.));row['calls']+=1;row['seconds']+=time.perf_counter()-begin;return result
  self.patches.append((bridge,'paste_native_roi',bridge.paste_native_roi));bridge.paste_native_roi=paste
  # optimized imports probability paste directly; adapter resolves its own global.
  self.patches.append((optimized,'paste_native_roi',optimized.paste_native_roi));optimized.paste_native_roi=paste
  try:yield self
  finally:
   for obj,name,original in reversed(self.patches):setattr(obj,name,original)


class Objective(nn.Module):
 def __init__(self,rt,base):
  super().__init__();self.joint=rt.joint;object.__setattr__(self,'rt',rt);object.__setattr__(self,'base',base)
  self.controller=rt.controller;self.full=False;self.last={}
 def forward(self,sample,ident):
  from data.ue_instance_targets import center_targets
  from rsprompter.uav.ubr_detector_integration import active_prediction_indices
  rt=self.rt;payload=rt.payload(ident);self.joint.prepare_uav(**payload);features=rt.satellite.extract_feat(sample['inputs'])
  gt=sample['data_samples'][0].gt_instances;boxes=getattr(gt.bboxes,'tensor',gt.bboxes);masks=gt.masks.to_tensor(dtype=torch.float32,device='cuda')
  primary=features[0][0].sum()*0
  if self.full:
   # Match the real runner: original task loss populates autocast cache BEFORE
   # checkpointed tail calls. Tail-only benchmarks disable that cache instead.
   with self.base.core.C.cached_extract(rt.satellite,features):losses=rt.satellite.loss(sample['inputs'],sample['data_samples'])
   inst,_=self.base.core.pilot.loss_fn(self.joint.uav.outputs,center_targets(boxes.detach().cpu().numpy()))
   assert 'loss_decoder_tail' not in losses
   terms=[v for name,values in losses.items() if 'loss' in name for v in (values if isinstance(values,(tuple,list)) else [values]) if torch.is_tensor(v) and v.requires_grad]
   primary=sum(terms)+inst
  prepared=self.controller.prepare(features,sample['data_samples'][0].metainfo);pool=rt.select_pool(features,payload['geometry'],prepared)
  active,assigned=active_prediction_indices(pool,boxes)
  tail=self.controller.training_loss(prepared,pool,boxes,masks,payload['geometry'])['loss_ubr']
  self.last=dict(active_ids=pool.candidate_ids[active].tolist(),assigned=assigned.tolist(),pool_ids=pool.candidate_ids.tolist(),
   pool_boxes_sha=tensor_sha(pool.boxes),pool_scores_sha=tensor_sha(pool.scores),source_ids=list(pool.source_ids),**self.controller.last_stats)
  return primary+tail


def grads(model):return {n:p.grad.detach().float().cpu().clone() for n,p in model.named_parameters() if p.grad is not None}
def compare_gradient(a,b,amp):
 if set(a)!=set(b):return dict(passed=False,reason='gradient parameter names differ')
 error=base_norm=0.;maximum=0.
 for name in a:
  d=(a[name]-b[name]).double();error+=float(d.square().sum());base_norm+=float(a[name].double().square().sum());maximum=max(maximum,float(d.abs().max()))
 relative=(error/max(base_norm,1e-30))**.5
 return dict(passed=relative<=(.02 if amp else .001),relative_L2=relative,max_absolute=maximum,reference_norm=base_norm**.5,
             tolerance=.02 if amp else .001,parameter_tensors=len(a))


def measured(ddp,model,rt,controller,sample,ident,amp,seed_state,old,fast,profile=False):
 print('FAST_GATE_CASE',dist.get_rank(),type(controller).__name__,getattr(controller,'active_microbatch',1),'AMP' if amp else 'FP32','profile' if profile else 'timed',flush=True)
 restore_rng(rt,seed_state);ddp.zero_grad(set_to_none=True);rt.clear();model.controller=controller;model.full=False;ddp.train()
 torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();rec=Recorder()
 @contextmanager
 def optional():
  if profile:
   with rec.install(old,controller):yield
  else:yield
 with optional():
  torch.cuda.synchronize();begin=time.perf_counter()
  with torch.autocast('cuda',dtype=torch.float16,enabled=amp,cache_enabled=False):loss=ddp(sample,ident)
  torch.cuda.synchronize();forward=time.perf_counter()-begin;rec.phase='backward';begin=time.perf_counter();loss.backward();torch.cuda.synchronize();backward=time.perf_counter()-begin
  values=grads(model)
 report=dict(loss=float(loss.detach()),forward_seconds=forward,backward_seconds=backward,total_seconds=forward+backward,
             peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
             metadata=dict(model.last),gradient_finite=all(torch.isfinite(v).all() for v in values.values()),
             profiled=profile,profile=rec.times,bands=rec.bands,contexts=rec.contexts)
 print('FAST_GATE_CASE_DONE',dist.get_rank(),report['total_seconds'],report['peak_reserved_mib'],flush=True)
 rt.clear();return report,values


def inference(rt,controller,sample,ident):
 rt.network.eval();rt.clear();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();begin=time.perf_counter()
 with torch.no_grad():
  payload=rt.payload(ident);rt.joint.prepare_uav(**payload);features=rt.satellite.extract_feat(sample['inputs']);prepared=controller.prepare(features,sample['data_samples'][0].metainfo)
  pool=rt.select_pool(features,payload['geometry'],prepared);output=controller.predict(prepared,pool,payload['geometry'])
 torch.cuda.synchronize();report=dict(seconds=time.perf_counter()-begin,instances=len(pool.boxes),peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
   last_stats=dict(controller.last_stats),active_source_counts={s:pool.source_ids.count(s) for s in set(pool.source_ids)})
 result={k:v.detach().cpu() for k,v in output.items()};rt.clear();return report,result


def output_compare(a,b,zero=False):
 error=float((a['probabilities']-b['probabilities']).abs().max()) if a['probabilities'].numel() else 0.
 changed=int((a['masks']!=b['masks']).sum());ids=torch.equal(a['candidate_ids'],b['candidate_ids']);boxes=torch.equal(a['boxes'],b['boxes']);scores=torch.equal(a['scores'],b['scores'])
 return dict(passed=ids and boxes and scores and changed==0 and error<=(0. if zero else 2e-5),probability_max_error=error,
             changed_binary_pixels=changed,ids_exact=ids,boxes_exact=boxes,scores_exact=scores)


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--allocator-cap-mib',type=int,default=16000)
 p.add_argument('--checkpoint',type=Path,default=Path('/data/wangcheng/checkpoint/ue_candidate_detail_ddp_20261026/detail/correct/last.pth'));a=p.parse_args()
 assert int(os.environ.get('WORLD_SIZE',0))==4;rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(local);torch.set_num_threads(2)
 torch.cuda.set_per_process_memory_fraction(a.allocator_cap_mib*2**20/torch.cuda.get_device_properties(local).total_memory)
 dist.init_process_group('nccl',timeout=timedelta(minutes=60));cpu=dist.new_group(backend='gloo',timeout=timedelta(minutes=60))
 if rank==0:a.output.mkdir(parents=True,exist_ok=False)
 dist.barrier();report=dict(rank=rank,pid=os.getpid(),world_size=4,scope='engineering_only_not_AP_or_training',cases={})
 try:
  from ue_ubr_runtime_20261027 import build_runtime,base
  from rsprompter.uav.ubr_detector_fast import FastUBRDetectorIntegration
  rt=build_runtime();old=rt.controller
  receipt=json.loads((a.checkpoint.parent/'training_complete.json').read_text());checksum=[sha(a.checkpoint) if rank==0 else None];dist.broadcast_object_list(checksum,0,group=cpu)
  assert receipt['checkpoint_sha256']==checksum[0] and receipt['epoch']==10 and receipt['world_size']==4
  cp=torch.load(a.checkpoint,map_location='cpu',weights_only=False);state={k:v for k,v in cp['model'].items() if 'decoder_tail_refiner' not in k};missing,unexpected=rt.joint.load_state_dict(state,strict=False)
  assert not unexpected and all('ubr_refiner' in k for k in missing)
  report['engineering_checkpoint_sha256']=checksum[0];report['engineering_checkpoint_role']='G E10 gate-only; not a proposed training initialization';report['initialization']=rt.initialization_audit
  with torch.random.fork_rng(devices=[local]):
   torch.manual_seed(2071);nn.init.normal_(old.refiner.out.weight,std=.005);nn.init.constant_(old.refiner.out.bias,.01)
  report['nonzero_UBR_head']=dict(std=.005,bias=.01,seed=2071)
  fast1=FastUBRDetectorIntegration.from_controller(old,active_microbatch=1,inference_microbatch=4);fast4=fast1
  report['execution_policy']=dict(training_microbatch=1,inference_microbatch=4,native_training_decode=1,native_evaluation_decode='original source/chunk32')
  ds=base.core.C.MinimalSatelliteDataset(base.core.previous.DATA,'train',augment=False);order=sorted(range(len(ds)),key=lambda i:len(ds.samples[i]['annotations']),reverse=True);idx=order[rank]
  ident=int(ds.samples[idx]['image_id']);sample=base.core.C.processed(rt.satellite,base.core.C.rtmdet_collate_fn([ds[idx]]),False)
  report['image_id']=ident;report['sources_before']=sources()
  if rank==0:
   import shutil
   for name in report['sources_before']:
    target=a.output/'frozen_source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(REPO/name,target)
  model=Objective(rt,base).cuda();ddp=DistributedDataParallel(model,device_ids=[local],output_device=local,find_unused_parameters=True,broadcast_buffers=False)
  seed_state=rng(rt);parameter_sha=base.core.C.state_digest(model);report['weight_sha_before']=parameter_sha
  for amp in (False,True):
   name='AMP' if amp else 'FP32';reference,reference_grad=measured(ddp,model,rt,old,sample,ident,amp,seed_state,old,fast4)
   assert reference['metadata']['active_count']==64,'Need natural full64 predicted matches, never GT insertion'
   reference_profile,_=measured(ddp,model,rt,old,sample,ident,amp,seed_state,old,fast4,profile=True)
   rows={'old':reference,'old_profile':reference_profile};report['cases'][name]=rows
   write(a.output/('rank%d_intermediate.json'%rank),report)
   for label,controller in [('fast1',fast1)]:
    row,gradient=measured(ddp,model,rt,controller,sample,ident,amp,seed_state,old,fast4)
    profiled,_=measured(ddp,model,rt,controller,sample,ident,amp,seed_state,old,fast4,profile=True)
    meta=row['metadata'];same=all(meta[k]==reference['metadata'][k] for k in ('active_ids','assigned','pool_ids','pool_boxes_sha','pool_scores_sha','source_ids'))
    loss_error=abs(row['loss']-reference['loss']);loss_ok=loss_error<=((2e-5 if amp else 2e-6)+abs(reference['loss'])*(.002 if amp else .0002))
    row['comparison']=dict(metadata_exact=same,band_exact=profiled['bands']==reference_profile['bands'],context_exact=profiled['contexts']==reference_profile['contexts'],
      loss_absolute_error=loss_error,loss_passed=loss_ok,gradient=compare_gradient(reference_grad,gradient,amp),speedup=reference['total_seconds']/row['total_seconds'])
    row['comparison']['passed']=same and row['comparison']['band_exact'] and row['comparison']['context_exact'] and loss_ok and row['comparison']['gradient']['passed'] and row['gradient_finite']
    rows[label]=row;rows[label+'_profile']=profiled;del gradient
    write(a.output/('rank%d_intermediate.json'%rank),report)
   report['cases'][name]=rows;del reference_grad
   write(a.output/('rank%d_intermediate.json'%rank),report)
  assert base.core.C.state_digest(model)==parameter_sha,'Equivalence benchmark changed weights'
  ddp.zero_grad(set_to_none=True);model.controller=fast4
  for zero in (False,True):
   print('FAST_GATE_INFERENCE',rank,'zero' if zero else 'nonzero',flush=True)
   saved={n:p.detach().clone() for n,p in old.refiner.out.named_parameters()}
   if zero:
    with torch.no_grad():
     for parameter in old.refiner.out.parameters():parameter.zero_()
   old_time,old_pred=inference(rt,old,sample,ident);new_time,new_pred=inference(rt,fast4,sample,ident)
   key='zero_inference' if zero else 'nonzero_inference';report[key]=dict(old=old_time,fast4=new_time,comparison=output_compare(old_pred,new_pred,zero),speedup=old_time['seconds']/new_time['seconds'])
   assert old_time['instances']==250 and new_time['instances']==250,'Full250 natural inference population required'
   with torch.no_grad():
    for n,parameter in old.refiner.out.named_parameters():parameter.copy_(saved[n])
  # Separate TRUE full-objective DDP AMP updates, not tail benchmark resource.
  import copy
  optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=.01);scaler=torch.amp.GradScaler('cuda',init_scale=32.)
  report['full_objective_updates']=[];model.full=True;model.controller=fast4
  for step in (1,2):
   print('FAST_GATE_FULL_OBJECTIVE',rank,step,flush=True)
   ddp.train();current=copy.deepcopy(sample)
   if rank==3 and step==1:current['data_samples'][0].gt_instances=current['data_samples'][0].gt_instances[:0]
   optimizer.zero_grad(set_to_none=True);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();begin=time.perf_counter()
   with torch.autocast('cuda',dtype=torch.float16):loss=ddp(current,ident)
   assert torch.isfinite(loss);scaler.scale(loss).backward();scaler.unscale_(optimizer);norm=torch.nn.utils.clip_grad_norm_(model.parameters(),.5,error_if_nonfinite=True)
   before=scaler.get_scale();scaler.step(optimizer);scaler.update();assert scaler.get_scale()>=before,'resource gate AMP skipped step; no success claim'
   torch.cuda.synchronize();hashes=all_reports(base.core.C.state_digest(model),cpu);assert len(set(hashes))==1
   report['full_objective_updates'].append(dict(step=step,loss=float(loss.detach()),seconds=time.perf_counter()-begin,amp_scale=scaler.get_scale(),preclip_norm=float(norm),
    peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,metadata=dict(model.last),replica_sha256=hashes[0]))
   rt.clear()
  report['sources_after']=sources();assert all(report['sources_after'].get(k)==v for k,v in report['sources_before'].items())
  report['passed']=all(report['cases'][mode]['fast1']['comparison']['passed'] for mode in ('FP32','AMP')) and report['zero_inference']['comparison']['passed'] and report['nonzero_inference']['comparison']['passed']
  report['correctness_peak_reserved_mib']=max(report['cases'][mode][which]['peak_reserved_mib'] for mode in ('FP32','AMP') for which in ('old','old_profile','fast1','fast1_profile'))
  report['official_fast_execution_peak_reserved_mib']=max([r['peak_reserved_mib'] for r in report['full_objective_updates']]+[report[x]['fast4']['peak_reserved_mib'] for x in ('zero_inference','nonzero_inference')])
  report['max_peak_reserved_mib']=report['official_fast_execution_peak_reserved_mib']
  report['max_all_measured_peak_reserved_mib']=max(report['correctness_peak_reserved_mib'],report['max_peak_reserved_mib'],report['zero_inference']['old']['peak_reserved_mib'],report['nonzero_inference']['old']['peak_reserved_mib'])
  report['reservation_peak_definition']='fast full-objective AMP successful updates and fast FP32 inference only; FP32 backward diagnostic separately disclosed'
  write(a.output/('rank%d.json'%rank),report);ranks=all_reports(report,cpu)
  if rank==0:write(a.output/'complete.json',dict(passed=all(r['passed'] for r in ranks),world_size=4,ranks=ranks,source_sha256=report['sources_after'],max_peak_reserved_mib=max(r['max_peak_reserved_mib'] for r in ranks), correctness_peak_reserved_mib=max(r['correctness_peak_reserved_mib'] for r in ranks), max_all_measured_peak_reserved_mib=max(r['max_all_measured_peak_reserved_mib'] for r in ranks),
    caveats=['single fixed batch, development engineering only','tail speed benchmark omits original task losses; full objective resources measured separately','profile times inclusive/overlapping and synchronized; speed ratios uninstrumented','64 matched prediction RoIs need not represent64 distinct GT instances']))
 except Exception:
  report['failed']=traceback.format_exc();write(a.output/('rank%d_failed.json'%rank),report);raise
 finally:dist.destroy_process_group()

if __name__=='__main__':main()
