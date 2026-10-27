"""Four-rank native RoI real-model engineering gate. No AP, no long training.

Same-pass decoder capture, original512 mask supervision, G and instance losses.
Engineering updates world4 x image1, not the formal accumulation/global8 budget.
"""
import argparse
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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
WORK=Path(__file__).resolve().parents[2];REPO=WORK.parent
sys.path[:0]=[str(WORK),str(WORK/'scripts/probes')]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
 path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');tmp.replace(path)
def gather(value,cpu):
 out=[None]*4;dist.all_gather_object(out,value,group=cpu);return out

def source_inventory():
 paths={Path(__file__).resolve()}
 for module in tuple(sys.modules.values()):
  name=getattr(module,'__file__',None)
  if name:
   path=Path(name).resolve()
   if path.is_file() and path.suffix=='.py' and REPO in path.parents:paths.add(path)
 return {str(path.relative_to(REPO)):sha(path) for path in sorted(paths)}

def grad_group(model,needle):
 pairs=[(n,p.grad) for n,p in model.named_parameters() if needle in n and p.grad is not None]
 return dict(tensors=len(pairs),nonzero_elements=sum(int(torch.count_nonzero(g)) for _,g in pairs),norm=float(sum((g.detach().float().square().sum() for _,g in pairs),torch.tensor(0.,device='cuda')).sqrt()))


def parity(rt,sample,ident,base):
 rt.network.eval();rt.shim.observe=True
 with torch.no_grad():
  payload=rt.payload(ident);rt.joint.prepare_uav(**payload);features=rt.satellite.extract_feat(sample['inputs']);head=rt.satellite.roi_head;xx=head._inject_extra_pe(features[0]);meta=sample['data_samples'][0].metainfo
  # Exact same model, weights, RoIs, batch; reference bypasses only the shim.
  boxes=torch.tensor([[16.25,20.5,83.75,93.25],[203.5,114.75,270.25,165.5]],device='cuda');rois=torch.cat((boxes.new_zeros((2,1)),boxes),1)
  def decode():return head._mask_forward(xx,rois,image_embeddings=features[1],image_positional_embeddings=features[2],high_res_features=features[3])['mask_preds']
  actual=decode();obs=rt.shim.observation;shim=head.mask_head.mask_decoder
  head.mask_head.mask_decoder=shim.base
  # NativeMaskHead deliberately requires shim: call original base class method
  # through temporary class swap ONLY for no-head engineering oracle.
  original_cls=head.mask_head.__class__
  from rsprompter.cspm.mask_head import RSPrompterAnchorMaskHeadSAM2
  head.mask_head.__class__=RSPrompterAnchorMaskHeadSAM2
  try:reference=decode()
  finally:head.mask_head.__class__=original_cls;head.mask_head.mask_decoder=shim
  actual_mask=head.mask_head._predict_by_feat_single(actual.clone(),boxes.clone(),torch.zeros(2,device='cuda',dtype=torch.long),meta,head.test_cfg,False)
  ref_mask=head.mask_head._predict_by_feat_single(reference.clone(),boxes.clone(),torch.zeros(2,device='cuda',dtype=torch.long),meta,head.test_cfg,False)
  assert torch.equal(actual,reference) and torch.equal(actual_mask,ref_mask)
  # Evidence ports operate ONLY after upper G extraction, no upstream hook changes.
  original_rgb=rt.joint.uav.native_aligned_rgb.detach().clone();original_support=rt.joint.uav.native_rgb_valid.clone()
  assert original_rgb.shape==(1,32,128,128) and original_support.dtype==torch.bool
  valid=payload['index']>=0;assert torch.equal(original_support,valid.any(1).reshape(1,1,128,128))
  assert rt.shim.call_context is None and rt.shim.captured is None
  result=dict(raw_logits_exact=True,binary_exact=True,probability_exact=True,native_hw=list(actual.shape[-2:]),
   ROI_ids_exact=torch.equal(obs['boxes'],boxes),RGB_support_independent_exact=True,RGB_shape=list(original_rgb.shape),
   feature_finite=bool(torch.isfinite(original_rgb).all()),same_upper_G=True,decoder_calls_added_per_mask_call=1)
 rt.clear();rt.shim.observe=False;rt.shim.observation={};return result


def run_loss(rt,network,sample,ident,base,mask_only=False):
 from data.ue_instance_targets import center_targets
 boxes=sample['data_samples'][0].gt_instances.bboxes.detach().cpu().numpy()
 losses=network(sample,**rt.payload(ident),instance_target=center_targets(boxes),center_loss_fn=base.core.pilot.loss_fn)
 assert 'loss_mask' in losses and 'loss_uav_instances' in losses
 assert not any('ubr' in k or 'decoder_tail' in k for k in losses)
 if mask_only:
  other=sum(v*0 for k,values in losses.items() if 'loss' in k and k!='loss_mask' for v in (values if isinstance(values,(tuple,list)) else [values]) if torch.is_tensor(v) and v.requires_grad)
  return losses['loss_mask']+other,{'loss_mask':float(losses['loss_mask'].detach())}
 total=sum(v for k,values in losses.items() if 'loss' in k for v in (values if isinstance(values,(tuple,list)) else [values]) if torch.is_tensor(v) and v.requires_grad)
 return total,{k:sum(float(v.detach()) for v in (values if isinstance(values,(tuple,list)) else [values]) if torch.is_tensor(v)) for k,values in losses.items() if 'loss' in k}


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--allocator-cap-mib',type=int,default=16000);p.add_argument('--engineering-checkpoint',type=Path)
 a=p.parse_args();assert int(os.environ.get('WORLD_SIZE',0))==4;rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(local);torch.set_num_threads(2)
 torch.cuda.set_per_process_memory_fraction(a.allocator_cap_mib*2**20/torch.cuda.get_device_properties(local).total_memory)
 dist.init_process_group('nccl',timeout=timedelta(minutes=30));cpu=dist.new_group(backend='gloo',timeout=timedelta(minutes=30))
 if rank==0:a.output.mkdir(parents=True,exist_ok=False)
 dist.barrier();report=dict(rank=rank,world_size=4,pid=os.getpid(),scope='engineering_only_no_AP_no_long_training',updates=[])
 try:
  from ue_ubr_native_runtime_20261027 import build_runtime,base
  rt=build_runtime('C_correct');report['initialization']=rt.initialization_audit
  assert rt.satellite.train_cfg.rcnn.mask_size==(512,512) or tuple(rt.satellite.train_cfg.rcnn.mask_size)==(512,512)
  assert not any('decoder_tail_refiner' in name or 'ubr_refiner' in name for name,_ in rt.network.named_parameters())
  if a.engineering_checkpoint:
   receipt=json.loads((a.engineering_checkpoint.parent/'training_complete.json').read_text());assert receipt['complete'] and receipt['epoch']==10 and receipt['world_size']==4
   assert sha(a.engineering_checkpoint)==receipt['checkpoint_sha256'];report['engineering_checkpoint']=rt.load_engineering_G(a.engineering_checkpoint)
  ds=base.core.C.MinimalSatelliteDataset(base.core.previous.DATA,'train',augment=False)
  order=sorted(range(len(ds)),key=lambda i:len(ds.samples[i]['annotations']),reverse=True);index=order[rank];ident=int(ds.samples[index]['image_id'])
  sample=base.core.C.processed(rt.satellite,base.core.C.rtmdet_collate_fn([ds[index]]),False);report['image_id']=ident
  report['parity']=parity(rt,sample,ident,base)
  report['sources_before']=source_inventory()
  if rank==0:
   import shutil
   for name in report['sources_before']:
    target=a.output/'frozen_source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(REPO/name,target)
  frozen={n:p.detach().cpu().clone() for n,p in rt.network.named_parameters() if not p.requires_grad}
  rt.network=DistributedDataParallel(rt.bare,device_ids=[local],output_device=local,find_unused_parameters=True,broadcast_buffers=False)
  optimizer=rt.optimizer;scaler=torch.amp.GradScaler('cuda',init_scale=512.)
  import copy
  for step in (1,2,3):
   rt.network.train();rt.set_epoch(1);current=copy.deepcopy(sample)
   if rank==3 and step==1:current['data_samples'][0].gt_instances=current['data_samples'][0].gt_instances[:0]
   rt.shim.calls=0;torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();begin=time.perf_counter()
   state=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(),loader=rt.generator.get_state())
   buffers={n:b.detach().clone() for n,b in rt.bare.named_buffers()}
   attempts=[]
   for attempt in range(7):
    if attempt:
     random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch']);torch.cuda.set_rng_state(state['cuda']);rt.generator.set_state(state['loader'])
     with torch.no_grad():
      for n,b in rt.bare.named_buffers():b.copy_(buffers[n])
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.float16):loss,terms=run_loss(rt,rt.network,current,ident,base)
    assert torch.isfinite(loss);scaler.scale(loss).backward();scaler.unscale_(optimizer)
    bad=[n for n,p in rt.bare.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
    allbad=gather(bad[:8],cpu);attempts.append(dict(scale=scaler.get_scale(),bad_parameters=allbad))
    if any(allbad):
     scaler.update(new_scale=scaler.get_scale()/2);rt.clear()
     if attempt==6:raise RuntimeError('AMP same-batch backoff exhausted')
     continue
    norm=torch.nn.utils.clip_grad_norm_(rt.bare.parameters(),.5,error_if_nonfinite=True)
    gradient=dict(native_out=grad_group(rt.bare,'native_refiner.out'),native_hidden=grad_group(rt.bare,'native_refiner.input'),
      native_all=grad_group(rt.bare,'native_refiner'),UAV=grad_group(rt.bare,'uav.'),decoder=grad_group(rt.bare,'mask_decoder.base.'))
    head=rt.satellite.roi_head.mask_head.native_refiner
    parameter_diagnostic=dict(out_weight_before=float(head.out.weight.detach().abs().max()),out_bias_before=float(head.out.bias.detach().abs().max()),
      out_weight_grad=float(head.out.weight.grad.detach().abs().max()),out_bias_grad=float(head.out.bias.grad.detach().abs().max()),
      out_weight_in_optimizer=any(head.out.weight is p for group in optimizer.param_groups for p in group['params']))
    old_scale=scaler.get_scale();scaler.step(optimizer);scaler.update();assert scaler.get_scale()>=old_scale
    parameter_diagnostic.update(out_weight_after=float(head.out.weight.detach().abs().max()),out_bias_after=float(head.out.bias.detach().abs().max()))
    break
   torch.cuda.synchronize();replicas=gather(base.core.C.state_digest(rt.bare),cpu);assert len(set(replicas))==1
   report['updates'].append(dict(step=step,loss=float(loss.detach()),loss_terms=terms,gradient=gradient,parameter_diagnostic=parameter_diagnostic,preclip_norm=float(norm),attempts=attempts,
     seconds=time.perf_counter()-begin,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
     decoder_calls=rt.shim.calls,last_native_observation=rt.shim.observation,native_head_parameters=sum(p.numel() for p in rt.satellite.roi_head.mask_head.native_refiner.parameters()),empty_GT=rank==3 and step==1,replica_sha256=replicas[0]))
   assert gradient['native_all']['norm']>0
   if step==1:assert gradient['native_out']['norm']>0
   if step>=2 and gradient['native_hidden']['norm']==0:
    write(a.output/('rank%d_hidden_diagnostic.json'%rank),report)
    raise RuntimeError('Native hidden gradient remains zero: '+repr(parameter_diagnostic))
   rt.clear();write(a.output/('rank%d_progress.json'%rank),report)
  # Isolate ORIGINAL main mask objective after native output has learned nonzero.
  # All ranks use real nonempty GT here, so no extra training objective is needed.
  rt.network.train();optimizer.zero_grad(set_to_none=True)
  with torch.autocast('cuda',dtype=torch.float16):mask_loss,_=run_loss(rt,rt.network,sample,ident,base,mask_only=True)
  scaler.scale(mask_loss).backward();scaler.unscale_(optimizer)
  report['main_mask_only_precision']=dict(autocast='FP16',gradient_scale=scaler.get_scale(),optimizer_step=False,other_task_loss_weights=0)
  report['main_mask_only_gradient']=dict(native=grad_group(rt.bare,'native_refiner'),
    hidden=grad_group(rt.bare,'native_refiner.input'),UAV_LoRA=grad_group(rt.bare,'uav.rgb_encoder'),
    decoder=grad_group(rt.bare,'mask_decoder.base.'))
  assert report['main_mask_only_gradient']['native']['norm']>0 and report['main_mask_only_gradient']['hidden']['norm']>0
  assert report['main_mask_only_gradient']['UAV_LoRA']['norm']>0 and report['main_mask_only_gradient']['decoder']['norm']>0
  optimizer.zero_grad(set_to_none=True);rt.clear()
  # Same batch, weights and sampling RNG; diagnostic full-objective comparison,
  # no optimizer steps or extra training budget. Bypass only native output head.
  from rsprompter.cspm.mask_head import RSPrompterAnchorMaskHeadSAM2
  mask_head=rt.satellite.roi_head.mask_head;shim=rt.shim;native_cls=mask_head.__class__
  rng_saved=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(),loader=rt.generator.get_state())
  report['diagnostic_full_objective_timing']={}
  for mode in ('A_nohead_bypass','C_native'):
   random.setstate(rng_saved['python']);np.random.set_state(rng_saved['numpy']);torch.set_rng_state(rng_saved['torch']);torch.cuda.set_rng_state(rng_saved['cuda']);rt.generator.set_state(rng_saved['loader'])
   optimizer.zero_grad(set_to_none=True);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();began=time.perf_counter()
   if mode=='A_nohead_bypass':mask_head.mask_decoder=shim.base;mask_head.__class__=RSPrompterAnchorMaskHeadSAM2
   try:
    with torch.autocast('cuda',dtype=torch.float16):diagnostic_loss,_=run_loss(rt,rt.bare,sample,ident,base)
    diagnostic_loss.backward();torch.cuda.synchronize()
    report['diagnostic_full_objective_timing'][mode]=dict(seconds=time.perf_counter()-began,loss=float(diagnostic_loss.detach()),peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,no_optimizer_step=True,reference='same current G/shared weights; bypass native head only')
   finally:
    mask_head.mask_decoder=shim;mask_head.__class__=native_cls
   rt.clear()
  optimizer.zero_grad(set_to_none=True)
  assert all(torch.equal(v,dict(rt.bare.named_parameters())[n].detach().cpu()) for n,v in frozen.items())
  rt.network.eval();rt.clear();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();rt.shim.calls=0
  with torch.no_grad():
   payload=rt.payload(ident);rt.joint.prepare_uav(**payload);features=rt.satellite.extract_feat(sample['inputs']);head=rt.satellite.roi_head;xx=head._inject_extra_pe(features[0]);meta=sample['data_samples'][0].metainfo
   # Use the exact established supplementary-mask consumer, forced250 stress.
   boxes=torch.tensor([[16.25,20.5,83.75,93.25]],device='cuda').repeat(250,1);scores=torch.full((250,),.1,device='cuda')
   rt.shim.observe=True;torch.cuda.synchronize();begin=time.perf_counter()
   result=base.core.consumer.masks(head,xx,features,meta,boxes,scores);torch.cuda.synchronize()
   observation=rt.shim.observation
   assert len(result['scores'])==250 and rt.shim.calls==8,'Original source consumer must execute250 with chunk32, no extra decode'
   assert not torch.equal(observation['raw'],observation['refined']),'Updated native logits not consumed'
   report['inference250']=dict(instances=250,forced_box_fixture=True,seconds=time.perf_counter()-begin,decoder_calls=rt.shim.calls,
    native_hw=observation['native_hw'],refined_logits_changed=True,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
    peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,original_consumer=True)
   rt.shim.observe=False;rt.shim.observation={};rt.clear()
  report['sources_after']=source_inventory();assert all(report['sources_after'].get(k)==v for k,v in report['sources_before'].items())
  report['passed']=True;report['max_peak_reserved_mib']=max([r['peak_reserved_mib'] for r in report['updates']]+[report['inference250']['peak_reserved_mib']])
  write(a.output/('rank%d.json'%rank),report);rows=gather(report,cpu)
  if rank==0:write(a.output/'complete.json',dict(passed=all(r['passed'] for r in rows),world_size=4,ranks=rows,source_sha256=report['sources_after'],max_peak_reserved_mib=max(r['max_peak_reserved_mib'] for r in rows),
    contract='native decoder128, original main-mask512 target, same upper G, instance loss retained; engineering global4 no accumulation'))
 except Exception:
  report['failed']=traceback.format_exc()
  try:write(a.output/('rank%d_failed.json'%rank),report)
  except Exception:(a.output/('rank%d_failed.txt'%rank)).write_text(report['failed'])
  raise
 finally:dist.destroy_process_group()

if __name__=='__main__':main()
