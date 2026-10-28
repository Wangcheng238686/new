"""Read-only four-rank fixed-global8 checkpoint gradient telemetry, no optimizer update."""
import argparse,hashlib,json,os,sys,time,traceback
from pathlib import Path
from datetime import timedelta
import numpy as np
import torch
import torch.distributed as dist
from ue_native_gradient_math_20261028 import partition_losses,parameter_group,summarize,direction,SHARED_GROUPS


def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n');t.replace(p)


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--arm',choices=['A_nohead','B_zero','C_correct'],required=True);p.add_argument('--epoch',type=int,choices=[1,10,25,50,75,100],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--allocator-cap-mib',type=int,default=14336);args=p.parse_args()
 assert int(os.environ.get('WORLD_SIZE',0))==4;rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);torch.set_num_threads(2)
 torch.cuda.set_per_process_memory_fraction(args.allocator_cap_mib*2**20/torch.cuda.get_device_properties(rank).total_memory)
 dist.init_process_group('nccl',timeout=timedelta(minutes=90));cpu=dist.new_group(backend='gloo',timeout=timedelta(minutes=90))
 if rank==0:args.output.mkdir(parents=True,exist_ok=False)
 dist.barrier();report={'complete':False,'arm':args.arm,'epoch':args.epoch,'rank':rank}
 try:
  manifest=json.loads((args.root/'launch_manifest.json').read_text());source=Path(manifest['runtime_source']);work=source/'portable_sam2_explicit_coarse'
  sys.path[:0]=[str(source/'sam2'),str(work),str(work/'scripts/probes')]
  import ue_ubr_native_full_run_20261028 as full
  from data.ue_instance_targets import center_targets
  full.native.configure();cp_path=args.root/args.arm/('epoch%d.pth'%args.epoch);checkpoint_sha=sha(cp_path)
  cp=torch.load(cp_path,map_location='cpu',weights_only=False);contract=json.loads((args.root/args.arm/'contract.json').read_text());assert cp['epoch']==args.epoch and cp['step']==args.epoch*267 and cp['contract']==contract
  for name,digest in contract['source_hashes'].items():assert sha(source/name)==digest,name
  rt=full.native.runtime(args.arm,False);rt.bare.load_state_dict(cp['model'],strict=True);del cp
  rt.bare.train();rt.set_epoch(args.epoch)
  # Always unscaled FP32 to avoid diagnosing FP16 activation underflow as conflict.
  initial=full.core.C.state_digest(rt.bare);buffers={n:b.detach().clone() for n,b in rt.bare.named_buffers()}
  named=[(n,p) for n,p in rt.bare.named_parameters() if p.requires_grad];params=[p for _,p in named];sizes=[p.numel() for p in params];offsets=np.cumsum([0]+sizes).tolist();total=offsets[-1]
  shared_indices=torch.cat([torch.arange(offsets[i],offsets[i+1],device='cuda') for i,(name,p) in enumerate(named) if parameter_group(name) in SHARED_GROUPS])
  vectors=torch.zeros((3,total),device='cuda',dtype=torch.float32);support=torch.zeros((3,len(params)),device='cuda',dtype=torch.int32)
  ds=full.core.C.MinimalSatelliteDataset(full.core.previous.DATA,'train',augment=False);indices=np.linspace(0,len(ds)-1,8,dtype=int).tolist();rows=[];calibration=[];loss_keys=None
  tasks_order=('detection','mask','instance_auxiliary');torch.cuda.reset_peak_memory_stats();began=time.time()
  def restore_buffers():
   with torch.no_grad():
    for name,b in rt.bare.named_buffers():b.copy_(buffers[name])
  def one(position,calibrate):
   restore_buffers();index=indices[position];full.core.C.seed(44000+index)
   sample=full.core.C.processed(rt.satellite,full.core.C.rtmdet_collate_fn([ds[index]]),True);ident=int(ds.samples[index]['image_id'])
   boxes=sample['data_samples'][0].gt_instances.bboxes.detach().cpu().numpy()
   losses=rt.bare(sample,**rt.payload(ident),instance_target=center_targets(boxes),center_loss_fn=full.core.pilot.loss_fn)
   tasks,keys=partition_losses(losses);v=torch.zeros_like(vectors);s=torch.zeros_like(support)
   for ti,name in enumerate(tasks_order):
    grads=torch.autograd.grad(tasks[name],params,retain_graph=calibrate or ti<2,allow_unused=True)
    for j,g in enumerate(grads):
     if g is not None:
      assert torch.isfinite(g).all(),('nonfinite',name,named[j][0]);v[ti,offsets[j]:offsets[j+1]]=g.detach().flatten().float();s[ti,j]=1
    del grads
   check=None
   if calibrate:
    reference=torch.autograd.grad(sum(tasks.values()),params,allow_unused=True);flat=torch.zeros(total,device='cuda')
    for j,g in enumerate(reference):
     if g is not None:flat[offsets[j]:offsets[j+1]]=g.detach().flatten().float()
    diff=(flat-v.sum(0));rel=float(diff.double().norm()/flat.double().norm().clamp_min(1e-30));maximum=float(diff.abs().max());assert rel<1e-4 or maximum<1e-5,(rel,maximum)
    check={'sum_task_vs_total_relative_L2':rel,'max_abs_error':maximum};del reference,flat,diff
   values={name:float(tasks[name].detach()) for name in tasks_order};per_image_direction=direction(v[0,shared_indices],v[1,shared_indices]);rt.clear();del losses,tasks,sample
   return v,s,keys,dict(position=position,image_id=ident,seed=44000+index,losses=values,shared_direction_unaveraged_image=per_image_direction),check
  for local in range(2):
   position=rank*2+local;v,s,keys,row,check=one(position,local==0)
   if loss_keys is None:loss_keys=keys
   assert keys==loss_keys
   vectors.add_(v/8);support.add_(s);rows.append(row)
   if local==0:
    repeat,ss,kk,rr,_=one(position,False);delta=repeat-v;rel=float(delta.double().norm()/v.double().norm().clamp_min(1e-30));mx=float(delta.abs().max());assert rel<1e-4 or mx<1e-5,(rel,mx)
    calibration.append(dict(check,repeat_seed_relative_L2=rel,repeat_seed_max_abs=mx,position=position));del repeat,ss,delta
   del v,s
  dist.all_reduce(vectors,op=dist.ReduceOp.SUM);dist.all_reduce(support,op=dist.ReduceOp.SUM);assert torch.isfinite(vectors).all()
  stats=summarize(named,vectors,support>0);restore_buffers();rt.clear()
  assert full.core.C.state_digest(rt.bare)==initial and all(p.grad is None for p in params) and not rt.optimizer.state
  assert sha(cp_path)==checkpoint_sha
  for name,digest in contract['source_hashes'].items():assert sha(source/name)==digest,name
  row=dict(report,complete=True,rows=rows,calibration=calibration,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,seconds=time.time()-began)
  write(args.output/('rank%d.json'%rank),row);allrows=[None]*4 if rank==0 else None;dist.gather_object(row,allrows,dst=0,group=cpu)
  if rank==0:
   write(args.output/'complete.json',dict(complete=True,arm=args.arm,epoch=args.epoch,checkpoint=str(cp_path),checkpoint_sha256=checkpoint_sha,world_size=4,global_images=8,
    protocol=dict(mode='FP32 train mode; augmentation disabled; dropout/RoI seed fixed per image; original task weights; buffers restored before each image',loss_keys=loss_keys,
       norm='L2 of global8 mean gradient; per-image sum/8 then rank SUM; preclip, unscaled, not AdamW step',cosine='fixed parameter group union with None=0; zero task norm=null; joint_connected_only supplementary',
       indices=indices,image_ids=[int(ds.samples[i]['image_id']) for i in indices],sampling_epochs=[1,10,25,50,75,100],split='train',scope='fixed checkpoint functions, not live stochastic AMP update gradients'),
    parameter_layout=[dict(name=n,shape=list(p.shape),elements=p.numel(),group=parameter_group(n)) for n,p in named],stats=stats,ranks=allrows,
    parameters_and_restored_buffers_unchanged=True,param_grad_all_none=True,no_optimizer_updates=True,
    observer_source_sha256={str(Path(__file__).resolve()):sha(__file__),str(Path(__file__).with_name('ue_native_gradient_math_20261028.py').resolve()):sha(Path(__file__).with_name('ue_native_gradient_math_20261028.py'))},
    limitations='single fixed8 unaugmented train batch; FP32 gradient proxy; negative cosine alone not proof of AP harm; instance auxiliary separate; no automatic weighting/gating/selection'))
 except Exception:
  write(args.output/('rank%d_failed.json'%rank),dict(report,failed=traceback.format_exc()));raise
 finally:dist.destroy_process_group()

if __name__=='__main__':main()
