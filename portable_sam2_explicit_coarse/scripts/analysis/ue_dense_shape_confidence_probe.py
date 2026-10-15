"""Three frozen dense prompt sources, actual original sparse always held fixed."""
import argparse,hashlib,math,subprocess,sys,time
from collections import defaultdict
from contextlib import nullcontext
from types import SimpleNamespace
import numpy as np
import torch
sys.path.insert(0,'/tmp')
import ue_rgb_content_probe as base
import ue_gt_prompt_oracle_probe as oracle
R=base.R;O=R/'dense_shape_confidence'
NAMES=['baseline','pred_binary_pm4','gt_binary_pm4']
class Replace:
 def __init__(self,head,target,mode):self.head=head;self.target=target;self.mode=mode;self.offset=0;self.ranges=[];self.originals=[]
 def call(self,module,args,output):
  a=self.offset;b=a+len(output);self.offset=b;self.ranges.append([a,b]);self.originals.append(output.detach().cpu());assert output.shape[1:]==(1,64,64)
  if self.mode=='baseline':return None
  binary=(output>=0) if self.mode=='pred_binary_pm4' else self.target[a:b,None].to(output.device).bool()
  value=(binary.to(output.dtype)*2-1)*4;assert value.shape==output.shape and torch.equal(value,(binary.to(output.dtype)*2-1)*4) and bool(((value==-4)|(value==4)).all())
  if self.mode=='pred_binary_pm4':assert torch.equal(value>=0,output>=0)
  return value
 def __enter__(self):self.h=self.head.shape_injector.mask_decoder.register_forward_hook(self.call);return self
 def __exit__(self,*a):
  self.h.remove()
  if a[0] is None:assert self.offset==len(self.target)
class DecoderGuard:
 def __init__(self,head,features,boxes,prompts=None):self.head=head;self.features=features;self.boxes=boxes;self.prompts=prompts;self.entries=[];self.offset=0;self.current=None
 def start(self,module,args,kwargs):self.current=kwargs['boxes'];assert self.current is not None
 def call(self,module,args,kwargs):
  n=len(self.current);a=self.offset;b=a+n;self.offset=b;assert torch.equal(self.current,self.boxes[a:b]);assert args[2].shape[1] in (2,6) and args[3].shape==(n,256,32,32)
  assert torch.equal(args[0],self.features[1].expand(n,-1,-1,-1));expected_pe=self.head.prompt_encoder.get_dense_pe().to(args[1]);assert torch.equal(args[1],expected_pe.expand(n,-1,-1,-1))
  assert set(kwargs)=={'high_res_features'} and len(kwargs['high_res_features'])==len(self.features[3][:2])
  for live,ref in zip(kwargs['high_res_features'],self.features[3][:2]):assert torch.equal(live,ref.expand(n,-1,-1,-1))
  assert len(args)==6 and args[4]==self.head.multimask_output and args[5] is False
  sparse=args[2] if self.prompts is None else self.prompts[len(self.entries)]['sparse'].to(args[2]);dense=args[3]
  assert torch.isfinite(sparse).all() and torch.isfinite(dense).all()
  # The actual decoder input sparse tensor is P even when upstream shape/points differ.
  self.entries.append(dict(sparse=sparse.detach().cpu().clone(),dense=dense.detach().cpu().clone(),boxes=self.current.detach().cpu().clone()))
  if self.prompts is not None:assert torch.equal(sparse.cpu(),self.prompts[len(self.entries)-1]['sparse'])
  return (*args[:2],sparse,dense,*args[4:]),kwargs
 def __enter__(self):self.hs=[self.head.register_forward_pre_hook(self.start,with_kwargs=True),self.head.mask_decoder.register_forward_pre_hook(self.call,with_kwargs=True)];return self
 def __exit__(self,*a):
  for h in self.hs:h.remove()
  if a[0] is None:assert self.offset==len(self.boxes)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--max-images',type=int,default=0);args=ap.parse_args();started=time.monotonic();torch.set_num_threads(2);torch.manual_seed(42);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 out=O/('smoke' if args.max_images else 'full');out.mkdir(exist_ok=True);expected=base.read(base.O/'correct_replay/run_manifest.json');cp=torch.load(expected['checkpoint'],map_location='cpu',weights_only=False);snap=base._snapshot(cp);ra=argparse.Namespace(sam2_ckpt=None,sam2_repo=None);base._resolve_sam2_repo(ra,snap);cfg,_=base._resolve_model_config(ra,snap);model=base._register_and_build(cfg);report=base._load_model_state(model,base._select_state_dict(cp,'ema'),allow_nonstrict=False);model.cuda().eval();model.uav_intervention='correct';del cp
 head=model.roi_head.mask_head;head.assert_no_mask_embedding_contract('dense shape/confidence');assert head.explicit_prompt_mode=='points_box_dense' and head.shape_injector.fusion_type=='roi_only'
 state=snap['data_config'];ds=base.UERGBDInstanceDataset(state['manifest'],'validation',state['geometry'],load_uav=True,cache_root='/tmp/uav_dual_dev_data/cache');assert ds.manifest_sha256==expected['manifest_sha256'];assert [r['index'] for r in ds.samples]==expected['image_ids']
 prior={k:defaultdict(list) for k in ('baseline','gt_binary_pm4')}
 for name,oldname in [('baseline','baseline'),('gt_binary_pm4','gt_dense_only')]:
  for r in base.read(R/'gt_prompt_factorial/full'/(oldname+'.json')):prior[name][r['image_id']].append(r)
 gtby={r['image_id']:r for r in base.read(R/'full_fixed_box_audit/dual_correct.json')['records'] if r['box_type']=='gt'};versions={k:t._version for k,t in model.named_parameters()};chunk=int(getattr(model.roi_head,'mask_roi_chunk_size',32) or 0);threshold=float(model.roi_head.test_cfg.mask_thr_binary);assert chunk==32
 m=dict(checkpoint=expected['checkpoint'],checkpoint_sha256=oracle.streaming_sha(expected['checkpoint']),architecture_contract=snap['architecture_contract'],trace_helper_sha256=base.file_sha(oracle.tracebase.__file__),weights='ema',strict=True,load_report=report,training=False,optimizer_created=False,variants=NAMES,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=base.ROOT,text=True).strip(),git_status_porcelain=subprocess.check_output(['git','status','--porcelain'],cwd=base.ROOT,text=True),script_sha256=base.file_sha(__file__),helper_script_sha256=base.file_sha(base.__file__),oracle_helper_sha256=base.file_sha(oracle.__file__),manifest_sha256=ds.manifest_sha256,preregistration_sha256=base.file_sha(O/'PREREGISTRATION.md'),mask_thr_binary=threshold,mask_logit_threshold=math.log(threshold/(1-threshold)),coarse_binary_threshold=0,amplitude=4,dense_shape_alpha=float(head._effective_shape_dense_alpha().detach().cpu()),dense_detach_input=bool(head.dense_prompt_cfg['detach_input']),mask_roi_chunk_size=chunk,actual_point_source='P',image_ids=[],complete=False)
 m['git_dirty']=bool(m['git_status_porcelain']);base.dump(out/'run_manifest.json',m);records={k:[] for k in NAMES};checks=[]
 with torch.inference_mode():
  for index,row in enumerate(ds.samples):
   if args.max_images and index>=args.max_images:break
   iid=row['index'];item=ds[index];payload=base.payload_hash(item['uav_payload']);rgbsha=base.tensor_sha(item['uav_payload']['rgb']);batch=base.ue_collate([item]);imgs,samples=base._build_data_samples(batch,torch.device('cuda'));processed=model.data_preprocessor(dict(inputs=imgs,data_samples=samples),training=False);sample=processed['data_samples'][0];features=model._extract_dual_feat(processed['inputs'],[sample]);tensors=list(features[0])+[features[1]]+([features[2]] if torch.is_tensor(features[2]) else [])+list(features[3] or ());hashes=[base.tensor_sha(t) for t in tensors]
   old=prior['baseline'][iid];n=len(old);boxes=np.array(gtby[iid]['boxes_xyxy'],dtype=np.float32);assert hashlib.sha256(boxes.tobytes()).hexdigest()==gtby[iid]['boxes_sha256'];assert n==len(boxes) and [r['gt_slot'] for r in old]==list(range(n));bt=torch.as_tensor(boxes,device='cuda');target=head.get_coarse_targets([SimpleNamespace(pos_priors=bt,pos_assigned_gt_inds=torch.arange(n,device='cuda'))],[sample.gt_instances],mask_size=(64,64)).cpu();assert bool(((target==0)|(target==1)).all());support=head._dense_embedding_box_support(bt,(32,32),torch.float32).sum((1,2,3)).cpu().numpy();truth=np.asarray(item['gt_masks'],dtype=bool);hc=[len(x._forward_hooks)+len(x._forward_pre_hooks) for x in head.modules()];p=None;base_masks=None;coarse=None;shams={}
   for name in NAMES:
    with Replace(head,target,name) as replace:
     with DecoderGuard(head,features,bt,p) as guard:masks=base.predict_masks(model,features,sample,boxes)
    sizes=[len(e['sparse']) for e in guard.entries];assert len(set(e['sparse'].shape[1] for e in guard.entries))==1;assert sizes==[min(chunk,n-a) for a in range(0,n,chunk)]
    original=torch.cat(replace.originals);sparse=torch.cat([e['sparse'] for e in guard.entries]);dense=torch.cat([e['dense'] for e in guard.entries]);no_mask=head.no_mask_embed.detach().cpu().reshape(1,256,1,1);norm=(dense-no_mask).float().flatten(1).norm(dim=1).numpy();assert bool((norm[support==0]==0).all())
    if name=='baseline':p=guard.entries;coarse=original;base_masks=masks.copy()
    else:assert torch.equal(original,coarse) and torch.equal(sparse,torch.cat([e['sparse'] for e in p]));assert np.array_equal(masks[support==0],base_masks[support==0])
    iou=(masks&truth).sum((1,2))/np.maximum((masks|truth).sum((1,2)),1)
    if name in prior:assert np.array_equal(iou,[r['gt_iou'] for r in prior[name][iid]]),('historical endpoint',iid,name)
    if args.max_images:
     # A second run with the same source and explicit baseline-sparse exchange is the actual-input sham.
     with Replace(head,target,name):
      with DecoderGuard(head,features,bt,p) as sham_guard:sham=base.predict_masks(model,features,sample,boxes)
     assert np.array_equal(sham,masks);assert all(torch.equal(a['sparse'],b['sparse']) and torch.equal(a['dense'],b['dense']) for a,b in zip(sham_guard.entries,guard.entries));shams[name]=True
    binary=(coarse[:,0]>=0) if name!='gt_binary_pm4' else target.bool();ti=target.bool();ci=(binary&ti).flatten(1).sum(1)/(binary|ti).flatten(1).sum(1).clamp(min=1);xor=np.logical_xor(masks,base_masks).sum((1,2))
    for j,z in enumerate(old):
     ident={k:z[k] for k in ('image_id','gt_slot','global_id','bbox_area','roi_level')};ss=base.tensor_sha(sparse[j]);assert ss==old[j]['actual_sparse_sha256'];ds_hash=base.tensor_sha(dense[j])
     if name in prior:assert ds_hash==prior[name][iid][j]['actual_dense_sha256']
     records[name].append(dict(**ident,mode=name,gt_iou=float(iou[j]),paste_xor_pixels=int(xor[j]),dense_support_cells=int(support[j]),dense_no_mask_delta_norm=float(norm[j]),actual_sparse_sha256=ss,baseline_sparse_sha256=old[j]['actual_sparse_sha256'],actual_dense_sha256=ds_hash,coarse_binary_iou=float(ci[j]),pred_shape_fg_pixels=int((coarse[j,0]>=0).sum()),source_shape_fg_pixels=int(binary[j].sum()),gt_shape_fg_pixels=int(target[j].sum()),sparse_token_count=int(sparse.shape[1])))
    print(iid,name,'GTIoU',float(iou.mean()),flush=True)
   assert hashes==[base.tensor_sha(t) for t in tensors] and payload==base.payload_hash(sample.uav) and rgbsha==base.tensor_sha(sample.uav.rgb);assert hc==[len(x._forward_hooks)+len(x._forward_pre_hooks) for x in head.modules()]
   checks.append(dict(image_id=iid,historical_endpoints_exact=True,actual_sparse_exact=True,actual_visual_values_exact=True,features_unchanged=True,geometry_rgb_unchanged=True,hooks_removed=True,original_coarse_exact=True,support_zero_masks_exact=True,binary_shape_preserved_exact=True,pm4_values_exact=True,actual_sparse_token_counts=[int(e['sparse'].shape[1]) for e in guard.entries],support0_count=int((support==0).sum()),chunk_sizes=sizes,target_ranges=replace.ranges,sham_pixel_exact=shams if args.max_images else None));m['image_ids'].append(iid)
   del p,guard,replace,masks,base_masks,coarse,original,sparse,dense,norm,features,tensors,hashes,target,support,processed,sample,samples,imgs,batch,item,bt
 assert versions=={k:t._version for k,t in model.named_parameters()};m.update(complete=len(checks)==153,processed_images=len(checks),ground_truth_instances=len(records['baseline']),parameter_versions_unchanged=True,elapsed_seconds=time.monotonic()-started)
 for name in NAMES:base.dump(out/(name+'.json'),records[name])
 base.dump(out/'run_manifest.json',m);base.dump(out/'invariance_checks.json',checks);print('DONE DENSE SHAPE CONFIDENCE',flush=True)
if __name__=='__main__':main()
