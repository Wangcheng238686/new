"""Frozen GT-shape conditional oracle; no training, RPN or AP evaluation."""
import argparse, hashlib, math, subprocess, sys, time
from collections import defaultdict
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,'/tmp')
import ue_rgb_content_probe as base
import ue_p2_prompt_trace as tracebase
Observer=tracebase.Observer;quality=tracebase.quality;delta_stats=tracebase.delta_stats;STAGES=tracebase.STAGES
R=base.R;O=R/'gt_prompt_oracle';NAMES=['baseline','joint_gt_prompt']

def streaming_sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()

class ShapeReplacement:
 def __init__(self,head,target,mode):self.head=head;self.target=target;self.mode=mode;self.offset=0;self.originals=[];self.handle=None;self.ranges=[]
 def replace(self,module,args,output):
  n=len(output);a=self.offset;b=a+n;self.offset=b;self.ranges.append([a,b]);self.originals.append(output.detach().cpu())
  assert output.shape[1:]==(1,64,64) and b<=len(self.target)
  if self.mode=='sham':return output.clone()
  replacement=(self.target[a:b,None].to(device=output.device,dtype=output.dtype)*2-1)*4
  assert replacement.shape==output.shape and torch.isfinite(replacement).all()
  return replacement
 def __enter__(self):self.handle=self.head.shape_injector.mask_decoder.register_forward_hook(self.replace);return self
 def __exit__(self,*unused):
  self.handle.remove()
  if unused[0] is None:assert self.offset==len(self.target)

def predict(model,features,sample,boxes,head,target,mode,observe=True):
 # Replacement is registered first so all observer callbacks see actual oracle input.
 from contextlib import nullcontext
 with (ShapeReplacement(head,target,mode) if mode!='baseline' else nullcontext(None)) as replace:
  with (Observer(head) if observe else nullcontext(None)) as observer:masks=base.predict_masks(model,features,sample,boxes)
 return masks,observer.merged() if observe else None,observer.entries if observe else [],replace

def main():
 p=argparse.ArgumentParser();p.add_argument('--max-images',type=int,default=0);args=p.parse_args();started=time.monotonic();torch.set_num_threads(2);torch.manual_seed(42);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 out=O/('smoke' if args.max_images else 'full');out.mkdir(exist_ok=True);expected=base.read(base.O/'correct_replay/run_manifest.json')
 checkpoint=torch.load(expected['checkpoint'],map_location='cpu',weights_only=False);snap=base._snapshot(checkpoint);ra=argparse.Namespace(sam2_ckpt=None,sam2_repo=None);base._resolve_sam2_repo(ra,snap);cfg,_=base._resolve_model_config(ra,snap);model=base._register_and_build(cfg);report=base._load_model_state(model,base._select_state_dict(checkpoint,'ema'),allow_nonstrict=False);model.cuda().eval();model.uav_intervention='correct';del checkpoint
 head=model.roi_head.mask_head;head.assert_no_mask_embedding_contract('GT shape oracle');assert head.shape_injector.fusion_type=='roi_only' and head.explicit_prompt_mode=='points_box_dense' and head.use_shape_dense
 state=snap['data_config'];ds=base.UERGBDInstanceDataset(state['manifest'],'validation',state['geometry'],load_uav=True,cache_root='/tmp/uav_dual_dev_data/cache');assert ds.manifest_sha256==expected['manifest_sha256'];assert [x['index'] for x in ds.samples]==expected['image_ids']
 prior=defaultdict(list)
 for x in base.read(R/'p2_prompt_trace/full/baseline.json'):prior[x['image_id']].append(x)
 gtboxes_by_image={x['image_id']:x for x in base.read(R/'full_fixed_box_audit/dual_correct.json')['records'] if x['box_type']=='gt'}
 versions={k:t._version for k,t in model.named_parameters()};status=subprocess.check_output(['git','status','--porcelain'],cwd=base.ROOT,text=True)
 threshold=float(model.roi_head.test_cfg.mask_thr_binary);logit_threshold=math.log(threshold/(1-threshold));chunk_size=int(getattr(model.roi_head,'mask_roi_chunk_size',32) or 0)
 manifest=dict(checkpoint=expected['checkpoint'],checkpoint_sha256=streaming_sha(expected['checkpoint']),weights='ema',strict=True,load_report=report,training=False,optimizer_created=False,variants=NAMES,protocol='GT common-box conditional oracle; no AP/RPN/bbox rerun',git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=base.ROOT,text=True).strip(),git_dirty=bool(status),git_status_porcelain=status,script_sha256=base.file_sha(__file__),helper_script_sha256=base.file_sha(base.__file__),trace_helper_sha256=base.file_sha(tracebase.__file__),manifest_sha256=ds.manifest_sha256,preregistration_sha256=base.file_sha(O/'PREREGISTRATION.md'),architecture_contract=snap['architecture_contract'],mask_thr_binary=threshold,mask_logit_threshold=logit_threshold,coarse_foreground_threshold=.5,oracle_logit_amplitude=4,oracle_input='binary actual get_coarse_targets 64x64 mapped to finite -4/+4; oracle coarse IoU is construction check, not learned improvement',dense_shape_alpha=float(head._effective_shape_dense_alpha().detach().cpu()),dense_detach_input=bool(head.dense_prompt_cfg['detach_input']),coarse_loss_weight=float(head.coarse_mask_loss_module.weight),shape_fusion_type=head.shape_injector.fusion_type,explicit_prompt_mode=head.explicit_prompt_mode,mask_roi_chunk_size=chunk_size,image_ids=[],complete=False)
 base.dump(out/'run_manifest.json',manifest);records={n:[] for n in NAMES};checks=[]
 with torch.inference_mode():
  for index,row in enumerate(ds.samples):
   if args.max_images and index>=args.max_images:break
   iid=row['index'];item=ds[index];before=base.payload_hash(item['uav_payload']);rgbsha=base.tensor_sha(item['uav_payload']['rgb']);batch=base.ue_collate([item]);imgs,samples=base._build_data_samples(batch,torch.device('cuda'));processed=model.data_preprocessor(dict(inputs=imgs,data_samples=samples),training=False);imgs=processed['inputs'];sample=processed['data_samples'][0]
   features=model._extract_dual_feat(imgs,[sample]);x,embedding,pe,high=features;tensors=list(x)+[embedding]+([pe] if torch.is_tensor(pe) else [])+list(high or ());hashes=[base.tensor_sha(t) for t in tensors]
   old=prior[iid];n=len(old);gtboxes=np.array(gtboxes_by_image[iid]['boxes_xyxy'],dtype=np.float32);assert hashlib.sha256(gtboxes.tobytes()).hexdigest()==gtboxes_by_image[iid]['boxes_sha256'];assert n==len(gtboxes) and [z['gt_slot'] for z in old]==list(range(n))
   box_tensor=torch.as_tensor(gtboxes,device='cuda');roi=torch.cat([box_tensor.new_zeros(n,1),box_tensor],dim=1);levels=model.roi_head.mask_roi_extractor.map_roi_levels(roi,model.roi_head.mask_roi_extractor.num_inputs).cpu().numpy();assert np.array_equal(levels,[z['roi_level'] for z in old])
   target=head.get_coarse_targets([SimpleNamespace(pos_priors=box_tensor,pos_assigned_gt_inds=torch.arange(n,device='cuda'))],[sample.gt_instances],mask_size=(64,64)).cpu();assert bool(((target==0)|(target==1)).all())
   support=head._dense_embedding_box_support(box_tensor,(32,32),torch.float32).sum((1,2,3)).cpu();base_trace=None;base_masks=None;chunk_sizes=None;sham_exact=None;off_exact=None
   hook_counts=[len(m._forward_hooks)+len(m._forward_pre_hooks) for m in head.modules()]
   for name in NAMES:
    masks,trace,entries,replacement=predict(model,features,sample,gtboxes,head,target,name)
    sizes=[len(z['roi']) for z in entries];assert len(sizes)==math.ceil(n/(chunk_size or n)) and sum(sizes)==n
    if chunk_sizes is None:chunk_sizes=sizes
    else:assert sizes==chunk_sizes
    assert torch.equal(trace['boxes'],torch.from_numpy(gtboxes))
    for k in STAGES:assert len(trace[k])==n and torch.isfinite(trace[k]).all(),k
    assert trace['coarse'].shape[1:]==(1,64,64) and trace['canvas'].shape[1:]==(1,128,128) and trace['dense'].shape[1:]==(256,32,32)
    truth=np.asarray(item['gt_masks'],dtype=bool);iou=(masks&truth).sum((1,2))/np.maximum((masks|truth).sum((1,2)),1)
    if name=='baseline':
     assert np.array_equal(iou,np.array([z['gt_iou'] for z in old])),('Historical baseline mismatch',iid);base_trace=trace;base_masks=masks.copy()
     if args.max_images:
      off,_,_,_=predict(model,features,sample,gtboxes,head,target,'baseline',False);assert np.array_equal(off,masks);off_exact=True
      sham,st,_,sr=predict(model,features,sample,gtboxes,head,target,'sham');assert np.array_equal(sham,masks)
      for k in STAGES+['coords','labels','local_yx','boxes','sparse_token_count']:assert torch.equal(st[k],trace[k]),('Sham stage differs',k,iid)
      assert torch.equal(torch.cat(sr.originals),trace['coarse']);sham_exact=True;del st,sr,sham,off
    else:
     assert torch.equal(torch.cat(replacement.originals),base_trace['coarse']) and torch.equal(trace['roi'],base_trace['roi'])
     assert torch.equal(trace['coarse'],(target[:,None]*2-1)*4)
    qstats=quality(trace,target);bqstats=quality(base_trace,target);stats={key:delta_stats(trace[key],base_trace[key]) for key in STAGES};bp=base_trace['coarse'].sigmoid();vp=trace['coarse'].sigmoid();coarse_prob_mae=(vp-bp).abs().flatten(1).mean(1);coarse_flip=(vp>=.5).ne(bp>=.5).float().flatten(1).mean(1)
    valid=trace['labels']>=0;bv=base_trace['labels']>=0;changed=trace['labels'].ne(base_trace['labels']) | (trace['coords'].ne(base_trace['coords']).any(2)&(valid|bv));changed&=(valid|bv);both=valid&bv;move=(trace['coords']-base_trace['coords']).float().norm(dim=2)*both
    pre_flip=(trace['sam_raw']>=logit_threshold).ne(base_trace['sam_raw']>=logit_threshold).float().flatten(1).mean(1);post_flip=(trace['tail_final']>=logit_threshold).ne(base_trace['tail_final']>=logit_threshold).float().flatten(1).mean(1)
    finalxor=np.logical_xor(masks,base_masks).sum((1,2));pd=(trace['pe_dense']-base_trace['pe_dense']).float().flatten(1).norm(dim=1);fd=(trace['dense']-base_trace['dense']).float().flatten(1).norm(dim=1);attenuation=torch.where(pd>1e-12,fd/pd,torch.full_like(pd,float('nan')))
    no_mask=head.no_mask_embed.detach().cpu().reshape(1,256,1,1);dense_no_base=(trace['dense']-no_mask).float().flatten(1).norm(dim=1);assert bool(dense_no_base[support==0].eq(0).all())
    original=base_trace['coarse'].float().flatten(1);orig_quantiles=torch.quantile(original,torch.tensor([0.,.05,.5,.95,1.]),dim=1).T
    for j,identity in enumerate(old):
     ident={k:identity[k] for k in ('image_id','gt_slot','global_id','bbox_area','roi_level')}
     r=dict(**ident,mode=name,gt_iou=float(iou[j]),coarse_iou=qstats['coarse_iou'][j],coarse_bce=qstats['coarse_bce'][j],predicted_coarse_iou=float(bqstats['coarse_iou'][j]),coarse_target_fg_pixels=int(target[j].sum()),oracle_input_exact=bool(name=='joint_gt_prompt'),original_coarse_logit_quantiles=orig_quantiles[j].tolist(),original_coarse_logit_mean=float(original[j].mean()),original_coarse_logit_std=float(original[j].std(unbiased=False)),dense_support_cells=int(support[j]),dense_no_mask_delta_norm=float(dense_no_base[j]),sparse_token_count=qstats['sparse_token_count'][j],sparse_token_count_changed=bool(trace['sparse_token_count'][j]!=base_trace['sparse_token_count'][j]),point_valid=qstats['point_valid'][j],point_correct=qstats['point_correct'][j],point_labels=qstats['point_labels'][j],point_coords=qstats['point_coords'][j],point_local_yx=qstats['point_local_yx'][j],point_changed_count=int(changed[j].sum()),point_both_valid_count=int(both[j].sum()),point_move_distance_sum=float(move[j].sum()),coarse_prob_mae=float(coarse_prob_mae[j]),coarse_binary_flip_fraction=float(coarse_flip[j]),sam_raw_binary_flip_fraction=float(pre_flip[j]),tail_final_binary_flip_fraction=float(post_flip[j]),paste_xor_pixels=int(finalxor[j]),dense_final_to_pe_delta_ratio=float(attenuation[j]) if torch.isfinite(attenuation[j]) else None,stages={key:{field:vals[j] for field,vals in stats[key].items()} for key in STAGES})
     records[name].append(r)
    print(iid,name,'GTIoU',float(iou.mean()),'point_changes',int(changed.sum()),flush=True)
    if name!='baseline':del trace
   assert hashes==[base.tensor_sha(t) for t in tensors] and base.payload_hash(sample.uav)==before and base.tensor_sha(sample.uav.rgb)==rgbsha
   assert hook_counts==[len(m._forward_hooks)+len(m._forward_pre_hooks) for m in head.modules()]
   checks.append(dict(image_id=iid,baseline_endpoint_iou_exact=True,observer_smoke_pixel_exact=off_exact,sham_smoke_pixel_exact=sham_exact,geometry_rgb_restored=True,features_unchanged=True,chunk_sizes=chunk_sizes,gt_slots_in_order=list(range(n)),oracle_target_ranges=replacement.ranges,hooks_removed=True,oracle_input_exact=True,roi_unchanged=True,original_coarse_unchanged=True));manifest['image_ids'].append(iid)
   del base_trace,base_masks,features,x,embedding,pe,high,tensors,hashes,masks,target,support,processed,sample,samples,imgs,batch,item,entries,replacement
   torch.cuda.empty_cache()
 assert versions=={k:t._version for k,t in model.named_parameters()};manifest.update(complete=len(checks)==153,processed_images=len(checks),ground_truth_instances=len(records['baseline']),parameter_versions_unchanged=True,elapsed_seconds=time.monotonic()-started)
 for n in NAMES:base.dump(out/(n+'.json'),records[n])
 base.dump(out/'run_manifest.json',manifest);base.dump(out/'invariance_checks.json',checks);print('DONE GT PROMPT ORACLE',flush=True)
if __name__=='__main__':main()
