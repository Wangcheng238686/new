"""Read-only stage tracing of the already evaluated frozen P2 transfer cells."""
import argparse
import hashlib
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0,'/tmp')
import ue_rgb_content_probe as base
R=base.R;O=R/'p2_prompt_trace';PREV=R/'p2_transfer_audit/full'
NAMES=['baseline','native_correct_q','native_donor_q','native_correct_one']
STAGES=['roi','coarse','canvas','pe_dense','sparse','dense','sam_raw','tail_final']

def cpu(x):return x.detach().cpu()

class Observer:
 def __init__(self,head):
  self.head=head;self.current=None;self.entries=[];self.handles=[]
 def put(self,key,value):
  assert self.current is not None and key not in self.current,key
  self.current[key]=cpu(value)
 def start(self,module,args,kwargs):
  assert self.current is None
  self.current={};self.put('roi',args[0]);self.put('boxes',kwargs['boxes'])
 def coarse(self,module,args,output):
  assert torch.equal(cpu(args[0]),self.current['roi']);self.put('coarse',output)
 def point(self,module,args,output):
  assert torch.equal(cpu(args[0]),self.current['coarse']);self.put('coords',output[0]);self.put('labels',output[1]);yx,labels=self.head.shape_point_miner.get_last_local_points();self.put('local_yx',yx);assert torch.equal(cpu(labels),self.current['labels'])
 def pe_before(self,module,args,kwargs):
  assert kwargs.get('masks') is not None
  self.put('canvas',kwargs['masks'])
 def pe_after(self,module,args,kwargs,output):self.put('pe_dense',output[1])
 def decoder_before(self,module,args,kwargs):
  sparse=args[2];n,count,c=sparse.shape;assert count in (2,6)
  self.current['sparse_token_count']=torch.full((n,),count,dtype=torch.long)
  if count==2:sparse=torch.cat([sparse.new_zeros(n,4,c),sparse],dim=1)
  self.put('sparse',sparse);self.put('dense',args[3])
 def decoder_after(self,module,args,output):self.put('sam_raw',output[0])
 def finish(self,module,args,kwargs,output):
  assert torch.equal(cpu(output.coarse_outputs['raw_logits']),self.current['coarse'])
  self.put('tail_final',output.mask_logits)
  for key in STAGES+['coords','labels','local_yx','boxes','sparse_token_count']:assert key in self.current,key
  self.entries.append(self.current);self.current=None
 def __enter__(self):
  h=self.head
  self.handles=[h.register_forward_pre_hook(self.start,with_kwargs=True),h.shape_injector.mask_decoder.register_forward_hook(self.coarse),h.shape_point_miner.register_forward_hook(self.point),h.prompt_encoder.register_forward_pre_hook(self.pe_before,with_kwargs=True),h.prompt_encoder.register_forward_hook(self.pe_after,with_kwargs=True),h.mask_decoder.register_forward_pre_hook(self.decoder_before,with_kwargs=True),h.mask_decoder.register_forward_hook(self.decoder_after),h.register_forward_hook(self.finish,with_kwargs=True)]
  return self
 def __exit__(self,*unused):
  for handle in self.handles:handle.remove()
 def merged(self):
  assert self.current is None and self.entries
  return {key:torch.cat([x[key] for x in self.entries],dim=0) for key in self.entries[0]}

def delta_stats(v,b):
 d=(v.float()-b.float()).flatten(1);bf=b.float().flatten(1)
 return dict(rms=d.square().mean(1).sqrt().tolist(),mae=d.abs().mean(1).tolist(),relative_l2=(d.norm(dim=1)/bf.norm(dim=1).clamp_min(1e-6)).tolist(),strict_changed=d.ne(0).any(1).tolist(),changed_gt1e6=d.square().mean(1).sqrt().gt(1e-6).tolist())

def quality(t,targets):
 logits=t['coarse'][:,0].float();binary=logits>=0;truth=targets>=.5;inter=(binary&truth).sum((1,2));union=(binary|truth).sum((1,2));n=len(truth);yx=t['local_yx'].long();labels=t['labels'];sampled=truth[torch.arange(n)[:,None],yx[...,0].clamp(0,63),yx[...,1].clamp(0,63)];valid=labels>=0;correct=torch.where(labels==1,sampled,~sampled)&valid
 return dict(coarse_iou=(inter/union.clamp_min(1)).tolist(),coarse_bce=F.binary_cross_entropy_with_logits(logits,targets.float(),reduction='none').mean((1,2)).tolist(),point_valid=valid.tolist(),point_correct=correct.tolist(),point_labels=labels.tolist(),point_coords=t['coords'].tolist(),point_local_yx=yx.tolist(),sparse_token_count=t['sparse_token_count'].tolist())

def main():
 p=argparse.ArgumentParser();p.add_argument('--max-images',type=int,default=0);args=p.parse_args();torch.set_num_threads(2);torch.manual_seed(42);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 out=O/('smoke' if args.max_images else 'full');out.mkdir(exist_ok=True);expected=base.read(base.O/'correct_replay/run_manifest.json');maps=base.read(base.O/'donor_maps.json');donors={x['target']:x for x in maps['maps']['donor_a']['records']}
 checkpoint=torch.load(expected['checkpoint'],map_location='cpu',weights_only=False);snap=base._snapshot(checkpoint);ra=argparse.Namespace(sam2_ckpt=None,sam2_repo=None);base._resolve_sam2_repo(ra,snap);cfg,_=base._resolve_model_config(ra,snap);model=base._register_and_build(cfg);report=base._load_model_state(model,base._select_state_dict(checkpoint,'ema'),allow_nonstrict=False);model.cuda().eval();model.uav_intervention='correct';del checkpoint
 head=model.roi_head.mask_head;head.assert_no_mask_embedding_contract('P2 readout trace');assert head.shape_injector.fusion_type=='roi_only' and head.explicit_prompt_mode=='points_box_dense' and head.use_shape_dense
 state=snap['data_config'];ds=base.UERGBDInstanceDataset(state['manifest'],'validation',state['geometry'],load_uav=True,cache_root='/tmp/uav_dual_dev_data/cache');assert ds.manifest_sha256==expected['manifest_sha256'];assert [x['index'] for x in ds.samples]==expected['image_ids'];fuser=model.fpn_fusion[0];assert model.fpn_levels==(1,2)
 prior={n:{x['image_id']:x for x in base.read(PREV/(n+'_fixed_box.json'))} for n in NAMES};oldids=base.read(PREV/'gt_identities.json');identities=defaultdict(list)
 for x in oldids:identities[x['image_id']].append(x)
 versions={k:t._version for k,t in model.named_parameters()};status=subprocess.check_output(['git','status','--porcelain'],cwd=base.ROOT,text=True)
 threshold=float(model.roi_head.test_cfg.mask_thr_binary);logit_threshold=math.log(threshold/(1-threshold));dense_alpha=float(head._effective_shape_dense_alpha().detach().cpu());chunk_size=int(getattr(model.roi_head,'mask_roi_chunk_size',32) or 0)
 manifest=dict(checkpoint=expected['checkpoint'],weights='ema',strict=True,load_report=report,training=False,optimizer_created=False,variants=NAMES,protocol='GT common-box read-only observation; no AP/RPN/bbox rerun',git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=base.ROOT,text=True).strip(),git_dirty=bool(status),git_status_porcelain=status,script_sha256=base.file_sha(__file__),helper_script_sha256=base.file_sha(base.__file__),manifest_sha256=ds.manifest_sha256,donor_maps_sha256=base.file_sha(base.O/'donor_maps.json'),preregistration_sha256=base.file_sha(O/'PREREGISTRATION.md'),architecture_contract=snap['architecture_contract'],mask_thr_binary=threshold,mask_logit_threshold=logit_threshold,coarse_foreground_threshold=.5,dense_shape_alpha=dense_alpha,dense_detach_input=bool(head.dense_prompt_cfg['detach_input']),coarse_loss_weight=float(head.coarse_mask_loss_module.weight),shape_fusion_type=head.shape_injector.fusion_type,explicit_prompt_mode=head.explicit_prompt_mode,mask_roi_chunk_size=chunk_size,image_ids=[],complete=False)
 base.dump(out/'run_manifest.json',manifest);records={n:[] for n in NAMES};checks=[]
 with torch.inference_mode():
  for index,row in enumerate(ds.samples):
   if args.max_images and index>=args.max_images:break
   iid=row['index'];item=ds[index];before=base.payload_hash(item['uav_payload']);rgbsha=base.tensor_sha(item['uav_payload']['rgb']);batch=base.ue_collate([item]);imgs,samples=base._build_data_samples(batch,torch.device('cuda'));processed=model.data_preprocessor(dict(inputs=imgs,data_samples=samples),training=False);imgs=processed['inputs'];sample=processed['data_samples'][0]
   captures=[]
   def capture(module,inputs,output):captures.append(output)
   hook=model.uav_bev.register_forward_hook(capture);original_rgb=sample.uav.rgb
   try:
    features=model._extract_dual_feat(imgs,[sample]);rgb=[]
    for rec in donors[iid]['rgb_files']:
     assert base.file_sha(rec['path'])==rec['sha256'];image=cv2.imread(rec['path']);assert image is not None and image.shape==(512,512,3);rgb.append(image[...,::-1].transpose(2,0,1))
    sample.uav.rgb=torch.from_numpy(np.stack(rgb).copy()).float().cuda();unused=model._extract_dual_feat(imgs,[sample]);del unused
   finally:sample.uav.rgb=original_rgb;hook.remove()
   assert len(captures)==2 and base.payload_hash(sample.uav)==before and base.tensor_sha(sample.uav.rgb)==rgbsha
   x,embedding,pe,high=features;p2=x[0];uc,q,c=model.bev_projection(captures[0],[sample.uav],p2.shape[-2:]);ud,qd,cd=model.bev_projection(captures[1],[sample.uav],p2.shape[-2:]);assert torch.equal(q,qd) and torch.equal(c,cd)
   nc,_=fuser(p2,uc,q,c);nd,_=fuser(p2,ud,q,c);rc=nc-p2;rd=nd-p2;modified={'baseline':p2,'native_correct_q':p2+.25*rc,'native_donor_q':p2+.25*rd,'native_correct_one':p2+rc};packs={n:((modified[n],*x[1:]),embedding,pe,high) for n in NAMES}
   tensors=list({id(t):t for t in list(x)+[embedding,uc,ud,q,c,rc,rd]+captures+list(modified.values())+([pe] if torch.is_tensor(pe) else [])+list(high or ())}.values());hashes=[base.tensor_sha(t) for t in tensors]
   boxes=np.array(prior['baseline'][iid]['mask_iou']);n=len(boxes);rows=identities[iid];assert len(rows)==n and [z['gt_slot'] for z in rows]==list(range(n));levels=np.array(prior['baseline'][iid]['roi_levels']);notp2=levels!=0
   # Recover identical GT boxes from the authoritative historical mask audit.
   gtboxes=np.array(gt_box_by_image[iid],dtype=np.float32);boxsha=hashlib.sha256(gtboxes.tobytes()).hexdigest();assert boxsha==prior['baseline'][iid]['boxes_sha256'];box_tensor=torch.as_tensor(gtboxes,device='cuda');roi=torch.cat([box_tensor.new_zeros(n,1),box_tensor],dim=1);assert np.array_equal(model.roi_head.mask_roi_extractor.map_roi_levels(roi,model.roi_head.mask_roi_extractor.num_inputs).cpu().numpy(),levels)
   target=head.get_coarse_targets([SimpleNamespace(pos_priors=box_tensor,pos_assigned_gt_inds=torch.arange(n,device='cuda'))],[sample.gt_instances],mask_size=(64,64)).cpu()
   support=head._dense_embedding_box_support(box_tensor,(32,32),torch.float32).sum((1,2,3)).cpu();base_trace=None;base_masks=None;chunk_sizes=None
   for name in NAMES:
    with Observer(head) as observer:masks=base.predict_masks(model,packs[name],sample,gtboxes)
    trace=observer.merged();sizes=[len(z['roi']) for z in observer.entries];assert len(sizes)==math.ceil(n/(chunk_size or n));assert sum(sizes)==n
    if chunk_sizes is None:chunk_sizes=sizes
    else:assert sizes==chunk_sizes
    assert torch.equal(trace['boxes'],torch.from_numpy(gtboxes))
    for k in STAGES:assert len(trace[k])==n and torch.isfinite(trace[k]).all(),k
    assert trace['coarse'].shape[1:]==(1,64,64) and trace['canvas'].shape[1:]==(1,128,128) and trace['dense'].shape[1:]==(256,32,32)
    truth=np.asarray(item['gt_masks'],dtype=bool);iou=(masks&truth).sum((1,2))/np.maximum((masks|truth).sum((1,2)),1);assert np.array_equal(iou,np.array(prior[name][iid]['mask_iou'])),('Historical endpoint mismatch',name,iid)
    if args.max_images:
     off=base.predict_masks(model,packs[name],sample,gtboxes);assert np.array_equal(masks,off),('Observer perturbed masks',name,iid)
    if name=='baseline':base_trace=trace;base_masks=masks.copy()
    else:
     for key in ['roi','coarse']:assert torch.equal(trace[key][notp2],base_trace[key][notp2])
     assert np.array_equal(masks[notp2],base_masks[notp2])
    qstats=quality(trace,target);stats={key:delta_stats(trace[key],base_trace[key]) for key in STAGES};bp=base_trace['coarse'].sigmoid();vp=trace['coarse'].sigmoid();coarse_prob_mae=(vp-bp).abs().flatten(1).mean(1);coarse_flip=(vp>=.5).ne(bp>=.5).float().flatten(1).mean(1)
    valid=trace['labels']>=0;bv=base_trace['labels']>=0;changed=trace['labels'].ne(base_trace['labels']) | (trace['coords'].ne(base_trace['coords']).any(2)&(valid|bv));changed&=(valid|bv);both=valid&bv;move=(trace['coords']-base_trace['coords']).float().norm(dim=2)*both
    pre_flip=(trace['sam_raw']>=logit_threshold).ne(base_trace['sam_raw']>=logit_threshold).float().flatten(1).mean(1);post_flip=(trace['tail_final']>=logit_threshold).ne(base_trace['tail_final']>=logit_threshold).float().flatten(1).mean(1)
    finalxor=np.logical_xor(masks,base_masks).sum((1,2));pd=(trace['pe_dense']-base_trace['pe_dense']).float().flatten(1).norm(dim=1);fd=(trace['dense']-base_trace['dense']).float().flatten(1).norm(dim=1);attenuation=torch.where(pd>1e-12,fd/pd,torch.full_like(pd,float('nan')))
    no_mask=head.no_mask_embed.detach().cpu().reshape(1,256,1,1);dense_no_base=(trace['dense']-no_mask).float().flatten(1).norm(dim=1);zero_support=support==0;assert bool(dense_no_base[zero_support].eq(0).all())
    for j,identity in enumerate(rows):
     r=dict(**identity,mode=name,gt_iou=float(iou[j]),coarse_iou=qstats['coarse_iou'][j],coarse_bce=qstats['coarse_bce'][j],dense_support_cells=int(support[j]),dense_no_mask_delta_norm=float(dense_no_base[j]),sparse_token_count=qstats['sparse_token_count'][j],sparse_token_count_changed=bool(trace['sparse_token_count'][j]!=base_trace['sparse_token_count'][j]),point_valid=qstats['point_valid'][j],point_correct=qstats['point_correct'][j],point_labels=qstats['point_labels'][j],point_coords=qstats['point_coords'][j],point_local_yx=qstats['point_local_yx'][j],point_changed_count=int(changed[j].sum()),point_both_valid_count=int(both[j].sum()),point_move_distance_sum=float(move[j].sum()),coarse_prob_mae=float(coarse_prob_mae[j]),coarse_binary_flip_fraction=float(coarse_flip[j]),sam_raw_binary_flip_fraction=float(pre_flip[j]),tail_final_binary_flip_fraction=float(post_flip[j]),paste_xor_pixels=int(finalxor[j]),dense_final_to_pe_delta_ratio=float(attenuation[j]) if torch.isfinite(attenuation[j]) else None,stages={key:{field:vals[j] for field,vals in stats[key].items()} for key in STAGES})
     records[name].append(r)
    print(iid,name,'GTIoU',float(iou.mean()),'point_changes',int(changed.sum()),flush=True)
    if name!='baseline':del trace
   assert hashes==[base.tensor_sha(t) for t in tensors] and base.payload_hash(sample.uav)==before and base.tensor_sha(sample.uav.rgb)==rgbsha
   checks.append(dict(image_id=iid,endpoint_iou_exact=True,nonp2_roi_coarse_paste_exact=True,observer_smoke_pixel_exact=bool(args.max_images),geometry_rgb_restored=True,features_unchanged=True,bev_hook_calls=len(captures),chunk_sizes=chunk_sizes,observer_hooks_removed=True));manifest['image_ids'].append(iid)
   del base_trace,base_masks,observer,packs,modified,features,x,embedding,pe,high,p2,uc,ud,q,qd,c,cd,nc,nd,rc,rd,captures,tensors,hashes,masks,target,support,processed,sample,samples,imgs,batch,item,original_rgb
   torch.cuda.empty_cache()
 assert versions=={k:t._version for k,t in model.named_parameters()};manifest.update(complete=len(checks)==153,processed_images=len(checks),ground_truth_instances=len(records['baseline']),parameter_versions_unchanged=True)
 for n in NAMES:base.dump(out/(n+'.json'),records[n])
 base.dump(out/'run_manifest.json',manifest);base.dump(out/'invariance_checks.json',checks);print('DONE P2 PROMPT TRACE',flush=True)

if __name__=='__main__':
 gt_box_by_image={}
 for row in base.read(R/'full_fixed_box_audit/dual_correct.json')['records']:
  if row['box_type']=='gt':gt_box_by_image[row['image_id']]=row['boxes_xyxy']
 main()
