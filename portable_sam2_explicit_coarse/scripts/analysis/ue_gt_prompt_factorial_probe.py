"""Frozen actual-decoder prompt exchange, four fixed cells. No training/AP."""
import argparse, hashlib, math, subprocess, sys, time
from collections import defaultdict
from contextlib import nullcontext
from types import SimpleNamespace
import numpy as np
import torch
sys.path.insert(0,'/tmp')
import ue_rgb_content_probe as base
import ue_gt_prompt_oracle_probe as oracle
R=base.R;O=R/'gt_prompt_factorial'
NAMES=['baseline','gt_point_only','gt_dense_only','joint_gt_prompt']
SOURCES={'baseline':('P','P'),'gt_point_only':('G','P'),'gt_dense_only':('P','G'),'joint_gt_prompt':('G','G')}

def visual_hash(args,kwargs):
 def walk(x):
  if torch.is_tensor(x):return base.tensor_sha(x)
  if isinstance(x,(list,tuple)):return [walk(y) for y in x]
  if isinstance(x,dict):return {k:walk(v) for k,v in x.items()}
  return x
 return walk([args[:2],args[4:],kwargs])

class Capture:
 def __init__(self,head):self.head=head;self.entries=[];self.current=None
 def start(self,module,args,kwargs):assert self.current is None;self.current={'boxes':kwargs['boxes'].detach().cpu().clone()}
 def points(self,module,args,output):
  yx,labels=self.head.shape_point_miner.get_last_local_points();assert torch.equal(labels,output[1]);self.current.update(coords=output[0].detach().cpu().clone(),labels=labels.detach().cpu().clone(),yx=yx.detach().cpu().clone())
 def decoder(self,module,args,kwargs):
  assert self.current is not None and args[2].shape[1] in (2,6)
  n=len(self.current['boxes']);assert args[2].shape==(n,args[2].shape[1],256) and args[3].shape==(n,256,32,32)
  assert torch.isfinite(args[2]).all() and torch.isfinite(args[3]).all()
  self.current.update(sparse=args[2].detach().cpu().clone(),dense=args[3].detach().cpu().clone(),visual_hash=visual_hash(args,kwargs));self.entries.append(self.current);self.current=None
 def __enter__(self):
  self.handles=[self.head.register_forward_pre_hook(self.start,with_kwargs=True),self.head.shape_point_miner.register_forward_hook(self.points),self.head.mask_decoder.register_forward_pre_hook(self.decoder,with_kwargs=True)];return self
 def __exit__(self,*unused):
  for h in self.handles:h.remove()
  if unused[0] is None:assert self.current is None and self.entries

class Exchange:
 def __init__(self,head,prompt_sources,point_source,dense_source,upstream_source):self.head=head;self.sources=prompt_sources;self.ps=point_source;self.ds=dense_source;self.us=upstream_source;self.offset=0;self.entries=[]
 def replace(self,module,args,kwargs):
  k=self.offset;self.offset+=1;p=self.sources[self.ps][k];d=self.sources[self.ds][k];u=self.sources[self.us][k]
  assert visual_hash(args,kwargs)==u['visual_hash']==p['visual_hash']==d['visual_hash']
  assert torch.equal(args[2].detach().cpu(),u['sparse']) and torch.equal(args[3].detach().cpu(),u['dense'])
  sparse=p['sparse'].to(args[2]);dense=d['dense'].to(args[3]);assert sparse.shape[1] in (2,6) and dense.shape==args[3].shape
  assert torch.equal(sparse.detach().cpu(),p['sparse']) and torch.equal(dense.detach().cpu(),d['dense'])
  self.entries.append(dict(sparse_sha256=[base.tensor_sha(z) for z in sparse],dense_sha256=[base.tensor_sha(z) for z in dense],token_count=sparse.shape[1],actual_source_values_exact=True,visual_args_unchanged=True))
  return (*args[:2],sparse,dense,*args[4:]),kwargs
 def __enter__(self):self.handle=self.head.mask_decoder.register_forward_pre_hook(self.replace,with_kwargs=True);return self
 def __exit__(self,*unused):
  self.handle.remove()
  if unused[0] is None:assert self.offset==len(self.sources[self.ps])==len(self.sources[self.ds])

def row_prompt_metadata(entries,target):
 records=[];offset=0
 for e in entries:
  n=len(e['sparse']);truth=target[offset:offset+n]>=.5;yx=e['yx'].long();labels=e['labels'];sample=truth[torch.arange(n)[:,None],yx[...,0].clamp(0,63),yx[...,1].clamp(0,63)];valid=labels>=0;correct=torch.where(labels==1,sample,~sample)&valid
  for j in range(n):records.append(dict(point_valid=valid[j].tolist(),point_correct=correct[j].tolist(),point_labels=labels[j].tolist(),point_coords=e['coords'][j].tolist(),point_local_yx=yx[j].tolist(),sparse_token_count=int(e['sparse'].shape[1]),sparse_sha256=base.tensor_sha(e['sparse'][j]),dense_sha256=base.tensor_sha(e['dense'][j])))
  offset+=n
 assert offset==len(target);return records

def source_hashes(sources):return {k:[(base.tensor_sha(e['sparse']),base.tensor_sha(e['dense'])) for e in v] for k,v in sources.items()}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--max-images',type=int,default=0);args=ap.parse_args();started=time.monotonic();torch.set_num_threads(2);torch.manual_seed(42);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 out=O/('smoke' if args.max_images else 'full');out.mkdir(exist_ok=True);expected=base.read(base.O/'correct_replay/run_manifest.json');cp=torch.load(expected['checkpoint'],map_location='cpu',weights_only=False);snap=base._snapshot(cp);ra=argparse.Namespace(sam2_ckpt=None,sam2_repo=None);base._resolve_sam2_repo(ra,snap);cfg,_=base._resolve_model_config(ra,snap);model=base._register_and_build(cfg);report=base._load_model_state(model,base._select_state_dict(cp,'ema'),allow_nonstrict=False);model.cuda().eval();model.uav_intervention='correct';del cp
 head=model.roi_head.mask_head;head.assert_no_mask_embedding_contract('GT prompt factorial');assert head.explicit_prompt_mode=='points_box_dense' and head.shape_injector.fusion_type=='roi_only'
 state=snap['data_config'];ds=base.UERGBDInstanceDataset(state['manifest'],'validation',state['geometry'],load_uav=True,cache_root='/tmp/uav_dual_dev_data/cache');assert ds.manifest_sha256==expected['manifest_sha256'];assert [x['index'] for x in ds.samples]==expected['image_ids']
 prior={k:defaultdict(list) for k in ('P','G')}
 for k,name in [('P','baseline'),('G','joint_gt_prompt')]:
  for x in base.read(R/'gt_prompt_oracle/full'/(name+'.json')):prior[k][x['image_id']].append(x)
 gtboxes_by_image={x['image_id']:x for x in base.read(R/'full_fixed_box_audit/dual_correct.json')['records'] if x['box_type']=='gt'};versions={k:t._version for k,t in model.named_parameters()};status=subprocess.check_output(['git','status','--porcelain'],cwd=base.ROOT,text=True);threshold=float(model.roi_head.test_cfg.mask_thr_binary);chunk_size=int(getattr(model.roi_head,'mask_roi_chunk_size',32) or 0)
 manifest=dict(checkpoint=expected['checkpoint'],checkpoint_sha256=oracle.streaming_sha(expected['checkpoint']),weights='ema',strict=True,load_report=report,training=False,optimizer_created=False,variants=NAMES,source_map={k:list(v) for k,v in SOURCES.items()},protocol='Actual post-neutralization sparse and post-gate dense decoder input exchange; GT common boxes; no RPN/AP',git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=base.ROOT,text=True).strip(),git_dirty=bool(status),git_status_porcelain=status,script_sha256=base.file_sha(__file__),helper_script_sha256=base.file_sha(base.__file__),oracle_helper_sha256=base.file_sha(oracle.__file__),trace_helper_sha256=base.file_sha(oracle.tracebase.__file__),manifest_sha256=ds.manifest_sha256,preregistration_sha256=base.file_sha(O/'PREREGISTRATION.md'),architecture_contract=snap['architecture_contract'],mask_thr_binary=threshold,mask_logit_threshold=math.log(threshold/(1-threshold)),oracle_logit_amplitude=4,actual_prompt_contract=dict(sparse_last_dimension=256,sparse_tokens=[2,6],dense_shape=[256,32,32]),dense_shape_alpha=float(head._effective_shape_dense_alpha().detach().cpu()),dense_detach_input=bool(head.dense_prompt_cfg['detach_input']),mask_roi_chunk_size=chunk_size,image_ids=[],complete=False)
 base.dump(out/'run_manifest.json',manifest);records={k:[] for k in NAMES};checks=[]
 with torch.inference_mode():
  for index,row in enumerate(ds.samples):
   if args.max_images and index>=args.max_images:break
   iid=row['index'];item=ds[index];before=base.payload_hash(item['uav_payload']);rgbsha=base.tensor_sha(item['uav_payload']['rgb']);batch=base.ue_collate([item]);imgs,samples=base._build_data_samples(batch,torch.device('cuda'));processed=model.data_preprocessor(dict(inputs=imgs,data_samples=samples),training=False);imgs=processed['inputs'];sample=processed['data_samples'][0];features=model._extract_dual_feat(imgs,[sample]);x,embedding,pe,high=features;tensors=list(x)+[embedding]+([pe] if torch.is_tensor(pe) else [])+list(high or ());hashes=[base.tensor_sha(t) for t in tensors]
   old=prior['P'][iid];n=len(old);gtboxes=np.array(gtboxes_by_image[iid]['boxes_xyxy'],dtype=np.float32);assert hashlib.sha256(gtboxes.tobytes()).hexdigest()==gtboxes_by_image[iid]['boxes_sha256'];assert n==len(gtboxes) and [z['gt_slot'] for z in old]==list(range(n));box_tensor=torch.as_tensor(gtboxes,device='cuda');target=head.get_coarse_targets([SimpleNamespace(pos_priors=box_tensor,pos_assigned_gt_inds=torch.arange(n,device='cuda'))],[sample.gt_instances],mask_size=(64,64)).cpu();assert bool(((target==0)|(target==1)).all());support=head._dense_embedding_box_support(box_tensor,(32,32),torch.float32).sum((1,2,3)).cpu().numpy();hook_counts=[len(m._forward_hooks)+len(m._forward_pre_hooks) for m in head.modules()]
   sources={};masks_by_name={};actual_hashes_by_name={};chunks=None
   for key,name in [('P','baseline'),('G','joint_gt_prompt')]:
    with (oracle.ShapeReplacement(head,target,'joint_gt_prompt') if key=='G' else nullcontext(None)) as replacement:
     with Capture(head) as capture:masks=base.predict_masks(model,features,sample,gtboxes)
    sources[key]=capture.entries;masks_by_name[name]=masks;sz=[len(e['sparse']) for e in capture.entries]
    if chunks is None:chunks=sz
    else:assert sz==chunks
    assert torch.equal(torch.cat([e['boxes'] for e in capture.entries]),torch.from_numpy(gtboxes));assert sz==[min(chunk_size,n-a) for a in range(0,n,chunk_size)]
    truth=np.asarray(item['gt_masks'],dtype=bool);iou=(masks&truth).sum((1,2))/np.maximum((masks|truth).sum((1,2)),1);assert np.array_equal(iou,[z['gt_iou'] for z in prior[key][iid]]),('Diagonal endpoint differs',iid,name)
   gt_target_ranges=replacement.ranges
   source_before=source_hashes(sources)
   for p,g in zip(sources['P'],sources['G']):
    assert p['visual_hash']==g['visual_hash'] and torch.equal(p['sparse'][:,-2:],g['sparse'][:,-2:]),('Visual/box tokens differ',iid)
   metadata={k:row_prompt_metadata(v,target) for k,v in sources.items()}
   for key,name in [('P','baseline'),('G','joint_gt_prompt')]:actual_hashes_by_name[name]=([r['sparse_sha256'] for r in metadata[key]],[r['dense_sha256'] for r in metadata[key]])
   for key in ('P','G'):
    for live,oldrow in zip(metadata[key],prior[key][iid]):
     for field in ('point_valid','point_correct','point_labels','point_coords','point_local_yx','sparse_token_count'):assert live[field]==oldrow[field],('Source metadata differs',key,field,iid)
   for name in ('gt_point_only','gt_dense_only'):
    ps,dsrc=SOURCES[name]
    with Exchange(head,sources,ps,dsrc,'P') as exchange:masks=base.predict_masks(model,features,sample,gtboxes)
    masks_by_name[name]=masks;actual=[e for e in exchange.entries];flat_sparse=[h for e in actual for h in e['sparse_sha256']];flat_dense=[h for e in actual for h in e['dense_sha256']]
    assert flat_sparse==[r['sparse_sha256'] for r in metadata[ps]] and flat_dense==[r['dense_sha256'] for r in metadata[dsrc]]
    actual_hashes_by_name[name]=(flat_sparse,flat_dense)
   assert np.array_equal(masks_by_name['gt_dense_only'][support==0],masks_by_name['baseline'][support==0]),('Support-zero dense-only negative control failed',iid)
   sham={}
   if args.max_images:
    for key,name in [('P','baseline'),('G','joint_gt_prompt')]:
     with (oracle.ShapeReplacement(head,target,'joint_gt_prompt') if key=='G' else nullcontext(None)):
      with Exchange(head,sources,key,key,key) as exchange:sham_masks=base.predict_masks(model,features,sample,gtboxes)
     assert np.array_equal(sham_masks,masks_by_name[name]),('Diagonal exchange sham changed masks',iid,name);sham[name]=True
   no_mask=head.no_mask_embed.detach().cpu().reshape(1,256,1,1);dense_norms={k:torch.cat([(e['dense']-no_mask).float().flatten(1).norm(dim=1) for e in v]).numpy() for k,v in sources.items()}
   for key in ('P','G'):assert bool((dense_norms[key][support==0]==0).all())
   for name in NAMES:
    ps,dsrc=SOURCES[name];masks=masks_by_name[name];iou=(masks&truth).sum((1,2))/np.maximum((masks|truth).sum((1,2)),1);xor=np.logical_xor(masks,masks_by_name['baseline']).sum((1,2))
    for j,oldrow in enumerate(old):
     ident={k:oldrow[k] for k in ('image_id','gt_slot','global_id','bbox_area','roi_level')};pm=metadata[ps][j];dm=metadata[dsrc][j]
     records[name].append(dict(**ident,mode=name,gt_iou=float(iou[j]),paste_xor_pixels=int(xor[j]),dense_support_cells=int(support[j]),actual_point_source=ps,actual_dense_source=dsrc,upstream_coarse_source=('G' if name=='joint_gt_prompt' else 'P'),sparse_token_count=pm['sparse_token_count'],actual_sparse_sha256=actual_hashes_by_name[name][0][j],source_sparse_sha256=pm['sparse_sha256'],actual_dense_sha256=actual_hashes_by_name[name][1][j],source_dense_sha256=dm['dense_sha256'],point_valid=pm['point_valid'],point_correct=pm['point_correct'],point_labels=pm['point_labels'],point_coords=pm['point_coords'],point_local_yx=pm['point_local_yx'],dense_no_mask_delta_norm=float(dense_norms[dsrc][j]),coarse_target_fg_pixels=int(target[j].sum())))
    print(iid,name,'GTIoU',float(iou.mean()),flush=True)
   assert source_before==source_hashes(sources) and hashes==[base.tensor_sha(t) for t in tensors] and base.payload_hash(sample.uav)==before and base.tensor_sha(sample.uav.rgb)==rgbsha;assert hook_counts==[len(m._forward_hooks)+len(m._forward_pre_hooks) for m in head.modules()]
   checks.append(dict(image_id=iid,diagonal_endpoints_exact=True,source_point_metadata_exact=True,source_prompt_hashes_unchanged=True,source_box_tokens_exact=True,actual_source_values_exact=True,actual_visual_args_unchanged=True,features_unchanged=True,geometry_rgb_unchanged=True,hooks_removed=True,dense_only_support0_paste_exact=True,support0_count=int((support==0).sum()),chunk_sizes=chunks,gt_source_target_ranges=gt_target_ranges,gt_slots_in_order=list(range(n)),diagonal_exchange_sham_pixel_exact=sham if args.max_images else None));manifest['image_ids'].append(iid)
   del sources,source_before,metadata,masks_by_name,masks,features,x,embedding,pe,high,tensors,hashes,target,support,processed,sample,samples,imgs,batch,item,capture,exchange,replacement
   torch.cuda.empty_cache()
 assert versions=={k:t._version for k,t in model.named_parameters()};manifest.update(complete=len(checks)==153,processed_images=len(checks),ground_truth_instances=len(records['baseline']),parameter_versions_unchanged=True,elapsed_seconds=time.monotonic()-started)
 for name in NAMES:base.dump(out/(name+'.json'),records[name])
 base.dump(out/'run_manifest.json',manifest);base.dump(out/'invariance_checks.json',checks);print('DONE GT PROMPT FACTORIAL',flush=True)
if __name__=='__main__':main()
