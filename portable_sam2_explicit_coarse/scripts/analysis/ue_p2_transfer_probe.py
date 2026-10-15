"""Frozen P3-fuser transfer to P2; fixed detection boxes isolate mask consumption."""
import argparse
import copy
import hashlib
import subprocess
import sys
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask as mu
sys.path.insert(0,'/tmp')
import ue_rgb_content_probe as base
R=base.R;O=R/'p2_transfer_audit'
NAMES=['baseline','native_correct_q','native_donor_q','upsample_correct_q','native_correct_one']

def band_iou(pred,truth):
 kernel=np.ones((7,7),np.uint8);outer=cv2.dilate(truth.astype(np.uint8),kernel,borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool);inner=cv2.erode(truth.astype(np.uint8),kernel,borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool);band=outer&~inner
 return float((pred&truth&band).sum()/max(((pred|truth)&band).sum(),1))

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--max-images',type=int,default=0);args=parser.parse_args();torch.set_num_threads(2);torch.manual_seed(42);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 out=O/('smoke' if args.max_images else 'full');out.mkdir(exist_ok=True);expected=base.read(base.O/'correct_replay/run_manifest.json');maps=base.read(base.O/'donor_maps.json');donors={v['target']:v for v in maps['maps']['donor_a']['records']}
 checkpoint=torch.load(expected['checkpoint'],map_location='cpu',weights_only=False);snap=base._snapshot(checkpoint);ra=argparse.Namespace(sam2_ckpt=None,sam2_repo=None);base._resolve_sam2_repo(ra,snap);cfg,_=base._resolve_model_config(ra,snap);model=base._register_and_build(cfg);report=base._load_model_state(model,base._select_state_dict(checkpoint,'ema'),allow_nonstrict=False);model.roi_head.mask_head.assert_no_mask_embedding_contract('Frozen P2 transfer');model.cuda().eval();model.uav_intervention='correct';del checkpoint
 assert model.fpn_levels==(1,2) and len(model.fpn_fusion)==2;fuser=model.fpn_fusion[0]
 state=snap['data_config'];ds=base.UERGBDInstanceDataset(state['manifest'],'validation',state['geometry'],load_uav=True,cache_root='/tmp/uav_dual_dev_data/cache');assert ds.manifest_sha256==expected['manifest_sha256']==maps['manifest_sha256'];assert [v['index'] for v in ds.samples]==expected['image_ids']
 old=defaultdict(list);gt=defaultdict(list);common={};stage=defaultdict(list)
 for v in base.read(base.O/'correct_replay/dt_records.json'):old[v['image_id']].append(v)
 for v in base.read(R/'dual_correct/gt_records.json'):gt[v['image_id']].append(v)
 for v in base.read(R/'full_stage_audit/dual_correct.json')['instances']:stage[v['image_id']].append(v)
 for v in base.read(R/'full_fixed_box_audit/dual_correct.json')['records']:
  if v['box_type']=='gt':common[v['image_id']]=v
 allgt=[];metas=[];alldt={n:[] for n in NAMES};exported={n:[] for n in NAMES};fixed={n:[] for n in NAMES};checks=[];identities=[]
 versions={k:t._version for k,t in model.named_parameters()};status=subprocess.check_output(['git','status','--porcelain'],cwd=base.ROOT,text=True);diff=subprocess.check_output(['git','diff','HEAD'],cwd=base.ROOT)
 manifest=dict(checkpoint=expected['checkpoint'],weights='ema',strict=True,load_report=report,training=False,optimizer_created=False,variants=NAMES,protocol='Fixed original detection boxes/scores; conditional segmentation AP, not full modified detector inference',primary='native_correct_q minus baseline',p3_alpha=float(fuser.alpha.detach().cpu()),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=base.ROOT,text=True).strip(),git_dirty=bool(status),git_status_porcelain=status,git_diff_sha256=hashlib.sha256(diff).hexdigest(),script_sha256=base.file_sha(__file__),helper_script_sha256=base.file_sha(base.__file__),manifest_sha256=ds.manifest_sha256,donor_maps_sha256=base.file_sha(base.O/'donor_maps.json'),preregistration_sha256=base.file_sha(O/'PREREGISTRATION.md'),architecture_contract=snap['architecture_contract'],image_ids=[],complete=False)
 base.dump(out/'run_manifest.json',manifest)
 with torch.inference_mode():
  for index,row in enumerate(ds.samples):
   if args.max_images and index>=args.max_images:break
   iid=row['index'];item=ds[index];before=base.payload_hash(item['uav_payload']);rgbsha=base.tensor_sha(item['uav_payload']['rgb']);batch=base.ue_collate([item]);imgs,samples=base._build_data_samples(batch,torch.device('cuda'));processed=model.data_preprocessor(dict(inputs=imgs,data_samples=samples),training=False);imgs=processed['inputs'];sample=processed['data_samples'][0]
   assert before==base.payload_hash(sample.uav) and rgbsha==base.tensor_sha(sample.uav.rgb)
   captures_bev=[];captures_p3=[]
   def capture_bev(module,inputs,output):captures_bev.append(output)
   def capture_p3(module,inputs,output):captures_p3.append((inputs[0],output[0]))
   hb=model.uav_bev.register_forward_hook(capture_bev);hf=fuser.register_forward_hook(capture_p3);original_rgb=sample.uav.rgb
   try:
    features=model._extract_dual_feat(imgs,[sample]);assert len(captures_bev)==len(captures_p3)==1
    rgb=[]
    for rec in donors[iid]['rgb_files']:
     assert base.file_sha(rec['path'])==rec['sha256'];raw=cv2.imread(rec['path']);assert raw is not None and raw.shape==(512,512,3);rgb.append(raw[...,::-1].transpose(2,0,1))
    sample.uav.rgb=torch.from_numpy(np.stack(rgb).copy()).float().cuda();assert before==base.payload_hash(sample.uav) and base.tensor_sha(sample.uav.rgb)!=rgbsha
    donor_features=model._extract_dual_feat(imgs,[sample]);del donor_features
   finally:
    sample.uav.rgb=original_rgb;hb.remove();hf.remove()
   assert len(captures_bev)==len(captures_p3)==2 and before==base.payload_hash(sample.uav) and rgbsha==base.tensor_sha(sample.uav.rgb)
   x,embedding,pe,high=features;p2=x[0];assert p2.shape==(1,512,128,128) and x[1].shape==(1,512,64,64)
   uc,q,c=model.bev_projection(captures_bev[0],[sample.uav],p2.shape[-2:]);ud,qd,cd=model.bev_projection(captures_bev[1],[sample.uav],p2.shape[-2:]);assert torch.equal(q,qd) and torch.equal(c,cd)
   native,_=fuser(p2,uc,q,c);donor,_=fuser(p2,ud,q,c);rc=native-p2;rd=donor-p2;r3=captures_p3[0][1]-captures_p3[0][0];ru=F.interpolate(r3,size=p2.shape[-2:],mode='bilinear',align_corners=False)*(c>0)
   valid=(c>0).to(p2.dtype);assert torch.equal(rc*(1-valid),torch.zeros_like(rc)) and torch.equal(rd*(1-valid),torch.zeros_like(rd))
   delta={'baseline':torch.zeros_like(rc),'native_correct_q':.25*rc,'native_donor_q':.25*rd,'upsample_correct_q':.25*ru,'native_correct_one':rc};p2s={n:p2+d for n,d in delta.items()};p2s['baseline']=p2
   packs={n:((p2s[n],*x[1:]),embedding,pe,high) for n in NAMES}
   for f in packs.values():
    assert f[1] is embedding and f[2] is pe and f[3] is high
    for a,b in zip(f[0][1:],x[1:]):assert a is b
   tensors=list({id(t):t for t in list(x)+[embedding,uc,ud,q,c,rc,rd,ru]+captures_bev+list(p2s.values())+([pe] if torch.is_tensor(pe) else [])+list(high or ())}.values());hashes=[base.tensor_sha(t) for t in tensors]
   proposals=model.rpn_head.predict(x,[sample],rescale=False);det=model.roi_head.predict_bbox(model.roi_head._inject_extra_pe(x),[sample.metainfo],proposals,rcnn_test_cfg=model.roi_head.test_cfg,rescale=False)[0];assert np.array_equal(det.bboxes.cpu().numpy(),base.xyxy(old[iid]));assert np.array_equal(det.scores.cpu().numpy(),np.array([v['score'] for v in old[iid]]))
   target=base._extract_instances_numpy(sample.gt_instances,(512,512));truth=np.stack([mu.decode(z['segmentation']).astype(bool) for z in gt[iid]]);assert np.array_equal(target['masks'].astype(bool),truth);allgt.append(base.arrays_small(target));meta=batch['img_metas'][0];metas.append(meta);boxes=np.array(common[iid]['boxes_xyxy'],dtype=np.float32);assert hashlib.sha256(boxes.tobytes()).hexdigest()==common[iid]['boxes_sha256'];rois=torch.cat([torch.zeros(len(boxes),1,device='cuda'),torch.as_tensor(boxes,device='cuda')],dim=1);levels=model.roi_head.mask_roi_extractor.map_roi_levels(rois,model.roi_head.mask_roi_extractor.num_inputs).cpu().numpy();notp2=levels!=0
   original_masks=None
   for n in NAMES:
    f=packs[n];fx=f[0];result=model.roi_head.predict_mask(model.roi_head._inject_extra_pe(fx),[sample.metainfo],[copy.deepcopy(det)],rescale=False,image_embeddings=embedding,image_positional_embeddings=pe,high_res_features=high)[0];pred=base._extract_instances_numpy(result,(512,512));records=[{k:v for k,v in z.items() if k!='file_name'} for z in base._prediction_records(pred,meta)];assert [(z['bbox'],z['score']) for z in records]==[(z['bbox'],z['score']) for z in old[iid]]
    masks=base.predict_masks(model,f,sample,boxes);iou=(masks&truth).sum(axis=(1,2))/np.maximum((masks|truth).sum(axis=(1,2)),1)
    if n=='baseline':
     assert records==old[iid] and np.array_equal(iou,np.array(common[iid]['mask_iou']));original_masks=masks.copy()
    else:assert np.array_equal(masks[notp2],original_masks[notp2]),('Non-P2 GT mask changed',n,iid)
    alldt[n].append(base.arrays_small(pred));exported[n].extend(records);fixed[n].append(dict(image_id=iid,boxes_sha256=common[iid]['boxes_sha256'],roi_levels=levels.tolist(),mask_iou=iou.tolist(),boundary_band_iou=[band_iou(p,t) for p,t in zip(masks,truth)]))
   assert hashes==[base.tensor_sha(t) for t in tensors] and before==base.payload_hash(sample.uav) and rgbsha==base.tensor_sha(sample.uav.rgb)
   for obj,level in zip(sorted(stage[iid],key=lambda z:z['gt_slot']),levels):identities.append(dict(image_id=iid,gt_slot=obj['gt_slot'],global_id=obj['global_id'],bbox_area=obj['bbox_area'],roi_level=int(level)))
   supported_count=valid.sum()*p2.shape[1];p2_supported_norm=(p2*valid).float().norm().clamp_min(1e-6)
   checks.append(dict(image_id=iid,donor=donors[iid]['donor'],bev_hook_calls=len(captures_bev),p3_hook_calls=len(captures_p3),hooks_removed=True,geometry_rgb_restored=True,feature_hashes_unchanged=True,only_p2_changed=True,bbox_scores_exact=True,baseline_exact=True,nonp2_gt_masks_exact=True,coverage_fraction=float(valid.float().mean()),p2_shape=list(p2.shape),p3_shape=list(x[1].shape),bev_shape=list(captures_bev[0].shape),residual_norm_ratio={n:float(d.float().norm()/p2.float().norm().clamp_min(1e-6)) for n,d in delta.items()},supported_rms_ratio={n:float(d.float().norm()/p2_supported_norm) if supported_count>0 else 0. for n,d in delta.items()}));manifest['image_ids'].append(iid)
   print(len(checks),iid,'GTIoU',*[round(float(np.mean(fixed[n][-1]['mask_iou'])),6) for n in NAMES],flush=True)
   del captures_bev,captures_p3,features,x,embedding,pe,high,p2,uc,ud,q,qd,c,cd,native,donor,rc,rd,r3,ru,valid,delta,p2s,packs,tensors,hashes,proposals,det,target,truth,rois,original_masks,masks,result,pred,f,fx,processed,sample,samples,imgs,batch,item,original_rgb
   torch.cuda.empty_cache()
 assert versions=={k:t._version for k,t in model.named_parameters()};metrics={}
 for n in NAMES:
  metrics[n]={}
  for kind in ['segm','bbox']:
   cg,cd=base.build_coco_gt_and_dt(allgt,alldt[n],metas,score_key='scores');metrics[n].update(base.run_coco_eval(cg,cd,kind))
  if n=='baseline' and not args.max_images:
   ref=base.read(base.O/'correct_replay/metrics.json')
   for k,v in metrics[n].items():assert abs(v-ref[k])<1e-12
  base.dump(out/(n+'_dt_records.json'),exported[n]);base.dump(out/(n+'_fixed_box.json'),fixed[n])
 for n in NAMES:
  for k,v in metrics[n].items():
   if k.startswith('bbox/'):assert v==metrics['baseline'][k]
 manifest.update(complete=len(checks)==153,processed_images=len(checks),ground_truth_instances=sum(len(v['labels']) for v in allgt),parameter_versions_unchanged=True)
 base.dump(out/'run_manifest.json',manifest);base.dump(out/'metrics.json',metrics);base.dump(out/'invariance_checks.json',checks);base.dump(out/'gt_identities.json',identities)
 print('DONE P2 TRANSFER',flush=True)
if __name__=='__main__':main()
