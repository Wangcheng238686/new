"""Freeze FPN; scale only the observed SAM fusion residual at the mask input."""
import argparse
import copy
import hashlib
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from pycocotools import mask as mu
sys.path.insert(0,'/tmp')
import ue_rgb_content_probe as base
R=base.R
O=R/'sam_strength_audit'
SCALES={'s0':0.0,'s05':0.5,'s1':1.0}

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--max-images',type=int,default=0);args=parser.parse_args()
 torch.set_num_threads(2);torch.manual_seed(42);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 out=O/('smoke' if args.max_images else 'full');out.mkdir(exist_ok=True)
 expected=base.read(base.O/'correct_replay/run_manifest.json');checkpoint=torch.load(expected['checkpoint'],map_location='cpu',weights_only=False);snap=base._snapshot(checkpoint);ra=argparse.Namespace(sam2_ckpt=None,sam2_repo=None);base._resolve_sam2_repo(ra,snap);cfg,_=base._resolve_model_config(ra,snap);model=base._register_and_build(cfg);report=base._load_model_state(model,base._select_state_dict(checkpoint,'ema'),allow_nonstrict=False)
 model.roi_head.mask_head.assert_no_mask_embedding_contract('Frozen SAM residual strength probe');model.cuda().eval();model.uav_intervention='correct';del checkpoint
 state=snap['data_config'];ds=base.UERGBDInstanceDataset(state['manifest'],'validation',state['geometry'],load_uav=True,cache_root='/tmp/uav_dual_dev_data/cache');assert ds.manifest_sha256==expected['manifest_sha256'];assert [x['index'] for x in ds.samples]==expected['image_ids']
 old=defaultdict(list);gt=defaultdict(list);common={}
 for v in base.read(base.O/'correct_replay/dt_records.json'):old[v['image_id']].append(v)
 for v in base.read(R/'dual_correct/gt_records.json'):gt[v['image_id']].append(v)
 for v in base.read(R/'full_fixed_box_audit/dual_correct.json')['records']:
  if v['box_type']=='gt':common[v['image_id']]=v
 allgt=[];metas=[];alldt={n:[] for n in SCALES};exported={n:[] for n in SCALES};fixed={n:[] for n in SCALES};checks=[]
 versions={k:t._version for k,t in model.named_parameters()};status=subprocess.check_output(['git','status','--porcelain'],cwd=base.ROOT,text=True);diff=subprocess.check_output(['git','diff','HEAD'],cwd=base.ROOT)
 manifest=dict(checkpoint=expected['checkpoint'],weights='ema',strict=True,load_report=report,training=False,optimizer_created=False,scales=SCALES,selection='s05 is preregistered primary comparison; s0 endpoint only',git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=base.ROOT,text=True).strip(),git_dirty=bool(status),git_status_porcelain=status,git_diff_sha256=hashlib.sha256(diff).hexdigest(),script_sha256=base.file_sha(__file__),helper_script_sha256=base.file_sha(base.__file__),manifest_sha256=ds.manifest_sha256,preregistration_sha256=base.file_sha(O/'PREREGISTRATION.md'),architecture_contract=snap['architecture_contract'],sam_alpha=float(model.sam_fusion.alpha.detach().cpu()),image_ids=[],complete=False)
 base.dump(out/'run_manifest.json',manifest)
 with torch.inference_mode():
  for index,row in enumerate(ds.samples):
   if args.max_images and index>=args.max_images:break
   iid=row['index'];item=ds[index];before=base.payload_hash(item['uav_payload']);rgbsha=base.tensor_sha(item['uav_payload']['rgb']);batch=base.ue_collate([item]);imgs,samples=base._build_data_samples(batch,torch.device('cuda'));processed=model.data_preprocessor(dict(inputs=imgs,data_samples=samples),training=False);imgs=processed['inputs'];sample=processed['data_samples'][0]
   assert base.payload_hash(sample.uav)==before and base.tensor_sha(sample.uav.rgb)==rgbsha
   captures=[]
   def capture(module,inputs,output):captures.append((inputs[0],output[0]))
   hook=model.sam_fusion.register_forward_hook(capture)
   try:features=model._extract_dual_feat(imgs,[sample])
   finally:hook.remove()
   assert len(captures)==1
   e0,e1=captures[0];assert features[1] is e1
   residual=e1-e0
   embeddings={'s0':e0,'s05':e0+.5*residual,'s1':e1}
   tensors=list({id(t):t for t in list(features[0])+[e0,e1,residual]+list(embeddings.values())+([features[2]] if torch.is_tensor(features[2]) else [])+list(features[3] or ())}.values());hashes=[base.tensor_sha(t) for t in tensors]
   x,embedding,pe,high=features;proposals=model.rpn_head.predict(x,[sample],rescale=False);det=model.roi_head.predict_bbox(model.roi_head._inject_extra_pe(x),[sample.metainfo],proposals,rcnn_test_cfg=model.roi_head.test_cfg,rescale=False)[0]
   assert np.array_equal(det.bboxes.cpu().numpy(),base.xyxy(old[iid]));assert np.array_equal(det.scores.cpu().numpy(),np.array([z['score'] for z in old[iid]]))
   target=base._extract_instances_numpy(sample.gt_instances,(512,512));truth=np.stack([mu.decode(z['segmentation']).astype(bool) for z in gt[iid]]);assert np.array_equal(target['masks'].astype(bool),truth)
   allgt.append(base.arrays_small(target));meta=batch['img_metas'][0];metas.append(meta);boxes=np.array(common[iid]['boxes_xyxy'],dtype=np.float32);assert hashlib.sha256(boxes.tobytes()).hexdigest()==common[iid]['boxes_sha256']
   detsha=base.tensor_sha(det.bboxes)+base.tensor_sha(det.scores)
   for n in SCALES:
    f=(x,embeddings[n],pe,high)
    result=model.roi_head.predict_mask(model.roi_head._inject_extra_pe(x),[sample.metainfo],[copy.deepcopy(det)],rescale=False,image_embeddings=embeddings[n],image_positional_embeddings=pe,high_res_features=high)[0]
    pred=base._extract_instances_numpy(result,(512,512));records=[{k:v for k,v in z.items() if k!='file_name'} for z in base._prediction_records(pred,meta)]
    assert [(z['bbox'],z['score']) for z in records]==[(z['bbox'],z['score']) for z in old[iid]]
    masks=base.predict_masks(model,f,sample,boxes);iou=(masks&truth).sum(axis=(1,2))/np.maximum((masks|truth).sum(axis=(1,2)),1)
    if n=='s1':assert records==old[iid] and np.array_equal(iou,np.array(common[iid]['mask_iou']))
    alldt[n].append(base.arrays_small(pred));exported[n].extend(records);fixed[n].append(dict(image_id=iid,boxes_sha256=common[iid]['boxes_sha256'],mask_iou=iou.tolist()))
   assert detsha==base.tensor_sha(det.bboxes)+base.tensor_sha(det.scores);assert hashes==[base.tensor_sha(t) for t in tensors];assert base.payload_hash(sample.uav)==before and base.tensor_sha(sample.uav.rgb)==rgbsha
   checks.append(dict(image_id=iid,geometry_rgb_unchanged=True,hook_calls=len(captures),hook_removed=True,feature_hashes_unchanged=True,bbox_scores_exact=True,original_s1_exact=True,gt_boxes_sha256=common[iid]['boxes_sha256'],residual_norm_ratio=float(residual.float().norm()/e0.float().norm().clamp_min(1e-6))))
   manifest['image_ids'].append(iid)
   print(len(checks),iid,'s0/s05/s1',*[round(float(np.mean(fixed[n][-1]['mask_iou'])),6) for n in SCALES],flush=True)
   del captures,e0,e1,residual,embeddings,tensors,hashes,features,x,embedding,pe,high,proposals,det,result,pred,masks,f,target,truth,processed,sample,samples,imgs,batch,item
   torch.cuda.empty_cache()
 assert versions=={k:t._version for k,t in model.named_parameters()}
 metrics={}
 for n in SCALES:
  metrics[n]={}
  for kind in ['segm','bbox']:
   cg,cd=base.build_coco_gt_and_dt(allgt,alldt[n],metas,score_key='scores');metrics[n].update(base.run_coco_eval(cg,cd,kind))
  if n=='s1' and not args.max_images:
   ref=base.read(base.O/'correct_replay/metrics.json')
   for k,v in metrics[n].items():assert abs(v-ref[k])<1e-12
  base.dump(out/(n+'_dt_records.json'),exported[n]);base.dump(out/(n+'_fixed_box.json'),fixed[n])
 for n in SCALES:
  for k,v in metrics[n].items():
   if k.startswith('bbox/'):assert v==metrics['s1'][k]
 manifest.update(complete=len(checks)==153,processed_images=len(checks),ground_truth_instances=sum(len(v['labels']) for v in allgt),parameter_versions_unchanged=True)
 base.dump(out/'run_manifest.json',manifest);base.dump(out/'metrics.json',metrics);base.dump(out/'invariance_checks.json',checks)
 print('DONE SAM STRENGTH',flush=True)
if __name__=='__main__':main()
