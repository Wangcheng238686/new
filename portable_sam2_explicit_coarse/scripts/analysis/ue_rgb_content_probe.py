"""Frozen dual EMA: replace only UAV RGB, assert geometry and replay invariants."""
import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
import torch
from pycocotools import mask as mu

ROOT=Path('/home/wangcheng2021/project/new-uav-dual/portable_sam2_explicit_coarse')
sys.path[:0]=[str(ROOT),str(ROOT/'inference')]
os.environ['RSPROMPTER_LIGHT_IMPORT']='0'
os.environ['SAM2_REPO']=str(ROOT.parent/'sam2')
from infer_from_checkpoint import _resolve_model_config,_resolve_sam2_repo,_register_and_build,_load_model_state,_select_state_dict,_snapshot
from eval_records import _build_data_samples,_extract_instances_numpy,_prediction_records
from data.ue_rgbd_dataset import UERGBDInstanceDataset,ue_collate
from data.ue_geometry_cache import YAWS
from mmengine.structures import InstanceData
from utils.coco_eval_utils import build_coco_gt_and_dt,run_coco_eval,_masks_to_rles

R=Path('/tmp/ue_uav_experiments/20261014_ad_seed42/final_analysis')
O=R/'rgb_content_audit'
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2))
def file_sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def tensor_sha(t):
 a=t.detach().cpu().contiguous().numpy()
 return hashlib.sha256(str((a.dtype,a.shape)).encode()+a.tobytes()).hexdigest()
def payload_hash(p):return {k:tensor_sha(v) if torch.is_tensor(v) else repr(v) for k,v in (p.all_items() if hasattr(p,'all_items') else p.items()) if k!='rgb'}
def xyxy(rows):return np.array([[x['bbox'][0],x['bbox'][1],x['bbox'][0]+x['bbox'][2],x['bbox'][1]+x['bbox'][3]] for x in rows],dtype=np.float32)
def predict_masks(model,features,sample,boxes):
 x,embedding,pe,high=features
 result=InstanceData(bboxes=torch.as_tensor(boxes,dtype=torch.float32,device='cuda'),labels=torch.zeros(len(boxes),dtype=torch.long,device='cuda'),scores=torch.ones(len(boxes),device='cuda'))
 return model.roi_head.predict_mask(model.roi_head._inject_extra_pe(x),[sample.metainfo],[result],rescale=True,image_embeddings=embedding,image_positional_embeddings=pe,high_res_features=high)[0].masks.cpu().numpy().astype(bool)
def arrays_small(x):
 y={k:v for k,v in x.items() if k!='masks'}
 y['rles']=_masks_to_rles(x['masks'])
 return y

def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['correct_replay','donor_a','donor_b'],required=True);p.add_argument('--max-images',type=int,default=0);args=p.parse_args()
 torch.set_num_threads(2);torch.manual_seed(42);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 out=O/(args.mode+('_smoke' if args.max_images else ''));out.mkdir(exist_ok=True)
 mapping=read(O/'donor_maps.json');expected=read(R/'dual_correct/run_manifest.json')
 cp=Path(expected['checkpoint']);checkpoint=torch.load(cp,map_location='cpu',weights_only=False);snap=_snapshot(checkpoint)
 resolve_args=argparse.Namespace(sam2_ckpt=None,sam2_repo=None);_resolve_sam2_repo(resolve_args,snap);config,_=_resolve_model_config(resolve_args,snap);model=_register_and_build(config)
 report=_load_model_state(model,_select_state_dict(checkpoint,'ema'),allow_nonstrict=False)
 model.roi_head.mask_head.assert_no_mask_embedding_contract('RGB content frozen EMA probe')
 model.cuda().eval();model.uav_intervention='correct';del checkpoint
 state=snap['data_config'];ds=UERGBDInstanceDataset(state['manifest'],'validation',state['geometry'],load_uav=True,cache_root='/tmp/uav_dual_dev_data/cache')
 assert ds.manifest_sha256==state['manifest_sha256']==mapping['manifest_sha256']
 assert [x['index'] for x in ds.samples]==mapping['image_ids']==expected['processed_image_ids']
 rows={x['index']:x for x in read(state['manifest'])['points']}
 donors={x['target']:x for x in mapping['maps'].get(args.mode,{}).get('records',[])}
 old=defaultdict(list);gt=defaultdict(list)
 for v in read(R/'dual_correct/dt_records.json'):old[v['image_id']].append(v)
 for v in read(R/'dual_correct/gt_records.json'):gt[v['image_id']].append(v)
 common=defaultdict(dict)
 for v in read(R/'full_fixed_box_audit/dual_correct.json')['records']:common[v['image_id']][v['box_type']]=v
 all_gt=[];all_dt=[];metas=[];dt_records=[];fixed=[];checks=[]
 versions={k:t._version for k,t in model.named_parameters()}
 git_status=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)
 git_diff=subprocess.check_output(['git','diff','HEAD'],cwd=ROOT)
 manifest=dict(git_dirty=bool(git_status),git_status_porcelain=git_status,git_diff_sha256=hashlib.sha256(git_diff).hexdigest(),mode=args.mode,checkpoint=str(cp),weights='ema',strict=True,load_report=report,training=False,optimizer_created=False,manifest_sha256=ds.manifest_sha256,donor_maps_sha256=file_sha(O/'donor_maps.json'),script_sha256=file_sha(__file__),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),preregistration_sha256=file_sha(O/'PREREGISTRATION.md'),source_yaws=list(YAWS),image_ids=[],processed_images=0,scope='Current frozen development subset validation, not independent test or full future dataset',architecture_contract=snap['architecture_contract'])
 dump(out/'run_manifest.json',manifest)
 with torch.inference_mode():
  for index,row in enumerate(ds.samples):
   if args.max_images and index>=args.max_images:break
   iid=row['index'];item=ds[index];before=payload_hash(item['uav_payload']);sat_sha=tensor_sha(item['img']);gt_sha=hashlib.sha256(item['gt_masks'].tobytes()+item['gt_bboxes'].numpy().tobytes()).hexdigest()
   donor=iid if args.mode=='correct_replay' else donors[iid]['donor'];donor_row=rows[donor]
   rgb=[];files=[]
   for yaw in YAWS:
    path=ds.uav_root/donor_row['uav_dir']/'oblique'/f'view_yaw_{yaw:03d}'/'oblique_rgb.0000.png'
    image=cv2.imread(str(path));assert image is not None and image.shape==(512,512,3)
    rgb.append(image[...,::-1].transpose(2,0,1));files.append(dict(path=str(path),sha256=file_sha(path)))
   replacement=torch.from_numpy(np.stack(rgb).copy()).float();original_sha=tensor_sha(item['uav_payload']['rgb'])
   if args.mode=='correct_replay':assert torch.equal(replacement,item['uav_payload']['rgb'])
   else:
    assert files==donors[iid]['rgb_files'];assert tensor_sha(replacement)!=original_sha
   item['uav_payload']['rgb']=replacement
   assert before==payload_hash(item['uav_payload']);assert sat_sha==tensor_sha(item['img']);assert gt_sha==hashlib.sha256(item['gt_masks'].tobytes()+item['gt_bboxes'].numpy().tobytes()).hexdigest()
   batch=ue_collate([item]);imgs,samples=_build_data_samples(batch,torch.device('cuda'));processed=model.data_preprocessor(dict(inputs=imgs,data_samples=samples),training=False);imgs=processed['inputs'];sample=processed['data_samples'][0]
   assert payload_hash(sample.uav)==before
   features=model._extract_dual_feat(imgs,[sample]);x,embedding,pe,high=features
   proposals=model.rpn_head.predict(x,[sample],rescale=False)
   result=model.roi_head.predict(x,proposals,[sample],rescale=False,image_embeddings=embedding,image_positional_embeddings=pe,high_res_features=high)[0]
   pred=_extract_instances_numpy(result,(512,512));target=_extract_instances_numpy(sample.gt_instances,(512,512));truth=np.stack([mu.decode(v['segmentation']).astype(bool) for v in gt[iid]])
   assert np.array_equal(target['masks'].astype(bool),truth)
   exported=list(_prediction_records(pred,batch['img_metas'][0]));exported=[{k:v for k,v in z.items() if k!='file_name'} for z in exported]
   check=dict(image_id=iid,donor=donor,geometry_hashes=before,satellite_sha256=sat_sha,target_sha256=gt_sha,original_rgb_sha256=original_sha,replacement_rgb_sha256=tensor_sha(replacement),rgb_files=files,geometry_unchanged=True)
   if args.mode=='correct_replay':
    assert exported==old[iid],('Full prediction replay failed',iid)
    check['full_prediction_exact']=True
   dt_records.extend(exported);all_gt.append(arrays_small(target));all_dt.append(arrays_small(pred));metas.append(batch['img_metas'][0])
   for box_type in ('gt','single_pred','dual_pred'):
    base=common[iid][box_type];boxes=np.array(base['boxes_xyxy'],dtype=np.float32);assert hashlib.sha256(boxes.tobytes()).hexdigest()==base['boxes_sha256']
    masks=predict_masks(model,features,sample,boxes);iou=(masks&truth).sum(axis=(1,2))/np.maximum((masks|truth).sum(axis=(1,2)),1)
    if args.mode=='correct_replay':assert np.array_equal(iou,np.array(base['mask_iou'])),('fixed box replay failed',iid,box_type)
    fixed.append(dict(image_id=iid,box_type=box_type,boxes_sha256=base['boxes_sha256'],mask_iou=iou.tolist()))
   checks.append(check);manifest['image_ids'].append(iid);manifest['processed_images']=len(checks)
   print(args.mode,len(checks),iid,'donor',donor,'det',len(exported),flush=True)
   del features,x,embedding,pe,high,proposals,result,pred,target,processed,sample,samples,imgs,batch,item
   torch.cuda.empty_cache()
 assert versions=={k:t._version for k,t in model.named_parameters()}
 metrics={}
 for kind in ('segm','bbox'):
  cg,cd=build_coco_gt_and_dt(all_gt,all_dt,metas,score_key='scores');metrics.update(run_coco_eval(cg,cd,kind))
 if args.mode=='correct_replay' and not args.max_images:
  for k,v in metrics.items():assert abs(v-expected['metrics'][k])<1e-12,(k,v,expected['metrics'][k])
 manifest.update(parameter_versions_unchanged=True,complete=(len(checks)==153),ground_truth_instances=sum(len(x['labels']) for x in all_gt),predicted_instances=len(dt_records))
 dump(out/'dt_records.json',dt_records);dump(out/'fixed_box.json',fixed);dump(out/'invariance_checks.json',checks);dump(out/'metrics.json',metrics);dump(out/'run_manifest.json',manifest)
 print('DONE',args.mode,json.dumps(metrics),flush=True)
if __name__=='__main__':main()
