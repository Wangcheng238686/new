"""Frozen 2x2 activation patching at independent FPN and SAM fusion outputs."""
import argparse
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import cv2
from pycocotools import mask as mu
sys.path.insert(0,'/tmp')
import ue_rgb_content_probe as base

R=base.R
O=R/'fusion_route_audit'
read=base.read
dump=base.dump

def equal_optional(a,b):
 if a is None or b is None:return a is None and b is None
 if torch.is_tensor(a):return torch.equal(a,b)
 return len(a)==len(b) and all(equal_optional(x,y) for x,y in zip(a,b))

def predict(model,features,sample,meta):
 x,embedding,pe,high=features
 proposals=model.rpn_head.predict(x,[sample],rescale=False)
 result=model.roi_head.predict(x,proposals,[sample],rescale=False,image_embeddings=embedding,image_positional_embeddings=pe,high_res_features=high)[0]
 arr=base._extract_instances_numpy(result,(512,512))
 export=[{k:v for k,v in z.items() if k!='file_name'} for z in base._prediction_records(arr,meta)]
 return arr,export

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--donor',choices=['donor_a','donor_b'],required=True);parser.add_argument('--max-images',type=int,default=0);args=parser.parse_args()
 torch.set_num_threads(2);torch.manual_seed(42);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 out=O/(args.donor+('_smoke' if args.max_images else ''));out.mkdir(exist_ok=True)
 expected=read(base.O/'correct_replay/run_manifest.json');maps=read(base.O/'donor_maps.json');mapping={v['target']:v for v in maps['maps'][args.donor]['records']}
 checkpoint=torch.load(expected['checkpoint'],map_location='cpu',weights_only=False);snap=base._snapshot(checkpoint);ra=argparse.Namespace(sam2_ckpt=None,sam2_repo=None);base._resolve_sam2_repo(ra,snap);cfg,_=base._resolve_model_config(ra,snap);model=base._register_and_build(cfg);report=base._load_model_state(model,base._select_state_dict(checkpoint,'ema'),allow_nonstrict=False)
 model.roi_head.mask_head.assert_no_mask_embedding_contract('Frozen fusion route probe');model.cuda().eval();model.uav_intervention='correct';del checkpoint
 state=snap['data_config'];ds=base.UERGBDInstanceDataset(state['manifest'],'validation',state['geometry'],load_uav=True,cache_root='/tmp/uav_dual_dev_data/cache')
 assert ds.manifest_sha256==maps['manifest_sha256']==expected['manifest_sha256'];assert [v['index'] for v in ds.samples]==expected['image_ids']
 diagonal={};common={};gt=defaultdict(list)
 for n,old in [('CC','correct_replay'),('DD',args.donor)]:
  dt=defaultdict(list)
  for v in read(base.O/old/'dt_records.json'):dt[v['image_id']].append(v)
  fixed={v['image_id']:v for v in read(base.O/old/'fixed_box.json') if v['box_type']=='gt'}
  diagonal[n]=(dt,fixed)
 for v in read(R/'full_fixed_box_audit/dual_correct.json')['records']:
  if v['box_type']=='gt':common[v['image_id']]=v
 for v in read(R/'dual_correct/gt_records.json'):gt[v['image_id']].append(v)
 allgt=[];metas=[];alldt={n:[] for n in ['CC','CD','DC','DD']};exported={n:[] for n in alldt};fixed={n:[] for n in alldt};checks=[]
 versions={k:t._version for k,t in model.named_parameters()};status=subprocess.check_output(['git','status','--porcelain'],cwd=base.ROOT,text=True)
 manifest=dict(donor=args.donor,checkpoint=expected['checkpoint'],weights='ema',strict=True,load_report=report,training=False,optimizer_created=False,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=base.ROOT,text=True).strip(),git_dirty=bool(status),git_status_porcelain=status,script_sha256=base.file_sha(__file__),helper_script_sha256=base.file_sha(base.__file__),manifest_sha256=ds.manifest_sha256,donor_maps_sha256=base.file_sha(base.O/'donor_maps.json'),preregistration_sha256=base.file_sha(O/'PREREGISTRATION.md'),image_ids=[],complete=False,architecture_contract=snap['architecture_contract'])
 dump(out/'run_manifest.json',manifest)
 with torch.inference_mode():
  for index,row in enumerate(ds.samples):
   if args.max_images and index>=args.max_images:break
   iid=row['index'];item=ds[index];before=base.payload_hash(item['uav_payload']);batch=base.ue_collate([item]);imgs,samples=base._build_data_samples(batch,torch.device('cuda'));processed=model.data_preprocessor(dict(inputs=imgs,data_samples=samples),training=False);imgs=processed['inputs'];sample=processed['data_samples'][0]
   assert before==base.payload_hash(sample.uav)
   cf=model._extract_dual_feat(imgs,[sample])
   rgb=[]
   for rec in mapping[iid]['rgb_files']:
    assert base.file_sha(rec['path'])==rec['sha256'];raw=cv2.imread(rec['path']);assert raw is not None and raw.shape==(512,512,3);rgb.append(raw[...,::-1].transpose(2,0,1))
   sample.uav.rgb=torch.from_numpy(np.stack(rgb).copy()).float().cuda()
   assert before==base.payload_hash(sample.uav)
   df=model._extract_dual_feat(imgs,[sample]);assert equal_optional(cf[2],df[2]) and equal_optional(cf[3],df[3])
   for level in range(len(cf[0])):
    if level not in model.fpn_levels:assert torch.equal(cf[0][level],df[0][level])
   packs=dict(CC=cf,CD=(cf[0],df[1],cf[2],cf[3]),DC=(df[0],cf[1],cf[2],cf[3]),DD=df)
   feature_tensors=list({id(t):t for f in [cf,df] for seq in [f[0],(f[1],),f[3] or ()] for t in seq}.values())
   feature_hashes=[base.tensor_sha(t) for t in feature_tensors]
   target=base._extract_instances_numpy(sample.gt_instances,(512,512));truth=np.stack([mu.decode(v['segmentation']).astype(bool) for v in gt[iid]]);assert np.array_equal(truth,target['masks'].astype(bool))
   allgt.append(base.arrays_small(target));meta=batch['img_metas'][0];metas.append(meta);boxes=np.array(common[iid]['boxes_xyxy'],dtype=np.float32);assert hashlib.sha256(boxes.tobytes()).hexdigest()==common[iid]['boxes_sha256']
   for n,f in packs.items():
    pred,records=predict(model,f,sample,meta);masks=base.predict_masks(model,f,sample,boxes);iou=(masks&truth).sum(axis=(1,2))/np.maximum((masks|truth).sum(axis=(1,2)),1)
    if n in diagonal:
     olddt,oldfixed=diagonal[n];assert records==olddt[iid],('diagonal output mismatch',n,iid);assert np.array_equal(iou,np.array(oldfixed[iid]['mask_iou'])),('diagonal IoU mismatch',n,iid)
    alldt[n].append(base.arrays_small(pred));exported[n].extend(records);fixed[n].append(dict(image_id=iid,boxes_sha256=common[iid]['boxes_sha256'],mask_iou=iou.tolist()))
   assert feature_hashes==[base.tensor_sha(t) for t in feature_tensors]
   check=dict(image_id=iid,donor=mapping[iid]['donor'],nonrgb_geometry_unchanged=True,pe_high_identical=True,unfused_fpn_identical=True,cc_exact=True,dd_exact=True,feature_hashes_unchanged=True)
   checks.append(check);manifest['image_ids'].append(iid)
   print(args.donor,len(checks),iid,'CC/CD/DC/DD',*[round(float(np.mean(fixed[n][-1]['mask_iou'])),6) for n in packs],flush=True)
   del cf,df,packs,feature_tensors,feature_hashes,f,target,truth,pred,masks,processed,sample,samples,imgs,batch,item
   torch.cuda.empty_cache()
 assert versions=={k:t._version for k,t in model.named_parameters()}
 metrics={}
 for n in alldt:
  metrics[n]={}
  for kind in ['segm','bbox']:
   cg,cd=base.build_coco_gt_and_dt(allgt,alldt[n],metas,score_key='scores');metrics[n].update(base.run_coco_eval(cg,cd,kind))
  if n in diagonal and not args.max_images:
   ref=read(base.O/('correct_replay' if n=='CC' else args.donor)/'metrics.json')
   for k,v in metrics[n].items():assert abs(v-ref[k])<1e-12
  dump(out/(n+'_dt_records.json'),exported[n]);dump(out/(n+'_fixed_box.json'),fixed[n])
 manifest.update(complete=(len(checks)==153),processed_images=len(checks),ground_truth_instances=sum(len(v['labels']) for v in allgt),parameter_versions_unchanged=True)
 dump(out/'metrics.json',metrics);dump(out/'invariance_checks.json',checks);dump(out/'run_manifest.json',manifest)
 print('DONE',args.donor,flush=True)
if __name__=='__main__':main()
