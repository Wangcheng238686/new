"""Frozen SAM residual translations, symmetric smoothing and local mask diagnostics."""
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
from rsprompter.uav.cameras import load_camera,localize_camera,backproject,project
from data.ue_geometry_cache import read_depth,YAWS
R=base.R;O=R/'spatial_local_audit'
SHIFTS={'shift4_right':(.25,0.),'shift4_left':(-.25,0.),'shift4_down':(0.,.25),'shift4_up':(0.,-.25)}
NAMES=['identity',*SHIFTS,'smooth_symmetric']

def translate(tensor,dx,dy):
 b,c,h,w=tensor.shape
 y=(torch.arange(h,device=tensor.device,dtype=tensor.dtype)+.5)*2/h-1-2*dy/h
 x=(torch.arange(w,device=tensor.device,dtype=tensor.dtype)+.5)*2/w-1-2*dx/w
 gy,gx=torch.meshgrid(y,x,indexing='ij');grid=torch.stack([gx,gy],dim=-1)[None].expand(b,-1,-1,-1)
 return F.grid_sample(tensor,grid,mode='bilinear',padding_mode='zeros',align_corners=False)

def translation_checks():
 p=torch.zeros(1,1,32,32);p[0,0,15,15]=1
 for name,(dx,dy) in SHIFTS.items():
  out=translate(p,dx,dy);expected=torch.zeros_like(p);expected[0,0,15,15]=.75;expected[0,0,15+int(np.sign(dy)),15+int(np.sign(dx))]+=.25
  assert torch.allclose(out,expected,atol=1e-6,rtol=0),(name,float((out-expected).abs().max()))
 edge=torch.zeros_like(p);edge[0,0,0,0]=1
 out=translate(edge,-.25,0);assert float(out[:,:,:,-1].abs().max())==0 and abs(float(out.sum())-.75)<1e-6
 # In the interior a unit linear slope must translate by a quarter cell.
 ramp=torch.arange(32,dtype=torch.float32)[None,None,None,:].expand(1,1,32,32)
 assert torch.allclose(translate(ramp,.25,0)[:,:,1:-1,1:-1],ramp[:,:,1:-1,1:-1]-.25,atol=1e-6,rtol=0)
 return dict(pulse_quarter_cell=True,direction_right_down=True,no_wrap=True,linear_ramp=True)

def camera_checks(row,ds):
 sat=ds.satellite_root/row['satellite_dir'];uav=ds.uav_root/row['uav_dir'];origin=np.array(row['origin_m']);sk,se,st,_=load_camera(sat);sle,slt=localize_camera(se,st,origin)
 maxima=[0.,0.,0.];count=0;view_counts=[]
 for yaw in YAWS:
  folder=uav/'oblique'/f'view_yaw_{yaw:03d}';k,e,t,_=load_camera(folder);le,lt=localize_camera(e,t,origin);depth=read_depth(folder/'depth_camera_z.0000.exr',512);y,x=np.mgrid[16:512:32,16:512:32];z=depth[y,x];valid=np.isfinite(z)&(z>0)&(z<10000);u=x[valid];v=y[valid];z=z[valid]
  assert len(z)>0,('No valid roundtrip depth samples',row['index'],yaw)
  view_counts.append(dict(yaw=yaw,valid_samples=len(z)))
  world=backproject(z,u,v,k,t);local=backproject(z,u,v,k,lt);uv,camera_z=project(local,k,le);absolute=np.max(np.abs(uv-np.column_stack([u,v])));zerror=np.max(np.abs(camera_z-z));a,az=project(world,sk,se);b,bz=project(local,sk,sle);visible=(az>0)&(bz>0)&np.isfinite(a).all(axis=1)&np.isfinite(b).all(axis=1)
  saterror=float(np.max(np.abs(a[visible]-b[visible]))) if visible.any() else 0.
  assert np.allclose(local,world-origin,atol=1e-8,rtol=0);maxima=np.maximum(maxima,[absolute,zerror,saterror]).tolist();count+=len(z)
 assert maxima[0]<1e-6 and maxima[1]<1e-6 and maxima[2]<1e-6
 return dict(per_view_valid_samples=view_counts,sampled_depth_pixels=count,max_same_camera_uv_error_px=maxima[0],max_same_camera_z_error_m=maxima[1],max_sat_world_local_uv_error_px=maxima[2])

def regions(mask):
 kernel=np.ones((7,7),np.uint8);inner=cv2.erode(mask.astype(np.uint8),kernel,borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool);outer=cv2.dilate(mask.astype(np.uint8),kernel,borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)
 return inner,outer,outer&~inner

def local_mask_metrics(pred,truth,region):
 inner,outer,band=region;intersection=int((pred&truth&band).sum());union=int(((pred|truth)&band).sum());n=int(inner.sum());area=int(truth.sum())
 return dict(boundary_band_iou=intersection/max(union,1),inner_fn_rate=float((inner&~pred).sum())/n if n else None,far_fp_per_gt_area=float((pred&~outer).sum())/max(area,1))

def main():
 p=argparse.ArgumentParser();p.add_argument('--max-images',type=int,default=0);args=p.parse_args();torch.set_num_threads(2);torch.manual_seed(42);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 out=O/('smoke' if args.max_images else 'full');out.mkdir(exist_ok=True);synthetic=translation_checks()
 expected=base.read(base.O/'correct_replay/run_manifest.json');checkpoint=torch.load(expected['checkpoint'],map_location='cpu',weights_only=False);snap=base._snapshot(checkpoint);ra=argparse.Namespace(sam2_ckpt=None,sam2_repo=None);base._resolve_sam2_repo(ra,snap);cfg,_=base._resolve_model_config(ra,snap);model=base._register_and_build(cfg);report=base._load_model_state(model,base._select_state_dict(checkpoint,'ema'),allow_nonstrict=False);model.roi_head.mask_head.assert_no_mask_embedding_contract('Spatial-local frozen diagnosis');model.cuda().eval();model.uav_intervention='correct';del checkpoint
 state=snap['data_config'];ds=base.UERGBDInstanceDataset(state['manifest'],'validation',state['geometry'],load_uav=True,cache_root='/tmp/uav_dual_dev_data/cache');assert ds.manifest_sha256==expected['manifest_sha256'];assert [x['index'] for x in ds.samples]==expected['image_ids']
 old=defaultdict(list);gt=defaultdict(list);common={};stage=defaultdict(list)
 for z in base.read(base.O/'correct_replay/dt_records.json'):old[z['image_id']].append(z)
 for z in base.read(R/'dual_correct/gt_records.json'):gt[z['image_id']].append(z)
 for z in base.read(R/'full_stage_audit/dual_correct.json')['instances']:stage[z['image_id']].append(z)
 for z in base.read(R/'full_fixed_box_audit/dual_correct.json')['records']:
  if z['box_type']=='gt':common[z['image_id']]=z
 allgt=[];metas=[];alldt={n:[] for n in NAMES};exported={n:[] for n in NAMES};fixed={n:[] for n in NAMES};checks=[];geometry=[]
 versions={k:t._version for k,t in model.named_parameters()};status=subprocess.check_output(['git','status','--porcelain'],cwd=base.ROOT,text=True)
 manifest=dict(checkpoint=expected['checkpoint'],weights='ema',strict=True,load_report=report,training=False,optimizer_created=False,variants=NAMES,shifts_feature_cells=SHIFTS,satellite_pixel_shift=4,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=base.ROOT,text=True).strip(),git_dirty=bool(status),git_status_porcelain=status,script_sha256=base.file_sha(__file__),helper_script_sha256=base.file_sha(base.__file__),manifest_sha256=ds.manifest_sha256,preregistration_sha256=base.file_sha(O/'PREREGISTRATION.md'),architecture_contract=snap['architecture_contract'],synthetic_checks=synthetic,image_ids=[],complete=False)
 base.dump(out/'run_manifest.json',manifest)
 with torch.inference_mode():
  for index,row in enumerate(ds.samples):
   if args.max_images and index>=args.max_images:break
   iid=row['index'];camera=camera_checks(row,ds);item=ds[index];before=base.payload_hash(item['uav_payload']);rgbsha=base.tensor_sha(item['uav_payload']['rgb']);batch=base.ue_collate([item]);imgs,samples=base._build_data_samples(batch,torch.device('cuda'));processed=model.data_preprocessor(dict(inputs=imgs,data_samples=samples),training=False);imgs=processed['inputs'];sample=processed['data_samples'][0]
   captures=[]
   def capture(module,inputs,output):captures.append((inputs[0],output[0],inputs[2],inputs[3]))
   hook=model.sam_fusion.register_forward_hook(capture)
   try:features=model._extract_dual_feat(imgs,[sample])
   finally:hook.remove()
   assert len(captures)==1;e0,e1,confidence,coverage=captures[0];assert e1 is features[1] and e1.shape[-2:]==(32,32)
   residual=e1-e0;valid=(coverage>0).to(residual.dtype);assert torch.equal((residual*(1-valid)),torch.zeros_like(residual));shifted={n:translate(residual,*xy)*valid for n,xy in SHIFTS.items()};smooth=sum(shifted.values())/4;embeddings={'identity':e1,**{n:e0+v for n,v in shifted.items()},'smooth_symmetric':e0+smooth}
   tensors=list({id(t):t for t in list(features[0])+[e0,e1,confidence,coverage,residual]+list(embeddings.values())+([features[2]] if torch.is_tensor(features[2]) else [])+list(features[3] or ())}.values());hashes=[base.tensor_sha(t) for t in tensors]
   x,embedding,pe,high=features;proposals=model.rpn_head.predict(x,[sample],rescale=False);det=model.roi_head.predict_bbox(model.roi_head._inject_extra_pe(x),[sample.metainfo],proposals,rcnn_test_cfg=model.roi_head.test_cfg,rescale=False)[0];assert np.array_equal(det.bboxes.cpu().numpy(),base.xyxy(old[iid]));assert np.array_equal(det.scores.cpu().numpy(),np.array([z['score'] for z in old[iid]]))
   target=base._extract_instances_numpy(sample.gt_instances,(512,512));truth=np.stack([mu.decode(z['segmentation']).astype(bool) for z in gt[iid]]);assert np.array_equal(truth,target['masks'].astype(bool));allgt.append(base.arrays_small(target));meta=batch['img_metas'][0];metas.append(meta);boxes=np.array(common[iid]['boxes_xyxy'],dtype=np.float32);assert hashlib.sha256(boxes.tobytes()).hexdigest()==common[iid]['boxes_sha256'];local_regions=[regions(m) for m in truth]
   payload=item['uav_payload'];pixels=payload['project_pixel'].numpy();mass=payload['project_mass'].numpy();quality=payload['project_quality'].numpy();assert len(pixels)==len(mass)==len(quality);fm=np.bincount(pixels,weights=mass,minlength=512**2).reshape(512,512);fq=np.bincount(pixels,weights=mass*quality,minlength=512**2).reshape(512,512)/np.maximum(fm,1e-6);supported=fm>0
   cq=F.interpolate(confidence.float(),size=(512,512),mode='nearest')[0,0].cpu().numpy();cc=F.interpolate(coverage.float(),size=(512,512),mode='nearest')[0,0].cpu().numpy();weak=(cc<.25)|(cq<.25);identities=sorted(stage[iid],key=lambda z:z['gt_slot']);assert len(identities)==len(truth)
   for slot,(gtmask,reg,identity) in enumerate(zip(truth,local_regions,identities)):
    sup=supported & gtmask;band=reg[2];ys,xs=np.where(gtmask)
    geometry.append(dict(image_id=iid,gt_slot=slot,global_id=identity['global_id'],bbox_area=identity['bbox_area'],fine_support=float(sup.sum()/gtmask.sum()),fine_quality_supported=float(fq[sup].mean()) if sup.any() else None,sam_coverage_gt=float(cc[gtmask].mean()),sam_confidence_gt=float(cq[gtmask].mean()),weak_coarse_fraction=float(weak[gtmask].mean()),boundary_fine_support=float(supported[band].mean()),clipped=bool(xs.min()==0 or xs.max()==511 or ys.min()==0 or ys.max()==511)))
   for n in NAMES:
    f=(x,embeddings[n],pe,high);result=model.roi_head.predict_mask(model.roi_head._inject_extra_pe(x),[sample.metainfo],[copy.deepcopy(det)],rescale=False,image_embeddings=embeddings[n],image_positional_embeddings=pe,high_res_features=high)[0];pred=base._extract_instances_numpy(result,(512,512));records=[{k:v for k,v in z.items() if k!='file_name'} for z in base._prediction_records(pred,meta)];assert [(z['bbox'],z['score']) for z in records]==[(z['bbox'],z['score']) for z in old[iid]]
    masks=base.predict_masks(model,f,sample,boxes);iou=(masks&truth).sum(axis=(1,2))/np.maximum((masks|truth).sum(axis=(1,2)),1)
    if n=='identity':assert records==old[iid] and np.array_equal(iou,np.array(common[iid]['mask_iou']))
    local=[local_mask_metrics(p,t,reg) for p,t,reg in zip(masks,truth,local_regions)];alldt[n].append(base.arrays_small(pred));exported[n].extend(records);fixed[n].append(dict(image_id=iid,boxes_sha256=common[iid]['boxes_sha256'],mask_iou=iou.tolist(),local_metrics=local))
   assert hashes==[base.tensor_sha(t) for t in tensors];assert before==base.payload_hash(sample.uav) and rgbsha==base.tensor_sha(sample.uav.rgb)
   checks.append(dict(image_id=iid,identity_exact=True,bbox_scores_exact=True,geometry_rgb_unchanged=True,feature_hashes_unchanged=True,hook_calls=len(captures),camera_checks=camera));manifest['image_ids'].append(iid)
   print(len(checks),iid,'IoU',*[round(float(np.mean(fixed[n][-1]['mask_iou'])),6) for n in NAMES],flush=True)
   del captures,e0,e1,confidence,coverage,residual,shifted,smooth,embeddings,tensors,hashes,features,x,embedding,pe,high,proposals,det,result,pred,masks,f,target,truth,processed,sample,samples,imgs,batch,item
   torch.cuda.empty_cache()
 assert versions=={k:t._version for k,t in model.named_parameters()};metrics={}
 for n in NAMES:
  metrics[n]={}
  for kind in ['segm','bbox']:
   cg,cd=base.build_coco_gt_and_dt(allgt,alldt[n],metas,score_key='scores');metrics[n].update(base.run_coco_eval(cg,cd,kind))
  if n=='identity' and not args.max_images:
   reference=base.read(base.O/'correct_replay/metrics.json')
   for k,v in metrics[n].items():assert abs(v-reference[k])<1e-12
  base.dump(out/(n+'_dt_records.json'),exported[n]);base.dump(out/(n+'_fixed_box.json'),fixed[n])
 for n in NAMES:
  for k,v in metrics[n].items():
   if k.startswith('bbox/'):assert v==metrics['identity'][k]
 manifest.update(complete=len(checks)==153,processed_images=len(checks),ground_truth_instances=sum(len(v['labels']) for v in allgt),parameter_versions_unchanged=True)
 base.dump(out/'run_manifest.json',manifest);base.dump(out/'metrics.json',metrics);base.dump(out/'invariance_checks.json',checks);base.dump(out/'local_geometry.json',geometry)
 print('DONE SPATIAL LOCAL',flush=True)
if __name__=='__main__':main()
