"""UNEXECUTED frozen CVT pooling+reconstruction diagnostic. Root approval required."""
import argparse,hashlib,json,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from mmcv.ops import RoIAlign
sys.path.insert(0,'/tmp');import ue_rgb_content_probe as b
sys.path.insert(0,'/tmp/ue_stage_shape_probe');from extract_stage_pair import matrix,project,feature_hash,streamsha
O=Path('/tmp/ue_projection_scale_probe');OLD=Path('/tmp/ue_stage_shape_probe/features')
REUSE=['sat32','sat_extra32','prob','target','boundary_weight','global_id','boxes','image_id','gt_slot']
def denominator(payload,size):
 pix=payload.project_pixel;stride=512//size;target=(pix//512//stride)*size+(pix%512//stride);d=torch.zeros(size*size,device=pix.device);d.index_add_(0,target,payload.project_mass*payload.project_quality);return d.reshape(1,1,size,size)
def delta(x,y):
 d=(x-y).float();return dict(rms=float(d.square().mean().sqrt()),mae=float(d.abs().mean()),relative_l2=float(d.norm()/y.float().norm().clamp_min(1e-6)))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');args=ap.parse_args();torch.set_num_threads(2);torch.manual_seed(42);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 old=b.read(OLD/'run_manifest.json');assert streamsha(old['checkpoint'])==old['checkpoint_sha256'];assert old['complete'] and old['processed_images']==281;by={(r['split'],r['image_id']):r for r in old['records']};cp=torch.load(old['checkpoint'],map_location='cpu',weights_only=False);snap=b._snapshot(cp);ra=argparse.Namespace(sam2_ckpt=None,sam2_repo=None);b._resolve_sam2_repo(ra,snap);cfg,_=b._resolve_model_config(ra,snap);m=b._register_and_build(cfg);load=b._load_model_state(m,b._select_state_dict(cp,'ema'),allow_nonstrict=False);m.cuda().eval();m.uav_intervention='correct';del cp
 assert snap['architecture_contract']==old['architecture_contract']
 for entry in old['projection_matrices']:assert b.tensor_sha(matrix(entry['in_dim'],entry['seed']))==entry['sha256']
 state=snap['data_config'];assert b.file_sha(state['manifest'])==old['manifest_sha256'];selected=old['selected_image_ids'];out=O/('smoke' if args.smoke else 'features');out.mkdir(exist_ok=True);versions={k:t._version for k,t in m.named_parameters()};started=time.monotonic();seen={'train':set(),'validation':set()}
 paths=[Path(__file__),Path(b.__file__),Path('/tmp/ue_stage_shape_probe/extract_stage_pair.py'),Path('/tmp/ue_stage_shape_probe/stage_pair_extraction_draft.py'),b.ROOT/'rsprompter/uav/projection.py',b.ROOT/'rsprompter/uav/cvt.py',b.ROOT/'rsprompter/detection/dual_detector.py']
 manifest=dict(complete=False,weights='ema',strict_load=load,training=False,optimizer=False,checkpoint=old['checkpoint'],checkpoint_sha256=streamsha(old['checkpoint']),manifest_sha256=old['manifest_sha256'],selected_image_ids=selected,architecture_contract=snap['architecture_contract'],source_sha256={str(p):b.file_sha(p) for p in paths},protocol_sha256=b.file_sha(O/'PREREGISTRATION_DRAFT.md'),old_manifest_sha256=b.file_sha(OLD/'run_manifest.json'),source_git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=b.ROOT,text=True).strip(),git_dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=b.ROOT,text=True)),projection_matrices=old['projection_matrices'],actual_pipeline=dict(coarse_hw=[32,32],fine_hw=[128,128],coarse_reconstruction='bilinear128 alignfalse',roi_output=[16,16],roi_spatial_scale=.25,shift_fine_cells=[32,32],common_support='fineD>0 & roll(fineD>0,32,32)',shared_geometry='oldp/xy/wh +finequality/finecoverage timesI'),records=[])
 b.dump(out/'run_manifest.json',manifest)
 with torch.inference_mode():
  for split in ('train','validation'):
   ds=b.UERGBDInstanceDataset(state['manifest'],split,state['geometry'],load_uav=True,cache_root='/tmp/uav_dual_dev_data/cache');rows=[(i,r) for i,r in enumerate(ds.samples) if r['index'] in set(selected[split])]
   if args.smoke:rows=rows[:3]
   for index,row in rows:
    if time.monotonic()-started>900:raise RuntimeError('Operational900sec cap')
    iid=row['index'];rec=by[(split,iid)];assert b.file_sha(rec['path'])==rec['sha256']
    with np.load(rec['path'],allow_pickle=False) as z:previous={k:z[k].copy() for k in z.files}
    item=ds[index];assert np.array_equal(item['gt_bboxes'].numpy(),previous['boxes']);n=len(previous['boxes']);assert n==rec['n_rois'] and np.array_equal(previous['gt_slot'],np.arange(n));assert np.all(previous['image_id']==iid)
    batch=b.ue_collate([item]);imgs,samples=b._build_data_samples(batch,torch.device('cuda'));proc=m.data_preprocessor(dict(inputs=imgs,data_samples=samples),training=False);sample=proc['data_samples'][0];before=b.payload_hash(sample.uav);rgb=b.tensor_sha(sample.uav.rgb);sat=b.tensor_sha(proc['inputs']);hookcounts=[(len(z._forward_hooks),len(z._forward_pre_hooks)) for z in m.modules()];capture=[]
    def hook(module,inputs,output):capture.append(output)
    h=m.uav_bev.register_forward_hook(hook)
    try:features=m._extract_dual_feat(proc['inputs'],[sample]);featurehash=feature_hash(features)
    finally:h.remove()
    del features;assert len(capture)==1;bev=capture[0];bevsha=b.tensor_sha(bev)
    coarse,qc,cc=m.bev_projection(bev,[sample.uav],(32,32));fine,qf,cf=m.bev_projection(bev,[sample.uav],(128,128));df=denominator(sample.uav,128);assert torch.equal(df>0,qf>0)
    if args.smoke:
     for size,qprod in [(32,qc),(128,qf)]:
      pixels=sample.uav.project_pixel.cpu().numpy();sources=sample.uav.project_source.cpu().numpy();weights=(sample.uav.project_mass*sample.uav.project_quality).cpu().numpy();stride=512//size;targetid=(pixels//512//stride)*size+pixels%512//stride;d64=np.bincount(targetid,weights=weights.astype(np.float64),minlength=size*size);assert np.array_equal(d64>0,qprod.cpu().numpy().ravel()>0)
     import cv2
     instances=b.read(ds.satellite_root/row['satellite_dir']/'instances.json')['instances'];mapping={int(z['local_instance_id']):int(z['global_instance_id']) for z in instances};rawids=cv2.imread(str(ds.satellite_root/row['satellite_dir']/'satellite_instance_id_clean.png'),cv2.IMREAD_UNCHANGED);localids=[int(z) for z in np.unique(rawids) if z];assert np.array_equal(np.stack([rawids==z for z in localids]),item['gt_masks'].astype(bool));assert np.array_equal(np.array([mapping[z] for z in localids]),previous['global_id'])
    valid=df>0;common=valid&valid.roll((32,32),(-2,-1));coarse128=F.interpolate(coarse,size=(128,128),mode='bilinear',align_corners=False);maps={'cvt32':coarse128*common,'cvt_shift32':coarse128.roll((32,32),(-2,-1))*common,'fine32':fine*common,'fine_shift32':fine.roll((32,32),(-2,-1))*common}
    flags=m.roi_head.mask_roi_extractor.roi_layers[0];manifest['actual_pipeline'].update(roi_aligned=bool(flags.aligned),roi_sampling_ratio=int(flags.sampling_ratio));roi=RoIAlign((16,16),spatial_scale=.25,sampling_ratio=int(flags.sampling_ratio),aligned=bool(flags.aligned));boxes=torch.from_numpy(previous['boxes']).cuda();rois=torch.cat([boxes.new_zeros(n,1),boxes],1)
    def flat(v):return roi(v,rois).permute(0,2,3,1).reshape(n,256,-1)
    features32={k:project(flat(v),1813) for k,v in maps.items()};quality=flat(qf*common).squeeze(-1);coverage=flat(cf*common).squeeze(-1);geometry=torch.from_numpy(previous['geometry']).cuda();geometry[...,-2]=quality;geometry[...,-1]=coverage
    arrays={k:previous[k] for k in REUSE};arrays.update({k:v.cpu().numpy().astype(np.float32) for k,v in features32.items()});arrays.update(geometry=geometry.cpu().numpy(),original_coverage=flat(cf).squeeze(-1).cpu().numpy(),common_coverage=coverage.cpu().numpy())
    for k in REUSE:assert np.array_equal(arrays[k],previous[k])
    for k,v in arrays.items():assert len(v)==n and np.isfinite(v).all(),k
    for k in ['sat32','sat_extra32','cvt32','cvt_shift32','fine32','fine_shift32']:assert arrays[k].shape==(n,256,32)
    assert arrays['geometry'].shape==(n,256,7)
    for k in ['prob','target','boundary_weight','original_coverage','common_coverage']:
     assert arrays[k].shape==(n,256) and ((arrays[k]>=0)&(arrays[k]<=1)).all(),k
    native=RoIAlign((16,16),spatial_scale=1/16,sampling_ratio=int(flags.sampling_ratio),aligned=bool(flags.aligned));route_delta=delta(roi(coarse128,rois),native(coarse,rois)) if args.smoke else None
    if args.smoke:
     off=m._extract_dual_feat(proc['inputs'],[sample]);assert feature_hash(off)==featurehash;del off
    assert bevsha==b.tensor_sha(bev) and before==b.payload_hash(sample.uav) and rgb==b.tensor_sha(sample.uav.rgb) and sat==b.tensor_sha(proc['inputs']);assert hookcounts==[(len(z._forward_hooks),len(z._forward_pre_hooks)) for z in m.modules()]
    seen[split].update(map(int,arrays['global_id']));assert not(seen['train']&seen['validation']);path=out/f'{split}_{iid}.npz';np.savez_compressed(path,**arrays)
    manifest['records'].append(dict(split=split,image_id=iid,n_rois=n,path=str(path),sha256=b.file_sha(path),source_cache_sha256=rec['sha256'],reused_arrays_exact=True,projection_payload_same=True,common_fine_support=True,hookon_off_features_exact=True if args.smoke else None,original_inputs_unchanged=True,hooks_removed=True,native_reconstruction_roi_delta=route_delta));b.dump(out/'run_manifest.json',manifest)
    if torch.cuda.max_memory_allocated()>20*1024**3:raise RuntimeError('Operational20GiB cap')
    if sum(p.stat().st_size for p in out.glob('*.npz'))>5*1024**3:raise RuntimeError('Operational5GiB cap')
    print(split,iid,n,'elapsed',round(time.monotonic()-started,1),flush=True)
    del previous,item,batch,imgs,samples,proc,sample,capture,bev,coarse,qc,cc,fine,qf,cf,df,valid,common,coarse128,maps,boxes,rois,features32,quality,coverage,geometry,arrays;torch.cuda.empty_cache()
 assert versions=={k:t._version for k,t in m.named_parameters()};manifest.update(complete=not args.smoke,parameter_versions_unchanged=True,processed_images=len(manifest['records']),n_rois=sum(r['n_rois'] for r in manifest['records']),split_global_ids={k:sorted(v) for k,v in seen.items()},visible_buildings_disjoint=True,elapsed_seconds=time.monotonic()-started,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated());b.dump(out/'run_manifest.json',manifest)
if __name__=='__main__':main()
