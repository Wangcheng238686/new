"""Frozen feature extraction ONLY; root approval required before execution."""
import argparse, hashlib, json, math, subprocess, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0,'/tmp')
import ue_rgb_content_probe as b
O=b.R/'uav_complementarity'

def streamsha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def matrix(c,seed,k=32):
 g=torch.Generator(device='cpu').manual_seed(seed);return torch.randn(c,k,generator=g)/math.sqrt(c)
def hashkey(i): return hashlib.sha256(('uav-complementarity-v1:'+str(i)).encode()).hexdigest()
def project(x,seed,k=32):
 w=matrix(x.shape[-1],seed,k);return x@w.to(x)
def main():
 p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');a=p.parse_args();torch.set_num_threads(2);torch.manual_seed(42);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 expected=b.read(b.R/'dual_correct/run_manifest.json');cp=torch.load(expected['checkpoint'],map_location='cpu',weights_only=False);snap=b._snapshot(cp);args=argparse.Namespace(sam2_ckpt=None,sam2_repo=None);b._resolve_sam2_repo(args,snap);cfg,_=b._resolve_model_config(args,snap);m=b._register_and_build(cfg);report=b._load_model_state(m,b._select_state_dict(cp,'ema'),allow_nonstrict=False);m.cuda().eval();m.uav_intervention='correct';del cp
 st=snap['data_config'];manifest=b.read(st['manifest']);assert b.file_sha(st['manifest'])==st['manifest_sha256']=='495841b2174242c87661e6b40dba2ddcca13d64b13798116efc8c933c679ca59';train=sorted(manifest['splits']['train'],key=hashkey)[:128];selected={'train':sorted(train),'validation':manifest['splits']['validation']};out=O/('smoke' if a.smoke else 'features');out.mkdir(exist_ok=True);versions={k:v._version for k,v in m.named_parameters()};started=time.monotonic();records=[]
 meta=dict(complete=False,script_sha256=b.file_sha(__file__),protocol_sha256=b.file_sha(O/'PREREGISTRATION.md'),checkpoint=expected['checkpoint'],checkpoint_sha256=streamsha(expected['checkpoint']),manifest_sha256=b.file_sha(st['manifest']),strict_load=report,weights='ema',training=False,optimizer=False,selected_image_ids=selected,projection_seeds=[1811,1812,1813],helper_sha256=streamsha(b.__file__),helper_path=b.__file__,architecture_contract=snap['architecture_contract'],git_dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=b.ROOT,text=True)),projection_matrices=[dict(seed=seed,in_dim=c,out_dim=32,sha256=b.tensor_sha(matrix(c,seed))) for c,seed in [(512,1811),(512,1812),(256,1813)]],source_git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=b.ROOT,text=True).strip(),scope='frozen dual-trained satellite path; not independent single-stream model',records=records)
 b.dump(out/'run_manifest.json',meta)
 with torch.inference_mode():
  for split in ['train','validation']:
   ds=b.UERGBDInstanceDataset(st['manifest'],split,st['geometry'],load_uav=True,cache_root='/tmp/uav_dual_dev_data/cache');wanted=set(selected[split]);rows=[(i,r) for i,r in enumerate(ds.samples) if r['index'] in wanted]
   if a.smoke:rows=rows[:3]
   for i,row in rows:
    if time.monotonic()-started>2100:raise RuntimeError('Operational 35min cap, incomplete not result-selected stop')
    item=ds[i];batch=b.ue_collate([item]);imgs,samples=b._build_data_samples(batch,torch.device('cuda'));proc=m.data_preprocessor(dict(inputs=imgs,data_samples=samples),training=False);sample=proc['data_samples'][0];capture={};payload_before=b.payload_hash(sample.uav);rgb_before=b.tensor_sha(sample.uav.rgb);sat_before=b.tensor_sha(proc['inputs']);hook_counts=(len(m.neck._forward_hooks),len(m.uav_bev._forward_hooks))
    def neckhook(mod,args,output):capture['neck']=tuple(t.clone() for t in output)
    def bevhook(mod,args,output):capture['bev']=output
    hn=m.neck.register_forward_hook(neckhook);hb=m.uav_bev.register_forward_hook(bevhook)
    try:m._extract_dual_feat(proc['inputs'],[sample])
    finally:hn.remove();hb.remove()
    assert hook_counts==(len(m.neck._forward_hooks),len(m.uav_bev._forward_hooks));source_hash=[b.tensor_sha(t) for t in (*capture['neck'],capture['bev'])]
    x=m.roi_head._inject_extra_pe(capture['neck']);boxes=item['gt_bboxes'].cuda();n=len(boxes);rois=torch.cat([boxes.new_zeros(n,1),boxes],1);extractor=m.roi_head.mask_roi_extractor;roi=extractor(x[:extractor.num_inputs],rois);head=m.roi_head.mask_head;assert head.shape_injector.fusion_type=='roi_only'
    rawcoarse=torch.cat([head.shape_injector.mask_decoder(roi[j:j+32]) for j in range(0,n,32)]);coarse=rawcoarse.sigmoid();target=head.get_coarse_targets([SimpleNamespace(pos_priors=boxes,pos_assigned_gt_inds=torch.arange(n,device='cuda'))],[sample.gt_instances],mask_size=(64,64))[:,None].float()
    assert coarse.shape==target.shape==(n,1,64,64)
    u,q,c=m.bev_projection(capture['bev'],[sample.uav],(128,128));layer=extractor.roi_layers[0];assert float(layer.spatial_scale)==.25;meta['actual_pipeline']=dict(roi_spatial_scale=float(layer.spatial_scale),roi_aligned=bool(layer.aligned),roi_sampling_ratio=int(layer.sampling_ratio),roi_output_size=list(layer.output_size),satellite_roi_shape=list(roi.shape[1:]),bev_shape=list(capture['bev'].shape[1:]),projected_uav_shape=list(u.shape[1:]),projection_hw=[128,128],coarse_shape=list(coarse.shape[1:]),shape_fusion_type=head.shape_injector.fusion_type,grid_hw=[8,8])
    ur=layer(u,rois);uxr=layer(u.roll((32,32),(-2,-1)),rois);qr=layer(q,rois);cr=layer(c,rois)
    def flat(t):return F.interpolate(t.float(),size=(8,8),mode='bilinear',align_corners=False).permute(0,2,3,1).reshape(n,64,-1)
    sat=flat(roi);uf=flat(ur);xf=flat(uxr);assert sat.shape[-1]==512 and uf.shape[-1]==xf.shape[-1]==256;s1=project(sat,1811);s2=project(sat,1812);ua=project(uf,1813);us=project(xf,1813)
    p8=F.adaptive_avg_pool2d(coarse,8).reshape(n,64);y8=F.adaptive_avg_pool2d(target,8).reshape(n,64);xy=torch.stack(torch.meshgrid((torch.arange(8,device='cuda')+.5)/8,(torch.arange(8,device='cuda')+.5)/8,indexing='ij'),-1).reshape(1,64,2).expand(n,-1,-1);wh=(boxes[:,2:]-boxes[:,:2]).clamp_min(1).log()[:,None].expand(-1,64,-1)
    geom=torch.cat([p8[...,None],xy,wh,flat(qr),flat(cr)],-1);assert geom.shape==(n,64,7)
    inst=b.read(ds.satellite_root/row['satellite_dir']/'instances.json')['instances'];mapping={int(z['local_instance_id']):int(z['global_instance_id']) for z in inst};import cv2
    ids=cv2.imread(str(ds.satellite_root/row['satellite_dir']/'satellite_instance_id_clean.png'),cv2.IMREAD_UNCHANGED);localids=[int(z) for z in np.unique(ids) if z];assert len(localids)==n and set(localids).issubset(mapping);recovered=np.stack([ids==z for z in localids]);assert np.array_equal(recovered,item['gt_masks'].astype(bool));gid=np.array([mapping[z] for z in localids],dtype=np.int64)
    noise=np.random.default_rng(int(hashkey(row['index'])[:8],16)).standard_normal((n,64,32)).astype(np.float32)
    arrays={k:v.cpu().numpy().astype(np.float32) for k,v in dict(sat32=s1,sat_extra32=s2,uav32=ua,shift32=us,geometry=geom,prob=p8,target=y8).items()};arrays.update(noise32=noise,global_id=gid,boxes=boxes.cpu().numpy(),image_id=np.full(n,row['index'],np.int64),gt_slot=np.arange(n,dtype=np.int64));path=out/f'{split}_{row["index"]}.npz';
    for key,value in arrays.items():assert len(value)==n and np.isfinite(value).all(),key
    for key in ['sat32','sat_extra32','uav32','shift32','noise32']:assert arrays[key].shape==(n,64,32),key
    assert arrays['geometry'].shape==(n,64,7) and arrays['prob'].shape==arrays['target'].shape==(n,64)
    assert len(set(gid))==n
    np.savez_compressed(path,**arrays)
    if a.smoke:
     observed=[]
     def hook(mod,args,output):observed.append(output.detach())
     h=head.shape_injector.mask_decoder.register_forward_hook(hook)
     try:
      m.uav_intervention='zero';features=m._extract_dual_feat(proc['inputs'],[sample]);b.predict_masks(m,features,sample,item['gt_bboxes'].numpy())
     finally:h.remove();m.uav_intervention='correct'
     assert torch.equal(torch.cat(observed),rawcoarse),'Direct coarse does not reproduce original zero-UAV head';del features,observed
    assert source_hash==[b.tensor_sha(t) for t in (*capture['neck'],capture['bev'])] and payload_before==b.payload_hash(sample.uav) and rgb_before==b.tensor_sha(sample.uav.rgb) and sat_before==b.tensor_sha(proc['inputs'])
    records.append(dict(source_features_unchanged=True,payload_rgb_satellite_unchanged=True,hooks_removed=True,split=split,image_id=row['index'],n_rois=n,path=str(path),sha256=b.file_sha(path),support_positive_rois=int((arrays['geometry'][:,:,-1].mean(1)>0).sum()),coarse_route='dual E72 satellite-only pre-UAV neck',smoke_coarse_exact=bool(a.smoke)))
    print(split,row['index'],n,'elapsed',round(time.monotonic()-started,1),flush=True);b.dump(out/'run_manifest.json',meta)
    del item,batch,imgs,samples,proc,sample,capture,x,boxes,rois,roi,rawcoarse,coarse,target,u,q,c,ur,uxr,qr,cr,sat,uf,xf,s1,s2,ua,us,p8,y8,xy,wh,geom,arrays;torch.cuda.empty_cache()
 assert versions=={k:v._version for k,v in m.named_parameters()};meta.update(complete=not a.smoke,parameter_versions_unchanged=True,processed_images=len(records),n_rois=sum(z['n_rois'] for z in records),elapsed_seconds=time.monotonic()-started);b.dump(out/'run_manifest.json',meta)
if __name__=='__main__':main()
