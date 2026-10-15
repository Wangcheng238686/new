"""UNEXECUTED frozen stage-pair extractor; root smoke/full approval required."""
import argparse, hashlib, json, math, subprocess, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0,'/tmp')
import ue_rgb_content_probe as b
sys.path.insert(0,'/tmp/ue_stage_shape_probe')
from stage_pair_extraction_draft import candidate_uniform_query, StreamingCovariance
from mmcv.ops import RoIAlign
O=Path('/tmp/ue_stage_shape_probe')

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
def feature_hash(pack):
 x,e,p,h=pack
 return [b.tensor_sha(t) for t in list(x)+[e]+([p] if torch.is_tensor(p) else [])+list(h or ())]
def main():
 p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');a=p.parse_args();torch.set_num_threads(2);torch.manual_seed(42);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 expected=b.read(b.R/'dual_correct/run_manifest.json');cp=torch.load(expected['checkpoint'],map_location='cpu',weights_only=False);snap=b._snapshot(cp);args=argparse.Namespace(sam2_ckpt=None,sam2_repo=None);b._resolve_sam2_repo(args,snap);cfg,_=b._resolve_model_config(args,snap);m=b._register_and_build(cfg);report=b._load_model_state(m,b._select_state_dict(cp,'ema'),allow_nonstrict=False);m.cuda().eval();m.uav_intervention='correct';del cp
 st=snap['data_config'];manifest=b.read(st['manifest']);assert b.file_sha(st['manifest'])==st['manifest_sha256']=='495841b2174242c87661e6b40dba2ddcca13d64b13798116efc8c933c679ca59';train=sorted(manifest['splits']['train'],key=hashkey)[:128];selected={'train':sorted(train),'validation':manifest['splits']['validation']};out=O/('smoke' if a.smoke else 'features');out.mkdir(exist_ok=True);cov={k:StreamingCovariance() for k in ['uniform','cvt']};versions={k:v._version for k,v in m.named_parameters()};started=time.monotonic();records=[];visible_ids={'train':set(),'validation':set()}
 meta=dict(complete=False,source_sha256={str(p):b.file_sha(p) for p in [b.ROOT/'rsprompter/uav/cvt.py',b.ROOT/'rsprompter/uav/projection.py',b.ROOT/'rsprompter/detection/dual_detector.py']},script_sha256=b.file_sha(__file__),protocol_sha256=b.file_sha(O/'PREREGISTRATION_DRAFT.md'),checkpoint=expected['checkpoint'],checkpoint_sha256=streamsha(expected['checkpoint']),manifest_sha256=b.file_sha(st['manifest']),strict_load=report,weights='ema',training=False,optimizer=False,selected_image_ids=selected,projection_seeds=[1811,1812,1813],helper_sha256=streamsha(b.__file__),helper_path=b.__file__,architecture_contract=snap['architecture_contract'],git_dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=b.ROOT,text=True)),projection_matrices=[dict(seed=seed,in_dim=c,out_dim=32,sha256=b.tensor_sha(matrix(c,seed))) for c,seed in [(512,1811),(512,1812),(256,1813)]],stage_helper_sha256=b.file_sha('/tmp/ue_stage_shape_probe/stage_pair_extraction_draft.py'),source_git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=b.ROOT,text=True).strip(),scope='frozen dual-trained satellite path; not independent single-stream model',records=records)
 b.dump(out/'run_manifest.json',meta)
 with torch.inference_mode():
  for split in ['train','validation']:
   ds=b.UERGBDInstanceDataset(st['manifest'],split,st['geometry'],load_uav=True,cache_root='/tmp/uav_dual_dev_data/cache');wanted=set(selected[split]);rows=[(i,r) for i,r in enumerate(ds.samples) if r['index'] in wanted]
   if a.smoke:rows=rows[:3]
   for i,row in rows:
    if time.monotonic()-started>900:raise RuntimeError('Operational 15min cap; incomplete not result-selected stop')
    if torch.cuda.max_memory_allocated()>20*1024**3:raise RuntimeError('Operational peak VRAM20GiB cap')
    item=ds[i];batch=b.ue_collate([item]);imgs,samples=b._build_data_samples(batch,torch.device('cuda'));proc=m.data_preprocessor(dict(inputs=imgs,data_samples=samples),training=False);sample=proc['data_samples'][0];capture={};payload_before=b.payload_hash(sample.uav);rgb_before=b.tensor_sha(sample.uav.rgb);sat_before=b.tensor_sha(proc['inputs']);hook_counts=[(len(z._forward_hooks),len(z._forward_pre_hooks)) for z in m.modules()]
    def neckhook(mod,args,output):capture['neck']=tuple(t.clone() for t in output)
    def firsthook(mod,args):
     query,tokens,payload,chunk,checkpoint=args;capture['uniform']=candidate_uniform_query(tokens,payload)
     capture['candidate_hook_calls']=capture.get('candidate_hook_calls',0)+1
     if a.smoke:
      ix=payload.candidate_index;ids=torch.where((ix>=0).any(1))[0];ids=ids[torch.linspace(0,len(ids)-1,min(16,len(ids)),device=ids.device).round().long()] if len(ids) else ids
      for qid in ids:
       valid=ix[qid][ix[qid]>=0];ref=tokens[payload.anchor_token[valid]].detach().double().cpu().mean(0);got=capture['uniform'].flatten(2)[0,:,qid].detach().double().cpu();assert torch.allclose(got,ref,atol=1e-6,rtol=1e-5), 'Candidate equal mean mismatch'
    def bevhook(mod,args,output):capture['bev']=output
    hn=m.neck.register_forward_hook(neckhook);hb=m.uav_bev.register_forward_hook(bevhook);hf=m.uav_bev.layers[0].register_forward_pre_hook(firsthook)
    try:
     frozen_output=m._extract_dual_feat(proc['inputs'],[sample]);frozen_output_hash=feature_hash(frozen_output)
     del frozen_output
    finally:hn.remove();hb.remove();hf.remove()
    assert capture['candidate_hook_calls']==1;assert hook_counts==[(len(z._forward_hooks),len(z._forward_pre_hooks)) for z in m.modules()];source_hash=[b.tensor_sha(t) for t in (*capture['neck'],capture['bev'],capture['uniform'])]
    x=m.roi_head._inject_extra_pe(capture['neck']);boxes=item['gt_bboxes'].cuda();n=len(boxes);rois=torch.cat([boxes.new_zeros(n,1),boxes],1);extractor=m.roi_head.mask_roi_extractor;roi=extractor(x[:extractor.num_inputs],rois);head=m.roi_head.mask_head;assert head.shape_injector.fusion_type=='roi_only'
    rawcoarse=torch.cat([head.shape_injector.mask_decoder(roi[j:j+32]) for j in range(0,n,32)]);coarse=rawcoarse.sigmoid();target=head.get_coarse_targets([SimpleNamespace(pos_priors=boxes,pos_assigned_gt_inds=torch.arange(n,device='cuda'))],[sample.gt_instances],mask_size=(64,64))[:,None].float()
    assert coarse.shape==target.shape==(n,1,64,64)
    u,q,c=m.bev_projection(capture['bev'],[sample.uav],(32,32));g,qg,cg=m.bev_projection(capture['uniform'],[sample.uav],(32,32));assert torch.equal(q,qg) and torch.equal(c,cg)
    original_layer=extractor.roi_layers[0];layer=RoIAlign((16,16),spatial_scale=1/16,sampling_ratio=int(original_layer.sampling_ratio),aligned=bool(original_layer.aligned));meta['actual_pipeline']=dict(satellite_roi_shape=list(roi.shape[1:]),satellite_sampling='production multilevelRoI14 then bilinear16, original PE512',uav_roi_shape=[256,16,16],uav_spatial_scale=1/16,uav_roi_aligned=bool(layer.aligned),uav_roi_sampling_ratio=int(layer.sampling_ratio),bev_shape=[256,128,128],projection_hw=[32,32],grid_hw=[16,16],shift='postprojection32 roll8cells bothaxes; all4U/C maps and sharedq/c multiplied by common I=(D>0)&roll(D>0), no ROI removed',uniform='production candidate valid slots equal weight afterDepthCVT.source; not GTselected or unaggregated')
    original_q=q;original_c=c;valid_projection=(q>0);common=valid_projection&valid_projection.roll((8,8),(-2,-1));shift_u=u.roll((8,8),(-2,-1));shift_g=g.roll((8,8),(-2,-1));u=u*common;g=g*common;shift_u=shift_u*common;shift_g=shift_g*common;q=q*common;c=c*common
    ur=layer(u,rois);uxr=layer(shift_u,rois);gr=layer(g,rois);gxr=layer(shift_g,rois);qr=layer(q,rois);cr=layer(c,rois);original_cr=layer(original_c,rois)
    def flat(t):return F.interpolate(t.float(),size=(16,16),mode='bilinear',align_corners=False).permute(0,2,3,1).reshape(n,256,-1)
    sat=flat(roi);uf=flat(ur);xf=flat(uxr);gf=flat(gr);gxf=flat(gxr);assert sat.shape[-1]==512 and uf.shape[-1]==gf.shape[-1]==256
    s1=project(sat,1811);s2=project(sat,1812);ua=project(uf,1813);us=project(xf,1813);ga=project(gf,1813);gs=project(gxf,1813)
    if split=='train':cov['uniform'].add(gf);cov['cvt'].add(uf)
    p8=F.adaptive_avg_pool2d(coarse,16).reshape(n,256);y8=F.adaptive_avg_pool2d(target,16).reshape(n,256);xy=torch.stack(torch.meshgrid((torch.arange(16,device='cuda')+.5)/16,(torch.arange(16,device='cuda')+.5)/16,indexing='ij'),-1).reshape(1,256,2).expand(n,-1,-1);wh=(boxes[:,2:]-boxes[:,:2]).clamp_min(1).log()[:,None].expand(-1,256,-1)
    dilation=F.max_pool2d(F.pad(target,(3,3,3,3),value=0),7,stride=1);erosion=-F.max_pool2d(-F.pad(target,(3,3,3,3),value=0),7,stride=1);boundary16=F.adaptive_avg_pool2d((dilation!=erosion).float(),16).reshape(n,256)
    geom=torch.cat([p8[...,None],xy,wh,flat(qr),flat(cr)],-1);assert geom.shape==(n,256,7)
    inst=b.read(ds.satellite_root/row['satellite_dir']/'instances.json')['instances'];mapping={int(z['local_instance_id']):int(z['global_instance_id']) for z in inst};import cv2
    ids=cv2.imread(str(ds.satellite_root/row['satellite_dir']/'satellite_instance_id_clean.png'),cv2.IMREAD_UNCHANGED);localids=[int(z) for z in np.unique(ids) if z];assert len(localids)==n and set(localids).issubset(mapping);recovered=np.stack([ids==z for z in localids]);assert np.array_equal(recovered,item['gt_masks'].astype(bool));gid=np.array([mapping[z] for z in localids],dtype=np.int64)
    # No new noise cell; six fixed cells only.
    arrays={k:v.cpu().numpy().astype(np.float32) for k,v in dict(sat32=s1,sat_extra32=s2,cvt32=ua,cvt_shift32=us,uniform32=ga,uniform_shift32=gs,geometry=geom,prob=p8,target=y8,boundary_weight=boundary16,original_coverage=flat(original_cr).squeeze(-1),common_coverage=flat(cr).squeeze(-1)).items()};arrays.update(global_id=gid,boxes=boxes.cpu().numpy(),image_id=np.full(n,row['index'],np.int64),gt_slot=np.arange(n,dtype=np.int64));path=out/f'{split}_{row["index"]}.npz';
    for key,value in arrays.items():assert len(value)==n and np.isfinite(value).all(),key
    for key in ['sat32','sat_extra32','cvt32','cvt_shift32','uniform32','uniform_shift32']:assert arrays[key].shape==(n,256,32),key
    assert arrays['geometry'].shape==(n,256,7) and arrays['prob'].shape==arrays['target'].shape==(n,256)
    assert len(set(gid))==n
    visible_ids[split].update(map(int,gid));assert not(visible_ids['train']&visible_ids['validation'])
    np.savez_compressed(path,**arrays)
    if sum(p.stat().st_size for p in out.glob('*.npz'))>5*1024**3:raise RuntimeError('Operational5GiB cache cap')
    if torch.cuda.max_memory_allocated()>20*1024**3:raise RuntimeError('Operational peak VRAM20GiB cap')
    if a.smoke:
     off=m._extract_dual_feat(proc['inputs'],[sample]);assert frozen_output_hash==feature_hash(off), 'Read-only hooks perturbed correct features';del off
     observed=[]
     def hook(mod,args,output):observed.append(output.detach())
     h=head.shape_injector.mask_decoder.register_forward_hook(hook)
     try:
      m.uav_intervention='zero';features=m._extract_dual_feat(proc['inputs'],[sample]);b.predict_masks(m,features,sample,item['gt_bboxes'].numpy())
     finally:h.remove();m.uav_intervention='correct'
     assert torch.equal(torch.cat(observed),rawcoarse),'Direct coarse does not reproduce original zero-UAV head';del features,observed
    assert source_hash==[b.tensor_sha(t) for t in (*capture['neck'],capture['bev'],capture['uniform'])] and payload_before==b.payload_hash(sample.uav) and rgb_before==b.tensor_sha(sample.uav.rgb) and sat_before==b.tensor_sha(proc['inputs'])
    records.append(dict(source_features_unchanged=True,payload_rgb_satellite_unchanged=True,hooks_removed=True,split=split,image_id=row['index'],n_rois=n,path=str(path),sha256=b.file_sha(path),support_positive_rois=int((arrays['geometry'][:,:,-1].mean(1)>0).sum()),coarse_route='dual E72 satellite-only pre-UAV neck',smoke_coarse_exact=bool(a.smoke),smoke_hookon_off_features_exact=bool(a.smoke),uniform_cvt_projection_quality_coverage_exact=True))
    print(split,row['index'],n,'elapsed',round(time.monotonic()-started,1),flush=True);b.dump(out/'run_manifest.json',meta)
    del item,batch,imgs,samples,proc,sample,capture,x,boxes,rois,roi,rawcoarse,coarse,target,u,q,c,g,qg,cg,valid_projection,common,shift_u,shift_g,original_q,original_c,original_cr,ur,uxr,gr,gxr,qr,cr,sat,uf,xf,gf,gxf,s1,s2,ua,us,ga,gs,p8,y8,xy,wh,geom,arrays;torch.cuda.empty_cache()
 assert versions=={k:v._version for k,v in m.named_parameters()};meta.update(complete=not a.smoke,parameter_versions_unchanged=True,processed_images=len(records),n_rois=sum(z['n_rois'] for z in records),elapsed_seconds=time.monotonic()-started);meta.update(split_global_ids={k:sorted(v) for k,v in visible_ids.items()},visible_buildings_disjoint=True,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),train_full256_covariance={k:v.finish() for k,v in cov.items()});b.dump(out/'run_manifest.json',meta)
if __name__=='__main__':main()
