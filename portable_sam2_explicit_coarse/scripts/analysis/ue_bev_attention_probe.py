"""Read-only exact-semantic CVT attention reconstruction, six fixed images."""
import argparse, hashlib, math, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,'/tmp')
import ue_rgb_content_probe as base
import ue_gt_prompt_oracle_probe as oracle
O=base.R/'bev_attention_audit';IDS=[1218,950,306,1216,948,505]

def feature_hash(features):
 x,e,p,h=features;return [base.tensor_sha(t) for t in list(x)+[e]+([p] if torch.is_tensor(p) else [])+list(h or ())]

class AttentionObserver:
 def __init__(self,model,out,iid):self.model=model;self.out=out;self.iid=iid;self.reports=[];self.handles=[]
 def run(self,layer,args,layer_index):
  query,tokens,payload,chunk,use_checkpoint=args;assert layer.heads==8 and not layer.training and not torch.is_autocast_enabled()
  assert layer.geometry['height_bin_m']==2 and layer.geometry['side_m']/layer.geometry['grid_size']/2==2
  before=base.payload_hash(payload);qh=base.tensor_sha(query);th=base.tensor_sha(tokens)
  xyz,token=payload.anchor_xyz,payload.anchor_token
  if not len(xyz):xyz=tokens.new_zeros(1,3);token=torch.zeros(1,dtype=torch.long,device=tokens.device)
  scale=xyz.new_tensor([layer.geometry['side_m']/2,layer.geometry['side_m']/2,100.]);position=layer.position(xyz/scale);token_grid=(layer.geometry['image_size']//layer.geometry['source_stride'])**2;position=position+layer.view(token//token_grid)
  q=layer.query(layer.norm(query));keys=layer.key(tokens);values=layer.value(tokens);indices=payload.candidate_index;n=len(query);width=layer.channels//layer.heads;assert n==16384 and width==32;assert all(t.dtype==torch.float32 for t in (q,keys,values,position));arrays={};sample_parts=[];errors=[]
  # Complete grid, not the legacy 256-query training diagnostic sample.
  all_has=(indices>=0).any(1);projected_grid=torch.zeros(n,dtype=torch.bool,device=query.device);projected_grid[payload.project_source]=True
  groups={'projected':projected_grid,'height_valid_nonprojected':payload.height_valid&all_has&~projected_grid,'height_invalid_candidate':~payload.height_valid&all_has,'no_candidate':~all_has}
  picks=[];sample_group_counts={}
  for name,mask in groups.items():
   ids=torch.where(mask)[0];take=min(4,len(ids));sample_group_counts[name]=take
   if take:picks.append(ids[torch.linspace(0,len(ids)-1,take,device=query.device).round().long()])
  sample_ids=torch.unique(torch.cat(picks))
  for start in range(0,n,256):
   stop=min(start+256,n);selected=indices[start:stop];valid=selected>=0;has=valid.any(1);safe=selected.clamp_min(0);atoken=token[safe];key=keys[atoken]+position[safe];value=values[atoken];delta=xyz[safe]-payload.reference_xyz[start:stop,None]
   relative=(delta/delta.new_tensor([layer.geometry['radius_m'],layer.geometry['radius_m'],10.])).clamp(-10,10);distance=relative[...,:2].square().sum(-1,keepdim=True);bias=layer.relative(torch.cat([relative,distance],-1))-.5*distance;bias=bias.permute(0,2,1).unsqueeze(2);bias=bias.masked_fill(~valid[:,None,None],float('-inf'));bias[~has,:,:,0]=0;assert bias.dtype==torch.float32
   qq=q[start:stop].reshape(stop-start,8,1,width);kk=key.reshape(stop-start,selected.shape[1],8,width).transpose(1,2);vv=value.reshape(stop-start,selected.shape[1],8,width).transpose(1,2);logits=(qq*kk).sum(-1)/math.sqrt(width)+bias.squeeze(2);wh=logits.softmax(-1);weights=wh.mean(1)*has[:,None];assert torch.isfinite(weights).all();assert torch.allclose(weights.sum(1),has.float(),atol=2e-6,rtol=0)
   height=delta[...,2].abs()<=2;core=(delta[...,:2].abs()<=2).all(-1);joint=height&core;count=valid.sum(1);offset=(weights[...,None]*delta).sum(1)
   def mass(mask):return (weights*mask).sum(1)
   entropy_heads=-(wh*wh.clamp_min(1e-30).log()).sum(2)/count.clamp_min(2)[:,None].float().log();entropy_heads=torch.where(count[:,None]>1,entropy_heads,torch.zeros_like(entropy_heads));entropy=entropy_heads.mean(1)
   entropy_headmean=-(weights*weights.clamp_min(1e-30).log()).sum(1)/count.clamp_min(2).float().log();entropy_headmean=torch.where(count>1,entropy_headmean,torch.zeros_like(entropy_headmean))
   data=dict(candidate_count=count,height_candidate_count=(valid&height).sum(1),xycore_candidate_count=(valid&core).sum(1),joint_candidate_count=(valid&joint).sum(1),height_mass=mass(height),xycore_mass=mass(core),joint_mass=mass(joint),view_mass=torch.stack([mass(atoken//token_grid==v) for v in range(5)],1),entropy_normalized=entropy,entropy_per_head=entropy_heads,entropy_headmean_distribution=entropy_headmean,weighted_signed_z_m=offset[:,2],weighted_abs_dz_m=mass(delta[...,2].abs()),weighted_signed_xy_m=offset[:,:2],weighted_xy_offset_norm_m=offset[:,:2].norm(dim=1),weighted_abs_xy_distance_m=mass(delta[...,:2].norm(dim=2)))
   uniform=valid.float()/count.clamp_min(1)[:,None];uo=(uniform[...,None]*delta).sum(1)
   data.update(uniform_entropy_normalized=(count>1).float(),uniform_height_mass=(uniform*height).sum(1),uniform_xycore_mass=(uniform*core).sum(1),uniform_joint_mass=(uniform*joint).sum(1),uniform_view_mass=torch.stack([(uniform*(atoken//token_grid==v)).sum(1) for v in range(5)],1),uniform_weighted_signed_z_m=uo[:,2],uniform_weighted_abs_dz_m=(uniform*delta[...,2].abs()).sum(1),uniform_weighted_signed_xy_m=uo[:,:2],uniform_weighted_xy_offset_norm_m=uo[:,:2].norm(dim=1),uniform_weighted_abs_xy_distance_m=(uniform*delta[...,:2].norm(dim=2)).sum(1))
   data['view_dominance']=data['view_mass'].max(1).values;data['uniform_view_dominance']=data['uniform_view_mass'].max(1).values
   for k,v in data.items():
    if v.is_floating_point():v=v.masked_fill(~has.reshape(-1,*([1]*(v.ndim-1))),float('nan'))
    arrays.setdefault(k,[]).append(v.detach().cpu().numpy())
   local_sample=sample_ids[(sample_ids>=start)&(sample_ids<stop)]-start
   if len(local_sample):
    s=local_sample;actual=layer.attend(q[start:stop][s],keys,values,position,payload.reference_xyz[start:stop][s],xyz,token,selected[s]);manual=(wh[s,:,None,:]@vv[s]).reshape(len(s),layer.channels)*has[s,None];err=(manual-actual).abs();rel=(manual-actual).norm()/actual.norm().clamp_min(1e-6);assert (float(err.max())<=1e-5 or float(rel)<=1e-5),('SDPA reconstruction mismatch',self.iid,layer_index,float(err.max()))
    errors.append(dict(max_abs=float(err.max()),relative_l2=float(rel)));sample_parts.append(dict(query_id=(s+start).cpu().numpy(),q=qq[s].cpu().numpy(),k=kk[s].cpu().numpy(),v=vv[s].cpu().numpy(),bias=bias[s].cpu().numpy(),weights_heads=wh[s].cpu().numpy(),has_candidate=has[s].cpu().numpy(),candidate_index=selected[s].cpu().numpy(),delta=delta[s].cpu().numpy(),attend_actual=actual.cpu().numpy(),attend_manual=manual.cpu().numpy()))
  result={k:np.concatenate(v) for k,v in arrays.items()};source=payload.project_source.cpu().numpy();mass=payload.project_mass.cpu().numpy();quality=payload.project_quality.cpu().numpy();raw=np.bincount(source,weights=mass,minlength=n);weighted=np.bincount(source,weights=mass*quality,minlength=n);projected=np.bincount(source,minlength=n)>0;assert np.all((payload.height_valid.cpu().numpy()&all_has.cpu().numpy())[projected])
  result.update(query_id=np.arange(n),has_candidate=all_has.cpu().numpy(),observation_support=payload.observation_support.cpu().numpy(),height_valid=payload.height_valid.cpu().numpy(),projected_mask=projected,project_source_mass=raw,project_source_mass_quality=weighted,project_source_mean_quality=np.divide(weighted,raw,out=np.zeros_like(weighted),where=raw>0),project_source_vote_count=np.bincount(source,minlength=n),reference_xyz=payload.reference_xyz.cpu().numpy())
  count=result['candidate_count'];den=np.maximum(count,1)
  for key in ('height','xycore','joint'):result[key+'_candidate_fraction']=result[key+'_candidate_count']/den
  path=self.out/f'attention_{self.iid}_layer{layer_index}.npz';np.savez_compressed(path,**result)
  sample_path=self.out/f'sdpa_sample_{self.iid}_layer{layer_index}.npz';np.savez_compressed(sample_path,**{k:np.concatenate([s[k] for s in sample_parts]) for k in sample_parts[0]})
  assert before==base.payload_hash(payload) and qh==base.tensor_sha(query) and th==base.tensor_sha(tokens)
  self.reports.append(dict(layer=layer_index,queries=n,heads=8,chunk=256,projected_unique_queries=int(projected.sum()),npz_sha256=base.file_sha(path),sample_npz_sha256=base.file_sha(sample_path),sdpa_max_abs=max(x['max_abs'] for x in errors),sdpa_max_relative_l2=max(x['relative_l2'] for x in errors),sample_queries=sum(len(s['query_id']) for s in sample_parts),input_hashes_unchanged=True,sample_group_counts=sample_group_counts,reconstruction_dtype='float32'))
 def __enter__(self):
  for i,layer in enumerate(self.model.uav_bev.layers):self.handles.append(layer.register_forward_pre_hook(lambda layer,args,i=i:self.run(layer,args,i)))
  return self
 def __exit__(self,*unused):
  for h in self.handles:h.remove()


def main():
 p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');args=p.parse_args();started=time.monotonic();torch.set_num_threads(2);torch.manual_seed(42);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;out=O/('smoke' if args.smoke else 'full');out.mkdir(exist_ok=True)
 expected=base.read(base.O/'correct_replay/run_manifest.json');cp=torch.load(expected['checkpoint'],map_location='cpu',weights_only=False);snap=base._snapshot(cp);ra=argparse.Namespace(sam2_ckpt=None,sam2_repo=None);base._resolve_sam2_repo(ra,snap);cfg,_=base._resolve_model_config(ra,snap);model=base._register_and_build(cfg);report=base._load_model_state(model,base._select_state_dict(cp,'ema'),allow_nonstrict=False);model.cuda().eval();model.uav_intervention='correct';del cp;assert len(model.uav_bev.layers)==2
 state=snap['data_config'];ds=base.UERGBDInstanceDataset(state['manifest'],'validation',state['geometry'],load_uav=True,cache_root='/tmp/uav_dual_dev_data/cache');assert ds.manifest_sha256==expected['manifest_sha256'];indices={r['index']:i for i,r in enumerate(ds.samples)};ids=IDS[:1] if args.smoke else IDS;versions={k:t._version for k,t in model.named_parameters()};status=subprocess.check_output(['git','status','--porcelain'],cwd=base.ROOT,text=True)
 source_paths=[Path(__file__),Path(base.__file__),Path(oracle.__file__),Path(oracle.tracebase.__file__)]+[base.ROOT/'rsprompter/uav'/f for f in ('cvt.py','diagnostics.py','projection.py','geometry.py','contract.py','candidates.py','surfaces.py','footprints.py','cameras.py','fusion.py')]+[base.ROOT/'data'/f for f in ('ue_geometry_cache.py','ue_rgbd_dataset.py','ue_payload.py')]+[base.ROOT/'rsprompter/detection/dual_detector.py',Path('/tmp/ue_bev_geometry_quality/summary.json')]
 manifest=dict(checkpoint=expected['checkpoint'],checkpoint_sha256=oracle.streaming_sha(expected['checkpoint']),weights='ema',strict=True,load_report=report,training=False,optimizer_created=False,manifest_sha256=ds.manifest_sha256,preregistration_sha256=base.file_sha(O/'PREREGISTRATION.md'),source_sha256={str(p):base.file_sha(p) for p in source_paths},git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=base.ROOT,text=True).strip(),git_dirty=bool(status),git_status_porcelain=status,architecture_contract=snap['architecture_contract'],geometry=state['geometry'],image_ids=ids,scope='fixed six development-subset images, not full validation/test',complete=False,records=[])
 rawgeo=base.read('/tmp/ue_bev_geometry_quality/summary.json');assert rawgeo['selected_ids']==IDS and rawgeo['manifest_sha256']==ds.manifest_sha256;geom_records={r['image_id']:r for r in rawgeo['records']}
 base.dump(out/'run_manifest.json',manifest)
 with torch.inference_mode():
  for iid in ids:
   item=ds[indices[iid]];cache=geom_records[iid]['cache'];assert base.file_sha(cache['path'])==cache['sha256']
   with np.load(cache['path'],allow_pickle=False) as cached:
    assert all(torch.equal(item['uav_payload'][k],torch.from_numpy(cached[k])) for k in cached.files)
   payload_before=base.payload_hash(item['uav_payload']);rgb=base.tensor_sha(item['uav_payload']['rgb']);batch=base.ue_collate([item]);imgs,samples=base._build_data_samples(batch,torch.device('cuda'));processed=model.data_preprocessor(dict(inputs=imgs,data_samples=samples),training=False);imgs=processed['inputs'];sample=processed['data_samples'][0];hooks=[len(m._forward_hooks)+len(m._forward_pre_hooks) for m in model.modules()]
   with AttentionObserver(model,out,iid) as observer:features=model._extract_dual_feat(imgs,[sample])
   fh=feature_hash(features);assert hooks==[len(m._forward_hooks)+len(m._forward_pre_hooks) for m in model.modules()];off_exact=None
   if args.smoke:
    other=model._extract_dual_feat(imgs,[sample]);assert feature_hash(other)==fh;off_exact=True;del other
   assert base.payload_hash(sample.uav)==payload_before and base.tensor_sha(sample.uav.rgb)==rgb;assert len(observer.reports)==2
   manifest['records'].append(dict(image_id=iid,layers=observer.reports,hookon_off_features_exact=off_exact,feature_sha256=fh,payload_sha256=payload_before,rgb_sha256=rgb,hooks_removed=True,payload_unchanged=True,geometry_cache=cache,geometry_cache_arrays_exact=True));base.dump(out/'run_manifest.json',manifest);print('DONE IMAGE',iid,observer.reports,flush=True)
   del features,observer,item,batch,imgs,samples,processed,sample;torch.cuda.empty_cache()
 assert versions=={k:t._version for k,t in model.named_parameters()};manifest.update(complete=not args.smoke,smoke_complete=args.smoke,parameter_versions_unchanged=True,elapsed_seconds=time.monotonic()-started);base.dump(out/'run_manifest.json',manifest);print('DONE ATTENTION AUDIT',flush=True)
if __name__=='__main__':main()
