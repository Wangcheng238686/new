"""CPU-only production-normalized CVT geometry-metric maps. No model forward."""
import hashlib,json,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
ROOT=Path('/home/wangcheng2021/project/new-uav-dual/portable_sam2_explicit_coarse');sys.path.insert(0,str(ROOT))
from rsprompter.uav.projection import BEVToSatellite
O=Path('/tmp/ue_bev_tail_association');A=Path('/tmp/ue_uav_experiments/20261014_ad_seed42/final_analysis/bev_attention_audit/full');IDS=[1218,950,306,1216,948,505];SIZES=[32,64,512]
FIELDS=['weighted_abs_dz_m','uniform_weighted_abs_dz_m','height_mass','uniform_height_mass','xycore_mass','uniform_xycore_mass','joint_mass','uniform_joint_mass','weighted_signed_z_m','uniform_weighted_signed_z_m','weighted_xy_offset_norm_m','uniform_weighted_xy_offset_norm_m']
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def tensor_sha(t):
 a=t.detach().cpu().contiguous().numpy();return hashlib.sha256(str((a.dtype,a.shape)).encode()+a.tobytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2))
def main():
 started=time.monotonic();torch.set_num_threads(2);m=read(A/'run_manifest.json');g=read('/tmp/ue_bev_geometry_quality/summary.json');assert m['image_ids']==IDS==g['selected_ids'];assert m['complete'] and m['parameter_versions_unchanged'];gm={r['image_id']:r for r in g['records']};am={r['image_id']:r for r in m['records']}
 status=__import__('subprocess').check_output(['git','status','--porcelain'],cwd=ROOT,text=True);result=dict(complete=False,gpu_used=False,network_forward=False,training=False,fit=False,image_ids=IDS,source_sha256={str(p):sha(p) for p in (Path(__file__),O/'PREREGISTRATION.md',A/'run_manifest.json',Path('/tmp/ue_bev_geometry_quality/summary.json'),ROOT/'rsprompter/uav/projection.py')},attention_git_commit=m['git_commit'],mapping_git_commit=__import__('subprocess').check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),git_status_porcelain=status,manifest_sha256=m['manifest_sha256'],checkpoint_sha256=m['checkpoint_sha256'],checkpoint=m['checkpoint'],weights=m['weights'],sizes=SIZES,fields=FIELDS,numerical_protocol='v2 independent sequential FP32 scatter bitexact gate; FP64 deviation descriptive, original gate failure preserved',records=[])
 assert sha(ROOT/'rsprompter/uav/projection.py')==m['source_sha256'][str(ROOT/'rsprompter/uav/projection.py')]
 projection=BEVToSatellite(512).cpu().eval()
 with torch.inference_mode():
  for iid in IDS:
   cache=gm[iid]['cache'];assert sha(cache['path'])==cache['sha256']==am[iid]['geometry_cache']['sha256']
   with np.load(cache['path'],allow_pickle=False) as z:payload_arrays={k:z[k].copy() for k in z.files}
   for key,a in payload_arrays.items():assert tensor_sha(torch.from_numpy(a))==am[iid]['payload_sha256'][key],('Payload mismatch',iid,key)
   src=payload_arrays['project_source'];pix=payload_arrays['project_pixel'];mass=payload_arrays['project_mass'].astype(np.float64);quality=payload_arrays['project_quality'].astype(np.float64);weight=mass*quality;assert np.all(mass>=0) and np.all(quality>=0) and np.isfinite(weight).all()
   pp=SimpleNamespace(**{k:torch.from_numpy(v) for k,v in payload_arrays.items()});cache_before={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in payload_arrays.items()}
   for layer in (0,1):
    path=A/f'attention_{iid}_layer{layer}.npz';info=am[iid]['layers'][layer];assert sha(path)==info['npz_sha256']
    with np.load(path,allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    assert np.array_equal(a['reference_xyz'],payload_arrays['reference_xyz']);assert np.array_equal(a['height_valid'],payload_arrays['height_valid']);assert np.array_equal(a['observation_support'],payload_arrays['observation_support']);assert np.array_equal(a['has_candidate'],(payload_arrays['candidate_index']>=0).any(1));assert np.array_equal(np.flatnonzero(a['projected_mask']),np.unique(src))
    n=len(a['query_id']);assert n==16384;valid=a['has_candidate'];assert np.all(valid[src]) and np.all(a['height_valid'][src])
    # No-candidate query metrics are NaN upstream. Replace only unused no-candidate
    # feature positions by 0 for passing dense-grid production pooling; source
    # references are all valid and therefore never consume these placeholders.
    vectors=np.stack([np.where(valid,a[k],0) for k in FIELDS],1);assert np.isfinite(vectors).all();bev=torch.from_numpy(vectors.astype(np.float32).T.copy()).reshape(1,len(FIELDS),128,128)
    for size in SIZES:
     stride=512//size;target=(pix//512//stride)*size+(pix%512//stride);d=np.bincount(target,weights=weight,minlength=size*size);raw=np.bincount(target,weights=mass,minlength=size*size);missing=d==0;den=np.maximum(d,1e-6);mapped=np.stack([np.bincount(target,weights=weight*vectors[src,j],minlength=size*size)/den for j in range(len(FIELDS))]);qmap=d/np.maximum(raw,1e-6);coverage=np.clip(raw/(stride*stride),0,1)
     actual,actual_q,actual_c=projection(bev,[pp],(size,size));actual=actual[0].numpy().reshape(len(FIELDS),-1);actual_q=actual_q.numpy().ravel();actual_c=actual_c.numpy().ravel()
     # v2 semantic gate: independent sequential FP32 scatter, original entry
     # order equals per-target stable sorted order. np.add.at is unbuffered.
     ww32=(payload_arrays['project_mass']*payload_arrays['project_quality']).astype(np.float32)
     den32=np.zeros(size*size,dtype=np.float32);raw32=np.zeros_like(den32);num32=np.zeros((size*size,len(FIELDS)),dtype=np.float32)
     np.add.at(den32,target,ww32);np.add.at(raw32,target,payload_arrays['project_mass']);np.add.at(num32,target,(ww32[:,None]*vectors[src].astype(np.float32)).astype(np.float32))
     mirror=(num32/np.maximum(den32,np.float32(1e-6))[:,None]).T
     mirror_q=den32/np.maximum(raw32,np.float32(1e-6));mirror_c=np.clip(raw32/np.float32(stride*stride),0,1)
     assert np.array_equal(mirror,actual),('FP32 independent scatter mismatch',iid,layer,size,float(np.max(np.abs(mirror-actual))))
     assert np.array_equal(mirror_q,actual_q) and np.array_equal(mirror_c,actual_c),('FP32 quality/coverage mismatch',iid,layer,size)
     assert np.all(actual[:,missing]==0) and np.all(actual_q[missing]==0) and np.all(actual_c[raw==0]==0)
     data={k:mapped[j].reshape(size,size) for j,k in enumerate(FIELDS)};data.update(abs_dz_excess=data['weighted_abs_dz_m']-data['uniform_weighted_abs_dz_m'],height_incompatible_mass=1-data['height_mass'],uniform_height_incompatible_mass=1-data['uniform_height_mass'],height_mass_excess=data['height_mass']-data['uniform_height_mass'],xycore_mass_excess=data['xycore_mass']-data['uniform_xycore_mass'],joint_mass_excess=data['joint_mass']-data['uniform_joint_mass'])
     for k in data:data[k]=np.where(missing.reshape(size,size),np.nan,data[k])
     data.update(production_fp32_abs_dz_excess=np.where(missing.reshape(size,size),np.nan,(actual[FIELDS.index('weighted_abs_dz_m')]-actual[FIELDS.index('uniform_weighted_abs_dz_m')]).reshape(size,size)),production_fp32_height_incompatible_mass=np.where(missing.reshape(size,size),np.nan,(1-actual[FIELDS.index('height_mass')]).reshape(size,size)))
     data.update(denominator_mass_quality=d.reshape(size,size),raw_mass=raw.reshape(size,size),mean_quality=qmap.reshape(size,size),coverage=coverage.reshape(size,size),missing_mask=missing.reshape(size,size),vote_count=np.bincount(target,minlength=size*size).reshape(size,size))
     output=O/f'maps_{iid}_layer{layer}_s{size}.npz';np.savez_compressed(output,**data)
     result['records'].append(dict(image_id=iid,layer=layer,size=size,stride=stride,source_unique=int(np.unique(src).size),supported_targets=int((~missing).sum()),missing_targets=int(missing.sum()),raw_positive_missing_targets=int(((raw>0)&missing).sum()),raw_mass_sum=float(raw.sum()),mass_quality_sum=float(d.sum()),normalization_max_sum_error=float(np.max(np.abs(np.bincount(target,weights=weight/den[target],minlength=size*size)[~missing]-d[~missing]/den[~missing]))) if (~missing).any() else 0,fp32_semantics_bitexact=True,original_fp64_gate_pass=bool(np.allclose(mapped,actual,atol=2e-5,rtol=2e-6)),original_fp64_gate_failed_values=int((np.abs(mapped-actual)>(2e-5+2e-6*np.abs(actual))).sum()),production_pool_max_abs=float(np.max(np.abs(mapped-actual))),production_pool_max_relative=float(np.max(np.abs(mapped-actual)/np.maximum(np.abs(actual),1e-6))),production_quality_max_abs=float(np.max(np.abs(qmap-actual_q))),production_coverage_max_abs=float(np.max(np.abs(coverage-actual_c))),path=str(output),sha256=sha(output),attention_npz_sha256=sha(path),geometry_cache=cache))
    print('DONE',iid,layer,flush=True)
   assert cache_before=={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in payload_arrays.items()}
 result.update(complete=True,elapsed_seconds=time.monotonic()-started);dump(O/'mapping_manifest.json',result);print('DONE CPU MAPPING',flush=True)
if __name__=='__main__':main()
