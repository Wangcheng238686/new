"""Independent raw-CM camera recomputation of the selected native audit points."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'
import json,hashlib
from pathlib import Path
import cv2
import numpy as np
cv2.setNumThreads(1)
R=Path('/data/wangcheng/checkpoint/ue_native_source_audit_20261022')
D=Path('/data/wangcheng/dataset/CitySample_SatUAV_TrainMinimal_v1_20261021')
ROOT=Path(__file__).resolve().parents[2]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def k_of(c):
 a=c['intrinsics'];return np.array([[a['fx'],0,a['cx']],[0,a['fy'],a['cy']],[0,0,1]],float)
m=json.loads((R/'manifest.json').read_text());gen=np.random.Generator(np.random.PCG64(20261022));expected=[]
for split in ('train','val'):
 ids=sorted(json.loads((D/'splits'/f'{split}.json').read_text()))
 expected += [dict(split=split,image_id=int(i)) for i in gen.choice(ids,4,replace=False)]
assert expected==m['samples']
rows=[];max_world=max_uv=max_z=0.
for sample in expected:
 iid=sample['image_id'];folder=D/'points'/f'{iid:06d}';a=json.loads((R/f'{iid:06d}_points.json').read_text());c=json.loads((folder/'cameras.json').read_text())
 cache=Path('/data/wangcheng/cache/ue_v1_native_geometry')/f'{iid:06d}.npz'
 check=json.loads((R/f'{iid:06d}_cache_check.json').read_text());assert check['reference_kind']=='cached-reference' and check['sha256']==sha(cache)
 for rel,h in a['source_hashes'].items():assert sha(folder/rel)==h
 # Load reported arrays solely to cross-check equality; do not use their selected
 # rays to find the independently projected nearest surface.
 with np.load(R/'sources'/f'{iid:06d}.npz') as f:
  with np.load(cache) as old:
   assert np.array_equal(f['support'],old['support']) and np.array_equal(f['fields'],old['fields'])
   stored=old['fields'].copy();support=old['support'].copy()
 target=set()
 for p in a['points']:
  x,y=p['target_uv'];target.add(y*512+x)
  for item in p['gradient_sources'].values():
   if item['valid']:xx,yy=item['neighbor_target_uv'];target.add(yy*512+xx)
 candidates={px:[] for px in target};raws={};us,vs=np.meshgrid(np.arange(512),np.arange(512));us=us.ravel();vs=vs.ravel()
 sk=k_of(c['satellite']);se=np.array(c['satellite']['world_to_camera_cv'])
 for vi,yaw in enumerate((0,72,144,216,288)):
  raw=cv2.imread(str(folder/f'uav/yaw_{yaw:03d}/depth_camera_z.exr'),cv2.IMREAD_UNCHANGED)[:,:,2].astype(float);raws[vi]=raw
  z=raw.ravel();ok=np.isfinite(z)&(z>0)&(z<1000000);u,v,z=us[ok],vs[ok],z[ok]
  cam=c['uav'][f'yaw_{yaw:03d}'];k=k_of(cam);t=np.array(cam['camera_to_world_cv'])
  xyz=np.stack(((u-k[0,2])*z/k[0,0],(v-k[1,2])*z/k[1,1],z,np.ones_like(z)),1)
  world=xyz@t.T;st=world@se.T;proj=st[:,:3]@sk.T;uv=proj[:,:2]/proj[:,2:]
  inside=(st[:,2]>0)&np.isfinite(uv).all(1)&(uv>=0).all(1)&(uv<512).all(1)
  uv,st,world,u,v,z=(x[inside] for x in (uv,st,world,u,v,z));pix=uv[:,1].astype(int)*512+uv[:,0].astype(int)
  for px in target:
   ix=np.flatnonzero(pix==px)
   if len(ix):candidates[px].append(dict(view=vi,z=st[ix,2]/100,height=(world[ix,2]-c['center_ue_cm'][2])/100))
 for p in a['points']:
  x,y=p['target_uv'];px=y*512+x;r=p['height_ray'];vi=r['view'];u,v=r['source_uv'];z=raws[vi][v,u]
  assert abs(z/100-r['source_z'])<1e-12
  cam=c['uav'][f'yaw_{(0,72,144,216,288)[vi]:03d}'];k=k_of(cam);t=np.array(cam['camera_to_world_cv'])
  ray=np.array([(u-k[0,2])*z/k[0,0],(v-k[1,2])*z/k[1,1],z,1.])
  w=t@ray;st=se@w;proj=sk@st[:3];uv=proj[:2]/proj[2]
  max_world=max(max_world,float(np.abs(w[:3]/100-r['world']).max()));max_uv=max(max_uv,float(np.abs(uv-r['sat_uv']).max()));max_z=max(max_z,abs(st[2]/100-r['sat_z']))
  assert np.array_equal(np.floor(uv).astype(int),[x,y])
  zmin=min(float(q['z'].min()) for q in candidates[px]);assert abs(zmin-st[2]/100)<1e-8
  count=sum(bool((q['z']<=zmin+1+1e-9).any()) for q in candidates[px])
  assert count==p['near1m_distinct_views'] and np.float16(count/5)==stored[3,y,x]
  hc=np.float32((w[2]-c['center_ue_cm'][2])/100)
  assert np.float16(hc/np.float32(100))==stored[0,y,x]
  for axis,ch in [('dx',1),('dy',2)]:
   item=p['gradient_sources'][axis]
   if item['valid']:
    nx,ny=item['neighbor_target_uv'];nr=item['neighbor_ray'];nz=min(float(q['z'].min()) for q in candidates[ny*512+nx]);assert abs(nz-nr['sat_z'])<1e-8
    nvi=nr['view'];nu,nv=nr['source_uv'];nrawz=raws[nvi][nv,nu]
    ncam=c['uav'][f'yaw_{(0,72,144,216,288)[nvi]:03d}'];nk=k_of(ncam);nt=np.array(ncam['camera_to_world_cv'])
    nray=np.array([(nu-nk[0,2])*nrawz/nk[0,0],(nv-nk[1,2])*nrawz/nk[1,1],nrawz,1.]);nw=nt@nray
    assert abs((nw[2]-c['center_ue_cm'][2])/100-nr['height'])<1e-8
    height=np.float32((nw[2]-c['center_ue_cm'][2])/100);value=np.float16(np.clip(np.float32(height-hc)/np.float32(10),-5,5))
    assert value==stored[ch,y,x],(iid,p['group'],axis,value,stored[ch,y,x])
   else:assert stored[ch,y,x]==0
 rows.append(dict(image_id=iid,points=len(a['points']),original_sources_rehashed=True,cache_support_fields_exact=True,independent_nearest_surface_and_viewcount_passed=True))
 assert len(a['points'])==3
 print('ROOT_VERIFIED',iid,flush=True)
assert max_world<1e-8 and max_uv<1e-8 and max_z<1e-8
report=dict(images=8,points=24,sampling_manifest_exact=True,rows=rows,max_world_error_m=max_world,max_projection_error_px=max_uv,max_satellite_z_error_m=max_z,
 method='Independent raw-centimeter homogeneous transforms, per-selected-target reprojection of all raw UAV depth pixels; no model forward',
 original_cache_exact=True,source_script_sha256=m['source_sha256'],verifier_sha256=sha(__file__),
 visual_review=dict(contact_images_viewed_by_root=8,status='no independently measured cross-view landmarks; no certification of pixel correspondence',
 notes='Repeated textures, planar roofs, street backgrounds and facades prevent unique fixed-point landmark identification. Do not promote visual plausibility to exact correspondence.'),
 caveats=['same rendering matrices/depth, not external ground truth','8 trainval images do not represent dense test','nearest raw observations are not complete target-view visibility certification'])
(R/'root_verification.json').write_text(json.dumps(report,indent=2)+'\n');print('COMPLETE ROOT VERIFICATION',max_uv)
