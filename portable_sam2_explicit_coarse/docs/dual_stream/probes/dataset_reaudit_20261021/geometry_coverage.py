import os
os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'
import cv2,json,hashlib
from pathlib import Path
import numpy as np
from pycocotools import mask as mu
from collections import defaultdict
R=Path('/data/wangcheng/dataset/CitySample_SatUAV_TrainMinimal_v1_20261021');O=Path('/tmp/ue_dataset_reaudit_20261021')
old=json.loads(Path('/tmp/ue_uav_experiments/20261014_ad_seed42/dataset/manifest.json').read_text());by={r['index']:r for r in old['points']}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
rows=[];k3=np.ones((3,3),np.uint8)
for split in ['train','val']:
 c=json.loads((R/'annotations'/f'{split}.json').read_text());anns=defaultdict(list)
 for a in c['annotations']:anns[a['image_id']].append(a)
 ids=json.loads((R/'splits'/f'{split}.json').read_text());ids=sorted(ids,key=lambda i:hashlib.sha256(f'dataset-geometry-20261021:{split}:{i}'.encode()).hexdigest())[:16]
 for iid in ids:
  p=R/'points'/f'{iid:06d}';cam=json.loads((p/'cameras.json').read_text());sat=cam['satellite'];sK=sat['intrinsics'];sT=np.array(sat['world_to_camera_cv']);zbuf=np.full(512*512,np.inf);points=[];hashrows=[];validfra=[];cmerr=0
  yy,xx=np.mgrid[:512,:512]
  for yaw in cam['uav_yaw_order']:
   folder=p/'uav'/f'yaw_{yaw:03d}';d=cv2.imread(str(folder/'depth_camera_z.exr'),-1);d=d[:,:,2] if d.ndim==3 else d
   u=cam['uav'][f'yaw_{yaw:03d}'];K=u['intrinsics'];T=np.array(u['camera_to_world_cv']);valid=np.isfinite(d)&(d>0)&(d<1e6);validfra.append(float(valid.mean()));z=d[valid].astype(np.float64)
   local=np.stack([(xx[valid]-K['cx'])*z/K['fx'],(yy[valid]-K['cy'])*z/K['fy'],z],1)
   world=local@T[:3,:3].T+T[:3,3];slocal=world@sT[:3,:3].T+sT[:3,3]
   uv=np.stack([sK['fx']*slocal[:,0]/slocal[:,2]+sK['cx'],sK['fy']*slocal[:,1]/slocal[:,2]+sK['cy']],1)
   ins=(slocal[:,2]>0)&(uv[:,0]>=-.5)&(uv[:,0]<511.5)&(uv[:,1]>=-.5)&(uv[:,1]<511.5)
   uv=np.floor(uv[ins]+.5).astype(int);pix=uv[:,1]*512+uv[:,0];zz=slocal[ins,2];np.minimum.at(zbuf,pix,zz);points.append((pix,zz))
   if iid in by:
    prior=Path(old['uav_root'])/by[iid]['uav_dir']/'oblique'/f'view_yaw_{yaw:03d}'
    if prior.is_dir():
     oc=json.loads((prior/'camera_extrinsics.json').read_text());ki=json.loads((prior/'camera_intrinsics.json').read_text());cmerr=max(cmerr,float(np.max(np.abs(np.array(oc['camera_to_world_cv'])-T))),max(abs(ki[k]-K[k]) for k in ('fx','fy','cx','cy')))
     hashrows.append(dict(yaw=yaw,rgb_same=sha(prior/'oblique_rgb.0000.png')==sha(folder/'rgb.png'),depth_same=sha(prior/'depth_camera_z.0000.exr')==sha(folder/'depth_camera_z.exr')))
  support=np.isfinite(zbuf).reshape(512,512);dil=cv2.dilate(support.astype(np.uint8),k3)>0
  viewcount=np.zeros(512*512,np.int32)
  for pix,z in points:
   agree=z<=zbuf[pix]+100 # within1m of nearest observed source surface
   viewcount[np.unique(pix[agree])]+=1
  boundary=np.zeros((512,512),bool);fg=np.zeros_like(boundary);covers=[]
  for ann in anns[iid]:
   a=ann['segmentation'];a=mu.frPyObjects(a,512,512) if isinstance(a['counts'],list) else a;m=mu.decode(a).astype(bool);fg|=m;boundary|=m&~(cv2.erode(m.astype(np.uint8),k3,borderType=cv2.BORDER_CONSTANT,borderValue=0)>0);covers.append(float((dil&m).sum()/max(m.sum(),1)))
  # Nearest observed surface only; not a true target-view visibility certification.
  z=zbuf.reshape(512,512);edges=np.zeros_like(boundary)
  for axis in (0,1):
   pair=support & np.roll(support,-1,axis=axis)
   with np.errstate(invalid='ignore'):gap=np.abs(z-np.roll(z,-1,axis=axis))
   e=pair&(gap>200);e[-1,:]=False if axis==0 else e[-1,:];e[:,-1]=False if axis==1 else e[:,-1];edges|=e
  ed=cv2.dilate(edges.astype(np.uint8),k3)>0;bd=boundary&dil
  rows.append(dict(index=iid,split=split,instances=len(covers),valid_depth_mean=float(np.mean(validfra)),
   image_support=float(support.mean()),fg_pixels=int(fg.sum()),fg_supported=int((fg&support).sum()),fg_supported_1px=int((fg&dil).sum()),
   boundary_pixels=int(boundary.sum()),boundary_supported=int((boundary&support).sum()),boundary_supported_1px=int(bd.sum()),
   boundary_near_2m_depth_step=int((bd&ed).sum()),supported_fg_views2=int((fg&(viewcount.reshape(512,512)>=2)).sum()),
   instance_cover50=int((np.array(covers)>=.5).sum()),old_source_comparison=hashrows,old_camera_max_abs=cmerr))
  print(split,iid,'fg_cover',round(float((fg&dil).sum()/max(fg.sum(),1)),3),'boundary_cover',round(float(bd.sum()/max(boundary.sum(),1)),3),flush=True)
summary=dict(protocol='hash-fixed16train+16val; no test labels/images read; raw full-resolution point projections; nearest UAV-observed surface only; 1px dilation is optimistic tolerance, not a calibration score; depth-step2m is geometric and not a semantic edge')
for split in ('train','val'):
 rr=[r for r in rows if r['split']==split];t={k:sum(r[k] for r in rr) for k in ('instances','fg_pixels','fg_supported','fg_supported_1px','boundary_pixels','boundary_supported','boundary_supported_1px','boundary_near_2m_depth_step','supported_fg_views2','instance_cover50')};t.update(images=len(rr),fg_coverage=t['fg_supported']/t['fg_pixels'],fg_coverage_1px=t['fg_supported_1px']/t['fg_pixels'],boundary_coverage=t['boundary_supported']/t['boundary_pixels'],boundary_coverage_1px=t['boundary_supported_1px']/t['boundary_pixels'],supported_boundary_near_2m_step=t['boundary_near_2m_depth_step']/max(t['boundary_supported_1px'],1));summary[split]=t
comparisons=[c for r in rows for c in r['old_source_comparison']];summary['same_source_check']=dict(views=len(comparisons),rgb_same=sum(c['rgb_same'] for c in comparisons),depth_same=sum(c['depth_same'] for c in comparisons),camera_max_abs=max(r['old_camera_max_abs'] for r in rows))
(O/'geometry_coverage_rows.json').write_text(json.dumps(rows,indent=2)+'\n');(O/'geometry_coverage_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
