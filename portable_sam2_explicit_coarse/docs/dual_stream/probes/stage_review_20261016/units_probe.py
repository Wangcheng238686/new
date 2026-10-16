import os
os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'
import json,hashlib
from pathlib import Path
import numpy as np,cv2
M=Path('/tmp/ue_uav_experiments/20261014_ad_seed42/dataset/manifest.json');m=json.loads(M.read_text());rows={r['index']:r for r in m['points']};ids=sorted(rows,key=lambda x:hashlib.sha256(('units-independent-20261016:'+str(x)).encode()).hexdigest())[:3]
def stats(a):
 a=np.asarray(a).ravel();a=a[np.isfinite(a)];return dict(n=len(a),p0=float(a.min()),p10=float(np.quantile(a,.1)),p50=float(np.median(a)),p90=float(np.quantile(a,.9)),p100=float(a.max()))
def camera(p):
 intr=json.loads((p/'camera_intrinsics.json').read_text());ex=json.loads((p/'camera_extrinsics.json').read_text());k=np.array([[intr['fx'],0,intr['cx']],[0,intr['fy'],intr['cy']],[0,0,1.]]);e=np.array(ex['world_to_camera_cv']);t=np.array(ex['camera_to_world_cv']);e[:3,3]/=100;t[:3,3]/=100;return intr,k,e,t
records=[]
for iid in ids:
 r=rows[iid];s=Path(m['satellite_root'])/r['satellite_dir'];u=Path(m['uav_root'])/r['uav_dir'];si,sk,se,st=camera(s);o=np.array(r['origin_m']);record=dict(id=iid,satellite_intr=si,satellite_center_m=st[:3,3].tolist(),views=[])
 def sp(p):
  c=p@se[:3,:3].T+se[:3,3];q=c@sk.T;return q[:,:2]/q[:,2:]
 center=o[None].copy();p=sp(center);px=sp(center+np.array([4,0,0]));py=sp(center+np.array([0,4,0]));record['satellite_4m_ground_px']=[float(np.linalg.norm(px-p)),float(np.linalg.norm(py-p))]
 for yaw in [0,72,144,216,288]:
  v=u/'oblique'/f'view_yaw_{yaw:03d}';intr,k,e,t=camera(v);raw=cv2.imread(str(v/'depth_camera_z.0000.exr'),-1);d=raw[...,2] if raw.ndim==3 else raw;yy,xx=np.mgrid[4:512:8,4:512:8];z=d[yy,xx]/100.;good=np.isfinite(z)&(z>0)&(z<10000);z=z[good];x=xx[good];y=yy[good];q=np.column_stack(((x-k[0,2])/k[0,0],(y-k[1,2])/k[1,1],np.ones(len(z))));ray=q@t[:3,:3].T;world=ray*z[:,None]+t[:3,3]
  sizes=[]
  for du,dv in [(8,0),(0,8),(16,0),(0,16)]:
   q2=q+np.array([du/k[0,0],dv/k[1,1],0]);ray2=q2@t[:3,:3].T;z2=(world[:,2]-t[2,3])/ray2[:,2];xy2=ray2[:,:2]*z2[:,None]+t[:2,3];sz=np.linalg.norm(xy2-world[:,:2],axis=1);sizes.append(stats(sz[z2>0]))
  packed=v/'depth_devicez_packed.0000.png';decode=None
  if packed.exists():
   png=cv2.imread(str(packed),-1)[...,:3][...,::-1].astype(float)/255.;pck=png[...,0]+png[...,1]/255+png[...,2]/65025;dz=1-pck;calc=10/dz;common=np.isfinite(d)&np.isfinite(calc)&(dz>1e-9);decode=dict(max_abs_cm=float(abs(calc[common]-d[common]).max()),median_ratio=float(np.median(calc[common]/d[common])))
  record['views'].append(dict(yaw=yaw,intr=intr,center_m=t[:3,3].tolist(),center_local_m=(t[:3,3]-o).tolist(),depth_raw_cm=stats(d[np.isfinite(d)&(d>0)]),sample_world_z_m=stats(world[:,2]),cameraZ_to_range_ratio=stats(np.linalg.norm(q,axis=1)),horizontal_plane_strides_m=sizes,inverse_error=float(abs(e@t-np.eye(4)).max()),packed_near10_check=decode))
 records.append(record)
out=dict(ids=ids,records=records,manifest_sha256=hashlib.sha256(M.read_bytes()).hexdigest());Path('/tmp/ue_units_independent_probe.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
