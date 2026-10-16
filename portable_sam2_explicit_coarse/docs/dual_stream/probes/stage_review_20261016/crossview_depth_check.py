import os
os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'
import json, numpy as np, cv2
from pathlib import Path
m=json.load(open('/tmp/ue_uav_experiments/20261014_ad_seed42/dataset/manifest.json')); ids=[1218,950,306,1216,948,505];out=[];metadata=[]
for iid in ids:
 row=next(r for r in m['points'] if r['index']==iid);base=Path(m['uav_root'])/row['uav_dir']/'oblique';views=[]
 for yaw in [0,72,144,216,288]:
  p=base/('view_yaw_%03d'%yaw);a=json.load(open(p/'camera_intrinsics.json'));b=json.load(open(p/'camera_extrinsics.json'));mta=json.load(open(p/'depth_scenecapture_metadata.json'));k=np.array([[a['fx'],0,a['cx']],[0,a['fy'],a['cy']],[0,0,1.]])
  e=np.array(b['world_to_camera_cv']);t=np.array(b['camera_to_world_cv']);e[:3,3]/=100;t[:3,3]/=100;d=cv2.imread(str(p/'depth_camera_z.0000.exr'),-1);d=(d[:,:,2] if d.ndim==3 else d).astype(float)/100
  pose=np.array([mta['camera_location_ue_cm'][v] for v in ['x','y','z']])/100;fov=2*np.degrees(np.arctan(256/a['fx']));metadata.append(dict(iid=iid,yaw=yaw,depth_pose_minus_camera_m=float(np.max(np.abs(pose-t[:3,3]))),fov_difference_deg=float(fov-mta['fov_x_deg']),capture_phase=mta.get('capture_phase'),controlled_plane_cm=mta.get('controlled_plane_distance_cm')))
  smooth=(cv2.dilate(d.astype('float32'),np.ones((3,3),np.uint8))-cv2.erode(d.astype('float32'),np.ones((3,3),np.uint8)))<1.
  views.append((k,e,t,d,smooth))
 for i,(k,e,t,d,sm) in enumerate(views):
  v,u=np.mgrid[8:504:16,8:504:16];z=d[v,u];cam=np.stack([(u-k[0,2])*z/k[0,0],(v-k[1,2])*z/k[1,1],z],axis=-1).reshape(-1,3);world=cam@t[:3,:3].T+t[:3,3]
  for j,(k2,e2,t2,d2,sm2) in enumerate(views):
   if i==j:continue
   cam2=world@e2[:3,:3].T+e2[:3,3];pix=cam2@k2.T;pix=pix[:,:2]/pix[:,2:];ij=np.rint(pix).astype(int);keep=(cam2[:,2]>0)&(ij[:,0]>=1)&(ij[:,0]<511)&(ij[:,1]>=1)&(ij[:,1]<511)&sm[v,u].ravel();ij=ij[keep];cz=cam2[keep,2];good=sm2[ij[:,1],ij[:,0]];err=cz[good]-d2[ij[good,1],ij[good,0]]
   out.append(dict(iid=iid,source=i,target=j,n=len(err),within_1m=int((np.abs(err)<=1).sum()),within_2m=int((np.abs(err)<=2).sum()),behind_2m=int((err>2).sum()),infront_2m=int((err< -2).sum()),median_signed_m=float(np.median(err)) if len(err) else None))
res=dict(protocol='six preexisting fixed images; directed independent-view depth reprojection; source grid16; round nearest target; both source/target local3x3 depth range<1m; report occlusion separately, no correspondence truth',pairs=out,metadata=metadata)
N=sum(x['n'] for x in out);res['aggregate']={k:sum(x[k] for x in out) for k in ['n','within_1m','within_2m','behind_2m','infront_2m']};res['aggregate']['within_1m_fraction']=res['aggregate']['within_1m']/N;res['aggregate']['within_2m_fraction']=res['aggregate']['within_2m']/N
Path('/tmp/ue_alignment_independent/crossview_depth.json').write_text(json.dumps(res,indent=2));print(json.dumps(res['aggregate']));print('pose_max',max(x['depth_pose_minus_camera_m'] for x in metadata),'fovmax',max(abs(x['fov_difference_deg']) for x in metadata))
