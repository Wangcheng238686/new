import os
os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'
import json
from pathlib import Path
import numpy as np
import cv2
from pycocotools import mask as mu
O=Path('/tmp/ue_roi_feasibility_20261016');U=O/'uav_evidence'
def load(p):return json.load(open(p))
m=load('/tmp/ue_uav_experiments/20261014_ad_seed42/dataset/manifest.json');gt={}
for g in load('/tmp/ue_uav_experiments/20261014_ad_seed42/final_analysis/single_best/gt_records.json'):gt.setdefault(g['image_id'],[]).append(g)
def camera(p):
 q=load(p/'camera_intrinsics.json');z=load(p/'camera_extrinsics.json');k=np.array([[q['fx'],0,q['cx']],[0,q['fy'],q['cy']],[0,0,1.]])
 t=np.array(z['camera_to_world_cv']);e=np.array(z['world_to_camera_cv']);t[:3,3]/=100;e[:3,3]/=100
 return k,e,t
checks=0;crosschecks=0
for c in load(U/'summary.json')['cases']:
 iid=c['image_id'];rec=next(p for p in m['points'] if p['index']==iid);sk,se,st=camera(Path(m['satellite_root'])/rec['satellite_dir'])
 masks=[mu.decode(gt[iid][s]['segmentation']).astype(bool) for s in c['slots']];negative=[np.roll(mm,c['nominal_shift_px'],axis=1) for mm in masks]
 cells=[set(),set()];views=[]
 for v in c['views']:
  p=Path(m['uav_root'])/rec['uav_dir']/'oblique'/f"view_yaw_{v['yaw']:03d}";k,e,t=camera(p);raw=cv2.imread(str(p/'depth_camera_z.0000.exr'),cv2.IMREAD_UNCHANGED);d=(raw[:,:,2] if raw.ndim==3 else raw).astype('float64')/100
  assert np.isfinite(d).all() and (d>0).all() and (d<10000).all()
  yy,xx=np.indices(d.shape);pix=np.vstack([xx.ravel(),yy.ravel(),np.ones(d.size)]);xyz=(np.linalg.inv(k)@pix)*d.ravel();world=t@np.vstack([xyz,np.ones(d.size)]);sat=se@world;uv=sk@sat[:3];coords=np.rint(uv[:2]/uv[2]).astype(int);inside=(sat[2]>0)&(coords[0]>=0)&(coords[0]<512)&(coords[1]>=0)&(coords[1]<512)
  for j in (0,1):
   a=np.zeros(d.size,bool);b=a.copy();a[inside]=masks[j][coords[1,inside],coords[0,inside]];b[inside]=negative[j][coords[1,inside],coords[0,inside]]
   expected=v['targets'][j];assert int(a.sum())==expected['source_pixels'];assert int(b.sum())==expected['shifted_pixels'];assert abs((a&b).sum()/max(1,(a|b).sum())-expected['source_shift_jaccard'])<1e-12
   assert len(np.unique(coords[:,a].T,axis=0))==expected['unique_satellite_pixels'];checks+=4
   xy=np.floor((world[:2,a].T-np.array(rec['origin_m'][:2]))/4).astype(int);cells[j].update(map(tuple,xy.tolist()))
  sm=(cv2.dilate(d.astype('float32'),np.ones((3,3),np.uint8))-cv2.erode(d.astype('float32'),np.ones((3,3),np.uint8)))<1
  views.append((k,e,t,d,sm))
 assert [len(s) for s in cells]==c['xy4m_screen_support_cells'] and len(cells[0]&cells[1])==c['xy4m_shared_cells']
 for z in c['crossview_depth']:
  k,e,t,d,sm=views[z['source']];k2,e2,t2,d2,sm2=views[z['target']];yy,xx=np.mgrid[8:504:16,8:504:16];pix=np.vstack([xx.ravel(),yy.ravel(),np.ones(xx.size)]);world=t@np.vstack([(np.linalg.inv(k)@pix)*d[yy,xx].ravel(),np.ones(xx.size)]);q=e2@world;pp=k2@q[:3];uv=np.rint(pp[:2]/pp[2]).astype(int);valid=(q[2]>0)&(uv[0]>=1)&(uv[0]<511)&(uv[1]>=1)&(uv[1]<511)&sm[yy,xx].ravel();uv=uv[:,valid];zz=q[2,valid];valid2=sm2[uv[1],uv[0]];err=zz[valid2]-d2[uv[1,valid2],uv[0,valid2]]
  assert [len(err),int((abs(err)<=2).sum()),int((err>2).sum()),int((err < -2).sum())]==[z[k] for k in ('n','within2m','behind2m','infront2m')];crosschecks+=1
res=load(O/'review_recompute.json');res.update(uav_independent_support_checks=checks,uav_independent_crossview_pairs=crosschecks,uav_all_30_views_valid_depth=True,uav_all_six_xy4m_sets_reproduced=True)
(O/'review_recompute.json').write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))
