"""Frozen, CPU-only source screen-correspondence diagnostic; no model imports."""
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/ue_roi_feasibility_20261016/mpl')
import sys,json,hashlib,collections
from pathlib import Path
import numpy as np,cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import importlib.util
ROOT=Path('/home/wangcheng2021/project/new-uav-dual/portable_sam2_explicit_coarse')
spec=importlib.util.spec_from_file_location('production_cameras',ROOT/'rsprompter/uav/cameras.py');cam=importlib.util.module_from_spec(spec);spec.loader.exec_module(cam)
from pycocotools import mask as mu
O=Path('/tmp/ue_roi_feasibility_20261016/uav_evidence')
manifestpath=Path('/tmp/ue_uav_experiments/20261014_ad_seed42/dataset/manifest.json')
def load(p):return json.load(open(p))
m=load(manifestpath);gt=collections.defaultdict(list)
for g in load('/tmp/ue_uav_experiments/20261014_ad_seed42/final_analysis/single_best/gt_records.json'):gt[g['image_id']].append(g)
sources={}
def register(p):
 p=Path(p);sources[str(p)]={'size':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
 return p
register(manifestpath);register(ROOT/'rsprompter/uav/cameras.py');register(O/'selected_cases.json');register(O/'PROTOCOL.md')
def rgb(p):return cv2.cvtColor(cv2.imread(str(register(p))),cv2.COLOR_BGR2RGB)
def depth(p):
 raw=cv2.imread(str(register(p)),cv2.IMREAD_UNCHANGED);d=raw[:,:,2] if raw.ndim==3 else raw
 assert d.shape==(512,512) and d.dtype==np.float32
 return d.astype(np.float64)/100

def camera(p):
 for f in ['camera_intrinsics.json','camera_extrinsics.json']:register(p/f)
 return cam.load_camera(p)

def contours(ax,masks,ls='solid',alpha=1):
 for j,mm in enumerate(masks):
  if mm.any() and not mm.all():ax.contour(mm,levels=[.5],colors=[['#00ffff','#ff50cf'][j]],linewidths=.9,linestyles=ls,alpha=alpha)
def draw(arrays,masks,neg,case,suffix,crops=False):
 fig,axs=plt.subplots(1,6,figsize=(24,4.4),layout='constrained')
 for j,ax in enumerate(axs):
  ax.imshow(arrays[j]);contours(ax,masks[j]);contours(ax,neg[j],'dotted',.75)
  if crops:
   yy,xx=np.where(masks[j][0]|masks[j][1]|neg[j][0]|neg[j][1])
   if len(xx):
    x0=max(0,xx.min()-24);x1=min(511,xx.max()+24);y0=max(0,yy.min()-24);y1=min(511,yy.max()+24)
    ax.set_xlim(x0,x1);ax.set_ylim(y1,y0)
  ax.set_title('Satellite' if j==0 else f'UAV yaw {[0,72,144,216,288][j-1]}');ax.set_axis_off()
 fig.suptitle(f"{case['image_id']} {case['category']} | cyan {case['global_ids'][0]}, magenta {case['global_ids'][1]} | solid: screen overlap; dotted: shifted negative\nNo satellite depth / UAV ID: projected support is not physical instance truth",fontsize=11)
 fig.savefig(O/f"case_{case['image_id']}_{suffix}.png",dpi=150);plt.close(fig)

result=[]
for case in load(O/'selected_cases.json'):
 iid=case['image_id'];r=next(r for r in m['points'] if r['index']==iid);sat=Path(m['satellite_root'])/r['satellite_dir'];uav=Path(m['uav_root'])/r['uav_dir']/'oblique';sk,se,st,sz=camera(sat)
 gm=[mu.decode(gt[iid][s]['segmentation']).astype(bool) for s in case['slots']]
 shift=32 if all(gt[iid][s]['bbox'][0]+gt[iid][s]['bbox'][2]+32<=512 for s in case['slots']) else -32
 nm=[]
 for mm in gm:
  nn=np.zeros_like(mm)
  if shift>0:nn[:,shift:]=mm[:,:-shift]
  else:nn[:,:shift]=mm[:,-shift:]
  nm.append(nn)
 risk=[]
 for mm in gm:
  n,l,stats,c=cv2.connectedComponentsWithStats(mm.astype('uint8'),8)
  areas=sorted(stats[1:,cv2.CC_STAT_AREA].tolist(),reverse=True)
  risk.append(dict(components8=n-1,component_areas=areas,secondary_pixels=sum(areas[1:]),caveat='Topology proxy only; connected is not confirmed clean'))
 arrays=[rgb(sat/'satellite_rgb.0000.png')];masks=[gm];neg=[nm];views=[];per=[];cells=[set(),set()]
 for yaw in [0,72,144,216,288]:
  v=uav/f'view_yaw_{yaw:03d}';k,e,t,size=camera(v);d=depth(v/'depth_camera_z.0000.exr');im=rgb(v/'oblique_rgb.0000.png');md=load(register(v/'depth_scenecapture_metadata.json'))
  vv,uu=np.mgrid[:512,:512];valid=np.isfinite(d)&(d>0)&(d<10000);world=cam.backproject(d[valid],uu[valid],vv[valid],k,t);pix,z=cam.project(world,sk,se);ij=np.rint(pix).astype(int);ok=(z>0)&(ij[:,0]>=0)&(ij[:,0]<512)&(ij[:,1]>=0)&(ij[:,1]<512)
  ui,vi=uu[valid],vv[valid];ws=[];ns=[];stats=[]
  for j in range(2):
   target=np.zeros(world.shape[0],bool);target[ok]=gm[j][ij[ok,1],ij[ok,0]];negative=np.zeros_like(target);negative[ok]=nm[j][ij[ok,1],ij[ok,0]]
   support=np.zeros((512,512),bool);support[vi[target],ui[target]]=True;nmask=np.zeros_like(support);nmask[vi[negative],ui[negative]]=True
   ws.append(support);ns.append(nmask)
   xy=np.floor((world[target,:2]-np.asarray(r['origin_m'][:2]))/4).astype(int);cells[j].update(map(tuple,xy.tolist()))
   stats.append(dict(source_pixels=int(target.sum()),shifted_pixels=int(negative.sum()),source_shift_jaccard=float(np.logical_and(target,negative).sum()/max(1,np.logical_or(target,negative).sum())),unique_satellite_pixels=len(set(map(tuple,ij[target].tolist())))))
  rp,rz=cam.project(world,k,e);roundtrip=float(np.max(np.abs(rp-np.column_stack([ui,vi]))))
  pose=np.array([md['camera_location_ue_cm'][x] for x in ['x','y','z']])/100
  per.append(dict(yaw=yaw,valid_depth_fraction=float(valid.mean()),roundtrip_max_px=roundtrip,depth_metadata_camera_pose_error_m=float(np.max(np.abs(pose-t[:3,3]))),depth_fov_error_deg=float(2*np.degrees(np.arctan(256/k[0,0]))-md['fov_x_deg']),targets=stats))
  arrays.append(im);masks.append(ws);neg.append(ns)
  smooth=(cv2.dilate(d.astype('float32'),np.ones((3,3),np.uint8))-cv2.erode(d.astype('float32'),np.ones((3,3),np.uint8)))<1
  views.append((k,e,t,d,smooth))
 cross=[]
 for i,(k,e,t,d,sm) in enumerate(views):
  yy,xx=np.mgrid[8:504:16,8:504:16];ww=cam.backproject(d[yy,xx].ravel(),xx.ravel(),yy.ravel(),k,t)
  for j,(k2,e2,t2,d2,sm2) in enumerate(views):
   if i==j:continue
   pp,zz=cam.project(ww,k2,e2);ij=np.rint(pp).astype(int);keep=(zz>0)&(ij[:,0]>=1)&(ij[:,0]<511)&(ij[:,1]>=1)&(ij[:,1]<511)&sm[yy,xx].ravel();ii=ij[keep];zz=zz[keep];good=sm2[ii[:,1],ii[:,0]];err=zz[good]-d2[ii[good,1],ii[good,0]]
   cross.append(dict(source=i,target=j,n=len(err),within2m=int((np.abs(err)<=2).sum()),behind2m=int((err>2).sum()),infront2m=int((err< -2).sum())))
 draw(arrays,masks,neg,case,'overview');draw(arrays,masks,neg,case,'context',True)
 cr=dict(case,nominal_shift_px=shift,topology_risks=risk,views=per,xy4m_screen_support_cells=[len(x) for x in cells],xy4m_shared_cells=len(cells[0]&cells[1]),crossview_depth=cross)
 result.append(cr);print(iid,[(v['yaw'],[x['source_pixels'] for x in v['targets']]) for v in per],flush=True)
(O/'summary.json').write_text(json.dumps(dict(protocol=str(O/'PROTOCOL.md'),cases=result,caveat='Screen overlap is not physical identity; no fit or model forward'),indent=2))
register(O/'run_probe.py');register(O/'freeze_cases.py');(O/'source_identity.json').write_text(json.dumps(sources,indent=2))
# Fixed descriptive aggregation and input provenance (no additional selection).
for f in ['/tmp/ue_adjacent_pair_20261016/outputs/pairs.json','/tmp/ue_adjacent_pair_20261016/outputs/instances.json','/tmp/ue_adjacent_pair_20261016/outputs/posthoc_sensitivity.json','/tmp/ue_uav_experiments/20261014_ad_seed42/final_analysis/single_best/gt_records.json']:
 register(f)
s=load(O/'summary.json')
for c in s['cases']:
 c['nonzero_support_views_by_target']=[sum(v['targets'][j]['source_pixels']>0 for v in c['views']) for j in range(2)]
 c['minmax_source_support_pixels_by_target']=[[min(v['targets'][j]['source_pixels'] for v in c['views']),max(v['targets'][j]['source_pixels'] for v in c['views'])] for j in range(2)]
 for v in c['views']:
  for j in range(2):
   v['targets'][j]['source_image_pixel_fraction']=v['targets'][j]['source_pixels']/(512*512)
   v['targets'][j]['screen_gt_pixel_fraction']=v['targets'][j]['unique_satellite_pixels']/float(mu.area(gt[c['image_id']][c['slots'][j]]['segmentation']))
xs=[x for c in s['cases'] for x in c['crossview_depth']]
s['crossview_aggregate']={k:sum(x[k] for x in xs) for k in ['n','within2m','behind2m','infront2m']}
s['max_roundtrip_px']=max(v['roundtrip_max_px'] for c in s['cases'] for v in c['views'])
s['max_depth_pose_error_m']=max(v['depth_metadata_camera_pose_error_m'] for c in s['cases'] for v in c['views'])
s['max_depth_fov_error_deg']=max(abs(v['depth_fov_error_deg']) for c in s['cases'] for v in c['views'])
s['selected_global_ids_distinct']=len(set(x for c in s['cases'] for x in c['global_ids']))
(O/'summary.json').write_text(json.dumps(s,indent=2));(O/'source_identity.json').write_text(json.dumps(sources,indent=2))
