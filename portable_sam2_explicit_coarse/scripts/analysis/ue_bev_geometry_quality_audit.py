import os
os.environ['RSPROMPTER_LIGHT_IMPORT']='1'
os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'
import sys,json,hashlib,time
from pathlib import Path
import numpy as np,cv2
ROOT=Path('/home/wangcheng2021/project/new-uav-dual/portable_sam2_explicit_coarse');sys.path.insert(0,str(ROOT))
from rsprompter.uav.cameras import load_camera,localize_camera,backproject,project
from rsprompter.uav.geometry import build_geometry
from rsprompter.uav.contract import geometry_signature
O=Path('/tmp/ue_bev_geometry_quality');M=Path('/tmp/ue_uav_experiments/20261014_ad_seed42/dataset/manifest.json');L=Path('/tmp/ue_uav_experiments/20261014_ad_seed42/dual/train_EXP-0006_pid745647.log')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
m=json.loads(M.read_text());assert sha(M)=='495841b2174242c87661e6b40dba2ddcca13d64b13798116efc8c933c679ca59';txt=L.read_text();cfg=json.JSONDecoder().raw_decode(txt[txt.index('{'):])[0]['geometry'];ids=sorted(m['splits']['validation'],key=lambda i:hashlib.sha256(('bev-geometry-quality-v1:'+str(i)).encode()).hexdigest())[:6]
rows={p['index']:p for p in m['points']};sources=[ROOT/'rsprompter/uav'/f'{n}.py' for n in ['geometry','surfaces','cameras','candidates','footprints','contract']];records=[];t0=time.monotonic();meta=dict(complete=False,manifest_sha256=sha(M),log_config=cfg,selected_ids=ids,script_sha256=sha(__file__),protocol_sha256=sha(O/'PREREGISTRATION.md'),source_sha256={str(p):sha(p) for p in sources},gt_used=False,gpu_used=False,records=records)
def dump(): (O/'summary.json').write_text(json.dumps(meta,indent=2))
def stats(a):
 return dict(n=len(a),mean=float(np.mean(a)),p50=float(np.quantile(a,.5)),p90=float(np.quantile(a,.9)),p99=float(np.quantile(a,.99)),gt1_fraction=float(np.mean(a>1)),gt4_fraction=float(np.mean(a>4)),gt8_fraction=float(np.mean(a>8))) if len(a) else dict(n=0)
def candidates(p,sel):
 ix=p['candidate_index'][sel];valid=ix>=0;safe=np.maximum(ix,0);delta=p['anchor_xyz'][safe]-p['reference_xyz'][sel,None];dz=np.abs(delta[...,2])[valid];xy=np.linalg.norm(delta[...,:2],axis=-1)[valid];core=(np.abs(delta[...,:2])<=2).all(-1)[valid];return dict(cells=int(sel.sum()),candidate_votes=int(valid.sum()),abs_dz_m=stats(dz),xy_m=stats(xy),height_compatible_fraction=float((dz<=2).mean()),xy_cell_core_fraction=float(core.mean()))
for iid in ids:
 r=rows[iid];origin=r['origin_m'];sat=Path(m['satellite_root'])/r['satellite_dir'];uav=Path(m['uav_root'])/r['uav_dir'];views=[uav/'oblique'/f'view_yaw_{yaw:03d}' for yaw in [0,72,144,216,288]];files=[p/f for p in [sat,*views] for f in ['camera_intrinsics.json','camera_extrinsics.json']]+[p/'depth_camera_z.0000.exr' for p in views];ident=[geometry_signature(cfg),list(map(float,origin)),m['split_audit']['floor_m']]
 for p in [sat,*views]:
  for f in ['camera_intrinsics.json','camera_extrinsics.json']:ident.append((p/f).read_text())
 for p in views:
  f=p/'depth_camera_z.0000.exr';s=f.stat();ident.append([str(f.resolve()),s.st_size,s.st_mtime_ns])
 key=hashlib.sha256(json.dumps(ident,sort_keys=True).encode()).hexdigest();cache=Path('/tmp/uav_dual_dev_data/cache')/f'{key}.npz';cams=[load_camera(p) for p in views];depths=[];pts=[];vi=[]
 for v,(p,cam) in enumerate(zip(views,cams)):
  raw=cv2.imread(str(p/'depth_camera_z.0000.exr'),cv2.IMREAD_UNCHANGED);z=(raw[...,2] if raw.ndim==3 else raw).astype(np.float64)/100;assert z.shape==(512,512);depths.append(z);k,e,tr,_=cam;_,tr=localize_camera(e,tr,origin);yy,xx=np.mgrid[2:512:4,2:512:4];zz=z[yy,xx];good=np.isfinite(zz)&(zz>0)&(zz<10000);p3=backproject(zz[good],xx[good],yy[good],k,tr);pts.append(p3);vi.append(np.full(len(p3),v,np.int64))
 p,prodstat=build_geometry(depths,cams,load_camera(sat),origin,cfg);cacheinfo=dict(path=str(cache),exists=cache.is_file(),recomputed=True)
 if cache.is_file():
  with np.load(cache) as old:
   assert set(old.files)==set(p);eq={k:bool(np.array_equal(old[k],p[k])) for k in p};assert all(eq.values()),eq
  cacheinfo.update(sha256=sha(cache),all_arrays_exact=True)
 points=np.concatenate(pts);viewsidx=np.concatenate(vi);ij=np.floor((points[:,:2]+256)/4).astype(np.int64);inside=((ij>=0)&(ij<128)).all(1);pp=points[inside];vvi=viewsidx[inside];ij=ij[inside];flat=ij[:,1]*128+ij[:,0];ref=p['reference_xyz'][flat].astype(np.float64);hv=p['height_valid'][flat];available=(p['candidate_index']>=0).any(1);projected=np.zeros(128*128,bool);projected[p['project_source']]=True
 # Reconstruct selected bin independently from raw point bins (reference float32 may straddle bin boundary).
 zb=np.floor(pp[:,2]/2).astype(np.int64);pairs,counts=np.unique(np.stack([flat,zb],1),axis=0,return_counts=True);top=np.full(128*128,np.iinfo(np.int64).min,np.int64);np.maximum.at(top,pairs[counts>=3,0],pairs[counts>=3,1]);selected=zb==top[flat];bins=np.bincount(pairs[:,0],minlength=128*128);qualbins=np.bincount(pairs[counts>=3,0],minlength=128*128)
 k,e,tr,_=load_camera(sat);e,tr=localize_camera(e,tr,origin);a,za=project(pp,k,e);bpoints=pp.copy();bpoints[:,2]=ref[:,2];bp,zbdepth=project(bpoints,k,e);c,zc=project(ref,k,e);common=hv&np.isfinite(a).all(1)&np.isfinite(bp).all(1)&np.isfinite(c).all(1)&(za>0)&(zbdepth>0)&(zc>0);inframe=((a>=0)&(a<512)).all(1);drz=np.linalg.norm(bp-a,axis=1);dxy=np.linalg.norm(c-bp,axis=1);dt=np.linalg.norm(c-a,axis=1);group={}
 for name,s in [('valid_height_all',common),('valid_height_satframe',common&inframe),('selected_bin_all',common&selected),('selected_bin_satframe',common&selected&inframe)]:group[name]=dict(n=int(s.sum()),z_change_px=stats(drz[s]),xy_center_px=stats(dxy[s]),total_px=stats(dt[s]))
 observed=p['observation_support'];eligible=observed&p['height_valid']&available;record=dict(image_id=iid,source_files_sha256={str(f):sha(f) for f in files},cache=cacheinfo,production_stats=prodstat,point_counts=dict(raw_valid=len(points),in_bev=int(inside.sum()),outside_bev=int((~inside).sum()),valid_height=int(hv.sum()),invalid_height=int((~hv).sum()),selected_bin=int(selected.sum()),valid_height_candidate=int((hv&available[flat]).sum()),valid_height_projected=int((hv&projected[flat]).sum())),cell_counts=dict(observed=int(observed.sum()),height_valid=int(p['height_valid'].sum()),eligible=int(eligible.sum()),projected=int(projected.sum()),multibin_observed=int(((bins>1)&observed).sum()),multiqualifyingbin_observed=int(((qualbins>1)&observed).sum())),displacement=group,candidates_eligible=candidates(p,eligible),candidates_projected=candidates(p,projected))
 np.savez_compressed(O/f'raw_geometry_{iid}.npz',flat=flat,point_view=vvi,selected_bin=selected,height_valid=hv,common=common,in_satellite_frame=inframe,z_change_px=drz,xy_center_px=dxy,total_px=dt,cell_zbin_count=bins,cell_qualifying_zbin_count=qualbins);records.append(record);dump();print(iid,'done elapsed',time.monotonic()-t0,flush=True)
meta.update(complete=True,elapsed_seconds=time.monotonic()-t0);dump()
