"""Read-only, CPU support/error audit of the fixed 459-image v1 test predictions."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[k]='1'
import json, hashlib, csv
from pathlib import Path
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
from pycocotools import mask as mu
cv2.setNumThreads(1)
ROOT=Path('/data/wangcheng/checkpoint/ue_v1_adapter_probe_20261021/test')
OUT=Path('/data/wangcheng/checkpoint/ue_test_weakness_20261022/support'); OUT.mkdir(parents=True,exist_ok=True)
CACHE=Path('/data/wangcheng/cache/ue_v1_local_adapter_geometry')
NATIVE=Path('/data/wangcheng/cache/ue_v1_native_geometry')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def decode(r):
 a=mu.decode(r['rles']); return a.astype(bool) if a.ndim==3 else a[:,:,None].astype(bool)
def bbox_iou(a,b):
 lo=np.maximum(a[:,None,:2],b[None,:,:2]);hi=np.minimum(a[:,None,2:],b[None,:,2:]);inter=np.maximum(hi-lo,0).prod(2)
 return inter/(np.maximum(a[:,2:]-a[:,:2],0).prod(1)[:,None]+np.maximum(b[:,2:]-b[:,:2],0).prod(1)[None,:]-inter+1e-9)
def frac(a,m):return float(a[m].mean()) if m.any() else None
paths={'baseline':ROOT/'baseline/disabled/records.json','native':ROOT/'native_geometry/correct/records.json','unit':ROOT/'unit/correct/records.json'}
rows={k:json.load(open(p)) for k,p in paths.items()}; assert all(len(v)==459 for v in rows.values())
per=[]; imgs=[]; cache_hash=hashlib.sha256();kernel=np.ones((3,3),np.uint8)
for j,base in enumerate(rows['baseline']):
 iid=base['image_id']; cp=CACHE/f'{iid:06d}.npz';npth=NATIVE/cp.name
 cache_hash.update((cp.name+sha(cp)+sha(npth)).encode())
 with np.load(cp) as a:idx=a['index'];geo=a['geom'].astype(np.float64)
 with np.load(npth) as a: native=a['support']; fields=a['fields'].astype(np.float32)
 valid=idx>=0;cell=np.broadcast_to(np.arange(16384)[:,None],idx.shape)[valid]
 uv=np.stack((cell%128*4+2,cell//128*4+2),1)+4*geo[valid,:2]
 uv=np.clip(uv,0,np.nextafter(512.,0));p=uv.astype(int)
 points=np.zeros((512,512),np.uint8);points[p[:,1],p[:,0]]=1
 near=cv2.dilate(points,kernel).astype(bool)
 grid=np.repeat(np.repeat(valid.any(1).reshape(128,128),4,0),4,1)
 nviews=np.stack([((idx//16384==v)&valid).any(1) for v in range(5)]).sum(0).reshape(128,128)
 multi=np.repeat(np.repeat(nviews>=2,4,0),4,1)
 gt=decode(base['gt']);ng=gt.shape[2];union=gt.any(2)
 inner=np.stack([cv2.erode(gt[:,:,g].astype(np.uint8),kernel,iterations=2,borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool) for g in range(ng)],2)
 boundary=(gt&~inner).any(2);interior=union&~boundary
 bb=np.array(base['gt']['bboxes']);db=np.array(base['dt']['bboxes'])
 bi=bbox_iou(bb,db);gi,di=linear_sum_assignment(1-bi); match=np.full(ng,-1,int);ok=bi[gi,di]>=.5;match[gi[ok]]=di[ok]
 result={};preds={}
 for name,rr in rows.items():
  r=rr[j]; assert r['image_id']==iid and r['gt']==base['gt']; assert r['dt']['bboxes']==base['dt']['bboxes'] and r['dt']['scores']==base['dt']['scores']
  dt=decode(r['dt']);preds[name]=dt.any(2);vals=np.full(ng,np.nan)
  for g in range(ng):
   if match[g]>=0:
    pm=dt[:,:,match[g]];gm=gt[:,:,g];vals[g]=(pm&gm).sum()/(pm|gm).sum()
  result[name]=vals
 im={'image_id':iid,'gt_instances':ng,'bbox_matched':int((match>=0).sum()),'native_gt_coverage':frac(native,union),'cell_gt_coverage':frac(grid,union),'retained_point_gt_coverage_approx':frac(points,union),'retained_1px_gt_coverage_approx':frac(near,union),'native_boundary_coverage':frac(native,boundary),'cell_boundary_coverage':frac(grid,boundary),'retained_1px_boundary_coverage_approx':frac(near,boundary),'ambiguous_rounded_anchor_coordinates':int((np.abs(uv-np.round(uv))<.001).any(1).sum()),'valid_candidates':int(valid.sum())}
 for zone,zmask in [('boundary2',boundary),('interior',interior)]:
  im[zone+'_pixels']=int(zmask.sum())
  for name,pm in preds.items():
   err=zmask&~pm;im[zone+'_'+name+'_FN']=int(err.sum())
   if name!='baseline':
    im[zone+'_'+name+'_corrected']=int((zmask&~preds['baseline']&pm).sum());im[zone+'_'+name+'_introduced']=int((zmask&preds['baseline']&~pm).sum())
  for typ,m in [('baseline_missed',zmask&~preds['baseline']),('baseline_correct',zmask&preds['baseline'])]:
   im[zone+'_'+typ+'_n']=int(m.sum());im[zone+'_'+typ+'_native_support_n']=int((m&native).sum());im[zone+'_'+typ+'_cell_support_n']=int((m&grid).sum());im[zone+'_'+typ+'_retained1px_n']=int((m&near).sum())
 imgs.append(im)
 for g in range(ng):
  m=gt[:,:,g];b=m&~inner[:,:,g];a=int(m.sum())
  per.append(dict(image_id=iid,instance=g,true_mask_area=a,area_group='small' if a<1024 else 'medium' if a<9216 else 'large',density_group='0-50' if ng<=50 else '51-100' if ng<=100 else '>100',bbox_matched=bool(match[g]>=0),native_coverage=frac(native,m),native_boundary_coverage=frac(native,b),cell_coverage=frac(grid,m),cell_boundary_coverage=frac(grid,b),retained1px_coverage_approx=frac(near,m),retained1px_boundary_coverage_approx=frac(near,b),multi_view_cell_coverage=frac(multi,m),near_surface_2views_coverage=frac(fields[3]>=.399,m),**{k+'_mask_iou':None if np.isnan(v[g]) else float(v[g]) for k,v in result.items()}))
 if j%50==0:print(j,iid,flush=True)
with open(OUT/'instances.csv','w') as f:
 w=csv.DictWriter(f,per[0].keys());w.writeheader();w.writerows(per)
json.dump(imgs,open(OUT/'images.json','w'),indent=2)
def describe(rr):
 out={'instances':len(rr),'bbox_unmatched':sum(not r['bbox_matched'] for r in rr)};matched=[r for r in rr if r['bbox_matched']]
 for key in ('native_coverage','native_boundary_coverage','cell_coverage','cell_boundary_coverage','retained1px_coverage_approx','retained1px_boundary_coverage_approx','multi_view_cell_coverage','near_surface_2views_coverage'):out[key]=float(np.mean([r[key] for r in rr if r[key] is not None]))
 out['matched_instances']=len(matched)
 for k in rows:out[k+'_mean_matched_mask_iou']=float(np.mean([r[k+'_mask_iou'] for r in matched]));out[k+'_matched_below_075']=sum(r[k+'_mask_iou']<.75 for r in matched)
 for k in ('native','unit'):
  d=np.array([r[k+'_mask_iou']-r['baseline_mask_iou'] for r in matched]);out[k+'_mean_paired_iou_delta']=float(d.mean());out[k+'_improved_001']=int((d>=.01).sum());out[k+'_worsened_001']=int((d<=-.01).sum())
 return out
summ={'all':describe(per),'native_coverage_groups':{},'true_mask_area_groups':{},'density_groups':{},'baseline_quality_groups':{}}
for name,lo,hi in [('<0.5',0,.5),('0.5-0.8',.5,.8),('>=0.8',.8,1.0001)]:summ['native_coverage_groups'][name]=describe([r for r in per if lo<=r['native_coverage']<hi])
for key,vals in [('area_group',['small','medium','large']),('density_group',['0-50','51-100','>100'])]:
 for v in vals:summ['true_mask_area_groups' if key=='area_group' else 'density_groups'][v]=describe([r for r in per if r[key]==v])
for name,pred in [('bbox_unmatched',lambda r:not r['bbox_matched']),('matched_iou<0.5',lambda r:r['bbox_matched'] and r['baseline_mask_iou']<.5),('matched_0.5-0.75',lambda r:r['bbox_matched'] and .5<=r['baseline_mask_iou']<.75),('matched_>=0.75',lambda r:r['bbox_matched'] and r['baseline_mask_iou']>=.75)]:
 rr=[r for r in per if pred(r)];summ['baseline_quality_groups'][name]={k:float(np.mean([r[k] for r in rr])) for k in ('native_coverage','cell_coverage','retained1px_coverage_approx','multi_view_cell_coverage')};summ['baseline_quality_groups'][name]['instances']=len(rr)
summ['pixel_totals']={key:sum(r[key] for r in imgs) for key in imgs[0] if key.startswith(('boundary2_','interior_'))}
summ['protocol']={'images':459,'record_sha256':{k:sha(p) for k,p in paths.items()},'script_sha256':sha(__file__),'cache_ordered_digest_sha256':cache_hash.hexdigest(),'matching':'Hungarian bbox IoU, accept >=0.5; same assignment all models, no scores threshold; not AP','support':'Native nearest observed surface mask; retained coordinates reconstructed from float16 geom, approximate ±0.001 px; grid support expanded 4x4; no visibility certification','boundary':'Each independent instance GT minus 2px erosion; union overlaps removed from interior; FN against predicted foreground union (not instance separation)','grouping':'Fixed native coverage <.5/.5-.8/>=.8; true mask area <1024/<9216/else; GT density <=50/<=100/else; no optimal test cutpoints','no_inference_or_training':True}
json.dump(summ,open(OUT/'summary.json','w'),indent=2);print(json.dumps(summ['all'],indent=2),flush=True)
