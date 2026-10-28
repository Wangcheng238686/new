"""Quantify existing bilinear mixing across height discontinuities; no training."""
import json,sys
from pathlib import Path
import numpy as np
import cv2
from pycocotools import mask as mu
cv2.setNumThreads(1)
r=Path('/data/wangcheng/checkpoint/ue_v1_adapter_probe_20261021/native_geometry/evaluation/correct')
# Locate the archived full-val output without guessing split membership.
root=Path('/data/wangcheng/checkpoint/ue_v1_adapter_probe_20261021/native_geometry')
candidates=[p for p in root.glob('evaluation/**/records.json') if len(json.loads(p.read_text()))==459]
assert candidates,candidates
rows=json.loads(candidates[0].read_text())
annotations=json.loads(Path('/data/wangcheng/dataset/CitySample_SatUAV_TrainMinimal_v1_20261021/annotations/val.json').read_text())
assert [x['image_id'] for x in rows]==[x['id'] for x in annotations['images']]
indices=np.linspace(0,458,8,dtype=int).tolist()
keys=['supported_samples','multiple_neighbor_samples','mixed_height_gt2m','mixed_height_gt5m','interpolated_height_off_nearest_gt1m']
total={region:{k:0 for k in keys} for region in ['all_roi_samples','gt_boundary2px_samples']};images=[]
for i in indices:
 row=rows[i];point=row['image_id'];cache=Path('/data/wangcheng/cache/ue_v1_native_geometry')/f'{point:06d}.npz'
 with np.load(cache) as d:height=d['fields'][0].astype(np.float64)*100;support=d['support'].astype(bool)
 gt=mu.decode(row['gt']['rles']);boundary=np.zeros((512,512),bool)
 for j in range(gt.shape[-1]):
  g=gt[:,:,j];boundary|=(cv2.dilate(g,np.ones((3,3),np.uint8),iterations=2)!=cv2.erode(g,np.ones((3,3),np.uint8),iterations=2,borderType=cv2.BORDER_CONSTANT,borderValue=0))
 b=np.asarray(row['dt']['bboxes']);line=(np.arange(128)+.5)/128
 y,x=np.meshgrid(line,line,indexing='ij');uv=b[:,None,None,:2]+np.stack((x,y),-1)[None]*(b[:,None,None,2:]-b[:,None,None,:2])
 px=uv[...,0]-.5;py=uv[...,1]-.5;x0=np.floor(px).astype(int);y0=np.floor(py).astype(int)
 values=[];valids=[];weights=[]
 for dy,dx in [(0,0),(0,1),(1,0),(1,1)]:
  xx=x0+dx;yy=y0+dy;inside=(xx>=0)&(xx<512)&(yy>=0)&(yy<512)
  xx=xx.clip(0,511);yy=yy.clip(0,511);valid=inside&support[yy,xx]
  w=(1-np.abs(px-(x0+dx)))*(1-np.abs(py-(y0+dy)))*valid
  values.append(height[yy,xx]);valids.append(valid);weights.append(w)
 values=np.stack(values);valids=np.stack(valids);weights=np.stack(weights)
 normalized_weights=weights/weights.sum(0).clip(1e-12)[None]
 material=valids&(normalized_weights>=.05)
 span=np.where(material,values,-np.inf).max(0)-np.where(material,values,np.inf).min(0)
 nearestx=np.floor(uv[...,0]).astype(int).clip(0,511);nearesty=np.floor(uv[...,1]).astype(int).clip(0,511)
 supported=support[nearesty,nearestx];multiple=material.sum(0)>=2
 interp=(values*weights).sum(0)/weights.sum(0).clip(1e-12)
 import torch
 import torch.nn.functional as F
 torch.set_num_threads(2)
 grid=torch.from_numpy(uv.reshape(1,-1,1,2)/256-1)
 image=torch.from_numpy(height)[None,None]
 sup=torch.from_numpy(support.astype(np.float64))[None,None]
 replay=(F.grid_sample(image,grid,align_corners=False)/F.grid_sample(sup,grid,align_corners=False).clamp_min(1e-12)).numpy().reshape(interp.shape)
 error=float(np.abs(replay-interp).max());assert error<1e-7,error
 events=dict(supported_samples=supported,multiple_neighbor_samples=supported&multiple,
  mixed_height_gt2m=supported&multiple&(span>2),mixed_height_gt5m=supported&multiple&(span>5),
  interpolated_height_off_nearest_gt1m=supported&(np.abs(interp-height[nearesty,nearestx])>1))
 stats={}
 for region,sel in [('all_roi_samples',np.ones_like(supported)),('gt_boundary2px_samples',boundary[nearesty,nearestx])]:
  stats[region]={k:int((v&sel).sum()) for k,v in events.items()}
  for k,v in stats[region].items():total[region][k]+=v
 images.append(dict(point=point,rois=len(b),counts=stats,numpy_vs_torch_max_error_m=error))
 print(point,stats,flush=True)
out=Path('/data/wangcheng/checkpoint/ue_native_sampling_audit_20261022');out.mkdir(parents=True,exist_ok=True)
(out/'summary.json').write_text(json.dumps(dict(selection='8 evenly spaced val indices, no AP selection',material_neighbor_definition='normalized bilinear weight>=.05; height span over material valid neighbors only',indices=indices,records_path=str(candidates[0]),images=images,counts=total,
 limitations='height-range is a discontinuity proxy, not satellite-camera depth or independently verified surface ID; GT band2px not Boundary AP; pooled ROI-grid samples repeat overlapping pixels'),indent=2)+'\n')
