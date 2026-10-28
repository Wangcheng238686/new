"""Fixed 32-image prediction error audit, no AP selection or training."""
import sys,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/probes'))
import ue_s2db_analysis as analysis
import numpy as np
import cv2
from pycocotools import mask as mu
cv2.setNumThreads(1)
r=Path('/data/wangcheng/checkpoint/ue_native_spatial_diagnostic_20261022')
paths=dict(satellite=Path('/data/wangcheng/checkpoint/ue_v1_adapter_probe_20261021/test/baseline/disabled/records.json'))
paths.update({k:r/k/'records.json' for k in ['correct','zero','zero_rms']})
rows={k:json.loads(p.read_text()) for k,p in paths.items()}
indices=np.linspace(0,458,32,dtype=int).tolist();summary={};per_image=[]
for i in indices:
 b=rows['satellite'][i];pairs,ug,ud=analysis.bbox_matching(b['gt']['bboxes'],b['dt']['bboxes'])
 arrays={k:mu.decode(v[i]['dt']['rles']).astype(bool) for k,v in rows.items()}
 gt=mu.decode(b['gt']['rles']).astype(bool)
 per=dict(image_id=b['image_id'],gt=len(b['gt']['rles']),matched=len(pairs),unmatched_gt=ug,unmatched_dt=ud)
 for key in arrays:
  stats=summary.setdefault(key,dict(instances=0,iou_sum=0.,better_than_satellite=0,worse_than_satellite=0,zones={z:dict(pixels=0,errors=0,corrected=0,introduced=0) for z in ['boundary','interior','exterior']}))
  for gi,di in pairs:
   g=gt[:,:,gi];pred=arrays[key][:,:,di];old=arrays['satellite'][:,:,di]
   iou=lambda p:float((p&g).sum()/max(int((p|g).sum()),1))
   v=iou(pred);v0=iou(old);stats['instances']+=1;stats['iou_sum']+=v
   stats['better_than_satellite']+=int(v>v0+1e-12);stats['worse_than_satellite']+=int(v<v0-1e-12)
   inner=cv2.erode(g.astype('uint8'),np.ones((3,3),np.uint8),iterations=2,borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)
   outer=cv2.dilate(g.astype('uint8'),np.ones((3,3),np.uint8),iterations=2,borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)
   box=np.asarray([b['gt']['bboxes'][gi],b['dt']['bboxes'][di]])
   lo=np.maximum(np.floor(box[:,:2].min(0)).astype(int)-8,0);hi=np.minimum(np.ceil(box[:,2:].max(0)).astype(int)+8,512)
   region=np.zeros((512,512),bool);region[lo[1]:hi[1],lo[0]:hi[0]]=1
   for z,mask in dict(boundary=outer&~inner,interior=inner,exterior=~outer).items():
    mask=mask&region;err=pred!=g;old_err=old!=g;t=stats['zones'][z]
    t['pixels']+=int(mask.sum());t['errors']+=int((mask&err).sum());t['corrected']+=int((mask&old_err&~err).sum());t['introduced']+=int((mask&~old_err&err).sum())
 per_image.append(per);print('IMAGE',i,flush=True)
for v in summary.values():v['mean_matched_iou']=v['iou_sum']/v['instances']
out=Path('/data/wangcheng/checkpoint/ue_native_path_diagnostic_20261022');out.mkdir(parents=True,exist_ok=True)
(out/'matched_error_audit.json').write_text(json.dumps(dict(selection='32 evenly spaced test indices, no metric selection',indices=indices,images=per_image,results=summary,matching='Hungarian frozen bbox IoU>=.5',scope='matched instance pixels; overlap repeats pixels, misses excluded; morphology2px is not Boundary AP band',source_paths={k:str(v) for k,v in paths.items()}),indent=2)+'\n')
