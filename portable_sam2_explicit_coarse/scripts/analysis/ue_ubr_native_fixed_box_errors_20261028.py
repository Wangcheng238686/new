"""Fixed-C-box paired mask errors; assignments independent of mask predictions.

Single-image bbox score-desc stable greedy one-to-one GT matching at IoU>=.5.
Uses original512 GT instance mask; boundary region=dilate14 minus erode14,
interior=erode14, background=(not dilate14)&(not GTunion), other-instance
region=(not dilate14)&GTunion. Per-instance pixel populations overlap
between DIFFERENT instances and are not unique image pixels. Unmatched GT and
predictions are counted, excluded from matched pixel denominators.
"""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from pycocotools import mask as mu
cv2.setNumThreads(1)


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()

def ious(a,b):
 a=np.asarray(a,np.float64).reshape(-1,4);b=np.asarray(b,np.float64).reshape(-1,4)
 inter=np.maximum(0,np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2])).prod(-1)
 union=(a[:,2:]-a[:,:2]).prod(-1)[:,None]+(b[:,2:]-b[:,:2]).prod(-1)[None]-inter
 return inter/np.maximum(union,1e-12)


def assignment(row):
 scores=np.asarray(row['dt']['scores']);quality=ious(row['dt']['bboxes'],row['gt']['bboxes']);used=set();matched=[];unmatched=[]
 for index in np.argsort(-scores,kind='stable'):
  value=quality[index].copy()
  if used:value[list(used)]=-1
  gt=int(np.argmax(value)) if len(value) else -1
  if gt>=0 and value[gt]>=.5:used.add(gt);matched.append((int(index),gt,float(value[gt])))
  else:unmatched.append(int(index))
 return matched,unmatched,[i for i in range(len(row['gt']['bboxes'])) if i not in used]


def empty_region():return dict(pixels=0,baseline_errors=0,candidate_errors=0,corrected=0,damaged=0,changed=0)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--baseline-records',type=Path,required=True);p.add_argument('--candidate-records',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--label',required=True);a=p.parse_args()
 assert not a.output.exists(),'Preserve completed analysis'
 before=json.loads(a.baseline_records.read_text());after=json.loads(a.candidate_records.read_text());assert len(before)==len(after)==459
 evidence={str(x.resolve()):sha(x) for x in [a.baseline_records,a.candidate_records,Path(__file__)]}
 result=dict(label=a.label,complete=False,images=459,matched_instances=0,unmatched_predictions=0,unmatched_GT=0,instances_predicted=0,instances_GT=0,
  regions={name:empty_region() for name in ['boundary','interior','background','other_instance']},mask_IoU=dict(baseline_sum=0.,candidate_sum=0.,improved=0,worsened=0,unchanged=0),
  bbox_matching='fixed C boxes+scores; descending-score stable greedy one-to-one GT IoU>=.5; masks never influence assignment',
  region_definition='GT binary mask512x512; square3x3 kernel14 iterations; boundary=dilate14-erode14, interior=erode14, background=~dilate14 & ~imageGTunion, other_instance=~dilate14 & imageGTunion; constant0 outside image. This double-sided band differs from BoundaryAP inner boundary.',
  denominator_scope='matched instance-pixels; overlaps across different instances count repeatedly, not unique image-pixels; unmatched excluded from pixel comparison',
  evidence_sha256=evidence,per_image=[])
 for b,c in zip(before,after):
  assert b['image_id']==c['image_id'] and b['gt']==c['gt']
  assert b['dt']['bboxes']==c['dt']['bboxes'] and b['dt']['scores']==c['dt']['scores'],'Fixed C boxes/scores required'
  assert b['dt']['labels']==c['dt']['labels']
  matched,unmatched,missed=assignment(c);result['matched_instances']+=len(matched);result['unmatched_predictions']+=len(unmatched);result['unmatched_GT']+=len(missed)
  result['instances_predicted']+=len(c['dt']['bboxes']);result['instances_GT']+=len(c['gt']['bboxes'])
  counts={name:empty_region() for name in result['regions']}
  union=np.zeros((512,512),dtype=bool)
  for rle in c['gt']['rles']:union|=mu.decode(rle).astype(bool)
  for index,gtindex,bbox_iou in matched:
   target=mu.decode(c['gt']['rles'][gtindex]).astype(bool);old=mu.decode(b['dt']['rles'][index]).astype(bool);new=mu.decode(c['dt']['rles'][index]).astype(bool)
   assert target.shape==old.shape==new.shape==(512,512)
   array=target.astype(np.uint8);kernel=np.ones((3,3),np.uint8)
   dilation=cv2.dilate(array,kernel,iterations=14,borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)
   erosion=cv2.erode(array,kernel,iterations=14,borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)
   regions={'boundary':dilation&~erosion,'interior':erosion,'background':~dilation&~union,'other_instance':~dilation&union}
   assert all(not np.any(regions[x]&regions[y]) for i,x in enumerate(regions) for y in list(regions)[i+1:])
   assert np.all(np.logical_or.reduce(list(regions.values())))
   error_old=old!=target;error_new=new!=target
   for name,region in regions.items():
    v=dict(pixels=int(region.sum()),baseline_errors=int((error_old&region).sum()),candidate_errors=int((error_new&region).sum()),corrected=int((error_old&~error_new&region).sum()),damaged=int((~error_old&error_new&region).sum()),changed=int(((old!=new)&region).sum()))
    assert v['baseline_errors']-v['candidate_errors']==v['corrected']-v['damaged']
    for key,value in v.items():counts[name][key]+=value;result['regions'][name][key]+=value
   old_iou=float((old&target).sum()/max(1,(old|target).sum()));new_iou=float((new&target).sum()/max(1,(new|target).sum()))
   result['mask_IoU']['baseline_sum']+=old_iou;result['mask_IoU']['candidate_sum']+=new_iou
   result['mask_IoU']['improved' if new_iou>old_iou else 'worsened' if new_iou<old_iou else 'unchanged']+=1
  result['per_image'].append(dict(image_id=c['image_id'],matched=len(matched),unmatched_predictions=len(unmatched),unmatched_GT=len(missed),regions=counts))
 n=result['matched_instances'];result['mask_IoU']['baseline_mean']=result['mask_IoU']['baseline_sum']/n if n else None;result['mask_IoU']['candidate_mean']=result['mask_IoU']['candidate_sum']/n if n else None
 for region in result['regions'].values():
  region['net_corrected']=region['corrected']-region['damaged'];region['baseline_error_rate']=region['baseline_errors']/region['pixels'] if region['pixels'] else None;region['candidate_error_rate']=region['candidate_errors']/region['pixels'] if region['pixels'] else None
 assert all(sha(path)==value for path,value in evidence.items());result['complete']=True;a.output.parent.mkdir(parents=True,exist_ok=True);tmp=a.output.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(a.output)
 print(json.dumps({k:v for k,v in result.items() if k!='per_image'}),flush=True)

if __name__=='__main__':main()
