import os
os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'
import cv2,json,hashlib
import numpy as np
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from pycocotools import mask as mu
from collections import defaultdict
O=Path('/tmp/ue_dataset_reaudit_20261021');N=Path('/data/wangcheng/dataset/CitySample_SatUAV_TrainMinimal_v1_20261021')
old=json.loads(Path('/tmp/ue_uav_experiments/20261014_ad_seed42/dataset/manifest.json').read_text())
by={r['index']:r for r in old['points']}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
newids={s:set(json.loads((N/'splits'/f'{s}.json').read_text())) for s in ('train','val','test')}
trans={s:{n:len(set(v)&ids) for n,ids in newids.items()} for s,v in old['splits'].items()}
rows=[]
for ns,osplit in [('train','train'),('val','validation')]:
 c=json.loads((N/'annotations'/f'{ns}.json').read_text());anns=defaultdict(list)
 for a in c['annotations']:anns[a['image_id']].append(a)
 selected=sorted(newids[ns]&set(old['splits'][osplit]))
 for index in selected:
  d=Path(old['satellite_root'])/by[index]['satellite_dir']; op=d/'satellite_instance_id_clean.png';im=cv2.imread(str(op),-1)
  rgb_same=sha(d/'satellite_rgb.0000.png')==sha(N/'points'/f'{index:06d}'/'satellite/rgb.png')
  if not rgb_same:
   rows.append(dict(index=index,old_split=osplit,new_split=ns,rgb_same=False));continue
  ids=np.unique(im);ids=ids[ids>0];oc=len(ids);nc=len(anns[index]);oa=np.array([(im==i).sum() for i in ids]);na=[];ints=[];fg=np.zeros(im.shape,bool)
  for ann in anns[index]:
   r=ann['segmentation'];r=mu.frPyObjects(r,512,512) if isinstance(r['counts'],list) else r;m=mu.decode(r).astype(bool);fg|=m;na.append(m.sum())
   counts=np.bincount(im[m].astype(np.int64),minlength=int(ids.max())+1);ints.append(counts[ids])
  inter=np.array(ints).T;na=np.array(na);union=oa[:,None]+na[None,:]-inter;iou=inter/np.maximum(union,1)
  oi,ni=linear_sum_assignment(-iou);v=iou[oi,ni];ofg=im>0
  # Conservative overlap graph diagnostics: each incident overlap>=32px and >=20% of the child's own area.
  one_old_many_new=((inter>=32)&(inter/np.maximum(na[None,:],1)>=.2)).sum(1)>=2
  many_old_one_new=((inter>=32)&(inter/np.maximum(oa[:,None],1)>=.2)).sum(0)>=2
  rows.append(dict(index=index,old_split=osplit,new_split=ns,rgb_same=True,old_count=oc,new_count=nc,
    old_fg=int(ofg.sum()),new_fg=int(fg.sum()),fg_intersection=int((ofg&fg).sum()),fg_union=int((ofg|fg).sum()),
    fg_added=int((fg&~ofg).sum()),fg_removed=int((ofg&~fg).sum()),matched_count=len(v),sum_matched_iou=float(v.sum()),
    matched_iou_lt05=int((v<.5).sum()),matched_iou_lt09=int((v<.9).sum()),old_with_multiple_new=int(one_old_many_new.sum()),
    new_with_multiple_old=int(many_old_one_new.sum())))
  print(ns,index,oc,nc,round(float(v.mean()),3),flush=True)
 del c,anns
summary=dict(protocol='CPU same-RGB old vs manual RLE; no test labels decoded; diagnostic graph thresholds not physical identity proof',split_transition=trans)
for split in ('train','val'):
 rr=[r for r in rows if r['new_split']==split];good=[r for r in rr if r['rgb_same']];tot={k:sum(r[k] for r in good) for k in ('old_count','new_count','old_fg','new_fg','fg_intersection','fg_union','fg_added','fg_removed','matched_count','sum_matched_iou','matched_iou_lt05','matched_iou_lt09','old_with_multiple_new','new_with_multiple_old')};tot.update(images=len(rr),same_rgb_images=len(good),foreground_iou=tot['fg_intersection']/max(tot['fg_union'],1),matched_iou_mean=tot['sum_matched_iou']/max(tot['matched_count'],1));summary[split]=tot
(O/'label_comparison_rows.json').write_text(json.dumps(rows,indent=2)+'\n');(O/'label_comparison_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
