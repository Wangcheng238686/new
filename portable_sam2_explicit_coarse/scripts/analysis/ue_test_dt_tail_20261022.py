"""Final-DT-only audit: cannot infer where proposals were dropped upstream."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
from pycocotools import mask as mu
p=Path('/data/wangcheng/checkpoint/ue_v1_adapter_probe_20261021/test/baseline/disabled/records.json');rows=json.loads(p.read_text());c=Counter();scores=[];counts=[];sub={k:Counter() for k in ['lt0.1','0.1to0.2','0.2to0.5','ge0.5']}
for r in rows:
 b=np.array(r['dt']['bboxes']);g=np.array(r['gt']['bboxes']);s=np.array(r['dt']['scores']);N=len(s);counts.append(N);scores.extend(s.tolist());c['images_at250']+=N==250
 b[:,2:]-=b[:,:2];g[:,2:]-=g[:,:2];i=mu.iou(b.tolist(),g.tolist(),[0]*len(g));selfi=mu.iou(b.tolist(),b.tolist(),[0]*N);order=np.argsort(-s,kind='stable');seen=set()
 for rank,j in enumerate(order):
  sc=s[j];key='lt0.1' if sc<.1 else '0.1to0.2' if sc<.2 else '0.2to0.5' if sc<.5 else 'ge0.5';ss=sub[key];ss['dt']+=1
  good=np.flatnonzero(i[j]>=.5);ss['no_gt_bbox_iou50']+=len(good)==0
  best=int(i[j].argmax()) if len(g) else -1
  ss['best_gt_seen_higher_score']+=bool(len(good) and best in seen)
  if len(good):seen.add(best)
  c['dt_with_higher_score_bbox_overlap50']+=bool(rank and (selfi[j,order[:rank]]>=.5).any())
  c['dt_with_higher_score_bbox_overlap75']+=bool(rank and (selfi[j,order[:rank]]>=.75).any())
res={'source':str(p),'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'counts':dict(c),'dt_count_min_median_max':[min(counts),float(np.median(counts)),max(counts)],'score_min_median_max':[min(scores),float(np.median(scores)),max(scores)],'score_strata':{k:dict(v) for k,v in sub.items()},'limitations':'All final DT only, no causal attribution to RPN/NMS/threshold; same-best-GT and bbox-overlap are duplicate proxies, not COCO FP labels.'}
Path('/data/wangcheng/checkpoint/ue_test_weakness_20261022/errors/dt_tail.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res,indent=2))
