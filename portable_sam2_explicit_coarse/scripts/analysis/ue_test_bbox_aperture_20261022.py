import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json,numpy as np,cv2
from pathlib import Path
from collections import Counter
from pycocotools import mask as mu
cv2.setNumThreads(1)
r=json.load(open('/data/wangcheng/checkpoint/ue_v1_adapter_probe_20261021/test/baseline/disabled/records.json'))
c=Counter();cases=[]
for row in r:
 b=np.asarray(row['dt']['bboxes']);g=row['gt']
 for j,rr in enumerate(g['rles']):
  m=mu.decode(rr);ys,xs=np.where(m);a=int(m.sum());x0,x1=int(xs.min()),int(xs.max())+1;y0,y1=int(ys.min()),int(ys.max())+1
  it=cv2.integral(m[y0:y1,x0:x1]);lo=np.floor(b[:,:2]-1).astype(int);hi=np.ceil(b[:,2:]+1).astype(int)
  lx=np.clip(lo[:,0]-x0,0,x1-x0);ly=np.clip(lo[:,1]-y0,0,y1-y0);hx=np.clip(hi[:,0]-x0,0,x1-x0);hy=np.clip(hi[:,1]-y0,0,y1-y0)
  cover=(it[hy,hx]-it[ly,hx]-it[hy,lx]+it[ly,lx])/a if len(b) else np.array([0]);best=float(cover.max());c['gt']+=1
  for t in [.5,.75,.9,.95]:c['no_candidate_aperture_cover_'+str(t)]+=best<t
  cases.append({'image_id':row['image_id'],'gt_index':j,'best_bbox_gt_coverage':best})
Path('/data/wangcheng/checkpoint/ue_test_weakness_20261022/errors/aperture.json').write_text(json.dumps({'counts':dict(c),'definition':'Upper bound on mask IoU if mask remains inside any predicted bounding rectangle expanded 1px conservatively; ignores uniqueness, classifier scores and false positives; not achievable AP or measured recall. roi_local confirmed in frozen config.','instances':cases},indent=2))
print(dict(c))
