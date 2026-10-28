"""Offline descriptive diagnostics; Hungarian pair counts are NOT COCO AP/AR."""
import os
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,csv,contextlib,io
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np,cv2
from scipy.optimize import linear_sum_assignment
from pycocotools import mask as mu
cv2.setNumThreads(1)
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.probes.ue_v1_adapter_analysis import evaluator
R=Path('/data/wangcheng/checkpoint/ue_v1_adapter_probe_20261021/test')
OUT=Path('/data/wangcheng/checkpoint/ue_test_weakness_20261022/errors');OUT.mkdir(parents=True,exist_ok=True)
paths={'baseline':R/'baseline/disabled/records.json','native_geometry':R/'native_geometry/correct/records.json','unit':R/'unit/correct/records.json'}
rows={k:json.loads(p.read_text()) for k,p in paths.items()}
summary={'sources':{k:{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for k,p in paths.items()},'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'images':len(rows['baseline']),'matching':'Hungarian maximum cardinality at bbox IoU>=0.5, then maximum sum IoU; all saved detections, no new score cutoff; frozen across models','size':'true mask area small<1024 medium<9216 else large; stored bbox area separately','pixel_zones':'2px GT dilate minus erode boundary; eroded GT interior; remaining background within union of GT/pred bounding boxes expanded 8px and clipped to image; each paired instance separately, so overlaps can repeat','thresholds':[.5,.75,.9,.95]}
agg=Counter();groups=defaultdict(Counter);changes={k:Counter() for k in ('native_geometry','unit')};pixel={k:Counter() for k in changes};gtcsv=[];imagecsv=[]
for n,b in enumerate(rows['baseline']):
    g=b['gt'];d=b['dt'];G=len(g['rles']);D=len(d['rles']);agg.update(gt=G,dt=D)
    gb=np.asarray(g['bboxes']);db=np.asarray(d['bboxes']);gxy=gb.copy();dxy=db.copy();gxy[:,2:]-=gxy[:,:2];dxy[:,2:]-=dxy[:,:2]
    bi=mu.iou(dxy.tolist(),gxy.tolist(),[0]*G) if D and G else np.zeros((D,G))
    cost=(bi>=.5).astype(float)*1000+bi
    di,gi=linear_sum_assignment(cost,maximize=True);pairs={int(j):int(i) for i,j in zip(di,gi) if bi[i,j]>=.5};used=set(pairs.values())
    agg['bbox_matched']+=len(pairs);agg['gt_unmatched']+=G-len(pairs);agg['dt_unmatched']+=D-len(used)
    if G:agg['gt_no_any_bbox_iou50']+=int((bi.max(axis=0)<.5).sum()) if D else G
    if D:agg['dt_no_any_bbox_iou50']+=int((bi.max(axis=1)<.5).sum()) if G else D
    agg['gt_count_exceeds250']+=max(0,G-250)
    gm=np.moveaxis(mu.decode(g['rles']).astype(bool),2,0).copy()
    dms={};ious={}
    for k,a in rows.items():
        a=a[n];assert a['image_id']==b['image_id'] and a['gt']==g
        assert all(a['dt'][x]==d[x] for x in ('bboxes','scores','labels'))
        dms[k]=np.moveaxis(mu.decode(a['dt']['rles']).astype(bool),2,0).copy() if D else np.zeros((0,512,512),bool)
        ious[k]=mu.iou(a['dt']['rles'],g['rles'],[0]*G) if D and G else np.zeros((D,G))
    im=Counter(image_id=b['image_id'],gt=G,dt=D,bbox_matched=len(pairs),gt_unmatched=G-len(pairs),dt_unmatched=D-len(used))
    for j in range(G):
        mask=gm[j];area=int(mask.sum());ba=float(gxy[j,2]*gxy[j,3]);ys,xs=np.where(mask);trunc=bool(np.any(mask[0]) or np.any(mask[-1]) or np.any(mask[:,0]) or np.any(mask[:,-1]));size='small' if area<1024 else 'medium' if area<9216 else 'large';density='le50' if G<=50 else '51to100' if G<=100 else 'gt100'
        gx=gb[j];other=gb.copy();near=((np.maximum(0,np.minimum(other[:,2],gx[2]+4)-np.maximum(other[:,0],gx[0]-4)))*(np.maximum(0,np.minimum(other[:,3],gx[3]+4)-np.maximum(other[:,1],gx[1]-4)))>0);near[j]=False;adj=bool(near.any())
        keys=[('all','all'),('size',size),('density',density),('truncated',str(trunc)),('bbox_neighbor4px',str(adj))]
        for key in keys:groups['/'.join(key)]['gt']+=1;groups['/'.join(key)]['bbox_matched']+=j in pairs
        rec={'image_id':b['image_id'],'gt_index':j,'mask_area':area,'bbox_area':ba,'size':size,'density':density,'truncated':trunc,'bbox_neighbor4px':adj,'dt_index':pairs.get(j,-1),'max_bbox_iou':float(bi[:,j].max()) if D else 0}
        if j not in pairs:gtcsv.append(rec);continue
        i=pairs[j];rec['bbox_iou']=float(bi[i,j]);vi={k:float(v[i,j]) for k,v in ious.items()}
        for k,v in vi.items():rec[k+'_mask_iou']=v
        for t in summary['thresholds']:
            tag=str(t);good=vi['baseline']>=t;agg['matched_mask_pass_'+tag]+=good;im['matched_mask_pass_'+tag]+=good
            for key in keys:groups['/'.join(key)]['matched_mask_pass_'+tag]+=good
            for k in changes:
                changes[k]['corrected_'+tag]+=not good and vi[k]>=t;changes[k]['new_error_'+tag]+=good and vi[k]<t
        im['baseline_iou_sum']+=vi['baseline'];agg['matched_iou_sum']+=vi['baseline']
        for k in changes:changes[k]['iou_delta_sum']+=vi[k]-vi['baseline'];changes[k]['improved_iou']+=vi[k]>vi['baseline']+1e-6;changes[k]['worsened_iou']+=vi[k]<vi['baseline']-1e-6
        er=cv2.erode(mask.astype('uint8'),np.ones((3,3),np.uint8),iterations=2).astype(bool);dil=cv2.dilate(mask.astype('uint8'),np.ones((3,3),np.uint8),iterations=2).astype(bool)
        x1,y1=np.maximum(0,np.floor(np.minimum(gb[j,:2],db[i,:2]))-8).astype(int);x2,y2=np.minimum(512,np.ceil(np.maximum(gb[j,2:],db[i,2:]))+8).astype(int)
        region=np.zeros((512,512),bool);region[y1:y2,x1:x2]=True
        sl=np.s_[y1:y2,x1:x2];zones={'boundary':(dil&~er)[sl],'interior':er[sl],'background':(~dil)[sl]};be=dms['baseline'][i][sl]!=mask[sl]
        for zone,z in zones.items():
            for k in changes:
                ce=dms[k][i][sl]!=mask[sl];pixel[k][zone+'_pixels']+=int(z.sum());pixel[k][zone+'_baseline_errors']+=int((z&be).sum());pixel[k][zone+'_corrected']+=int((z&be&~ce).sum());pixel[k][zone+'_new_errors']+=int((z&~be&ce).sum())
        gtcsv.append(rec)
    imagecsv.append(dict(im))
    if n%50==0:print(n,flush=True)
summary['hungarian']=dict(agg);summary['groups']={k:dict(v) for k,v in groups.items()};summary['fixed_pair_changes']={k:dict(v) for k,v in changes.items()};summary['pixel_changes']={k:dict(v) for k,v in pixel.items()}
# Official score-greedy matching diagnostics, kept distinct from Hungarian.
summary['coco_matching']={}
for kind in ('bbox','segm'):
    e=evaluator(rows['baseline'],kind=kind);cs={}
    for t in summary['thresholds']:
        ix=int(np.argmin(abs(e.params.iouThrs-t)));c=Counter()
        for a in e.evalImgs[:len(rows['baseline'])]:
            if a is None:continue
            c['gt']+=int(np.logical_not(a['gtIgnore']).sum());c['dt']+=len(a['dtScores']);c['matched_gt']+=int(((a['gtMatches'][ix]>0)&np.logical_not(a['gtIgnore'])).sum());c['unmatched_dt']+=int(((a['dtMatches'][ix]==0)&np.logical_not(a['dtIgnore'][ix])).sum())
        cs[str(t)]=dict(c)
    summary['coco_matching'][kind]=cs
for name,table in [('instances',gtcsv),('images',imagecsv)]:
    fields=sorted(set().union(*(a.keys() for a in table)))
    with (OUT/(name+'.csv')).open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(table)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
