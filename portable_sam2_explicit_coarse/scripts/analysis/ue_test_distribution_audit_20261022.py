"""Offline v1 split audit. No training or inference; frozen existing records only."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import sys,json,hashlib,contextlib,io,copy
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.probes.ue_v1_adapter_analysis import evaluator
DATA=Path('/data/wangcheng/dataset/CitySample_SatUAV_TrainMinimal_v1_20261021')
RUN=Path('/data/wangcheng/checkpoint/ue_v1_adapter_probe_20261021')
OUT=Path('/data/wangcheng/checkpoint/ue_test_weakness_20261022/distribution');OUT.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def q(a):return {str(p):float(np.percentile(a,p)) for p in (0,25,50,75,90,95,100)} if len(a) else {}
def mean(a):return float(np.mean(a)) if len(a) else None
def area_group(a):return 'small' if a<1024 else 'medium' if a<9216 else 'large'
def density(n):return '<=50' if n<=50 else '51-100' if n<=100 else '>100'
samples=[json.loads(s) for s in (DATA/'samples.jsonl').read_text().splitlines()]
sample_map={s['index']:s for s in samples}
images={};summaries={};sources={str(DATA/'samples.jsonl'):sha(DATA/'samples.jsonl')}
for source in (ROOT/'utils/coco_eval_utils.py',ROOT/'scripts/probes/ue_v1_adapter_analysis.py'):
 sources[str(source)]=sha(source)
for split in ('train','val','test'):
 p=DATA/'annotations'/f'{split}.json';sources[str(p)]=sha(p);d=json.loads(p.read_text());by=defaultdict(list)
 for a in d['annotations']:by[a['image_id']].append(a)
 anomalies=Counter();cross=Counter();allareas=[];allbbox=[];allrows=[];exact_duplicates=[]
 for im in d['images']:
  k=im['id'];aa=by[k];n=len(aa);edge=[];seen={}
  for a in aa:
   x,y,w,h=a['bbox'];area=a['area'];cnt=a['segmentation']['counts'];rlearea=sum(cnt[1::2]);allareas.append(area);allbbox.append(w*h)
   anomalies['rle_area_mismatch']+=int(area!=rlearea);anomalies['rle_length_mismatch']+=int(sum(cnt)!=512*512)
   anomalies['nonpositive_area']+=int(area<=0);anomalies['bbox_outside']+=int(x<0 or y<0 or x+w>512 or y+h>512 or w<=0 or h<=0)
   anomalies['tiny_mask_lt16']+=int(area<16);anomalies['thin_bbox_le4']+=int(min(w,h)<=4)
   hit=x<=0 or y<=0 or x+w>=512 or y+h>=512;edge.append(hit)
   cross[(area_group(area),area_group(w*h))]+=1
   key=tuple(cnt)
   if key in seen:exact_duplicates.append([k,seen[key],a['id']])
   seen[key]=a['id']
  s=sample_map[k];row=dict(image_id=k,split=split,n=n,density=density(n),edge_n=sum(edge),edge_fraction=mean(edge),mask_small=sum(a['area']<1024 for a in aa),mask_tiny_lt256=sum(a['area']<256 for a in aa),center_m=[v/100 for v in s['center_ue_cm'][:2]],block_250m=[int(np.floor(v/25000)) for v in s['center_ue_cm'][:2]])
  images[k]=row;allrows.append(row)
 summaries[split]=dict(images=len(allrows),instances=len(d['annotations']),instances_per_image=q([r['n'] for r in allrows]),mean_instances=mean([r['n'] for r in allrows]),density_images=dict(Counter(r['density'] for r in allrows)),mask_area=q(allareas),bbox_area=q(allbbox),mask_size_counts=dict(Counter(area_group(x) for x in allareas)),bbox_size_counts=dict(Counter(area_group(x) for x in allbbox)),area_group_confusion={f'{a}->{b}':n for (a,b),n in cross.items()},edge_instances=sum(r['edge_n'] for r in allrows),anomalies=dict(anomalies),exact_rle_duplicates=exact_duplicates,spatial_range_m={axis:[min(r['center_m'][j] for r in allrows),max(r['center_m'][j] for r in allrows)] for j,axis in enumerate(('x','y'))})
 print(split,summaries[split]['density_images'],flush=True)
(OUT/'image_metadata.json').write_text(json.dumps(list(images.values()),indent=2))
# Nearest point centers are proximity only, not footprint overlap nor asset independence.
train=np.array([r['center_m'] for r in images.values() if r['split']=='train'])
for split in ('val','test'):
 centers=np.array([r['center_m'] for r in images.values() if r['split']==split]);dist=np.sqrt(((centers[:,None,:]-train[None,:,:])**2).sum(-1)).min(1)
 summaries[split]['nearest_train_center_m']=q(dist)

def metrics(ev):
 p=ev.eval['precision'];r=ev.eval['recall']
 def avg(a):return mean(a[a>=0])
 return dict(AP=avg(p[:,:,:,0,-1]),AP50=avg(p[0,:,:,0,-1]),AP75=avg(p[5,:,:,0,-1]),AP_s=avg(p[:,:,:,1,-1]),AP_m=avg(p[:,:,:,2,-1]),AP_l=avg(p[:,:,:,3,-1]),AR250=avg(r[:,:,0,-1]))
def submetrics(ev,positions):
 orig=ev.evalImgs;params=copy.deepcopy(ev.params);n=len(params.imgIds)
 if not positions:return None
 ev.evalImgs=[orig[a*n+i] for a in range(4) for i in positions]
 # Relabel image ids in parameter snapshots to make accumulation positional.
 old_evparams=ev._paramsEval
 ev._paramsEval=copy.deepcopy(params);ev._paramsEval.imgIds=list(range(len(positions)))
 pp=copy.deepcopy(ev._paramsEval)
 with contextlib.redirect_stdout(io.StringIO()):ev.accumulate(pp)
 result=metrics(ev);ev.evalImgs=orig;ev._paramsEval=old_evparams;ev.params=params
 return result
results={}
for split,path in [('val',RUN/'evaluation/disabled/records.json'),('test',RUN/'test/baseline/disabled/records.json')]:
 rows=json.loads(path.read_text());sources[str(path)]=sha(path);ids=[int(r['image_id']) for r in rows];assert len(rows)==459
 info={'detection_count':dict(per_image=q([len(r['dt']['scores']) for r in rows]),images_at_250=sum(len(r['dt']['scores'])==250 for r in rows),images_dt_less_than_gt=sum(len(r['dt']['scores'])<len(r['gt']['labels']) for r in rows))};
 groups={'all':list(range(459))}
 for label in ('<=50','51-100','>100'):groups['density_'+label]=[i for i,k in enumerate(ids) if images[k]['density']==label]
 for edge in (False,True):groups['edge_fraction_'+('>=.25' if edge else '<.25')]=[i for i,k in enumerate(ids) if (images[k]['edge_fraction']>=.25)==edge]
 for block in sorted(set(tuple(images[k]['block_250m']) for k in ids)):
  pos=[i for i,k in enumerate(ids) if tuple(images[k]['block_250m'])==block]
  if len(pos)>=15:groups['block250m_'+str(block)]=pos
 for kind in ('segm','bbox'):
  print('evaluate',split,kind,flush=True);ev=evaluator(rows,kind=kind)
  info[kind]={}
  for name,pos in groups.items():
   mm=submetrics(ev,pos)
   info[kind][name]=dict(images=len(pos),instances=sum(images[ids[i]]['n'] for i in pos),metrics=mm)
  # GT recall with standard greedy score-ranked matching across all GT (same matching, no subgroup rematching).
  gtinfo={a['id']:a for a in ev.cocoGt.dataset['annotations']};strata=defaultdict(lambda:[0,0,0,0.])
  for e in ev.evalImgs[:459]:
   if e is None:continue
   for j,gid in enumerate(e['gtIds']):
    a=gtinfo[gid];x,y,w,h=a['bbox'];hit=x<=0 or y<=0 or x+w>=512 or y+h>=512
    from pycocotools import mask as mu
    maskarea=float(mu.area(a['segmentation']))
    for name in ['all','edge' if hit else 'nonedge','true_mask_'+area_group(maskarea),'bbox_'+area_group(a['area'])]:
     arr=strata[name];arr[0]+=1;arr[1]+=int(e['gtMatches'][0,j]>0);arr[2]+=int(e['gtMatches'][5,j]>0);arr[3]+=float((e['gtMatches'][:,j]>0).mean())
  info[kind]['instance_recall']={k:dict(GT=n,recall50=a/n,recall75=b/n,recall50_95=c/n) for k,(n,a,b,c) in strata.items()}
 # Extra offline grouping correction only; existing frozen evaluator is untouched.
 from pycocotools.cocoeval import COCOeval
 from pycocotools import mask as mu
 for a in ev.cocoGt.dataset['annotations']:a['area']=float(mu.area(a['segmentation']))
 corrected={}
 for kind in ('segm','bbox'):
  ce=COCOeval(ev.cocoGt,ev.cocoDt,kind);ce.params.maxDets=[1,10,250]
  with contextlib.redirect_stdout(io.StringIO()):ce.evaluate();ce.accumulate()
  corrected[kind]=metrics(ce)
  assert abs(corrected[kind]['AP']-info[kind]['all']['metrics']['AP'])<1e-12
 info['offline_true_mask_area_grouping']=corrected
 # Re-evaluate one subgroup directly to calibrate positional accumulation.
 pos=groups['density_<=50']
 if pos:
  direct=evaluator([rows[i] for i in pos]);assert abs(metrics(direct)['AP']-info['segm']['density_<=50']['metrics']['AP'])<1e-12
 results[split]=info
 (OUT/'audit.json').write_text(json.dumps(dict(split_summary=summaries,baseline=results,sources_sha256=sources),indent=2))
 print('done',split,flush=True)

result=json.loads((OUT/'audit.json').read_text());result['analysis_source_sha256']=sha(Path(__file__));(OUT/'audit.json').write_text(json.dumps(result,indent=2))
