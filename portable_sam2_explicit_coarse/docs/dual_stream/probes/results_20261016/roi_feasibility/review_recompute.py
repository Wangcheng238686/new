import json
from collections import defaultdict
from pathlib import Path
from pycocotools import mask as mu
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
import numpy as np
O=Path('/tmp/ue_roi_feasibility_20261016');P=Path('/tmp/ue_adjacent_pair_20261016/outputs');S=Path('/tmp/ue_uav_experiments/20261014_ad_seed42/final_analysis')
def load(p):return json.load(open(p))
gt=defaultdict(list)
for g in load(S/'single_best/gt_records.json'):gt[g['image_id']].append(g)
frag={(x['image_id'],x['gt_slot']) for x in load(P/'posthoc_sensitivity.json')['fragment_instances'] if x['mode']=='dual_correct'}
used=set();picked=[]
for cat in ('fragment_failure','other_failure','both_success'):
 n=0
 for p in sorted(load(P/'pairs.json'),key=lambda p:(p['image_id'],p['slots'])):
  iid=p['image_id'];ss=p['slots']
  good=True
  for s in ss:
   g=gt[iid][s];x,y,w,h=g['bbox']
   good &= min(x,y)>0 and x+w<512 and y+h<512 and mu.area(g['segmentation'])>=1024
  if not good or used.intersection(p['physical_pair']):continue
  d=p['modes']['dual_correct']['0.5']['mask_pair_success'];a=p['modes']['single_best']['0.5']['mask_pair_success'];f=any((iid,s) in frag for s in ss)
  choose={'fragment_failure':not d and f,'other_failure':not d and not f,'both_success':d and a}[cat]
  if choose:
   picked.append((cat,iid,ss));used.update(p['physical_pair']);n+=1
   if n==2:break
saved=load(O/'uav_evidence/selected_cases.json')
assert picked==[(r['category'],r['image_id'],r['slots']) for r in saved]
res={'selection_independent_reproduction':True,'selected':picked,'unique_buildings':len(used)}
(O/'review_recompute.json').write_text(json.dumps(res,indent=2));print(res)
rows=load(O/'candidates/pair_records.json');idx=defaultdict(list)
for r in rows:idx[r['mode'],r['image_id']].append(r)
def bm(a,b):
 a=np.asarray(a);b=np.asarray(b);z=np.clip(np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2]),0,None).prod(2)
 return z/((a[:,2:]-a[:,:2]).prod(1)[:,None]+(b[:,2:]-b[:,:2]).prod(1)[None,:]-z)
def dedup(xs):
 out=[]
 for x in xs:
  if x is not None and all(max(abs(float(v)-float(w)) for v,w in zip(x,y))>1e-4 for y in out):out.append(x)
 return out
checks=0;witnesses=0;matching_cardinalities=0
for mode in ('single_best','dual_correct'):
 stages=defaultdict(list);dets=defaultdict(list)
 for x in load(S/'full_stage_audit'/f'{mode}.json')['instances']:stages[x['image_id']].append(x)
 for x in load(S/mode/'dt_records.json'):dets[x['image_id']].append(x)
 for iid,gs in stages.items():
  gs.sort(key=lambda x:x['gt_slot']);g=[x['bbox_xyxy'] for x in gs]
  f=dedup([[d['bbox'][0],d['bbox'][1],d['bbox'][0]+d['bbox'][2],d['bbox'][1]+d['bbox'][3]] for d in dets[iid]])
  e=dedup([x['eligible_best_box'] for x in gs]);u=dedup(e+f)
  sets={'E':e,'F':f,'U':u};m={k:bm(g,v) for k,v in sets.items()};mut={k:bm(v,v) for k,v in sets.items()}
  for t in (.5,.75):
   for k in m:
    adj=m[k]>=t; mm=maximum_bipartite_matching(csr_matrix(adj),perm_type='column'); cardinality=int((mm>=0).sum());matching_cardinalities+=1
    # Producer's matches appear as pair booleans only; independently compare final image cardinality against original instance matches.
    if k=='F':
     old=[x for x in load(P/'instances.json') if x['image_id']==iid]
     assert cardinality==sum(x['modes'][mode][str(t)]['box_matched'] for x in old)
  for r in idx[mode,iid]:
   t=r['threshold'];s1,s2=r['slots']
   for k in m:
    aa=np.flatnonzero(m[k][s1]>=t);bb=np.flatnonzero(m[k][s2]>=t)
    feasible=any(a!=b for a in aa for b in bb)
    compat=any(a!=b and mut[k][a,b]<=.5 for a in aa for b in bb)
    assert r[k]['pair_local_feasible']==feasible and r[k]['pair_local_nms_compatible']==compat,(mode,iid,r['slots'],k)
    checks+=1
   if 'witness' in r:
    w=r['witness'];a,b=w['indices'];assert a!=b and np.allclose([u[a],u[b]],w['boxes'])
    assert m['U'][s1,a]>=t and m['U'][s2,b]>=t and mut['U'][a,b]<=.5
    witnesses+=1
res['candidate_pair_set_independent_checks']=checks;res['candidate_witness_checks']=witnesses;res['independent_sparse_matching_cardinalities']=matching_cardinalities
summary=load(O/'candidates/summary.json')
for st,condition in [('all',lambda x:True),('interior_nonsmall',lambda x:x['interior_nonsmall']),('risk_flagged',lambda x:x['risk']),('risk_unflagged',lambda x:not x['risk']),('interior_nonsmall_unflagged',lambda x:x['interior_nonsmall'] and not x['risk'])]:
 for mode in ('single_best','dual_correct'):
  for t in (.5,.75):
   rr=[r for r in rows if r['mode']==mode and r['threshold']==t and condition(r)];ss=summary['strata'][st][mode][str(t)]
   assert len(rr)==ss['appearances']
   for key,value in ss['counts'].items():assert sum(r[key] for r in rr)==value
res['candidate_all_summary_counts_reproduced']=True
(O/'review_recompute.json').write_text(json.dumps(res,indent=2));print(res)
