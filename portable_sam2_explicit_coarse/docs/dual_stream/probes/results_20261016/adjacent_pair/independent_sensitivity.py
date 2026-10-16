import json,itertools,collections
from pathlib import Path
import numpy as np
from pycocotools import mask as M
R=Path('/tmp/ue_uav_experiments/20261014_ad_seed42/final_analysis'); O=Path('/tmp/ue_adjacent_pair_20261016/outputs')
def read(p):return json.loads(p.read_text())
def group(rows):
 d=collections.defaultdict(list)
 for r in rows:d[r['image_id']].append(r)
 return d
def decode(rows):return M.decode([dict(size=r['segmentation']['size'],counts=r['segmentation']['counts'].encode()) for r in rows]).transpose(2,0,1).astype(bool)
gt=group(read(R/'single_best/gt_records.json')); pairs=read(O/'pairs.json'); records={(r['image_id'],r['gt_slot']):r for r in read(O/'instances.json')}; border={}; small={}; fragments={}
for name in ['single_best','dual_correct']:
 dt=group(read(R/name/'dt_records.json'))
 for image_id,gr in gt.items():
  gm=decode(gr); dm=decode(dt[image_id]); da=dm.sum((1,2))
  for i,m in enumerate(gm):
   key=(image_id,i); border[key]=bool(m[0].any() or m[-1].any() or m[:,0].any() or m[:,-1].any());small[key]=bool(m.sum()<1024)
   found=[]
   if not records[key]['modes'][name]['0.5']['mask_matched']:
    intersections=dm[:,m].sum(1); cov=intersections/m.sum(); purity=intersections/np.maximum(da,1)
    cand=np.flatnonzero((cov>=.25)&(purity>=.5))
    for j,k in itertools.combinations(cand,2):
     inter=np.logical_and(dm[j],dm[k]).sum();union=da[j]+da[k]-inter
     if inter/max(union,1)>.1:continue
     if np.logical_and(m,np.logical_or(dm[j],dm[k])).sum()/m.sum()>=.5:
      found=[int(j),int(k)];break
   fragments[(name,image_id,i)]=found
actual=read(O/'posthoc_sensitivity.json')
listed={(r['mode'],r['image_id'],r['gt_slot']):r['prediction_indices'] for r in actual['fragment_instances']}
assert listed=={k:v for k,v in fragments.items() if v}
for st,v in actual['strata'].items():
 selected=[]
 for p in pairs:
  keys=[(p['image_id'],i) for i in p['slots']];b=any(border[k] for k in keys);s=any(small[k] for k in keys)
  include={'all':True,'any_border':b,'neither_border':not b,'any_small':s,'neither_small':not s,'interior_nonsmall':not b and not s}[st]
  if include:selected.append(p)
 assert len(selected)==v['appearances']
 assert len({tuple(p['physical_pair']) for p in selected})==v['unique_physical_pairs']
 for name in ['single_best','dual_correct']:
  observed=v['modes'][name]
  for t in ['0.5','0.75']:
   vals=[p['modes'][name][t] for p in selected]
   for metric,count in observed[t].items():
    if metric=='mask_fail_gtbox_both_pass':expected=sum(not x['mask_pair_success'] and x['gt_box_both_masks_pass'] for x in vals)
    else:expected=sum(x[metric] for x in vals)
    assert expected==count,(st,name,t,metric,count,expected)
  expected=sum(not p['modes'][name]['0.5']['mask_pair_success'] and any(fragments[(name,p['image_id'],i)] for i in p['slots']) for p in selected)
  assert expected==observed['mask05_failure_with_fragment_proxy']
report={'verdict':'PASS','method':'All fragment identities recomputed directly from decoded Boolean mask intersections/unions; border from mask edge pixels; full strata counts independently recomputed','fragment_instances_by_mode':dict(collections.Counter(k[0] for k in listed)),'strata':actual['strata']}
Path('/tmp/ue_adjacent_pair_20261016/independent_sensitivity_numbers.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
