import json
from pathlib import Path
import numpy as np
from pycocotools import mask as mu
P=Path('/tmp/ue_adjacent_pair_20261016/outputs');O=Path('/tmp/ue_roi_feasibility_20261016/uav_evidence')
def load(p):return json.load(open(p))
gt={}
for g in load('/tmp/ue_uav_experiments/20261014_ad_seed42/final_analysis/single_best/gt_records.json'):gt.setdefault(g['image_id'],[]).append(g)
frag={(r['image_id'],r['gt_slot']) for r in load(P/'posthoc_sensitivity.json')['fragment_instances'] if r['mode']=='dual_correct'}
pairs=sorted(load(P/'pairs.json'),key=lambda x:(x['image_id'],x['slots']));used=set();out=[]
for category in ['fragment_failure','other_failure','both_success']:
 count=0
 for p in pairs:
  iid=p['image_id'];rows=[gt[iid][s] for s in p['slots']]
  if any(g['bbox'][0]==0 or g['bbox'][1]==0 or g['bbox'][0]+g['bbox'][2]==512 or g['bbox'][1]+g['bbox'][3]==512 or mu.area(g['segmentation'])<1024 for g in rows):continue
  if used.intersection(p['physical_pair']):continue
  dual=p['modes']['dual_correct']['0.5']['mask_pair_success'];single=p['modes']['single_best']['0.5']['mask_pair_success'];f=any((iid,s) in frag for s in p['slots'])
  yes={'fragment_failure':not dual and f,'other_failure':not dual and not f,'both_success':dual and single}[category]
  if not yes:continue
  out.append(dict(category=category,image_id=iid,slots=p['slots'],global_ids=p['physical_pair'],fragment_slots=[s for s in p['slots'] if (iid,s) in frag],dual_box_success=p['modes']['dual_correct']['0.5']['box_pair_success']))
  used.update(p['physical_pair']);count+=1
  if count==2:break
(O/'selected_cases.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
