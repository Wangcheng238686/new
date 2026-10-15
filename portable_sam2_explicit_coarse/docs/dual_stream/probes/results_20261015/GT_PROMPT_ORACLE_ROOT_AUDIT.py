"""Root review: recompute reported effects directly from paired records."""
from pathlib import Path
import json,statistics,collections
import numpy as np
p=Path('/tmp/ue_uav_experiments/20261014_ad_seed42/final_analysis/gt_prompt_oracle')
a=json.loads((p/'full/baseline.json').read_text());b=json.loads((p/'full/joint_gt_prompt.json').read_text());s=json.loads((p/'summary.json').read_text());m=json.loads((p/'full/run_manifest.json').read_text())
assert len(a)==len(b)==4496 and m['complete']
images=collections.defaultdict(list);buildings=collections.defaultdict(list)
for x,y in zip(a,b):
 assert (x['image_id'],x['gt_slot'],x['global_id'])==(y['image_id'],y['gt_slot'],y['global_id'])
 d=y['gt_iou']-x['gt_iou'];images[x['image_id']].append(d);buildings[x['global_id']].append(d)
assert len(images)==153 and len(buildings)==237
main=s['main_comparison'];delta=statistics.fmean(y['gt_iou']-x['gt_iou'] for x,y in zip(a,b));be=statistics.fmean(statistics.fmean(v) for v in buildings.values())
assert abs(delta-main['delta_mean'])<1e-12 and abs(be-main['building_equal_weight_delta'])<1e-12
rng=np.random.default_rng(1760);draws=rng.integers(0,153,size=(1000,153));ids=m['image_ids'];bs=[]
for draw in draws:
 total=0.;count=0
 for j in draw:
  vals=images[ids[j]];total+=sum(vals);count+=len(vals)
 bs.append(total/count)
ci=np.percentile(bs,[2.5,97.5]);assert np.allclose(ci,main['image_paired_percentile95'],rtol=0,atol=1e-12)
selectors={'all':lambda x:True,'small':lambda x:x['bbox_area']<1024,'medium':lambda x:1024<=x['bbox_area']<9216,'large':lambda x:x['bbox_area']>=9216,'P2':lambda x:x['roi_level']==0,'nonP2':lambda x:x['roi_level']!=0,'dense_support_zero':lambda x:x['dense_support_cells']==0,'dense_support_nonzero':lambda x:x['dense_support_cells']>0}
for name,pred in selectors.items():
 dif=[y['gt_iou']-x['gt_iou'] for x,y in zip(a,b) if pred(x)];g=main['by_group'][name];assert len(dif)==g['n'] and abs(statistics.fmean(dif)-g['mean'])<1e-12
 assert sum(d>.002 for d in dif)==g['improved_gt002'] and sum(d<-.002 for d in dif)==g['worse_gt002']
 for mode,rows in [('baseline',a),('joint_gt_prompt',b)]:
  rr=[r for r in rows if pred(r)];out=s['groups'][mode][name];assert len(rr)==out['n']
  assert abs(statistics.fmean(r['gt_iou'] for r in rr)-out['final_iou'])<1e-12
  assert abs(statistics.fmean(r['coarse_iou'] for r in rr)-out['coarse_iou_diagnostic'])<1e-12
  assert sum(r['paste_xor_pixels']>0 for r in rr)==out['paste_changed_rois']
  for i,slot in enumerate(['p1','p2','n1','n2']):
   z=out['point_slots'][slot];assert sum(r['point_valid'][i] for r in rr)==z['valid'] and sum(r['point_correct'][i] for r in rr)==z['correct']
  for stage,z in out['stages'].items():
   assert sum(r['stages'][stage]['strict_changed'] for r in rr)==z['strict_changed_rois']
   assert sum(r['stages'][stage]['changed_gt1e6'] for r in rr)==z['rms_gt1e6_rois']
   assert abs(statistics.fmean(r['stages'][stage]['rms'] for r in rr)-z['mean_rms'])<1e-12
passed=delta>=.005 and ci[0]>0;assert main['gate']['passed']==passed
print(json.dumps({'verdict':'PASS','delta':delta,'ci':ci.tolist(),'building_equal_delta':be,'gate_passed':bool(passed),'groups':main['by_group']},indent=2))
