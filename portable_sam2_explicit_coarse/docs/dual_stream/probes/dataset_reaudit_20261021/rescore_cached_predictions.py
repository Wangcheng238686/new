"""Change validation labels only; replay immutable predictions, no model execution."""
import copy,json
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools import mask as mu
from pycocotools.cocoeval import COCOeval
R=Path('/tmp/ue_uav_experiments/20261014_ad_seed42/final_analysis');O=Path('/tmp/ue_dataset_reaudit_20261021')
old=json.loads((R/'single_best/gt_records.json').read_text());images=json.loads((R/'single_best/images.json').read_text());ids={a['image_id'] for a in old};print('image_format',str(images[0])[:300],flush=True)
new=[a for a in json.loads(Path('/data/wangcheng/dataset/CitySample_SatUAV_TrainMinimal_v1_20261021/annotations/val.json').read_text())['annotations'] if a['image_id'] in ids]
results={}
for label,records in [('old',old),('new',new)]:
 annotations=[]
 for i,a in enumerate(records,1):
  a=copy.deepcopy(a);a['id']=i;a['area']=a['bbox'][2]*a['bbox'][3];a['iscrowd']=0
  if isinstance(a['segmentation']['counts'],list):a['segmentation']=mu.frPyObjects(a['segmentation'],512,512)
  annotations.append(a)
 gt=COCO();gt.dataset=dict(images=[dict(id=i,height=512,width=512) for i in sorted(ids)],categories=[dict(id=1,name='building')],annotations=annotations);gt.createIndex()
 results[label]={}
 for arm in ['single_best','dual_correct']:
  dt=gt.loadRes(json.loads((R/arm/'dt_records.json').read_text()));metrics={}
  for kind in ['segm','bbox']:
   e=COCOeval(gt,dt,kind);e.params.maxDets=[1,10,100];e.evaluate();e.accumulate();e.summarize();metrics[kind]=float(e.stats[0])
  results[label][arm]=metrics
  if label=='old':
   baseline=json.loads((R/arm/'metrics.json').read_text())
   for kind in ['segm','bbox']:assert abs(metrics[kind]-baseline[f'{kind}/mAP'])<1e-10,(arm,kind,metrics[kind],baseline[f'{kind}/mAP'])
 results[label]['dual_minus_single']={kind:results[label]['dual_correct'][kind]-results[label]['single_best'][kind] for kind in ['segm','bbox']}
results.update(protocol='Replay same immutable old E66single/E72dual EMA predictions on same153 validation RGB; old labels exactly reproduce official archived AP100; new manual labels only changed; GT area=bbox area as historical evaluator; no refit/checkpoint selection/test labels/images.')
(O/'cached_predictions_rescore.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))
