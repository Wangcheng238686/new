import cv2,json,hashlib
import numpy as np
from pathlib import Path
from collections import defaultdict
from pycocotools import mask as mu
O=Path('/tmp/ue_dataset_reaudit_20261021');root=Path('/tmp/ue_uav_experiments/20261014_ad_seed42');m=json.loads((root/'dataset/manifest.json').read_text());by={r['index']:r for r in m['points']};gt=defaultdict(list)
for a in json.loads((root/'final_analysis/single_best/gt_records.json').read_text()):gt[a['image_id']].append(a)
def key(x):return hashlib.sha256(x.astype(np.uint8).tobytes()).hexdigest()
checks=[]
for i in m['splits']['validation']:
 p=Path(m['satellite_root'])/by[i]['satellite_dir']/'satellite_instance_id_clean.png';im=cv2.imread(str(p),-1)
 current=sorted(key(im==k) for k in np.unique(im) if k>0);historical=sorted(key(mu.decode(a['segmentation'])) for a in gt[i]);checks.append(dict(image_id=i,exact_mask_multiset=current==historical,instances=len(current)))
report=dict(images=len(checks),instances=sum(x['instances'] for x in checks),exact_images=sum(x['exact_mask_multiset'] for x in checks),source=str(root/'final_analysis/single_best/gt_records.json'),checks=checks)
(O/'historical_gt_verification.json').write_text(json.dumps(report,indent=2)+'\n');assert report['exact_images']==153;print({k:v for k,v in report.items() if k!='checks'})
