"""Paired val proposal/complement analysis; never final AP or test evaluation."""
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

ROOT=Path('/data/wangcheng/checkpoint/ue_geometry_proposal_pilot_20261024')
REFERENCE=Path('/data/wangcheng/checkpoint/ue_rpn_geometry_20261023/val/budget/records.json')


def overlaps(a,b):
    if not len(b):return np.zeros((len(a),0))
    lo=np.maximum(a[:,None,:2],b[None,:,:2]);hi=np.minimum(a[:,None,2:],b[None,:,2:])
    inter=np.maximum(hi-lo,0).prod(2)
    area=lambda x:np.maximum(x[:,2:]-x[:,:2],0).prod(1)
    return inter/np.maximum(area(a)[:,None]+area(b)[None]-inter,1e-12)


def main():
    jobs=json.loads((ROOT/'launch.json').read_text())
    while not all((ROOT/j['content']/'result.json').exists() for j in jobs):
        for j in jobs:
            if (ROOT/j['content']/'result.json').exists():continue
            try:os.kill(j['pid'],0)
            except ProcessLookupError:raise RuntimeError(f"Missing result after worker exit: {j}")
        time.sleep(10)
    base=json.loads(REFERENCE.read_text());assert len(base)==459
    proposal={arm:json.loads((ROOT/arm/'val_records.json').read_text()) for arm in ['geometry','constant']}
    rows=[]
    for index,b in enumerate(base):
        a=np.asarray(b['gt']['bboxes'],np.float64).reshape(-1,4)
        row=dict(image_id=b['image_id'],gt=len(a))
        for arm,records in proposal.items():
            r=records[index];assert r['image_id']==b['image_id'] and r['counts']['gt']==len(a)
            ov=overlaps(a,np.asarray(r['boxes'],np.float64).reshape(-1,4))
            best=ov.max(1) if ov.shape[1] else np.zeros(len(a))
            for t in [.5,.75]:
                good=best>=t
                capacity=int((maximum_bipartite_matching(csr_matrix(ov>=t),perm_type='column')>=0).sum()) if ov.shape[1] else 0
                prefix=f'{arm}_{t}'
                row[prefix+'_any']=int(good.sum());row[prefix+'_capacity']=capacity
                for stage in ['rpn_cap','final_cap']:
                    old=np.asarray(b['stages'][stage]['best_iou'])>=t
                    row[f'{prefix}_adds_to_sat2000_{stage}']=int((good&~old).sum())
        rows.append(row)
    metrics={};rng=np.random.default_rng(20261021);draws=rng.integers(0,len(rows),(500,len(rows)))
    den=np.asarray([r['gt'] for r in rows]);bootden=den[draws].sum(1)
    for t in [.5,.75]:
        for suffix in ['any','capacity','adds_to_sat2000_rpn_cap','adds_to_sat2000_final_cap']:
            key=f'{t}_{suffix}'
            a=np.array([r[f'geometry_{key}'] for r in rows]);b=np.array([r[f'constant_{key}'] for r in rows]);delta=a-b
            metrics[key]=dict(geometry=int(a.sum()),constant=int(b.sum()),difference=int(delta.sum()),denominator_GT=int(den.sum()),
                difference_recall_fraction=float(delta.sum()/den.sum()),
                paired_image_bootstrap95=np.quantile(delta[draws].sum(1)/bootden,[.025,.975]).tolist())
    (ROOT/'complement_per_image.json').write_text(json.dumps(rows))
    (ROOT/'complement_summary.json').write_text(json.dumps(dict(complete=True,images=459,metrics=metrics,
        bootstrap=500,seed=20261021,reference=str(REFERENCE),
        scope='Single-seed val proposal feasibility. Satellite reference is pre2000/post1000. '
              'Adds count previously uncovered GT with an independent geometry proposal; not merged-detector inference, '
              'not AP gain, no downstream RoI/SAM validation. Any coverage may reuse boxes; standalone capacity is maximum-cardinality matching.'),indent=2))
    print(json.dumps(metrics,indent=2),flush=True)


if __name__=='__main__':main()
