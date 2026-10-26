"""Wait for both cohorts, then audit the paired 2x2 injection/content interaction."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from ue_geometry_mechanism_schedule import sha,dump
from ue_v1_adapter_analysis import evaluator,precision_ap,sample_ap


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
    ladder=Path('/data/wangcheng/checkpoint/ue_attention_ladder_20261022')
    names=['post_correct','post_constant','pre_correct','pre_constant']
    folders=[r/'test/correct/correct',r/'test/constant/constant',ladder/'test/L0_correct/correct',ladder/'test/L0_constant/constant']
    train=[r/'correct',r/'constant',ladder/'L0_correct',ladder/'L0_constant']
    required=[f/'evaluation_identity.json' for f in folders]+[t/'parameter_changes.json' for t in train]
    out=r/'interaction';out.mkdir(exist_ok=True)
    hashes={str(p):sha(p) for p in [Path(__file__),Path(__file__).with_name('ue_v1_adapter_analysis.py')]}
    while not all(p.exists() for p in required):
        dump(out/'state.json',dict(stage='waiting_for_complete_cohorts',missing=[str(p) for p in required if not p.exists()]))
        time.sleep(30)
    assert all(sha(p)==h for p,h in hashes.items()),'Interaction numeric sources changed while waiting'
    contracts=[json.loads((t/'frozen_training_contract.json').read_text()) for t in train]
    for key in ['base_sha256','satellite_state_sha256','archive_hashes','initial_fingerprint','init_file_sha256',
                'initialization_seed','data_seed','epoch_seed_formula','epochs','updates','batch','accum','lr','warmup',
                'cosine_min_factor','weight_decay','clip','amp','ema','selection','max_dets']:
        assert all(c[key]==contracts[0][key] for c in contracts),key
    for epoch in range(1,11):
        rows=[json.loads((t/f'epoch{epoch}_summary.json').read_text()) for t in train]
        assert all(x['sample_order']==rows[0]['sample_order'] and x['update']==epoch*267 for x in rows)
    rows=[json.loads((f/'records.json').read_text()) for f in folders]
    for other in rows[1:]:
        assert len(other)==len(rows[0])==459
        for left,right in zip(rows[0],other):
            assert left['image_id']==right['image_id'] and left['gt']==right['gt']
            for k in ['bboxes','labels','scores']:assert left['dt'][k]==right['dt'][k]
    rng=np.random.default_rng(20261021);samples=rng.integers(0,459,(500,459))
    result=dict(complete=False,identity_and_sample_order_verified=True,records_sha256={n:sha(f/'records.json') for n,f in zip(names,folders)},
        source_sha256=hashes,bootstrap_seed=20261021,bootstrap_samples=500,metrics={})
    vectors={}
    for metric,ratio in [('MaskAP',None),('BoundaryAP_2pct',.02),('BoundaryAP_1pct',.01)]:
        ev=[evaluator(x,ratio=ratio) for x in rows];points=np.array([precision_ap(e) for e in ev])
        assert all(abs(sample_ap(e,np.arange(459))-v)<1e-12 for e,v in zip(ev,points))
        draws=np.array([[sample_ap(e,ix) for e in ev] for ix in samples])
        values=draws[:,0]-draws[:,1]-draws[:,2]+draws[:,3];delta=points[0]-points[1]-points[2]+points[3]
        vectors[metric]=values
        result['metrics'][metric]=dict(points=dict(zip(names,points.tolist())),interaction=float(delta),ci95=np.quantile(values,[.025,.975]).tolist())
        dump(out/'partial.json',result);print(metric,result['metrics'][metric],flush=True)
    result['complete']=True
    result['limitation']='Single seed, development test, image bootstrap does not cover training variance/spatial correlation'
    np.savez_compressed(out/'bootstrap.npz',**vectors);dump(out/'interaction.json',result)
    dump(out/'state.json',dict(stage='complete'))


if __name__=='__main__':main()
