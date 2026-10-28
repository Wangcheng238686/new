"""Shared-reference Mask AP bootstrap for fixed-checkpoint input diagnostics."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import ue_joint_encoding_analysis as original
import ue_joint_encoding_fast_analysis as fast

CORRECT=Path('/data/wangcheng/checkpoint/ue_geometry_conditional_online_20261024/geometry/test')
MODES=Path('/data/wangcheng/checkpoint/ue_geometry_content_intervention_20261024')
DONOR=Path('/data/wangcheng/checkpoint/ue_geometry_whole_map_donor_20261024')
OUT=Path('/data/wangcheng/checkpoint/ue_geometry_intervention_analysis_20261024')


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    ref_path=CORRECT/'combined_records.json';good=json.loads(ref_path.read_text())
    ref_identity=json.loads((CORRECT/'identity.json').read_text())
    samples=np.random.default_rng(20261021).integers(0,459,(500,459))
    ev=original.evaluator(good,kind='segm');point=original.precision_ap(ev)
    assert abs(fast.fast_sample_ap(ev,np.arange(459))-point)<1e-12
    boot=np.asarray([fast.fast_sample_ap(ev,s) for s in samples])
    result={}
    for mode,folder,pid in [
        ('constant',MODES/'constant/geometry/test',json.loads((MODES/'launch.json').read_text())[0]['pid']),
        ('shuffle',MODES/'shuffle/geometry/test',json.loads((MODES/'launch.json').read_text())[1]['pid']),
        ('whole_map_donor',DONOR/'geometry/test',json.loads((DONOR/'launch.json').read_text())['pid'])]:
        while not (folder/'identity.json').exists():
            try:os.kill(pid,0)
            except ProcessLookupError:raise RuntimeError(f'Intervention ended without identity: {mode}')
            time.sleep(10)
        identity=json.loads((folder/'identity.json').read_text())
        assert identity['complete'] and identity['images']==459
        assert identity['checkpoint_sha256']==ref_identity['checkpoint_sha256']
        assert (folder/'prefix_records.json').read_bytes()==(CORRECT/'prefix_records.json').read_bytes()
        records=json.loads((folder/'combined_records.json').read_text())
        assert len(records)==459 and all(a['image_id']==b['image_id'] and a['gt']==b['gt'] for a,b in zip(good,records))
        bad=original.evaluator(records,kind='segm');value=original.precision_ap(bad)
        assert abs(fast.fast_sample_ap(bad,np.arange(459))-value)<1e-12
        differences=boot-np.asarray([fast.fast_sample_ap(bad,s) for s in samples])
        record=dict(images=459,metric='MaskAP250',correct=point,intervention=value,delta=point-value,
            paired_image_bootstrap95=np.quantile(differences,[.025,.975]).tolist(),bootstrap=500,seed=20261021,
            same_checkpoint=True,same_satellite_prefix=True,reference_sha256=sha(ref_path),candidate_sha256=sha(folder/'combined_records.json'),
            source_sha256=sha(Path(__file__)),reference_function_calibration=fast.audits.copy(),
            scope='Fixed-checkpoint input dependence; controls are OOD or unpaired-map diagnostics, not physically exact counterfactuals. '
                  'No model or threshold selection from these results; main net/content reports retain four metrics separately.')
        (OUT/f'{mode}.json').write_text(json.dumps(record,indent=2));result[mode]=record
        print(mode,json.dumps(record),flush=True)
    (OUT/'summary.json').write_text(json.dumps(dict(complete=True,comparisons=result),indent=2))


if __name__=='__main__':main()
