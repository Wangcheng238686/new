"""Paired AP with changing detections; GT and image identities must stay fixed."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from ue_v1_adapter_analysis import evaluator,precision_ap,sample_ap


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();base=json.loads(a.baseline.read_text());cand=json.loads(a.candidate.read_text())
    assert len(base)==len(cand)==459
    assert [r['image_id'] for r in base]==[r['image_id'] for r in cand]
    assert all(b['gt']==c['gt'] for b,c in zip(base,cand))
    rng=np.random.default_rng(20261021);samples=rng.integers(0,459,(500,459))
    result=dict(images=459,gt_and_images_exact=True,detection_changes_allowed=True,
        baseline_sha256=sha(a.baseline),candidate_sha256=sha(a.candidate),metrics={},
        analysis_sha256=sha(__file__),estimator_sha256=sha(Path(__file__).with_name('ue_v1_adapter_analysis.py')),
        scope='Paired image bootstrap; single training seed, development-test multi-route exploration; no training-variance guarantee')
    for name,kind,ratio in [('MaskAP','segm',None),('BBoxAP','bbox',None),('BoundaryAP_2pct','segm',.02),('BoundaryAP_1pct','segm',.01)]:
        eb,ec=evaluator(base,kind=kind,ratio=ratio),evaluator(cand,kind=kind,ratio=ratio)
        vb,vc=precision_ap(eb),precision_ap(ec)
        assert abs(sample_ap(eb,np.arange(459))-vb)<1e-12 and abs(sample_ap(ec,np.arange(459))-vc)<1e-12
        diffs=[sample_ap(ec,ix)-sample_ap(eb,ix) for ix in samples]
        result['metrics'][name]=dict(baseline=vb,candidate=vc,delta=vc-vb,paired_image_bootstrap95=np.quantile(diffs,[.025,.975]).tolist(),bootstrap_samples=500,seed=20261021)
        print(name,result['metrics'][name],flush=True)
    result['clear_mask_gain']=result['metrics']['MaskAP']['delta']>=.002 and result['metrics']['MaskAP']['paired_image_bootstrap95'][0]>0
    a.out.parent.mkdir(exist_ok=True,parents=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
