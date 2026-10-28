"""Paired E10 validation content effects and differences in differences."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
from ue_joint_encoding_analysis import evaluator, precision_ap
from ue_joint_encoding_fast_analysis import fast_sample_ap, audits
from ue_rgb_mechanism_run_20261025 import ROOT, VARIANTS


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);a=p.parse_args()
    specs={'satellite':('satellite','correct','matched'),
           'base_correct':('baseline','correct','matched'),
           'base_constant':('baseline','constant','matched')}
    contrasts={'baseline_content':{'base_correct':1,'base_constant':-1}}
    for v in VARIANTS:
        for content,mode,suffix in [('correct','matched','correct'),('constant','matched','constant'),
                                   ('correct','donor','donor'),('correct','constant','constant_input')]:
            specs[v+'_'+suffix]=(v,content,mode)
        contrasts[v+'_content']={v+'_correct':1,v+'_constant':-1}
        contrasts[v+'_content_interaction']={v+'_correct':1,v+'_constant':-1,'base_correct':-1,'base_constant':1}
        contrasts[v+'_correct_net']={v+'_correct':1,'base_correct':-1}
        contrasts[v+'_satellite_net']={v+'_correct':1,'satellite':-1}
        contrasts[v+'_donor_dependency']={v+'_correct':1,v+'_donor':-1}
        contrasts[v+'_constant_dependency']={v+'_correct':1,v+'_constant_input':-1}
    data={}
    for key,(v,c,m) in specs.items():
        out=a.root/'evaluation'/v/c/m
        identity=json.loads((out/'complete.json').read_text())
        assert identity['images']==459 and identity['split']=='val' and identity['precision']=='FP32'
        data[key]=json.loads((out/'records.json').read_text())
    reference=data['base_correct']
    for rows in data.values():
        assert [r['image_id'] for r in rows]==[r['image_id'] for r in reference]
        assert all(b['gt']==c['gt'] for b,c in zip(reference,rows))
    samples=np.random.default_rng(20261021).integers(0,459,(500,459))
    result=dict(split='val',selection='E10 fixed, not validation best',training_seed=44,
        images=459,bootstrap=500,bootstrap_seed=20261021,paths={},
        limitations='early matched100epoch-schedule prefix; not full-budget final ranking; single-seed development validation')
    out=a.root/'analysis';out.mkdir(exist_ok=True)
    for path in ('combined','original_path'):
        result['paths'][path]={}
        for metric,kind,ratio in [('MaskAP','segm',None),('BBoxAP','bbox',None),
                                   ('BoundaryAP2','segm',.02),('BoundaryAP1','segm',.01)]:
            points,boot={},{}
            for key,rows in data.items():
                mapped=[dict(r,dt=r['original_path']) for r in rows] if path=='original_path' else rows
                ev=evaluator(mapped,kind=kind,ratio=ratio);points[key]=precision_ap(ev)
                assert abs(fast_sample_ap(ev,np.arange(459))-points[key])<1e-12
                boot[key]=np.array([fast_sample_ap(ev,ix) for ix in samples])
                del ev
                print(path,metric,key,points[key],flush=True)
            comparisons={}
            for name,weights in contrasts.items():
                difference=sum(weights[k]*points[k] for k in weights)
                draws=sum(weights[k]*boot[k] for k in weights)
                comparisons[name]=dict(coefficients=weights,delta=difference,
                    paired_image_bootstrap95=np.quantile(draws,[.025,.975]).tolist())
            result['paths'][path][metric]=dict(points=points,comparisons=comparisons)
            (out/'summary.partial.json').write_text(json.dumps(result,indent=2)+'\n')
    result['fast_accumulation_calibration']=audits
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
