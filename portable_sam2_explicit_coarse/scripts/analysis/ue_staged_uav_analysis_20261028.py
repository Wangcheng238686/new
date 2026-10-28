"""Explicit CPU-only staged test analysis. Never a loss, gate or model selector.

Validation selects checkpoints/base before test exposure. This script reads only
fixed completed evaluations, performs500 same-image paired bootstraps, and writes
independent descriptive artifacts. No fixed E10 or hardcoded source-count audit.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[_key]='1'
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np
WORK=Path(__file__).resolve().parents[2]
METRICS=(('MaskAP','segm',None),('BBoxAP','bbox',None),('BoundaryAP_2pct','segm',.02),('BoundaryAP_1pct','segm',.01))
SEED=20261021
DRAWS=500


def read(path):return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp.'+str(os.getpid()))
    with temp.open('x') as stream:json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    os.replace(temp,path)


def resolve_stage(manifest,stage):
    """Pure recipe/contrast resolution; positive differences favor full evidence."""
    if stage not in ('structure','modality','views'):raise ValueError('Unknown stage')
    members=manifest['stages'][stage]
    if stage=='structure':
        if members!=['S','U','T','C']:raise ValueError('Expected complete S/U/T/C structure stage')
        names=list(members)
        weights={label:{plus:1,minus:-1} for label,plus,minus in [('U-S','U','S'),('T-S','T','S'),('C-S','C','S'),('C-U','C','U'),('C-T','C','T')]}
        weights['C-U-T+S']={'C':1,'U':-1,'T':-1,'S':1}
    else:
        lock=manifest.get('base_lock')
        if not lock or lock['base_arm'] not in ('U','T','C'):raise ValueError('Locked validation-selected U/T/C base required')
        base=lock['base_arm'];expected=['mod_constant','mod_no_geometry'] if stage=='modality' else ['views_1','views_3']
        if members!=expected:raise ValueError('Unexpected follow-up stage members')
        names=[base]+list(members)
        weights={base+'-'+name:{base:1,name:-1} for name in members}
        recipe=manifest['recipes'][base]
        if recipe['uav_view_ids']!=[0,72,144,216,288] or recipe['rgb_content']!='correct' or recipe['geometry_content']!='on':
            raise ValueError('Locked base must be complete correct five-view content')
    if any(name not in manifest['recipes'] for name in names):raise ValueError('Recipe unresolved')
    return dict(stage=stage,names=names,recipes={n:manifest['recipes'][n] for n in names},contrasts=weights,
                interaction='same draws C-U-T+S; U includes interaction, instance objective and supplementary candidates' if stage=='structure' else None)


def validate_records(data,images=459):
    reference=None
    for name,records in data.items():
        if len(records)!=images or len({r['image_id'] for r in records})!=images:raise ValueError('Expected unique full%d records: '%images+name)
        if reference is None:reference=records
        if [r['image_id'] for r in records]!=[r['image_id'] for r in reference]:raise ValueError('Paired image order differs: '+name)
        if any(r['gt']!=b['gt'] for r,b in zip(records,reference)):raise ValueError('Paired GT differs: '+name)
        if any('dt' not in r for r in records):raise ValueError('Missing combined prediction records')


def controller_module():
    path=WORK/'scripts/experiments/ue_staged_uav_control.py'
    spec=importlib.util.spec_from_file_location('staged_analysis_control',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def load_inputs(root,manifest,plan,controller):
    controller.validate(manifest)
    if Path(manifest['root']).resolve()!=root.resolve():raise ValueError('Manifest root mismatch')
    evidence={str((root/'launch_manifest.json').resolve()):sha(root/'launch_manifest.json')}
    data={};checkpoints={}
    for name in plan['names']:
        if not controller.accepted(manifest,'train',name) or not controller.accepted(manifest,'eval',name):
            raise RuntimeError('Accepted full training and fixed best-checkpoint evaluation required: '+name)
        train=root/name/'training_complete.json';receipt=root/'evaluation/test'/name/'complete.json'
        checkpoint=root/name/'best_model.pth';records=root/'evaluation/test'/name/'records.json'
        digest=sha(checkpoint);trained=read(train);evaluated=read(receipt)
        if digest!=trained['best_checkpoint_sha256'] or digest!=evaluated['checkpoint_sha256']:raise RuntimeError('Checkpoint receipt mismatch: '+name)
        checkpoints[name]=dict(sha256=digest,best=trained.get('best'),recipe_sha256=manifest['recipes'][name]['recipe_sha256'])
        for path in (train,receipt,checkpoint,records):evidence[str(path.resolve())]=digest if path==checkpoint else sha(path)
        data[name]=read(records)
    validate_records(data)
    return data,evidence,checkpoints


def contrast_result(points,boot,weights):
    point=sum(weight*points[name]['AP'] for name,weight in weights.items())
    samples=sum(weight*boot[name] for name,weight in weights.items())
    return dict(weights=weights,delta=point,paired_image_bootstrap95=np.quantile(samples,[.025,.975]).tolist(),
                bootstrap_samples=DRAWS,seed=SEED)


def run(root,stage,plan_only=False):
    manifest=read(root/'launch_manifest.json');plan=resolve_stage(manifest,stage)
    if plan_only:
        print(json.dumps(dict(plan_only=True,plan=plan,bootstrap_samples=DRAWS,seed=SEED,metrics=[m[0] for m in METRICS]),indent=2));return
    output=root/'analysis'/stage/'summary.json'
    if output.exists():raise RuntimeError('Preserve completed analysis: '+str(output))
    output.parent.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (output.parent/'analysis.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if output.exists():raise RuntimeError('Analysis completed while acquiring lock')
        controller=controller_module();data,evidence,checkpoints=load_inputs(root,manifest,plan,controller)
        sys.path[:0]=[str(WORK),str(WORK/'scripts/probes'),str(WORK/'scripts/analysis')]
        import ue_joint_encoding_fast_analysis as fast
        from ue_compact_joint_paired_20261024 import point_metrics
        analysis_sources={str(Path(p).resolve()):sha(p) for p in (Path(__file__),Path(controller.__file__),Path(fast.__file__),Path(fast.original.__file__),WORK/'scripts/analysis/ue_compact_joint_paired_20261024.py',WORK/'utils/coco_eval_utils.py')}
        samples=np.random.default_rng(SEED).integers(0,459,(DRAWS,459));fast.audits.clear();started=time.time()
        report=dict(complete=False,stage=stage,view='combined',images=459,split='test',max_dets=250,
            plan=plan,checkpoints=checkpoints,input_sha256=evidence,analysis_source_sha256=analysis_sources,
            source_sha256=manifest['source_sha256'],git_commit=manifest['git_commit'],base_lock=manifest.get('base_lock'),
            bootstrap_samples=DRAWS,bootstrap_seed=SEED,metrics={},
            metric_definition='COCO AP50:95 maxDets250; boundary AP uses min(maskIoU,inner-boundaryIoU), radius round(ratio*image diagonal): 512square2%=14px,1%=7px',
            selection='fixed per-recipe validation best; analysis never changes training/checkpoint/base selection',
            limitations='single training seed; nominal image bootstrap excludes training variance, spatial dependence and multiple comparisons; U is a system package; geometry-off retains measured correspondence/support')
        for metric,kind,ratio in METRICS:
            points={};boot={};prior=len(fast.audits)
            for name,rows in data.items():
                print('MATCH',stage,metric,name,flush=True);ev=fast.original.evaluator(rows,kind=kind,ratio=ratio)
                point=point_metrics(ev)
                if abs(fast.fast_sample_ap(ev,np.arange(459))-point['AP'])>=1e-12:raise RuntimeError('Point accumulator calibration failed')
                values=np.empty(DRAWS,np.float64)
                for index,indices in enumerate(samples):
                    values[index]=fast.fast_sample_ap(ev,indices)
                    if (index+1)%100==0:print('BOOTSTRAP',stage,metric,name,index+1,flush=True)
                points[name]=point;boot[name]=values;del ev
                report['in_progress']=dict(metric=metric,completed_recipes=list(points))
                write(output.with_name('summary.partial.json'),report)
            if len(fast.audits)-prior!=3*len(data):raise RuntimeError('Missing reference accumulation checks')
            report['metrics'][metric]=dict(points=points,contrasts={label:contrast_result(points,boot,weights) for label,weights in plan['contrasts'].items()})
            report['accumulation_reference_checks']=list(fast.audits);report['elapsed_seconds']=time.time()-started
            report.pop('in_progress',None);write(output.with_name('summary.partial.json'),report)
            print('RESULT',stage,metric,json.dumps(report['metrics'][metric]['contrasts']),flush=True)
        if any(r['max_abs_error']>=1e-12 for r in fast.audits):raise RuntimeError('Reference calibration mismatch')
        for path,digest in {**evidence,**analysis_sources}.items():
            if sha(path)!=digest:raise RuntimeError('Input/source changed during analysis: '+path)
        controller.validate(manifest);report['complete']=True;write(output,report)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--stage',required=True,choices=['structure','modality','views']);parser.add_argument('--plan-only',action='store_true')
    args=parser.parse_args();run(args.root.resolve(),args.stage,args.plan_only)

if __name__=='__main__':main()
