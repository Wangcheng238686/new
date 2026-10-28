"""Fixed-checkpoint native geometry interventions; no training or model selection."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts/probes'),str(ROOT)]
MODES=['correct','zero','shuffle','zero_rms','shuffle_rms']
BASE=Path('/data/wangcheng/checkpoint/ue_minimal_single_20261021_EXP-0017/checkpoints/best_model_epoch73.pth')
ADAPTER=Path('/data/wangcheng/checkpoint/ue_v1_adapter_probe_20261021/native_geometry/last.pth')
DATA=Path('/data/wangcheng/dataset/CitySample_SatUAV_TrainMinimal_v1_20261021')
CACHE=Path('/data/wangcheng/cache/ue_v1_native_geometry')
OLD=Path('/data/wangcheng/checkpoint/ue_v1_adapter_probe_20261021/test')


def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def intervene(fields,support,mode,point):
    import numpy as np
    out=fields.copy()
    if mode.startswith('zero'):out[:]=0
    elif mode.startswith('shuffle'):
        valid=np.flatnonzero(support.reshape(-1));rng=np.random.default_rng(20261022+int(point))
        perm=rng.permutation(len(valid));flat=out.reshape(4,-1)
        flat[:,valid]=fields.reshape(4,-1)[:,valid[perm]]
    assert np.isfinite(out).all() and (out[:,~support]==0).all()
    return out


def gate():
    import numpy as np
    support=np.array([[1,0,1],[1,1,0]],dtype=bool)
    fields=np.arange(24,dtype=np.float32).reshape(4,2,3);fields[:,~support]=0
    a=intervene(fields,support,'shuffle',17);b=intervene(fields,support,'shuffle_rms',17)
    assert np.array_equal(a,b) and not np.array_equal(a,fields)
    assert sorted(map(tuple,a[:,support].T))==sorted(map(tuple,fields[:,support].T))
    assert np.array_equal(intervene(fields,support,'correct',17),fields)
    assert not intervene(fields,support,'zero',17).any()
    return dict(joint_field_multiset_exact=True,support_unchanged=True,deterministic=True,invalid_zero=True)


def evaluate(a):
    import ue_v1_adapter as common
    import numpy as np
    import torch
    from rsprompter.uav.native_geometry_probe import NativeGeometryAdapter,prepare_native

    class DiagnosticAdapter(NativeGeometryAdapter):
        def encode(self,unused,support,fields,high):
            super().encode(unused,support,fields,high)
            self.calls=[]
        def _inject(self,module,args,base):
            if not self.enabled or self.context is None:return base
            support,reference,high=self.context;q=self.query(high)
            def delta(fields):return torch.cat([self._chunk(q,fields,support,b,base.shape[-1]) for b in self.boxes.split(4)])
            ref=delta(reference)
            altered=ref if a.mode=='correct' else delta(self.altered)
            rr=ref.float().square().mean().sqrt();ar=altered.float().square().mean().sqrt()
            scale=rr/ar.clamp_min(1e-12) if a.mode.endswith('_rms') else torch.ones_like(rr)
            used=altered*scale
            applied=self.alpha*used
            nr=ref.float().flatten().norm();na=altered.float().flatten().norm()
            cosine=(ref.float().flatten()@altered.float().flatten())/(nr*na).clamp_min(1e-12)
            self.stats=dict(alpha=float(self.alpha),support_fraction=float(support.mean()),
                source_delta_rms=float(used.float().square().mean().sqrt()),
                applied_delta_rms=float(applied.float().square().mean().sqrt()),
                base_rms=float(base.float().square().mean().sqrt()),rois=len(base),support_grid=512,
                raw_intervention_rms=float(ar),reference_delta_rms=float(rr),rms_match_scale=float(scale),
                residual_cosine_to_correct=float(cosine),
                residual_change_rms=float((used-ref).float().square().mean().sqrt()))
            self.calls.append(dict(self.stats))
            return base+applied

    common.seed(44)
    a.base=BASE;a.data_root=DATA;a.cache=CACHE;a.adapter=ADAPTER
    a.source_mode='fine';a.route='highres';a.seed=44;a.appearance='correct';a.eval_split='test';a.max_images=0
    cp=torch.load(ADAPTER,map_location='cpu',weights_only=False)
    assert cp['contract']['base_sha256']==sha(BASE)
    model,unused,snapshot=common.make_model(a)
    for h in unused.handles:h.remove()
    del unused;torch.cuda.empty_cache()
    common.seed(45);adapter=DiagnosticAdapter().cuda();adapter.attach(model)
    adapter.load_state_dict(cp['adapter'],strict=True);del cp
    before=common.state_digest(model)
    summary=[]
    def payload(folder,cache,appearance='correct',donor=None):
        p=prepare_native(folder,cache)
        with np.load(p) as d:
            support=d['support'].astype(bool);fields=d['fields'].astype(np.float32)
        changed=intervene(fields,support,a.mode,int(folder.name))
        adapter.altered=torch.from_numpy(changed.copy())[None].cuda()
        return None,torch.from_numpy(support.astype(np.float32))[None,None].cuda(),torch.from_numpy(fields)[None].cuda()
    common.payload=payload
    # Snapshot all mask calls, rather than overwriting the legacy last-call telemetry.
    original_predict=model.predict
    def predict(*args,**kw):
        result=original_predict(*args,**kw)
        summary.append(dict(calls=list(adapter.calls)))
        return result
    model.predict=predict
    common.evaluate(a,model,adapter,common.MinimalSatelliteDataset(DATA,'test'),'correct',a.out)
    assert common.state_digest(model)==before
    records=json.loads((a.out/'records.json').read_text());assert len(summary)==len(records)==459
    for row,calls in zip(records,summary):calls['image_id']=row['image_id']
    write(a.out/'all_call_telemetry.json',summary)
    old=json.loads((OLD/'native_geometry/correct/records.json').read_text())
    for b,c in zip(old,records):
        assert b['image_id']==c['image_id'] and b['gt']==c['gt']
        for k in ['bboxes','labels','scores']:assert b['dt'][k]==c['dt'][k]
        if a.mode=='correct':assert b['dt']==c['dt'],'correct predictions differ from archived checkpoint'
    write(a.out/'diagnostic_identity.json',dict(intervention=a.mode,split='test',images=459,
        checkpoint_sha256=sha(ADAPTER),base_sha256=sha(BASE),runner_sha256=sha(__file__),
        satellite_state_unchanged=True,detector_gt_exact=True,
        correct_predictions_exact_to_archive=a.mode=='correct',gate=gate(),
        scope='fixed checkpoint diagnostic only; RMS-matched variants are not new model scores'))


def run(a):
    a.out.mkdir(parents=True,exist_ok=True)
    assert not (a.out/'plan.json').exists(),'Use a new immutable diagnostic folder'
    plan=dict(modes=MODES,seed=20261022,split='test',images=459,base_sha256=sha(BASE),
        adapter_sha256=sha(ADAPTER),runner_sha256=sha(__file__),gpus=a.gpus,
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'scripts/probes/ue_v1_adapter.py',ROOT/'scripts/probes/ue_v1_adapter_analysis.py',ROOT/'rsprompter/uav/native_geometry_probe.py']},
        registered_utc=time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime()),gate=gate(),
        permutation='per-image seeded joint four-vector shuffle among target supported pixels; support/field multiset preserved',
        rms_matching='per actual mask-head call, match pre-alpha residual RMS to correct fields, no clipping; diagnostic only',
        comparisons='each intervention vs fixed correct; correct vs satellite baseline',
        limitations='shuffle destroys spatial autocorrelation and derivative consistency, and is out of distribution; zero changes input distribution; no retraining')
    write(a.out/'plan.json',plan)
    (a.out/'runner.source.py').write_text(Path(__file__).read_text())
    jobs=list(MODES);results={};active={}
    while jobs or active:
        for gpu in a.gpus:
            if gpu in active or not jobs:continue
            # Co-residency allowed; leave >=8GiB visible headroom for the inference worker.
            used,total=map(int,subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.used,memory.total','--format=csv,noheader,nounits'],text=True).strip().split(','))
            if total-used<8192:continue
            mode=jobs.pop(0);env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
            log=(a.out/f'{mode}.log').open('a')
            p=subprocess.Popen([sys.executable,__file__,'eval','--mode',mode,'--out',str(a.out/mode)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            log.close();active[gpu]=(p,mode)
        for gpu,(p,mode) in list(active.items()):
            if p.poll() is not None:
                results[mode]=dict(returncode=p.returncode,identity=(a.out/mode/'diagnostic_identity.json').exists());del active[gpu]
                if p.returncode:write(a.out/'state.json',dict(stage='failed',results=results));raise RuntimeError(mode)
        write(a.out/'state.json',dict(stage='evaluating',owner_pid=os.getpid(),queued=jobs,
            active={str(g):dict(pid=p.pid,mode=m) for g,(p,m) in active.items()},results=results))
        if jobs or active:time.sleep(5)
    def analysis(item):
        name,baseline,candidate=item;path=a.out/f'{name}_paired_analysis.json'
        env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        with (a.out/f'{name}_analysis.log').open('w') as f:
            subprocess.run([sys.executable,str(ROOT/'scripts/probes/ue_v1_adapter_analysis.py'),'--baseline',str(baseline),'--candidate',str(candidate),'--output',str(path),'--split','test'],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        return name
    comparisons=[(m,a.out/'correct/records.json',a.out/m/'records.json') for m in MODES if m!='correct']
    comparisons.append(('correct_vs_satellite',OLD/'baseline/disabled/records.json',a.out/'correct/records.json'))
    write(a.out/'state.json',dict(stage='analysis',owner_pid=os.getpid(),results=results))
    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(analysis,comparisons))
    write(a.out/'state.json',dict(stage='complete',owner_pid=os.getpid(),results=results,analyses=[x[0] for x in comparisons]))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run','eval']);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--mode',choices=MODES,default='correct');p.add_argument('--gpus',nargs='+',type=int,default=[0,2]);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    if a.action=='run':run(a)
    else:evaluate(a)

if __name__=='__main__':main()
