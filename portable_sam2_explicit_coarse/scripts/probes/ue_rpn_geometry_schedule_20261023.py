"""Measured shared-GPU owner for the fixed three-arm RPN geometry study."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import torch
import ue_rpn_geometry_probe_20261023 as run

R=run.STUDY;ROOT=run.ROOT;PY='/data/wangcheng/envs/cvt2/bin/python'
def read(p):return json.loads(p.read_text())
def sha(p):return run.file_sha(p)
def alive(pid):
    p=Path('/proc')/str(pid)/'status'
    return p.exists() and '\nState:\tZ' not in p.read_text()

def collect(jobs,manifest):
    init=torch.load(R/'common_init.pth',map_location='cpu',weights_only=False)
    orders=[];audits={}
    for arm in ['geometry','constant','zero']:
        folder=R/arm;cp=torch.load(folder/'last.pth',map_location='cpu',weights_only=False)
        assert cp['epoch']==10 and cp['step']==2670
        c=cp['contract'];assert c['initial_fingerprint']==init['fingerprint']
        assert c['init_file_sha256']==sha(R/'common_init.pth') and c['base_sha256']==sha(run.BASE)
        assert c['archive_hashes']==run.archive_identity(run.DATA)
        assert all(sha(ROOT/p)==h for p,h in c['source_hashes'].items())
        rows=[json.loads(s) for s in (folder/'train.jsonl').read_text().splitlines()]
        assert len(rows)==2670 and [r['update'] for r in rows]==list(range(1,2671))
        order=[]
        for epoch in range(1,11):
            s=read(folder/f'epoch{epoch}_summary.json')
            assert s['update']==epoch*267 and s['microbatches']==2136 and s['satellite_state_sha256']==run.FROZEN_SHA
            assert hashlib.sha256(json.dumps(s['sample_order']).encode()).hexdigest()==s['sample_order_sha256']
            assert all(r['epoch_seed']==1044000+epoch for r in rows[(epoch-1)*267:epoch*267])
            order.append(s['sample_order_sha256'])
        orders.append(order);audits[arm]=dict(checkpoint_sha256=sha(folder/'last.pth'),updates=2670,epoch_sample_order=order)
    assert all(o==orders[0] for o in orders)
    for j in jobs.values():
        if j['key'].startswith('eval_'):
            p=Path(j['artifact']);d=read(p);assert d['images']==459 and d['satellite_state_unchanged']
            assert d['nms_pre']==(2000 if 'budget' in j['key'] else 1000)
            assert d['rpn_postcap']==1000 and d['max_dets']==[1,10,250]
            if '--adapter' in j['cmd']:
                ap=Path(j['cmd'][j['cmd'].index('--adapter')+1]);assert d['checkpoint_sha256']==sha(ap)
    lines=['# UAV几何辅助RPN：固定三臂结果','','全部E10/2670及逐轮样本/冻结来源已核定。单seed、开发test名义区间，不证明训练重复性。','',
        '|对比|Mask AP增量（百分点）|95%区间|BBox AP增量（百分点）|','|---|---:|---|---:|']
    for key in ['G_sat','G_K','G_Z','K_sat','Z_sat','budget_sat','dependency_constant','dependency_zero','dependency_shuffle']:
        d=read(R/'analysis'/(key+'.json'));m=d['metrics']['MaskAP'];b=d['metrics']['BBoxAP']
        lines.append(f"|{key}|{100*m['delta']:+.4f}|{[round(100*x,4) for x in m['paired_image_bootstrap95']]}|{100*b['delta']:+.4f}|")
    lines.extend(['','RPN/最终召回与完整阶段每GT最大IoU、一对一容量保存在每项records.json；检测输出允许变化。',
        'G_sat与G_K为主要归因比较；K/Z保留目标support和卫星query。预算2000是纯卫星候选诊断，不是UAV模型。',
        '读取点几何不能代表整栋建筑。RPN召回改善不保证经过原RoI NMS后仍保留，也不能替代最终Mask AP。'])
    (R/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    run.dump(R/'complete_audit_summary.json',dict(complete=True,training=audits,matching_initialization_and_sample_order=True,
        manifest_sha256=sha(R/'manifest.json'),source_unchanged=all(sha(ROOT/p)==h for p,h in manifest['source_hashes'].items())))


def main():
    R.mkdir(exist_ok=True);lock=(R/'owner.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    gatefile=R/'gate/gpu_gate.json'
    while not gatefile.exists():
        run.dump(R/'state.json',dict(stage='waiting_engineering_gate',owner_pid=os.getpid()));time.sleep(10)
    gate=read(gatefile);assert gate['passed'] and gate['anchor_targets_and_sampling_exact'] and gate['resume_mini_epochs_exact']
    assert all(sha(ROOT/p)==h for p,h in gate['source_hashes'].items())
    paths=run.SOURCES+['scripts/probes/ue_rpn_geometry_schedule_20261023.py','scripts/probes/ue_rpn_geometry_analysis_20261023.py','scripts/probes/ue_v1_adapter_analysis.py','scripts/probes/ue_rpn_rank_reachability_20261023.py']
    hashes={p:sha(ROOT/p) for p in paths}
    for p,h in read(Path('/data/wangcheng/checkpoint/ue_s2db_probe_20261022/manifest.json'))['source_hashes'].items():
        assert sha(ROOT/p)==h;hashes[p]=h
    manifest=dict(study='rpn_native_geometry_objectness_v1',arms=['geometry','constant','zero'],
        constants_sha256=sha(Path('/data/wangcheng/checkpoint/ue_geometry_mechanism_20261022/common_constants.json')),
        init_sha256=sha(R/'common_init.pth'),gate_sha256=sha(gatefile),source_hashes=hashes,
        epochs=10,updates=2670,selection='E10 last',main_comparisons=['G_sat','G_K'],
        additional_comparisons=['G_Z','K_sat','Z_sat','budget_sat','dependency_constant','dependency_zero','dependency_shuffle'],
        no_ema=True,epoch_seed_formula='1044000+epoch',data=run.archive_identity(run.DATA),
        reserve_MiB=int(max(3000,gate['peak_reserved_MiB']*1.25)),device_margin_MiB=2048,
        semantic_limits='Only RPN logits trained; all original parameters frozen; unchanged proposal budgets except explicit SAT-only2000',
        telemetry='Training latest-forward 5-level sums only; all ranking/GT-stage observations offline')
    if (R/'manifest.json').exists():assert read(R/'manifest.json')==manifest
    else:run.dump(R/'manifest.json',manifest)
    # Initialization parameters predate gate revisions; bind the final code hash separately.
    run.dump(R/'final_source_freeze.json',dict(source_hashes=hashes,initialization_file_unchanged=True,init_sha256=manifest['init_sha256']))
    jobs={};script=str(ROOT/'scripts/probes/ue_rpn_geometry_probe_20261023.py')
    def add(k,cmd,artifact,deps=(),gpu=True,priority=10):
        jobs[k]=dict(key=k,cmd=cmd,artifact=str(artifact),dependencies=list(deps),gpu_required=gpu,priority=priority,status='pending')
    for arm in manifest['arms']:
        train='train_'+arm;folder=R/arm
        add(train,[PY,script,'train','--content',arm,'--out',str(folder)],folder/'last.pth',priority=0)
        for split in ['val','test']:
            modes=['correct' if arm=='geometry' else arm]
            if arm=='geometry' and split=='test':modes+=['constant','zero','shuffle']
            for mode in modes:
                dest=R/split/arm/mode;k=f'eval_{split}_{arm}_{mode}'
                add(k,[PY,script,'eval','--adapter',str(folder/'last.pth'),'--mode',mode,'--eval-split',split,'--out',str(dest)],dest/'evaluation_identity.json',[train],priority=5 if mode==modes[0] else 9)
    add('rank_reachability',[PY,str(ROOT/'scripts/probes/ue_rpn_rank_reachability_20261023.py')],R/'rank/reachability.json',priority=1)
    for split in ['val','test']:
        for name,n in [('baseline',1000),('budget',2000)]:
            dest=R/split/name
            add(f'eval_{split}_{name}',[PY,script,'eval','--mode','disabled','--eval-split',split,'--nms-pre',str(n),'--out',str(dest)],dest/'evaluation_identity.json',priority=7)
    def compare(key,left,right,deps):
        output=R/'analysis'/(key+'.json')
        add('analysis_'+key,[PY,str(ROOT/'scripts/probes/ue_rpn_geometry_analysis_20261023.py'),'--baseline',str(left),'--candidate',str(right),'--out',str(output)],output,deps,gpu=False,priority=12)
    base=R/'test/baseline/records.json';g=R/'test/geometry/correct/records.json';k=R/'test/constant/constant/records.json';z=R/'test/zero/zero/records.json'
    for key,left,right,deps in [
        ('G_sat',base,g,['eval_test_baseline','eval_test_geometry_correct']),
        ('G_K',k,g,['eval_test_constant_constant','eval_test_geometry_correct']),
        ('G_Z',z,g,['eval_test_zero_zero','eval_test_geometry_correct']),
        ('K_sat',base,k,['eval_test_baseline','eval_test_constant_constant']),
        ('Z_sat',base,z,['eval_test_baseline','eval_test_zero_zero']),
        ('budget_sat',base,R/'test/budget/records.json',['eval_test_baseline','eval_test_budget'])]:compare(key,left,right,deps)
    for mode in ['constant','zero','shuffle']:
        compare('dependency_'+mode,R/'test/geometry'/mode/'records.json',g,['eval_test_geometry_'+mode,'eval_test_geometry_correct'])
    if (R/'state.json').exists() and 'jobs' in read(R/'state.json'):
        old=read(R/'state.json');assert set(old['jobs'])==set(jobs);jobs=old['jobs']
    state=dict(stage='running',owner_pid=os.getpid(),jobs=jobs);children={};streams={}
    def save():run.dump(R/'state.json',state)
    save()
    while True:
        assert all(sha(ROOT/p)==h for p,h in hashes.items()),'Frozen source changed'
        for key,j in jobs.items():
            if j['status']!='running':continue
            rc=children[key].poll() if key in children else (None if alive(j['pid']) else 0)
            if rc is None:continue
            if key in streams:streams.pop(key).close()
            children.pop(key,None)
            j.update(status='complete' if rc==0 and Path(j['artifact']).exists() else 'failed',returncode=rc)
            print('END',key,j['status'],flush=True);save()
        if any(j['status']=='failed' for j in jobs.values()):state['stage']='failure_requires_audit';save()
        else:
            gpu_lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
            free={int(a):int(b) for a,b in (line.split(',') for line in gpu_lines.splitlines())}
            running=[j for j in jobs.values() if j['status']=='running'];busy={j['gpu'] for j in running if j['gpu_required']}
            cpu=sum(not j['gpu_required'] for j in running)
            for j in sorted(jobs.values(),key=lambda j:j['priority']):
                if j['status']!='pending' or not all(jobs[d]['status']=='complete' for d in j['dependencies']):continue
                gpu=None
                if j['gpu_required']:
                    available=[i for i in free if i not in busy and free[i]>=manifest['reserve_MiB']+2048]
                    if not available:continue
                    gpu=max(available,key=lambda i:free[i]);busy.add(gpu);free[gpu]-=manifest['reserve_MiB']
                elif cpu>=2:continue
                else:cpu+=1
                path=R/'logs'/(j['key']+'.log');path.parent.mkdir(exist_ok=True)
                stream=path.open('ab');env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu))
                proc=subprocess.Popen(j['cmd'],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                children[j['key']]=proc;streams[j['key']]=stream;j.update(status='running',gpu=gpu,pid=proc.pid,log=str(path))
                print('START',j['key'],gpu,proc.pid,flush=True);save()
        if all(j['status']=='complete' for j in jobs.values()):
            collect(jobs,manifest);state['stage']='complete';save();return
        time.sleep(10)

if __name__=='__main__':main()
