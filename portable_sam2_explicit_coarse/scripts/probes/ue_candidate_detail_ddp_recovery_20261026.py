"""Audited control-plane recovery; keep all original numerical sources immutable."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from ue_candidate_detail_ddp_20261026 import ROOT, WORK, PROTOCOL, hashes, base

PYTHON = '/data/wangcheng/envs/cvt2/bin/python'
RUN = WORK/'scripts/probes/ue_candidate_detail_ddp_20261026.py'
ANALYSIS = WORK/'scripts/probes/ue_candidate_detail_ddp_analysis_20261026.py'


def launch_command(action, variant, content, split='val', mode='matched'):
    return [PYTHON, '-u', '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
        '--nproc_per_node=4', str(RUN), action, '--variant', variant, '--content', content,
        '--split', split, '--mode', mode, '--root', str(ROOT)]


def main():
    global ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    ROOT = parser.parse_args().root
    ROOT.mkdir(parents=True, exist_ok=True)
    lock = (ROOT/'owner.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert (ROOT/'state.json').exists(), 'Recovery requires original execution state'
    old_state = json.loads((ROOT/'state.json').read_text())
    assert not Path('/proc', str(old_state['owner_pid'])).exists(), 'Original owner still exists'
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            cmd = (proc/'cmdline').read_bytes().split(b'\0')
        except OSError:
            continue
        if any(x.endswith(b'/ue_candidate_detail_ddp_20261026.py') for x in cmd):
            assert str(ROOT).encode() not in cmd, 'Existing numerical child requires explicit audit'
    engineering = json.loads((ROOT/'engineering_complete.json').read_text())
    repro = json.loads((ROOT/'reproducibility.json').read_text())
    source = hashes()
    assert engineering['passed'] and engineering['source_hashes'] == source
    assert repro['git_sources_exact'] and repro['source_hashes'] == source
    local_sources = repro['local_source_sha256']
    assert all(base.core.C.file_sha(WORK.parent/n) == sha for n,sha in local_sources.items())
    assert all(base.core.C.file_sha(n) == sha for n,sha in repro['epoch_order_source_sha256'].items())
    working = engineering['per_GPU_reservation_MiB']
    jobs = {}
    def add(name, command, marker, dependencies=(), launch_record=None):
        jobs[name] = dict(command=command, marker=str(marker), dependencies=list(dependencies),
            launch_record=str(launch_record) if launch_record else None, status='pending')
    training = []
    for variant,content in PROTOCOL['training_jobs']:
        name = variant+'_'+content+'_train'
        training.append(name)
        add(name, launch_command('train',variant,content), ROOT/variant/content/'training_complete.json',
            launch_record=ROOT/'rank_launches'/f'train_{variant}_{content}_val_matched.json')
    for split in ['val','test']:
        evaluations = []
        for variant,content in PROTOCOL['training_jobs']:
            modes = ['matched','constant','donor'] if (variant,content) == ('both','correct') else ['matched']
            for mode in modes:
                name = '_'.join([variant,content,split,mode])
                add(name, launch_command('eval',variant,content,split,mode),
                    ROOT/'evaluation'/split/variant/content/mode/'complete.json',
                    training if split == 'test' else [variant+'_'+content+'_train'],
                    ROOT/'rank_launches'/f'eval_{variant}_{content}_{split}_{mode}.json')
                evaluations.append(name)
        add('analysis_'+split, [PYTHON,'-u',str(ANALYSIS),'--split',split,'--root',str(ROOT)],
            ROOT/'analysis'/split/'summary.json', evaluations)
    manifest = dict(protocol=PROTOCOL, source_hashes=source, local_source_sha256=local_sources,
        git_commit=repro['git_commit'], engineering_sha256=base.core.C.file_sha(ROOT/'engineering_complete.json'),
        reproducibility_sha256=base.core.C.file_sha(ROOT/'reproducibility.json'),
        groups=10, per_group_world_size=4, tasks=len(jobs), per_GPU_reservation_MiB=working,
        GPU_indices=[0,1,2,3], margin_MiB=PROTOCOL['device_margin_MiB'],
        schedule='admit a group only when every GPU has headroom; no fixed concurrent-group cap')
    original_manifest = json.loads((ROOT/'manifest.json').read_text())
    assert manifest == original_manifest, 'Original contract changed'
    original_plan = json.loads((ROOT/'job_plan.json').read_text())
    assert jobs == original_plan, 'Original task DAG changed'
    recovered = []
    for name,job in jobs.items():
        marker = Path(job['marker'])
        if not marker.exists():
            if name.endswith('_train'):
                assert not (marker.parent/'last.pth').exists(), 'Partial train needs separate resume gate'
            continue
        receipt = json.loads(marker.read_text())
        if name.endswith('_train'):
            variant,content,_ = name.split('_')
            assert receipt['complete'] and receipt['epoch'] == 10 and receipt['updates'] == 2670
            assert receipt['world_size'] == 4 and receipt['contract']['source_hashes'] == source
            assert receipt['contract'] == json.loads((marker.parent/'contract.json').read_text())
            assert receipt['checkpoint_sha256'] == base.core.C.file_sha(marker.parent/'epoch10.pth')
            for epoch in range(1,11):
                e = json.loads((marker.parent/f'epoch{epoch}.json').read_text())
                assert e['epoch'] == epoch and e['step'] == epoch*267 and e['world_size'] == 4
                assert len(e['sample_order']) == len(set(e['sample_order'])) == 2136
        else:
            raise AssertionError('Unexpected preexisting evaluation requires separate audit')
        job.update(status='complete', recovered_from_verified_receipt=True)
        recovered.append(name)
    archive = ROOT/'recovery_20261026'
    archive.mkdir(exist_ok=False)
    shutil.copyfile(ROOT/'state.json', archive/'previous_state.json')
    shutil.copyfile(ROOT/'owner_launch.json', archive/'previous_owner_launch.json')
    own_source = base.core.C.file_sha(__file__)
    base.core.C.dump(archive/'audit.json', dict(previous_owner_pid=old_state['owner_pid'],
        previous_owner_absent=True, numerical_children_absent=True,
        recovered_completed_jobs=recovered, source_sha256=own_source,
        frozen_local_sources_verified=len(local_sources), training_git_commit=repro['git_commit'],
        control_git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=WORK,text=True).strip(),
        root_cause='Original owner disappeared without exit receipt; cause not established',
        started=time.time(), new_owner_pid=os.getpid()))
    shutil.copyfile(__file__, archive/Path(__file__).name)
    base.core.C.dump(ROOT/'owner_recovery_launch.json', dict(pid=os.getpid(),started=time.time(),
        source_sha256=own_source,recovered_jobs=recovered))
    children,streams = {},{}
    while True:
        assert base.core.C.file_sha(__file__) == own_source
        for name,child in list(children.items()):
            rc = child.poll()
            if rc is None:
                continue
            streams.pop(name).close()
            del children[name]
            marker = Path(jobs[name]['marker'])
            passed = rc == 0 and marker.exists()
            if passed and name.endswith('_train'):
                receipt = json.loads(marker.read_text())
                passed = receipt.get('world_size') == 4 and receipt.get('updates') == 2670 and receipt.get('epoch') == 10
            jobs[name].update(status='complete' if passed else 'failed', returncode=rc)
            print('END',name,rc,jobs[name]['status'],flush=True)
        failed = [n for n,j in jobs.items() if j['status'] == 'failed']
        if not failed:
            assert hashes() == source
            assert all(base.core.C.file_sha(WORK.parent/n) == sha for n,sha in local_sources.items())
            assert all(base.core.C.file_sha(n) == sha for n,sha in repro['epoch_order_source_sha256'].items())
            raw = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free',
                '--format=csv,noheader,nounits'],text=True)
            free = {int(r.split(',')[0]):int(r.split(',')[1])-PROTOCOL['device_margin_MiB'] for r in raw.splitlines()}
            raw = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory',
                '--format=csv,noheader,nounits'],text=True)
            used = {int(r.split(',')[0]):int(r.split(',')[1]) for r in raw.splitlines() if r.split(',')[1].strip().isdigit()}
            for name in children:
                record = jobs[name]['launch_record']
                if record is None:
                    continue
                path = Path(record)
                rank_pids = json.loads(path.read_text())['ranks'] if path.exists() else []
                for gpu in range(4):
                    pid = next((r['pid'] for r in rank_pids if r['local_rank'] == gpu), None)
                    free[gpu] -= max(0,working-used.get(pid,0))
            cpu_running = any(jobs[n]['launch_record'] is None for n in children)
            for name,job in jobs.items():
                if job['status'] != 'pending' or any(jobs[d]['status'] != 'complete' for d in job['dependencies']):
                    continue
                env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
                if job['launch_record']:
                    if any(free[g] < working for g in range(4)):
                        continue
                    for gpu in range(4):
                        free[gpu] -= working
                    env['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
                else:
                    if cpu_running:
                        continue
                    cpu_running = True
                    env['CUDA_VISIBLE_DEVICES'] = ''
                log = ROOT/'logs'/(name+'.log')
                log.parent.mkdir(exist_ok=True)
                stream = log.open('w')
                child = subprocess.Popen(job['command'],cwd=WORK,env=env,stdin=subprocess.DEVNULL,
                    stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                children[name],streams[name] = child,stream
                job.update(status='running',launcher_pid=child.pid,log=str(log),
                    GPU_indices=[0,1,2,3] if job['launch_record'] else [])
                print('START',name,'four GPUs' if job['launch_record'] else 'CPU',child.pid,flush=True)
        complete = all(j['status'] == 'complete' for j in jobs.values())
        stage = 'complete' if complete else ('failure_requires_audit' if failed else 'running')
        base.core.C.dump(ROOT/'state.json',dict(stage=stage,owner_pid=os.getpid(),jobs=jobs))
        if complete:
            base.core.C.dump(ROOT/'complete.json',dict(complete=True,tasks=len(jobs),git_commit=repro['git_commit'],
                final_scientific_review_required=True))
            return
        if failed and not children:
            raise RuntimeError(failed)
        time.sleep(15)


if __name__ == '__main__':
    main()
