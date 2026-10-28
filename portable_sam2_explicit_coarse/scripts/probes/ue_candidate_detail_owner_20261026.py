"""Non-preemptive four-GPU co-renting, admitted by measured VRAM not slots."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from ue_candidate_detail_run_20261026 import ROOT, WORK, PROTOCOL, hashes, base

PYTHON = '/data/wangcheng/envs/cvt2/bin/python'
RUN = WORK/'scripts/probes/ue_candidate_detail_run_20261026.py'
ANALYSIS = WORK/'scripts/probes/ue_candidate_detail_analysis_20261026.py'


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    lock = (ROOT/'owner.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (ROOT/'state.json').exists(), 'Existing execution requires explicit recovery review'
    source = hashes()
    drivers = {str(p.relative_to(WORK)):base.core.C.file_sha(p) for p in [Path(__file__), ANALYSIS,
        WORK/'scripts/probes/ue_joint_encoding_analysis.py', WORK/'scripts/probes/ue_joint_encoding_fast_analysis.py']}
    gates = {}
    for v in PROTOCOL['variants']:
        p = ROOT/'gates'/v/'gpu_gate.json'
        g = json.loads(p.read_text())
        assert g['passed'] and g['source_hashes'] == source
        assert g['peak_reserved_MiB'] <= PROTOCOL['allocator_cap_MiB']
        gates[str(p)] = base.core.C.file_sha(p)
    cpu = json.loads((ROOT/'cpu_gate.json').read_text())
    assert cpu['passed'] and cpu['source_hashes'] == source
    gate_files = [ROOT/'cpu_gate.json', ROOT/'analysis_kernel_gate.json']
    for p in gate_files:
        receipt = json.loads(p.read_text())
        assert receipt['passed'] and receipt['source_hashes'] == source
        if 'analysis_sha256' in receipt:
            assert receipt['analysis_sha256'] == base.core.C.file_sha(ANALYSIS)
        gates[str(p)] = base.core.C.file_sha(p)
    working = PROTOCOL['allocator_cap_MiB']+512
    jobs = {}
    def add(name, command, marker, dependencies=(), gpu=True):
        jobs[name] = dict(command=command, marker=str(marker), dependencies=list(dependencies),
                          gpu=gpu, status='pending')
    training = []
    for variant,content in PROTOCOL['training_jobs']:
        name = variant+'_'+content+'_train'
        training.append(name)
        add(name, [PYTHON, '-u', str(RUN), 'train', '--variant', variant, '--content', content],
            ROOT/variant/content/'training_complete.json')
    evaluations = {'val':[], 'test':[]}
    for split in ('val', 'test'):
        for variant,content in PROTOCOL['training_jobs']:
            for mode in (['matched', 'constant', 'donor'] if (variant,content) == ('both','correct') else ['matched']):
                name = '_'.join((variant, content, split, mode))
                deps = training if split == 'test' else [variant+'_'+content+'_train']
                add(name, [PYTHON, '-u', str(RUN), 'eval', '--variant', variant, '--content', content,
                           '--split', split, '--mode', mode],
                    ROOT/'evaluation'/split/variant/content/mode/'complete.json', deps)
                evaluations[split].append(name)
        for content in ('satellite', 'correct', 'constant', 'plane'):
            name = 'reference_'+content+'_'+split
            add(name, [PYTHON, '-u', str(RUN), 'reference', '--content', content, '--split', split],
                ROOT/'reference'/split/content/'complete.json', training if split == 'test' else [])
            evaluations[split].append(name)
        add('analysis_'+split, [PYTHON, '-u', str(ANALYSIS), '--split', split],
            ROOT/'analysis'/split/'summary.json', evaluations[split], gpu=False)
    manifest = dict(source_hashes=source, driver_hashes=drivers, gate_sha256=gates,
        common_init_sha256=base.core.C.file_sha(base.ROOT/'common_init.pth'),
        protocol=PROTOCOL, tasks=len(jobs), train_jobs=7,
        working_reservation_MiB=working, margin_MiB=PROTOCOL['device_margin_MiB'],
        no_fixed_GPU_slot_cap=True, test_requires_all_seven_trainings=True,
        existing_owners_not_preempted=True)
    base.core.C.dump(ROOT/'manifest.json', manifest)
    base.core.C.dump(ROOT/'job_plan.json', jobs)
    for name in {**source, **drivers}:
        target = ROOT/'frozen_source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(WORK/name, target)
    base.core.C.dump(ROOT/'owner_launch.json', dict(pid=os.getpid(), started=time.time(),
        start_ticks=Path('/proc/self/stat').read_text().rsplit(') ',1)[1].split()[19],
        source_sha256=base.core.C.file_sha(__file__)))
    children, streams = {}, {}
    while True:
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
                passed = receipt.get('complete') and receipt.get('epoch') == 10 and receipt.get('updates') == 2670
            jobs[name].update(status='complete' if passed else 'failed', returncode=rc)
            print('END', name, rc, jobs[name]['status'], flush=True)
        failed = [n for n,j in jobs.items() if j['status'] == 'failed']
        if not failed:
            assert hashes() == source
            assert all(base.core.C.file_sha(WORK/n) == sha for n,sha in drivers.items())
            raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free',
                '--format=csv,noheader,nounits'], text=True)
            available = {int(row.split(',')[0]):int(row.split(',')[1])-PROTOCOL['device_margin_MiB']
                         for row in raw.splitlines()}
            raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
                '--format=csv,noheader,nounits'], text=True)
            used = {int(row.split(',')[0]):int(row.split(',')[1]) for row in raw.splitlines()
                    if row.split(',')[1].strip().isdigit()}
            for name,child in children.items():
                if jobs[name]['gpu']:
                    available[jobs[name]['gpu_index']] -= max(0, working-used.get(child.pid, 0))
            cpu_count = sum(not jobs[n]['gpu'] for n in children)
            for name,job in jobs.items():
                if job['status'] != 'pending' or any(jobs[d]['status'] != 'complete' for d in job['dependencies']):
                    continue
                env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
                if job['gpu']:
                    eligible = [g for g,v in available.items() if v >= working]
                    if not eligible:
                        continue
                    gpu = max(eligible, key=available.get)
                    available[gpu] -= working
                    env['CUDA_VISIBLE_DEVICES'] = str(gpu)
                    job['gpu_index'] = gpu
                else:
                    if cpu_count >= 1:
                        continue
                    cpu_count += 1
                    env['CUDA_VISIBLE_DEVICES'] = ''
                log = ROOT/'logs'/(name+'.log')
                log.parent.mkdir(exist_ok=True)
                stream = log.open('w')
                child = subprocess.Popen(job['command'], cwd=WORK, env=env, stdin=subprocess.DEVNULL,
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                job.update(status='running', pid=child.pid, log=str(log))
                children[name], streams[name] = child, stream
                print('START', name, job.get('gpu_index'), child.pid, flush=True)
        complete = all(j['status'] == 'complete' for j in jobs.values())
        stage = 'complete' if complete else ('failure_requires_audit' if failed else 'running')
        base.core.C.dump(ROOT/'state.json', dict(stage=stage, owner_pid=os.getpid(), jobs=jobs))
        if complete:
            base.core.C.dump(ROOT/'complete.json', dict(complete=True, jobs=len(jobs),
                manifest_sha256=base.core.C.file_sha(ROOT/'manifest.json'),
                foreground_scientific_audit_required=True))
            return
        if failed and not children:
            raise RuntimeError(failed)
        time.sleep(15)


if __name__ == '__main__':
    main()
