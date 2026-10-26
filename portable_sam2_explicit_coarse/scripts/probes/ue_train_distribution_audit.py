"""Read-only full-train evaluation of completed N2 / N2-shared adapters.

Reuses the original numerical evaluation functions, bypassing only their CLI's
val/test split restriction. No training, augmentation, or checkpoint selection.
"""
import argparse
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
PYTHON = '/data/wangcheng/envs/cvt2/bin/python'
DEFAULT_OUT = Path('/data/wangcheng/checkpoint/ue_train_distribution_audit_20261024')
INIT = Path('/data/wangcheng/checkpoint/ue_attention_ladder_20261022/common_init.pth')
FAMILIES = {
    'N2': ('ue_post_activation_probe', Path('/data/wangcheng/checkpoint/ue_post_activation_20261023')),
    'N2_shared': ('ue_post_activation_shared_probe', Path('/data/wangcheng/checkpoint/ue_post_activation_shared_20261023')),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def family_init(family):
    if family == 'N2_shared':
        return FAMILIES[family][1] / 'common_init.pth'
    return INIT


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def evaluate(args):
    import torch
    torch.set_num_threads(2)
    total = torch.cuda.get_device_properties(0).total_memory / 2**20
    torch.cuda.set_per_process_memory_fraction(5500 / total)
    run = importlib.import_module(FAMILIES[args.family][0])
    adapter = FAMILIES[args.family][1] / args.content / 'last.pth'
    original = argparse.Namespace(
        action='eval', direction='N2', content=args.content,
        init=family_init(args.family), adapter=adapter if args.mode != 'disabled' else None,
        data_root=run.DATA, base=run.BASE,
        cache=Path('/data/wangcheng/cache/ue_v1_s2db_geometry_v2'),
        eval_split='train', mode=args.mode, max_images=0, out=args.out,
        resume_from=None,
    )
    run.seed(44)
    model, branch, snapshot = run.make_model(original)
    # make_model checks archive / frozen satellite / initialization identity;
    # evaluate checks E10 / 2670 / training content / base SHA and disables aug.
    run.evaluate(original, model, branch)
    path = args.out / args.mode / 'evaluation_identity.json'
    identity = json.loads(path.read_text())
    assert identity['split'] == 'train' and identity['images'] == 2142
    assert run.state_digest(model) == run.FROZEN_SHA
    identity.update(audit_source_sha256=sha(__file__), augmentation=False,
                    checkpoint_selection='existing E10 last; no training-set selection',
                    frozen_satellite_after_evaluation_exact=True,
                    allocator_limit_mib=5500)
    dump(path, identity)


def metrics(args):
    # Point metrics first: image bootstrap on the training set cannot measure
    # generalization or training-seed uncertainty and is deliberately omitted.
    from ue_joint_encoding_analysis import evaluator, precision_ap
    rows = json.loads(args.records.read_text())
    assert len(rows) == 2142 and len({r['image_id'] for r in rows}) == 2142
    result = dict(split='train', images=len(rows),
                  instances=sum(len(r['gt']['labels']) for r in rows),
                  input_sha256=sha(args.records), metrics={},
                  in_sample_diagnostic_only=True, max_dets=250,
                  boundary_definition='min(mask IoU, inner boundary IoU); band 2% / 1% image diagonal')
    for name, kind, ratio in [('MaskAP', 'segm', None), ('BBoxAP', 'bbox', None),
                              ('BoundaryAP_2pct', 'segm', .02), ('BoundaryAP_1pct', 'segm', .01)]:
        ev = evaluator(rows, kind=kind, ratio=ratio)
        result['metrics'][name] = precision_ap(ev)
        print(name, result['metrics'][name], flush=True)
        dump(args.out / 'point_metrics.partial.json', result)
        del ev
    dump(args.out / 'point_metrics.json', result)


def schedule(args):
    r = args.out
    r.mkdir(parents=True, exist_ok=True)
    lock = (r / 'owner.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (r / 'state.json').exists(), 'Refuse duplicate owner / implicit resume'
    # Lock numerical dependencies by content; never edit another active runner.
    sources = {str(Path(__file__).relative_to(ROOT)): sha(__file__)}
    checkpoints = {str(INIT): sha(INIT)}
    for family, (module, folder) in FAMILIES.items():
        # Read the archived gate source inventory without importing torch in owner.
        gate = folder / 'gate/gpu_gate.json'
        for name, expected in json.loads(gate.read_text())['source_hashes'].items():
            assert sha(ROOT / name) == expected, name
            sources[name] = expected
        checkpoints[str(family_init(family))] = sha(family_init(family))
        manifest = folder / 'manifest.json'
        if manifest.exists():
            prior = json.loads(manifest.read_text()).get('source_hashes', {})
            for name, expected in prior.items():
                assert sha(ROOT / name) == expected, name
                sources[name] = expected
        script = ROOT / 'scripts/probes' / (module + '.py')
        sources[str(script.relative_to(ROOT))] = sha(script)
        for content in ('correct', 'constant'):
            cp = folder / content / 'last.pth'
            checkpoints[str(cp)] = sha(cp)
    analysis = ROOT / 'scripts/probes/ue_joint_encoding_analysis.py'
    sources[str(analysis.relative_to(ROOT))] = sha(analysis)
    jobs = {}
    specs = [('baseline', 'N2', 'correct', 'disabled'),
             ('N2_correct', 'N2', 'correct', 'correct'),
             ('N2_constant_trained', 'N2', 'constant', 'constant'),
             ('N2_correct_constant_input', 'N2', 'correct', 'constant'),
             ('N2_correct_donor_input', 'N2', 'correct', 'donor'),
             ('shared_correct', 'N2_shared', 'correct', 'correct'),
             ('shared_constant_trained', 'N2_shared', 'constant', 'constant')]
    for key, family, content, mode in specs:
        out = r / 'predictions' / key
        cmd = [PYTHON, str(Path(__file__).resolve()), 'eval', '--family', family,
               '--content', content, '--mode', mode, '--out', str(out)]
        jobs[key] = dict(cmd=cmd, status='pending', gpu=True, dependencies=[],
                         artifact=str(out / mode / 'evaluation_identity.json'))
        metric_out = r / 'metrics' / key
        jobs[key + '_metrics'] = dict(
            cmd=[PYTHON, str(Path(__file__).resolve()), 'metrics', '--records',
                 str(out / mode / 'records.json'), '--out', str(metric_out)],
            status='pending', gpu=False, dependencies=[key],
            artifact=str(metric_out / 'point_metrics.json'))
    dump(r / 'manifest.json', dict(source_hashes=sources, checkpoint_hashes=checkpoints,
         evaluation_split='train', images=2142, no_training=True, augmentation=False,
         owned_gpu_concurrency=2, allowed_gpus=[0, 1], minimum_free_mib=8500,
         allocator_cap_mib=5500, cpu_metric_concurrency=1,
         purpose='training-distribution fit versus capacity/content; does not prove dataset causality'))
    state = dict(owner_pid=os.getpid(), stage='running', jobs=jobs)
    children, streams = {}, {}
    while True:
        for key, child in list(children.items()):
            rc = child.poll()
            if rc is None:
                continue
            streams.pop(key).close()
            del children[key]
            jobs[key]['returncode'] = rc
            jobs[key]['status'] = 'complete' if rc == 0 and Path(jobs[key]['artifact']).exists() else 'failed'
            print('END', key, jobs[key]['status'], rc, flush=True)
        if any(j['status'] == 'failed' for j in jobs.values()):
            state['stage'] = 'failure_requires_audit'
        else:
            for name, expected in sources.items():
                assert sha(ROOT / name) == expected, 'Source changed: ' + name
            free = {}
            rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free',
                                            '--format=csv,noheader,nounits'], text=True)
            for row in rows.splitlines():
                gpu, memory = map(int, row.split(','))
                free[gpu] = memory
            owned = {jobs[k].get('gpu_index') for k in children if jobs[k]['gpu']}
            cpu_busy = any(not jobs[k]['gpu'] for k in children)
            for key, job in jobs.items():
                if job['status'] != 'pending' or any(jobs[d]['status'] != 'complete' for d in job['dependencies']):
                    continue
                env = os.environ.copy()
                env.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
                if job['gpu']:
                    candidates = [g for g in (0, 1) if g not in owned and free[g] >= 8500]
                    if not candidates:
                        continue
                    gpu = candidates[0]
                    env['CUDA_VISIBLE_DEVICES'] = str(gpu)
                    job['gpu_index'] = gpu
                    owned.add(gpu)
                    job['admission_free_mib'] = free[gpu]
                else:
                    if cpu_busy:
                        continue
                    env['CUDA_VISIBLE_DEVICES'] = ''
                    cpu_busy = True
                logs = r / 'logs'
                logs.mkdir(exist_ok=True)
                stream = (logs / (key + '.log')).open('w')
                child = subprocess.Popen(job['cmd'], env=env, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                job.update(status='running', pid=child.pid)
                children[key], streams[key] = child, stream
                print('START', key, child.pid, flush=True)
        if all(j['status'] == 'complete' for j in jobs.values()):
            state['stage'] = 'complete'
            values = {key: json.loads((r / 'metrics' / key / 'point_metrics.json').read_text())
                      for key, *_ in specs}
            pairs = [('N2_net', 'baseline', 'N2_correct'),
                     ('N2_content', 'N2_constant_trained', 'N2_correct'),
                     ('N2_constant_net', 'baseline', 'N2_constant_trained'),
                     ('N2_input_constant', 'N2_correct_constant_input', 'N2_correct'),
                     ('N2_input_donor', 'N2_correct_donor_input', 'N2_correct'),
                     ('shared_net', 'baseline', 'shared_correct'),
                     ('shared_content', 'shared_constant_trained', 'shared_correct')]
            comparison = {key: {metric: (values[right]['metrics'][metric] - value) * 100
                               for metric, value in values[left]['metrics'].items()}
                          for key, left, right in pairs}
            dump(r / 'summary.json', dict(results=values, deltas_percentage_points=comparison,
                 limitation='in-sample, single-seed diagnostic; no causal dataset attribution'))
        dump(r / 'state.json', state)
        if state['stage'] == 'complete' or (state['stage'] == 'failure_requires_audit' and not children):
            break
        time.sleep(15)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['eval', 'metrics', 'schedule'])
    p.add_argument('--out', type=Path, default=DEFAULT_OUT)
    p.add_argument('--family', choices=list(FAMILIES), default='N2')
    p.add_argument('--content', choices=['correct', 'constant'], default='correct')
    p.add_argument('--mode', choices=['correct', 'constant', 'donor', 'disabled'], default='correct')
    p.add_argument('--records', type=Path)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dict(eval=evaluate, metrics=metrics, schedule=schedule)[args.action](args)


if __name__ == '__main__':
    main()
