"""User-requested train/test evaluation of immutable current validation-best weights.

This supplemental snapshot is separate from the locked full100epoch study.
No training, checkpoint selection from test, or changes to the numerical runner.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

WORK = Path(__file__).resolve().parents[2]
PYTHON = '/data/wangcheng/envs/cvt2/bin/python'
STUDY = Path('/data/wangcheng/checkpoint/ue_compact_joint_20261024')
ARMS = ('satellite', 'rgbd', 'constant_rgb', 'reference_plane')
SCRIPT = Path(__file__).resolve()


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def runner():
    sys.path[:0] = [str(WORK), str(WORK / 'scripts/probes')]
    import ue_compact_joint_run_20261024 as run
    return run


def verify_sources(root):
    manifest = read(root / 'manifest.json')
    for path, expected in manifest['evaluation_sources'].items():
        assert sha(path) == expected, path
    for rel, expected in manifest['training_sources'].items():
        assert sha(WORK / rel) == expected, rel
        assert sha(STUDY / 'source' / rel) == expected, rel
    return manifest


def prepare(root):
    assert not (root / 'manifest.json').exists(), 'Use a new snapshot directory'
    run = runner()
    import torch
    sources = run.sources()
    assert read(STUDY / 'initialization.json')['source_hashes'] == sources
    root.mkdir(parents=True, exist_ok=True)
    source = root / 'source' / SCRIPT.name
    source.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SCRIPT, source)
    analysis_sources = [WORK / 'scripts/probes/ue_joint_encoding_fast_analysis.py',
                        WORK / 'scripts/probes/ue_joint_encoding_analysis.py',
                        WORK / 'utils/coco_eval_utils.py']
    evaluation_sources = {str(SCRIPT): sha(SCRIPT), str(source): sha(source)}
    for path in analysis_sources:
        copy = root / 'source' / path.relative_to(WORK)
        copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, copy)
        evaluation_sources.update({str(path): sha(path), str(copy): sha(copy)})
    snapshots = {}
    for arm in ARMS:
        origin = STUDY / arm
        destination = root / 'checkpoints' / (arm + '.pth')
        destination.parent.mkdir(exist_ok=True)
        # The trainer copies best_model non-atomically. Reject a concurrent write.
        for attempt in range(4):
            best = read(origin / 'best_validation_metrics.json')
            before = (origin / 'best_model.pth').stat()
            shutil.copyfile(origin / 'best_model.pth', destination)
            after = (origin / 'best_model.pth').stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                continue
            cp = torch.load(destination, map_location='cpu', weights_only=False)
            if cp['epoch'] != best['epoch'] or best != read(origin / 'best_validation_metrics.json'):
                del cp
                continue
            assert cp['contract'] == read(origin / 'contract.json')
            assert cp['best'] == dict(epoch=best['epoch'], score=best['metrics']['segm/mAP'])
            assert cp['step'] == cp['epoch'] * 267
            assert sha(destination) == sha(origin / 'best_model.pth')
            completed = max(int(m[1]) for p in origin.glob('epoch*.json')
                            for m in [re.fullmatch(r'epoch(\d+)\.json', p.name)] if m)
            history = [read(origin / ('epoch%d.json' % e)) for e in range(1, completed + 1)]
            selected = max(history, key=lambda d: d['validation']['segm/mAP'])
            assert selected['epoch'] == cp['epoch'], 'Best changed during snapshot; rerun prepare'
            assert selected['validation'] == best['metrics']
            snapshots[arm] = dict(checkpoint=str(destination), checkpoint_sha256=sha(destination),
                selected_epoch=cp['epoch'], selected_step=cp['step'], validation=best['metrics'],
                completed_epochs_at_snapshot=completed, contract=cp['contract'],
                selected_epoch_record_sha256=sha(origin / ('epoch%d.json' % cp['epoch'])))
            del cp
            break
        else:
            raise RuntimeError('Could not capture stable validation-best for ' + arm)
    dump(root / 'manifest.json', dict(created=time.time(), study=str(STUDY), arms=snapshots,
        training_sources=sources, evaluation_sources=evaluation_sources,
        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=WORK, text=True).strip(),
        splits=dict(test=459, train=2142), precision='FP32', max_dets=250,
        selection='Current validation-best combined Mask AP250, ties earlier; copied before evaluation',
        scope='User-authorized supplemental snapshot; budgets differ; official full100 evaluation remains separate',
        limitations='One seed; development-exposed v1; train is in-sample; no test-based tuning; not a full100 matched-budget claim'))
    print('SNAPSHOT READY', {a: (v['selected_epoch'], v['completed_epochs_at_snapshot'])
                             for a, v in snapshots.items()}, flush=True)


def evaluate(root, arm, split, limit=None):
    manifest = verify_sources(root)
    row = manifest['arms'][arm]
    assert sha(row['checkpoint']) == row['checkpoint_sha256']
    run = runner()
    import numpy as np
    import torch
    runtime = run.Runtime(STUDY, arm)
    assert run.contract(runtime) == row['contract']
    cp = runtime.restore(row['checkpoint'], row['contract'], training=False)
    assert cp['epoch'] == row['selected_epoch']
    del cp
    runtime.network.eval()
    before = run.core.previous.tensor_digest(runtime.network.state_dict())
    imported_sources = {}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if not filename or not filename.endswith('.py'):
            continue
        path = Path(filename).resolve()
        if WORK.parent not in path.parents or not path.is_file():
            continue
        imported_sources[str(path)] = sha(path)
        copy = root / 'runtime_sources' / arm / split / imported_sources[str(path)][:16] / path.relative_to(WORK.parent)
        copy.parent.mkdir(parents=True, exist_ok=True)
        if not copy.exists():
            shutil.copyfile(path, copy)
        assert sha(copy) == imported_sources[str(path)], str(path)
    dataset = run.core.C.MinimalSatelliteDataset(run.core.previous.DATA, split, augment=False)
    assert len(dataset) == manifest['splits'][split]
    count = min(limit, len(dataset)) if limit else len(dataset)
    out = root / ('gates' if limit else 'evaluations') / arm / split
    assert not (out / 'complete.json').exists(), 'Never overwrite a completed evaluation'
    records, metas = [], []
    began = time.time()
    with torch.inference_mode(), torch.autocast('cuda', enabled=False):
        for index in range(count):
            ident = dataset.samples[index]['image_id']
            batch = run.core.C.rtmdet_collate_fn([dataset[index]])
            sample = run.core.C.processed(runtime.satellite, batch)
            if runtime.joint is not None:
                payload = runtime.payload(ident)
                runtime.joint.prepare_uav(**payload)
                adapter, geometry = runtime.joint.uav, payload['geometry']
            else:
                adapter, geometry = SimpleNamespace(enabled=False), None
            features = runtime.satellite.extract_feat(sample['inputs'])
            prefix, combined, accepted = run.core.inference_outputs(
                runtime.satellite, adapter, sample, features, geometry)
            records.append(dict(image_id=ident,
                gt=run.core.C.serialized(run.core.C.gt_arrays(sample['data_samples'][0].gt_instances)),
                dt=run.core.C.serialized(combined), geometry_accepted=accepted,
                original_path=run.core.C.serialized(prefix)))
            metas.append(batch['img_metas'][0])
            runtime.clear()
            if (index + 1) % 25 == 0 or index + 1 == count:
                dump(out / 'progress.json', dict(pid=os.getpid(), arm=arm, split=split,
                    images_done=index + 1, images_total=count, elapsed_seconds=time.time() - began))
                print('EVAL', arm, split, index + 1, '/', count, flush=True)
    assert before == run.core.previous.tensor_digest(runtime.network.state_dict())
    assert not runtime.optimizer.state and all(p.grad is None for p in runtime.network.parameters())
    assert all(sha(path) == expected for path, expected in imported_sources.items())
    gt = [{**r['gt'], 'bboxes': np.asarray(r['gt']['bboxes'], np.float32),
           'labels': np.asarray(r['gt']['labels'], np.int64)} for r in records]
    dt = [run.core.consumer.arrays(r['dt']) for r in records]
    cg, cd = run.core.C.build_coco_gt_and_dt(gt, dt, metas, score_key='scores')
    metrics = {}
    for kind in ('segm', 'bbox'):
        metrics.update(run.core.C.run_coco_eval(cg, cd, kind, max_dets=[1, 10, 250]))
    metrics = {k.replace('AR@100', 'AR@250'): v for k, v in metrics.items()}
    dump(out / 'records.json', records)
    dump(out / 'metrics.json', metrics)
    verify_sources(root)
    assert sha(row['checkpoint']) == row['checkpoint_sha256']
    dump(out / 'complete.json', dict(complete=True, arm=arm, split=split, images=count,
        checkpoint_sha256=row['checkpoint_sha256'], selected_epoch=row['selected_epoch'],
        manifest_sha256=sha(root / 'manifest.json'), records_sha256=sha(out / 'records.json'),
        metrics_sha256=sha(out / 'metrics.json'), precision='FP32', max_dets=250,
        augmentation=False, optimizer_updates=0, state_unchanged=True,
        imported_sources_sha256=imported_sources,
        peak_reserved_MiB=torch.cuda.max_memory_reserved() / 2**20,
        purpose='engineering_subset' if limit else 'supplemental_snapshot'))


def analyze(root, split):
    manifest = verify_sources(root)
    sys.path.insert(0, str(WORK / 'scripts/probes'))
    import ue_joint_encoding_fast_analysis as fast
    import numpy as np
    rows, evidence, points, boot = {}, {}, {}, {}
    count = manifest['splits'][split]
    samples = np.random.default_rng(20261021).integers(0, count, (500, count))
    for arm in ARMS:
        folder = root / 'evaluations' / arm / split
        complete = read(folder / 'complete.json')
        assert complete['purpose'] == 'supplemental_snapshot' and complete['images'] == count
        assert complete['manifest_sha256'] == sha(root / 'manifest.json')
        assert complete['checkpoint_sha256'] == manifest['arms'][arm]['checkpoint_sha256']
        assert complete['records_sha256'] == sha(folder / 'records.json')
        assert complete['metrics_sha256'] == sha(folder / 'metrics.json')
        rows[arm] = read(folder / 'records.json')
        assert len(rows[arm]) == len({r['image_id'] for r in rows[arm]}) == count
        if arm != 'satellite':
            assert [r['image_id'] for r in rows[arm]] == [r['image_id'] for r in rows['satellite']]
            assert all(a['gt'] == b['gt'] for a, b in zip(rows[arm], rows['satellite']))
        evidence[str(folder / 'records.json')] = sha(folder / 'records.json')
        points[arm] = dict(combined={}, original_path={})
        for name, kind, ratio in [('MaskAP', 'segm', None), ('BBoxAP', 'bbox', None),
                                  ('BoundaryAP_2pct', 'segm', .02), ('BoundaryAP_1pct', 'segm', .01)]:
            ev = fast.original.evaluator(rows[arm], kind=kind, ratio=ratio)
            point = fast.original.precision_ap(ev)
            points[arm]['combined'][name] = point
            if ratio is None:
                expected = read(folder / 'metrics.json')[kind + '/mAP']
                assert abs(point - expected) < 1e-12
            if name == 'MaskAP':
                assert abs(fast.fast_sample_ap(ev, np.arange(count)) - point) < 1e-12
                boot[arm] = np.array([fast.fast_sample_ap(ev, ix) for ix in samples])
            del ev
        prefix = [dict(image_id=r['image_id'], gt=r['gt'], dt=r['original_path']) for r in rows[arm]]
        for name, kind in [('MaskAP', 'segm'), ('BBoxAP', 'bbox')]:
            ev = fast.original.evaluator(prefix, kind=kind)
            points[arm]['original_path'][name] = fast.original.precision_ap(ev)
            del ev
        del rows[arm]
        if arm == 'satellite':
            # Keep only GT/IDs for following arm identity checks, not all predictions.
            rows['satellite'] = [dict(image_id=r['image_id'], gt=r['gt'])
                                 for r in read(folder / 'records.json')]
        print('ANALYZED', split, arm, points[arm]['combined'], flush=True)
    pairs = [('satellite', a) for a in ARMS if a != 'satellite']
    pairs += [('constant_rgb', 'rgbd'), ('reference_plane', 'rgbd')]
    comparisons = {}
    for baseline, candidate in pairs:
        delta = boot[candidate] - boot[baseline]
        comparisons[candidate + '_minus_' + baseline] = dict(
            MaskAP_delta=points[candidate]['combined']['MaskAP'] - points[baseline]['combined']['MaskAP'],
            paired_image_bootstrap95=np.quantile(delta, [.025, .975]).tolist())
    assert len(fast.audits) == 12
    assert all(sha(p) == value for p, value in evidence.items())
    dump(root / 'analysis' / (split + '.json'), dict(complete=True, split=split, images=count,
        points=points, comparisons=comparisons, bootstrap=500, bootstrap_seed=20261021,
        reference_checks=fast.audits, evidence_sha256=evidence,
        manifest_sha256=sha(root / 'manifest.json'), limitations=manifest['limitations'],
        budget_scope=manifest['scope'], uncertainty_scope='Paired image bootstrap for combined Mask AP only; not training-seed repeatability'))


def owner(root):
    lock = (root / 'owner.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    verify_sources(root)
    assert not (root / 'owner_launch.json').exists(), 'No implicit scheduler recovery'
    dump(root / 'owner_launch.json', dict(pid=os.getpid(), started=time.time(), gpu=2,
        tasks=8, command=sys.argv, source_sha256=sha(SCRIPT)))
    analyses = {}
    done = []
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='2', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    for split in ('test', 'train'):
        for arm in ARMS:
            while True:
                raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free',
                    '--format=csv,noheader,nounits'], text=True)
                free = {int(x.split(',')[0]): int(x.split(',')[1]) for x in raw.splitlines()}
                if free[2] >= 8192:
                    break
                dump(root / 'owner_state.json', dict(stage='waiting_for_gpu2_memory', done=done,
                    next_task=arm + '_' + split, free_MiB=free[2], required_MiB=8192))
                time.sleep(15)
            verify_sources(root)
            command = [PYTHON, '-u', str(SCRIPT), 'eval', '--root', str(root),
                       '--arm', arm, '--split', split]
            logs = root / 'logs'
            logs.mkdir(exist_ok=True)
            with (logs / (arm + '_' + split + '.log')).open('w') as stream:
                child = subprocess.Popen(command, cwd=WORK, env=env, stdout=stream,
                                         stderr=subprocess.STDOUT, start_new_session=True)
            dump(root / 'owner_state.json', dict(stage='evaluating', done=done, arm=arm,
                split=split, pid=child.pid, gpu=2, command=command))
            code = child.wait()
            if code:
                dump(root / 'failed.json', dict(task=arm + '_' + split, returncode=code,
                    pid=child.pid, done=done, reason='No automatic retry; inspect preserved log'))
                raise RuntimeError('Evaluation failed: ' + arm + '_' + split)
            marker = root / 'evaluations' / arm / split / 'complete.json'
            assert read(marker)['images'] == read(root / 'manifest.json')['splits'][split]
            done.append(arm + '_' + split)
        command = [PYTHON, '-u', str(SCRIPT), 'analyze', '--root', str(root), '--split', split]
        with (root / 'logs' / (split + '_analysis.log')).open('w') as stream:
            analyses[split] = subprocess.Popen(command, cwd=WORK, env=env, stdout=stream,
                stderr=subprocess.STDOUT, start_new_session=True)
    dump(root / 'owner_state.json', dict(stage='analyzing', done=done,
        analysis_pids={s: p.pid for s, p in analyses.items()}))
    for split, process in analyses.items():
        assert process.wait() == 0, 'Analysis failed; inspect ' + split
        assert read(root / 'analysis' / (split + '.json'))['complete']
    dump(root / 'complete.json', dict(complete=True, evaluations=done, analyses=list(analyses),
        manifest_sha256=sha(root / 'manifest.json')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'eval', 'analyze', 'owner'])
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--arm', choices=ARMS, default='rgbd')
    parser.add_argument('--split', choices=['test', 'train'], default='test')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.root)
    elif args.action == 'eval':
        evaluate(args.root, args.arm, args.split, args.limit)
    elif args.action == 'analyze':
        analyze(args.root, args.split)
    else:
        owner(args.root)


if __name__ == '__main__':
    main()
