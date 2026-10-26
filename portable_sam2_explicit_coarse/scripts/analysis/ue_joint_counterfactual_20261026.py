"""Fixed E67 RGB-D test interventions A-E; no fitting or checkpoint selection."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

WORK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ue_joint_snapshot_eval_20261026 as evaluation

SCRIPT = Path(__file__).resolve()
BASE = Path('/data/wangcheng/checkpoint/ue_joint_train_test_snapshot_20261026')
GROUPS = dict(A='correct', B='constant_rgb', C='donor_rgb', D='reference_plane', E='disable_uav')


def prepare(root):
    assert not (root / 'manifest.json').exists(), 'Use a new immutable intervention root'
    old = evaluation.verify_sources(BASE)
    row = copy.deepcopy(old['arms']['rgbd'])
    assert row['selected_epoch'] == 67
    baseline = BASE / 'evaluations/rgbd/test'
    done = evaluation.read(baseline / 'complete.json')
    assert done['images'] == 459 and done['precision'] == 'FP32'
    assert done['checkpoint_sha256'] == row['checkpoint_sha256']
    assert done['records_sha256'] == evaluation.sha(baseline / 'records.json')
    assert done['metrics_sha256'] == evaluation.sha(baseline / 'metrics.json')
    records = evaluation.read(baseline / 'records.json')
    ids = [r['image_id'] for r in records]
    assert len(ids) == len(set(ids)) == 459
    donor = {str(ident): ids[(index + len(ids) // 2) % len(ids)] for index, ident in enumerate(ids)}
    assert all(int(k) != v for k, v in donor.items())
    cp = root / 'checkpoints/rgbd_E67.pth'
    cp.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(row['checkpoint'], cp)
    assert evaluation.sha(cp) == row['checkpoint_sha256']
    row['checkpoint'] = str(cp)
    saved_source = root / 'source' / SCRIPT.name
    saved_source.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SCRIPT, saved_source)
    sources = dict(old['evaluation_sources'])
    sources.update({str(SCRIPT): evaluation.sha(SCRIPT), str(saved_source): evaluation.sha(saved_source)})
    common = dict(old, arms=dict(rgbd=row), splits=dict(test=459), evaluation_sources=sources,
        selection='Previously frozen RGB-D E67 validation-best checkpoint, identical for A-E',
        scope='Fixed-checkpoint input/route sensitivity, not retraining modality ablation',
        limitations='C keeps target geometry with donor RGB and is physically inconsistent. D replaces all measured-depth-derived inputs jointly. E measures co-adapted checkpoint dependence, not separately trained satellite AP. One seed, development-exposed test, nominal image bootstrap only.')
    for group in GROUPS:
        evaluation.dump(root / 'groups' / group / 'manifest.json', common)
    evaluation.dump(root / 'manifest.json', dict(created=time.time(), groups=GROUPS,
        checkpoint=cp.as_posix(), checkpoint_sha256=row['checkpoint_sha256'], selected_epoch=67,
        baseline=str(baseline), baseline_complete_sha256=evaluation.sha(baseline / 'complete.json'),
        baseline_records_sha256=evaluation.sha(baseline / 'records.json'),
        baseline_metrics_sha256=evaluation.sha(baseline / 'metrics.json'), donor_map=donor,
        test_ids=ids, source_sha256=sources, precision='FP32', max_dets=250, images=459,
        bootstrap=500, bootstrap_seed=20261021, scope=common['scope'], limitations=common['limitations']))
    evaluation.dump(root / 'groups/A/complete.json', dict(complete=True, group='A', images=459,
        checkpoint_sha256=row['checkpoint_sha256'], raw=str(baseline), reused=True,
        records_sha256=done['records_sha256'], metrics_sha256=done['metrics_sha256'],
        baseline_complete_sha256=evaluation.sha(baseline / 'complete.json')))
    print('FIXED E67 A-E PREPARED', row['checkpoint_sha256'], flush=True)


def verify(root):
    manifest = evaluation.read(root / 'manifest.json')
    for path, expected in manifest['source_sha256'].items():
        assert evaluation.sha(path) == expected, path
    assert evaluation.sha(manifest['checkpoint']) == manifest['checkpoint_sha256']
    return manifest


def evaluate(root, group, limit=None):
    manifest = verify(root)
    folder = root / 'groups' / group
    assert limit or group != 'A', 'A uses already completed identical-checkpoint test evaluation'
    run = evaluation.runner()
    import cv2
    import numpy as np
    import torch
    from rsprompter.uav.local_adapter_probe import YAWS
    original_runtime = run.Runtime
    seen, read_paths, identity_paths, donor_rgb = [], set(), set(), {}
    field_preserved = []
    infer_checks = []
    measured = [Path('/data/wangcheng/cache/ue_v1_native_geometry'),
                Path('/data/wangcheng/cache/ue_v1_s2db_geometry_v2')]
    plane = Path('/data/wangcheng/cache/ue_v1_reference_plane_20261024')
    data_root = run.core.previous.DATA

    def audit(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, Path)):
            return
        path = Path(os.fsdecode(args[0])).absolute()
        is_measured = any(p == path or p in path.parents for p in measured)
        is_plane = plane == path or plane in path.parents
        is_uav = data_root in path.parents and ('/uav/' in str(path) or path.name == 'cameras.json')
        if (is_measured or is_plane) and path.suffix != '.npz':
            # Runtime contract hashes manifests even when the UAV route is off.
            # Identity metadata is not a geometry tensor consumed by inference.
            identity_paths.add(str(path))
            return
        if group == 'D' and is_measured:
            raise AssertionError('Reference-plane intervention read measured cache: ' + str(path))
        if group == 'D' and is_uav and path.suffix == '.exr':
            raise AssertionError('Reference-plane intervention read measured depth: ' + str(path))
        if group == 'E' and (is_measured or is_plane or is_uav):
            raise AssertionError('Disabled UAV route read UAV input: ' + str(path))
        if is_measured or is_plane or is_uav:
            read_paths.add(str(path))

    sys.addaudithook(audit)
    original_imread = cv2.imread

    def checked_imread(path, *args, **kwargs):
        # Native OpenCV I/O does not emit Python's open audit event.
        audit('open', (str(path),))
        return original_imread(path, *args, **kwargs)

    cv2.imread = checked_imread

    class IntervenedRuntime(original_runtime):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if group == 'E':
                self.joint.uav.enabled = False
                self.joint.uav.internal = False
                self.joint.clear()
                self.joint.prepare_uav = lambda **unused: None

        def payload(self, ident):
            seen.append(int(ident))
            if group == 'E':
                assert not self.joint.uav.enabled and not self.joint.uav.internal
                return dict(geometry=None)
            if group == 'D':
                value = self.uav_inputs.load(ident, geometry_mode='reference_plane', rgb_mode='correct')
                provenance = value.pop('provenance')
                assert provenance['measured_depth'] is False
                return value
            value = super().payload(ident)
            retained = {k: value[k] for k in ('geometry', 'index', 'anchor')}
            if group == 'B':
                value['rgb'] = self.joint.uav.rgb_mean.expand_as(value['rgb']).clone()
                assert torch.count_nonzero((value['rgb'] - self.joint.uav.rgb_mean) / self.joint.uav.rgb_std) == 0
            elif group == 'C':
                donor = manifest['donor_map'][str(ident)]
                images = []
                for yaw in YAWS:
                    path = data_root / 'points' / ('%06d' % donor) / ('uav/yaw_%03d/rgb.png' % yaw)
                    image = cv2.imread(str(path))
                    assert image is not None and image.shape == (512, 512, 3)
                    images.append(image[:, :, ::-1].transpose(2, 0, 1).copy())
                    donor_rgb[str(path)] = evaluation.sha(path)
                value['rgb'] = torch.from_numpy(np.stack(images)).to(value['rgb'])
            assert all(value[k] is tensor for k, tensor in retained.items())
            field_preserved.append(int(ident))
            return value

    run.Runtime = IntervenedRuntime
    original_inference = run.core.inference_outputs

    def checked_inference(model, adapter, sample, features, geometry):
        prefix, combined, accepted = original_inference(model, adapter, sample, features, geometry)
        if group == 'E':
            assert not adapter.enabled and not adapter.internal
            assert adapter.context is None and adapter.outputs is None and accepted == 0
            # Keep the formal path's existing score mapping even with UAV disabled.
            expected = copy.deepcopy(prefix)
            expected['scores'] = np.asarray(run.core.consumer.trusted_score(prefix['scores']), np.float64)
            assert run.core.C.serialized(expected) == run.core.C.serialized(combined)
            infer_checks.append(True)
        return prefix, combined, accepted

    run.core.inference_outputs = checked_inference
    evaluation.evaluate(folder, 'rgbd', 'test', limit=limit)
    raw = folder / ('gates' if limit else 'evaluations') / 'rgbd/test'
    count = min(limit, 459) if limit else 459
    assert seen == manifest['test_ids'][:count]
    if group in ('A', 'B', 'C'):
        assert field_preserved == seen
    if group == 'E':
        assert len(infer_checks) == count and not read_paths
    if group == 'C':
        assert len(donor_rgb) == count * 5
    inputs = {p: evaluation.sha(p) for p in sorted(read_paths)}
    assert all(evaluation.sha(p) == h for p, h in donor_rgb.items())
    if group == 'A':
        assert evaluation.read(raw / 'records.json') == evaluation.read(Path(manifest['baseline']) / 'records.json')[:count]
    verify(root)
    evaluation.dump(folder / ('gate.json' if limit else 'complete.json'), dict(complete=True,
        group=group, intervention=GROUPS[group], images=count, raw=str(raw),
        checkpoint_sha256=manifest['checkpoint_sha256'], selected_epoch=67,
        records_sha256=evaluation.sha(raw / 'records.json'), metrics_sha256=evaluation.sha(raw / 'metrics.json'),
        raw_complete_sha256=evaluation.sha(raw / 'complete.json'), payload_ids=seen,
        target_fields_preserved=len(field_preserved), disabled_inference_checks=len(infer_checks),
        input_files_sha256=inputs, donor_rgb_sha256=donor_rgb,
        identity_metadata_sha256={p: evaluation.sha(p) for p in sorted(identity_paths)},
        source_sha256=evaluation.sha(SCRIPT), manifest_sha256=evaluation.sha(root / 'manifest.json')))


def analyze(root):
    manifest = verify(root)
    sys.path.insert(0, str(WORK / 'scripts/probes'))
    import ue_joint_encoding_fast_analysis as fast
    import numpy as np
    points, boot, evidence = {}, {}, {}
    samples = np.random.default_rng(20261021).integers(0, 459, (500, 459))
    baseline_gt = None
    for group in GROUPS:
        completed = evaluation.read(root / 'groups' / group / 'complete.json')
        assert completed['complete'] and completed['images'] == 459
        assert completed['checkpoint_sha256'] == manifest['checkpoint_sha256']
        raw = Path(completed['raw'])
        assert evaluation.sha(raw / 'records.json') == completed['records_sha256']
        assert evaluation.sha(raw / 'metrics.json') == completed['metrics_sha256']
        evidence[str(raw / 'records.json')] = completed['records_sha256']
        rows = evaluation.read(raw / 'records.json')
        assert [r['image_id'] for r in rows] == manifest['test_ids']
        gt = [r['gt'] for r in rows]
        if baseline_gt is None:
            baseline_gt = gt
        assert gt == baseline_gt
        points[group] = dict(combined={}, original_path={})
        for name, kind, ratio in [('MaskAP', 'segm', None), ('BBoxAP', 'bbox', None),
                                  ('BoundaryAP_2pct', 'segm', .02), ('BoundaryAP_1pct', 'segm', .01)]:
            ev = fast.original.evaluator(rows, kind=kind, ratio=ratio)
            point = fast.original.precision_ap(ev)
            points[group]['combined'][name] = point
            if ratio is None:
                assert abs(point - evaluation.read(raw / 'metrics.json')[kind + '/mAP']) < 1e-12
            if name == 'MaskAP':
                assert abs(fast.fast_sample_ap(ev, np.arange(459)) - point) < 1e-12
                boot[group] = np.array([fast.fast_sample_ap(ev, ix) for ix in samples])
            del ev
        prefix = [dict(image_id=r['image_id'], gt=r['gt'], dt=r['original_path']) for r in rows]
        for name, kind in [('MaskAP', 'segm'), ('BBoxAP', 'bbox')]:
            ev = fast.original.evaluator(prefix, kind=kind)
            points[group]['original_path'][name] = fast.original.precision_ap(ev)
            del ev
        points[group]['mean_predictions'] = float(np.mean([len(r['dt']['scores']) for r in rows]))
        points[group]['mean_appended_candidates'] = float(np.mean([r['geometry_accepted'] for r in rows]))
        evaluation.dump(root / 'analysis/progress.json', dict(last_completed_group=group, points=points))
        print('COUNTERFACTUAL POINTS', group, points[group], flush=True)
        del rows
    comparisons = {}
    for group in ('B', 'C', 'D', 'E'):
        comparisons['A_minus_' + group] = dict(
            MaskAP_delta=points['A']['combined']['MaskAP'] - points[group]['combined']['MaskAP'],
            paired_image_bootstrap95=np.quantile(boot['A'] - boot[group], [.025, .975]).tolist())
    assert len(fast.audits) == 15
    assert all(evaluation.sha(p) == h for p, h in evidence.items())
    evaluation.dump(root / 'analysis/complete.json', dict(complete=True, points=points,
        comparisons=comparisons, bootstrap=500, bootstrap_seed=20261021,
        reference_checks=fast.audits, evidence_sha256=evidence,
        manifest_sha256=evaluation.sha(root / 'manifest.json'), limitations=manifest['limitations'],
        uncertainty_scope='Combined Mask AP only, paired images; not training-seed variance'))


def owner(root):
    import fcntl
    lock = (root / 'owner.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    verify(root)
    assert not (root / 'owner_launch.json').exists()
    for group in GROUPS:
        assert evaluation.read(root / 'groups' / group / 'gate.json')['complete']
    evaluation.dump(root / 'owner_launch.json', dict(pid=os.getpid(), started=time.time(), gpu=2,
        source_sha256=evaluation.sha(SCRIPT), tasks=['B', 'C', 'D', 'E'], reused_baseline='A'))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='2', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    (root / 'logs').mkdir(exist_ok=True)
    done = ['A']
    for group in ('B', 'C', 'D', 'E'):
        while True:
            raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free',
                '--format=csv,noheader,nounits'], text=True)
            free = {int(line.split(',')[0]): int(line.split(',')[1]) for line in raw.splitlines()}
            if free[2] >= 6144:
                break
            evaluation.dump(root / 'owner_state.json', dict(stage='waiting_for_gpu2_memory',
                next_group=group, done=done, free_MiB=free[2], required_MiB=6144))
            time.sleep(15)
        verify(root)
        command = [evaluation.PYTHON, '-u', str(SCRIPT), 'eval', '--root', str(root), '--group', group]
        with (root / 'logs' / (group + '.log')).open('w') as stream:
            process = subprocess.Popen(command, cwd=WORK, env=env, stdout=stream,
                stderr=subprocess.STDOUT, start_new_session=True)
        evaluation.dump(root / 'owner_state.json', dict(stage='evaluating', group=group,
            pid=process.pid, done=done, gpu=2, command=command))
        if process.wait() != 0:
            evaluation.dump(root / 'failed.json', dict(group=group, returncode=process.returncode, done=done))
            raise RuntimeError('Intervention failed; inspect log, no automatic retry: ' + group)
        assert evaluation.read(root / 'groups' / group / 'complete.json')['images'] == 459
        done.append(group)
    evaluation.dump(root / 'owner_state.json', dict(stage='analyzing', done=done))
    analyze(root)
    evaluation.dump(root / 'complete.json', dict(complete=True, done=done,
        analysis_sha256=evaluation.sha(root / 'analysis/complete.json')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'eval', 'analyze', 'owner'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--group', choices=list(GROUPS))
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.root)
    elif args.action == 'eval':
        assert args.group
        evaluate(args.root, args.group, args.limit)
    elif args.action == 'analyze':
        analyze(args.root)
    else:
        owner(args.root)


if __name__ == '__main__':
    main()
