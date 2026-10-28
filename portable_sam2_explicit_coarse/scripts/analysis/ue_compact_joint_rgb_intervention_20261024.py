"""Post-training fixed-RGBD-checkpoint RGB interventions; never train or select.

Only replaces the five RGB images. Target geometry/index/anchor remain intact.
A donor RGB scene with target geometry is physically inconsistent diagnostic input.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

WORK = Path(__file__).resolve().parents[2]
ROOT = Path('/data/wangcheng/checkpoint/ue_compact_joint_20261024')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--mode', choices=['constant_rgb', 'donor_rgb'])
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    if args.plan_only:
        print(json.dumps(dict(modes=['constant_rgb', 'donor_rgb'], arm='rgbd',
            selection='original full100epoch validation-best; no new selection',
            requires='all four full trainings and official tests complete',
            donor='same test split order, (index+459//2)%459, five views jointly',
            preserved=['target measured geometry', 'target support', 'target source anchor and index',
                       'target intrinsics/extrinsics', 'target satellite image'],
            caveat='RGB-only donor is physically inconsistent, not a scene counterfactual'), indent=2))
        return
    assert args.mode
    for arm in ('satellite', 'rgbd', 'constant_rgb', 'reference_plane'):
        for name in ('training_complete.json', 'test/complete.json'):
            p = args.root / arm / name
            assert p.exists() and json.loads(p.read_text())['complete'], p
    sys.path.insert(0, str(WORK / 'scripts/probes'))
    import ue_compact_joint_run_20261024 as run
    import cv2
    import numpy as np
    import torch
    from rsprompter.uav.local_adapter_probe import YAWS
    root = args.root
    checkpoint = root / 'rgbd/best_model.pth'
    checkpoint_sha = digest(checkpoint)
    training = json.loads((root / 'rgbd/training_complete.json').read_text())
    official = json.loads((root / 'rgbd/test/complete.json').read_text())
    assert checkpoint_sha == training['best_checkpoint_sha256'] == official['checkpoint_sha256']
    assert training['epoch'] == 100 and training['updates'] == 26700
    out = root / 'fixed_rgb_intervention' / args.mode
    assert not out.exists(), 'Preserve every started intervention; review incomplete runs explicitly'
    out.mkdir(parents=True)
    runtime = run.Runtime(root, 'rgbd')
    contract = run.contract(runtime)
    assert contract == training['contract']
    runtime.restore(checkpoint, contract, training=False)
    runtime.network.eval()
    initial_state = run.core.C.state_digest(runtime.network)
    dataset = run.core.C.MinimalSatelliteDataset(run.core.previous.DATA, 'test')
    ids = [int(x['image_id']) for x in dataset.samples]
    assert len(ids) == len(set(ids)) == 459
    donor = {ident: ids[(i + len(ids) // 2) % len(ids)] for i, ident in enumerate(ids)}
    assert all(k != v for k, v in donor.items())
    original_payload = runtime.payload
    seen, rgb_sources = [], {}

    def payload(ident):
        value = original_payload(ident)
        retained = {name: value[name] for name in ('geometry', 'index', 'anchor')}
        if args.mode == 'constant_rgb':
            value['rgb'] = torch.tensor([123.675, 116.28, 103.53],
                device=value['rgb'].device).view(1, 3, 1, 1).expand_as(value['rgb']).clone()
        else:
            folder = run.core.previous.DATA / 'points' / ('%06d' % donor[int(ident)])
            images = []
            for yaw in YAWS:
                path = folder / ('uav/yaw_%03d/rgb.png' % yaw)
                image = cv2.imread(str(path))
                assert image is not None and image.shape == (512, 512, 3)
                images.append(image[:, :, ::-1].transpose(2, 0, 1).copy())
                rgb_sources[str(path)] = digest(path)
            value['rgb'] = torch.from_numpy(np.stack(images)).to(value['rgb'])
        assert value['rgb'].shape == (5, 3, 512, 512)
        assert all(value[name] is tensor for name, tensor in retained.items())
        seen.append(int(ident))
        return value

    runtime.payload = payload
    run.evaluate(runtime, 'test', out / 'raw', checkpoint_sha256=checkpoint_sha)
    assert seen == ids and len(set(seen)) == 459
    assert run.core.C.state_digest(runtime.network) == initial_state
    assert digest(checkpoint) == checkpoint_sha
    assert run.sources() == contract['source_hashes']
    raw = json.loads((out / 'raw/complete.json').read_text())
    assert raw['images'] == 459 and raw['checkpoint_sha256'] == checkpoint_sha
    assert all(digest(path) == expected for path, expected in rgb_sources.items())
    source = Path(__file__).resolve()
    (out / source.name).write_bytes(source.read_bytes())
    run.core.C.dump(out / 'complete.json', dict(complete=True, mode=args.mode, arm='rgbd',
        selection='official validation-best after full100epochs', checkpoint_sha256=checkpoint_sha,
        model_state_unchanged=True, images=459, split='test', precision='FP32', max_dets=250,
        preserved_inputs=['target geometry including support', 'target anchor', 'target index',
                          'target calibration', 'target satellite'],
        donor_map=donor if args.mode == 'donor_rgb' else None, rgb_source_sha256=rgb_sources,
        diagnostic_scope='constant/donor RGB only; donor keeps original geometry and is physically inconsistent',
        source_sha256=digest(source), training_source_hashes=contract['source_hashes'],
        raw_records_sha256=digest(out / 'raw/records.json'),
        raw_metrics_sha256=digest(out / 'raw/metrics.json')))


if __name__ == '__main__':
    main()
