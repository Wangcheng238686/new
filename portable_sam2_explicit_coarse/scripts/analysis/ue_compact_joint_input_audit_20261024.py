"""Actual reference-cache bytes, source cameras and center-target equivalence."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

WORK = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(WORK), str(WORK / 'scripts/probes')]
import ue_compact_joint_run_20261024 as run
from data.ue_instance_targets import center_targets
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=run.ROOT)
    args = p.parse_args()
    folder = run.PLANE
    complete = json.loads((folder / 'BUILD_COMPLETE.json').read_text())
    manifest = json.loads((folder / 'manifest.json').read_text())
    assert complete['complete'] and complete['images'] == 3060
    assert run.core.C.file_sha(folder / 'manifest.json') == complete['manifest_sha256']
    assert len(complete['rows']) == len({row['image_id'] for row in complete['rows']}) == 3060
    assert len(list(folder.glob('*.npz'))) == 3060
    for name, sha in manifest['source_hashes'].items():
        assert run.core.C.file_sha(WORK / name) == run.core.C.file_sha(folder / 'source' / name) == sha
    def verify(row):
        assert row['measured_depth_read'] is False and row['relative_reference_z_m'] == 0.
        assert row['source_hashes'] == manifest['source_hashes']
        ident = row['image_id']
        cache = folder / f'{ident:06d}.npz'
        camera = run.core.previous.DATA / 'points' / f'{ident:06d}' / 'cameras.json'
        assert run.core.C.file_sha(cache) == row['cache_sha256'], cache
        assert run.core.C.file_sha(camera) == row['camera_sha256'], camera
        return cache.stat().st_size
    with ThreadPoolExecutor(max_workers=4) as pool:
        sizes = list(pool.map(verify, complete['rows']))
    # Old target creator is used ONLY as a reference in this audit, never in
    # the no-depth training pipeline (it reads an unused measured x field).
    old = run.core.pilot.Data('train')
    choices = [0, max(range(len(old)), key=lambda i: len(old.boxes[i])), len(old) - 1]
    targets = []
    for i in choices:
        expected = old[i]
        actual = center_targets(old.boxes[i], device='cpu')
        for key, value in actual.items():
            assert torch.equal(value[0], expected[key]), (i, key)
        targets.append(dict(image_id=old.ids[i], instances=len(old.boxes[i]), heat_reg_valid_exact=True))
    assert len(old.ids) == 2142
    output = dict(passed=True, reference_files=3060, total_reference_bytes=sum(sizes),
        actual_cache_and_camera_SHA_all_match=True, source_snapshots_exact=True,
        reference_manifest_sha256=run.core.C.file_sha(folder / 'manifest.json'),
        build_complete_sha256=run.core.C.file_sha(folder / 'BUILD_COMPLETE.json'),
        satellite_only_targets=targets, source_hashes=run.sources(),
        audit_source_sha256=run.core.C.file_sha(__file__),
        scope='Input provenance/bytes and training target engineering, not depth content AP or external calibration proof')
    run.core.C.dump(args.root / 'input_audit.json', output)
    print('COMPACT JOINT INPUT AUDIT PASS', output, flush=True)


if __name__ == '__main__':
    main()
