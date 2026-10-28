"""Final actual-byte audit of the frozen archive and all three geometry caches."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

WORK = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(WORK), str(WORK / 'scripts/probes')]
import ue_compact_joint_run_20261024 as run


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=run.ROOT)
    args = p.parse_args()
    root = args.root
    out = root / 'final_input_identity.json'
    assert not out.exists()
    contracts = []
    for arm in run.PROTOCOL['arms']:
        done = json.loads((root / arm / 'training_complete.json').read_text())
        assert done['complete'] and done['epoch'] == 100 and done['updates'] == 26700
        contracts.append(done['contract'])
    data = run.core.previous.DATA
    archive = run.core.C.archive_identity(data)
    assert all(c['data_identity'] == archive for c in contracts)
    files = []
    for line in (data / 'SHA256SUMS.txt').read_text().splitlines():
        expected, relative = line.split('  ', 1)
        path = data / relative
        assert path.resolve().is_relative_to(data.resolve())
        files.append((path, expected))
    assert len(files) == len({str(p) for p, _ in files}) and len(files) >= 3060*12
    measured_manifest = run.core.OUT / 'input_cache_sha256.json'
    assert all(c['measured_cache_manifest_sha256'] == run.core.C.file_sha(measured_manifest) for c in contracts)
    measured = json.loads(measured_manifest.read_text())
    assert measured['complete']
    cache_files = []
    for key, folder in [('native', '/data/wangcheng/cache/ue_v1_native_geometry'),
                        ('anchors', '/data/wangcheng/cache/ue_v1_s2db_geometry_v2')]:
        values = measured['caches'][key]
        assert len(values) == len(list(Path(folder).glob('*.npz'))) == 3060
        cache_files.extend((Path(folder)/name, expected) for name, expected in values.items())
    plane = json.loads((run.PLANE / 'BUILD_COMPLETE.json').read_text())
    assert plane['complete'] and plane['images'] == 3060
    assert run.core.C.file_sha(run.PLANE / 'manifest.json') == plane['manifest_sha256']
    assert all(c['plane_manifest_sha256'] == plane['manifest_sha256'] for c in contracts)
    assert len(plane['rows']) == len({r['image_id'] for r in plane['rows']}) == 3060
    assert len(list(run.PLANE.glob('*.npz'))) == 3060
    for row in plane['rows']:
        assert row['measured_depth_read'] is False and row['relative_reference_z_m'] == 0
        cache_files.append((run.PLANE / ('%06d.npz' % row['image_id']), row['cache_sha256']))
    def verify(item):
        path, expected = item
        assert run.core.C.file_sha(path) == expected, path
        return path.stat().st_size
    with ThreadPoolExecutor(max_workers=4) as pool:
        archive_sizes = list(pool.map(verify, files))
        cache_sizes = list(pool.map(verify, cache_files))
    assert run.core.C.archive_identity(data) == archive
    initial = json.loads((root / 'initialization.json').read_text())
    assert run.sources() == initial['source_hashes']
    destination = root / 'final_input_identity_source.py'
    destination.write_bytes(Path(__file__).read_bytes())
    run.core.C.dump(out, dict(complete=True, archive_files=len(files), geometry_cache_files=len(cache_files),
        archive_bytes=sum(archive_sizes), geometry_cache_bytes=sum(cache_sizes),
        actual_SHA_all_match=True, archive_identity=archive, source_hashes=run.sources(),
        diagnostic_source_sha256=run.core.C.file_sha(Path(__file__)),
        scope='actual bytes against pre-existing frozen manifests after training; no AP, no label decoding, no physical calibration certification'))


if __name__ == '__main__':
    main()
