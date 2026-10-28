"""Build an independent no-measured-depth cache from cameras and a fixed plane."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

WORK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORK))
import numpy as np
from rsprompter.uav.geometry_inputs import reference_plane_depth, candidate_arrays, native_arrays
from rsprompter.uav.local_adapter_probe import file_sha

VERSION = 'reference_plane_sampling_origin_v1'


def dump(path, value):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('/data/wangcheng/dataset/CitySample_SatUAV_TrainMinimal_v1_20261021'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--workers', type=int, choices=[1, 2, 3, 4], default=2)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (args.out / 'build.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    sources = {name: file_sha(WORK / name) for name in (
        'rsprompter/uav/geometry_inputs.py', 'rsprompter/uav/local_adapter_probe.py',
        'scripts/probes/ue_reference_plane_cache_20261024.py')}
    splits = {name: json.loads((args.data / 'splits' / f'{name}.json').read_text())
              for name in ('train', 'val', 'test')}
    split_sha = {name: file_sha(args.data / 'splits' / f'{name}.json') for name in splits}
    ids = sorted({ident for values in splits.values() for ident in values})
    assert len(ids) == sum(map(len, splits.values())) == 3060
    manifest = dict(version=VERSION, source_hashes=sources, data=str(args.data),
        split_sha256=split_sha, split_counts={key: len(value) for key, value in splits.items()},
        relative_reference_z_m=0., reference='horizontal plane at renderer sampling-origin world Z; not inferred ground',
        inputs=['cameras.json including sampling origin'],
        forbidden=['measured EXR depth', 'old measured candidate/anchor/field/support caches'],
        format='combined index/geom/anchor/fields/support npz, all rebuilt from the declared prior')
    manifest_path = args.out / 'manifest.json'
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == manifest, 'Cache contract changed'
    else:
        dump(manifest_path, manifest)
    for name, sha in sources.items():
        path = args.out / 'source' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((WORK / name).read_bytes())
        assert file_sha(path) == sha
    def forbid_measured(event, values):
        if event != 'open' or not values or not isinstance(values[0], (str, bytes)):
            return
        path = os.fsdecode(values[0])
        if path.endswith('.exr') or any(x in path for x in (
                '/ue_v1_native_geometry/', '/ue_v1_s2db_geometry_v2/', '/ue_v1_local_adapter_geometry/')):
            raise AssertionError('No-depth builder attempted measured input access: ' + path)
    sys.addaudithook(forbid_measured)
    def build(ident):
        camera_file = args.data / 'points' / f'{ident:06d}' / 'cameras.json'
        camera_sha = file_sha(camera_file)
        path = args.out / f'{ident:06d}.npz'
        if path.exists():
            with np.load(path) as z:
                old = json.loads(str(z['report']))
                assert str(z['version']) == VERSION and old['camera_sha256'] == camera_sha
                assert old['source_hashes'] == sources
                assert z['index'].shape == (16384, 8) and z['fields'].shape == (4, 512, 512)
            return dict(image_id=ident, cache_sha256=file_sha(path), **old)
        cameras = json.loads(camera_file.read_text())
        depth = reference_plane_depth(cameras, 0.)
        index, geom, anchor = candidate_arrays(cameras, depth)
        fields, support = native_arrays(cameras, depth)
        assert np.isfinite(geom).all() and np.isfinite(anchor).all() and np.isfinite(fields).all()
        report = dict(camera_sha256=camera_sha, source_hashes=sources, measured_depth_read=False,
            candidates=int((index >= 0).sum()), candidate_cells=int((index >= 0).any(1).sum()),
            supported_pixels=int(support.sum()), relative_reference_z_m=0.)
        tmp = path.with_name(path.name + '.tmp')
        with tmp.open('wb') as file:
            np.savez_compressed(file, index=index, geom=geom, anchor=anchor, fields=fields,
                                support=support, report=json.dumps(report), version=VERSION)
        tmp.replace(path)
        return dict(image_id=ident, cache_sha256=file_sha(path), **report)
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(build, ids):
            rows.append(row)
            if len(rows) % 25 == 0:
                dump(args.out / 'progress.json', dict(pid=os.getpid(), images=len(rows), total=len(ids)))
                print('PLANE CACHE', len(rows), len(ids), flush=True)
    assert all(file_sha(WORK / name) == sha for name, sha in sources.items())
    assert all(file_sha(args.data / 'splits' / f'{name}.json') == sha for name, sha in split_sha.items())
    dump(args.out / 'BUILD_COMPLETE.json', dict(complete=True, images=len(rows), rows=rows,
        manifest_sha256=file_sha(manifest_path), measured_inputs_forbidden_by_audit_hook=True))
    print('REFERENCE PLANE CACHE COMPLETE', len(rows), flush=True)


if __name__ == '__main__':
    main()
