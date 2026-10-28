"""Verify pure geometry conversion and a depth-free reference-plane control."""
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import argparse
import json
from pathlib import Path
import sys

WORK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORK))
import cv2
import numpy as np
from rsprompter.uav.geometry_inputs import reference_plane_depth, candidate_arrays, native_arrays
from rsprompter.uav.local_adapter_probe import YAWS, file_sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    data = Path('/data/wangcheng/dataset/CitySample_SatUAV_TrainMinimal_v1_20261021')
    val = json.loads((data / 'splits/val.json').read_text())
    ids = [val[i] for i in (0, len(val) // 2, len(val) - 1)]
    source = WORK / 'rsprompter/uav/geometry_inputs.py'
    hashes = {str(source): file_sha(source), str(Path(__file__)): file_sha(__file__)}
    blocked = [False]
    attempts = []
    def deny_io(event, arguments):
        if event == 'open' and blocked[0]:
            attempts.append(repr(arguments))
            raise AssertionError('Pure reference-plane conversion attempted file I/O')
    sys.addaudithook(deny_io)
    rows = []
    for ident in ids:
        folder = data / 'points' / f'{ident:06d}'
        cameras = json.loads((folder / 'cameras.json').read_text())
        depths = np.stack([cv2.imread(str(folder / f'uav/yaw_{yaw:03d}/depth_camera_z.exr'),
                            cv2.IMREAD_UNCHANGED)[:, :, 2].astype(np.float64) / 100 for yaw in YAWS])
        index, geom, anchor = candidate_arrays(cameras, depths)
        fields, support = native_arrays(cameras, depths)
        with np.load(Path('/data/wangcheng/cache/ue_v1_s2db_geometry_v2') / f'{ident:06d}.npz') as z:
            assert np.array_equal(index, z['index'])
            assert np.array_equal(geom, z['geom'])
            assert np.array_equal(anchor, z['anchor'])
        with np.load(Path('/data/wangcheng/cache/ue_v1_native_geometry') / f'{ident:06d}.npz') as z:
            assert np.array_equal(fields, z['fields'])
            assert np.array_equal(support, z['support'])
        # The depth-free path receives ONLY camera metadata and the declared
        # sampling plane. Every file open is rejected while it executes.
        blocked[0] = True
        try:
            plane = reference_plane_depth(cameras, relative_z_m=0.)
            pi, pg, pa = candidate_arrays(cameras, plane)
            pf, ps = native_arrays(cameras, plane)
            invalid = np.full_like(plane, np.nan)
            ei, _, _ = candidate_arrays(cameras, invalid)
            ef, es = native_arrays(cameras, invalid)
            assert not es.any() and not ef.any() and (ei == -1).all()
        finally:
            blocked[0] = False
        assert np.isfinite(pa).all() and np.isfinite(pg).all() and np.isfinite(pf).all()
        assert (pi >= 0).any() and ps.any()
        assert not np.array_equal(pi, index), 'Control failed to alter measured-depth correspondence'
        valid = pi >= 0
        tokens = pi[valid] % 16384
        assert np.array_equal((pa[..., 1][valid].astype(int) // 4) * 128 + pa[..., 0][valid].astype(int) // 4, tokens)
        uv = pa[..., 3:5][valid]
        cells = np.broadcast_to(np.arange(16384)[:, None], pi.shape)[valid]
        assert np.array_equal((uv[:, 1] // 4).astype(int) * 128 + (uv[:, 0] // 4).astype(int), cells)
        np.savez_compressed(args.out / f'{ident:06d}_plane.npz', index=pi, geom=pg, anchor=pa,
                            fields=pf, support=ps, version='reference_plane_sampling_origin_v1')
        rows.append(dict(image_id=ident, camera_sha256=file_sha(folder / 'cameras.json'),
            measured_conversion_exact=True, reference_build_no_file_reads=True,
            measured_candidate_count=int((index >= 0).sum()), plane_candidate_count=int(valid.sum()),
            measured_supported_pixels=int(support.sum()), plane_supported_pixels=int(ps.sum()),
            index_slots_changed=int((pi != index).sum())))
    assert not attempts
    assert all(file_sha(path) == sh for path, sh in hashes.items())
    result = dict(complete=True, rows=rows, source_hashes=hashes, all_invalid_fallback=True,
        reference_relative_z_m=0., reference_inputs=['camera metadata', 'declared sampling-origin plane'],
        measured_inputs_used_to_generate_reference=False,
        scope='3 predetermined val indices, conversion/cache engineering gate only; not depth-ablation training/AP or actual ground estimation')
    (args.out / 'complete.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
