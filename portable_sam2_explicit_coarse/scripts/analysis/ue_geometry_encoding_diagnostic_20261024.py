"""Train-only read-only audit of numerical geometry encoding and support masks.

Quantifies representation ambiguity and scale; cannot establish AP causation.
No labels, validation/test samples, model parameters or training code are changed.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

DATA = Path('/data/wangcheng/dataset/CitySample_SatUAV_TrainMinimal_v1_20261021')
CACHE = Path('/data/wangcheng/cache/ue_v1_native_geometry')
OUT = Path('/data/wangcheng/checkpoint/ue_compact_joint_20261024/geometry_encoding_train_audit.json')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert not OUT.exists()
    split_path = DATA / 'splits/train.json'
    ids = json.loads(split_path.read_text())
    assert len(ids) == len(set(ids)) == 2142
    sums = np.zeros(4, np.float64)
    squares = sums.copy()
    counts = dict(supported_pixels=0, all_pixels=0, occupied64_cells=0, all64_cells=0,
                  full64_cells=0)
    axes = {name: dict(valid_pairs=0, supported_origins=0, unknown_pairs=0,
                      zero_valid_pairs=0, zero_supported_origins=0, clipped_valid_pairs=0)
            for name in ('dx', 'dy')}
    files, occupancies = {}, []
    for ident in ids:
        p = CACHE / ('%06d.npz' % int(ident))
        expected = sha(p)
        with np.load(p) as z:
            assert str(z['version']) == 'native512_height_dx_dy_viewcount_v1'
            f = z['fields'].astype(np.float32)
            s = z['support'].astype(bool)
        assert f.shape == (4, 512, 512) and s.shape == (512, 512)
        assert np.isfinite(f).all() and not np.any(f[:, ~s])
        n = int(s.sum())
        values = f[:, s].astype(np.float64)
        sums += values.sum(1)
        squares += np.square(values).sum(1)
        counts['supported_pixels'] += n
        counts['all_pixels'] += s.size
        fill = s.reshape(64, 8, 64, 8).sum((1, 3))
        occupied = fill > 0
        counts['occupied64_cells'] += int(occupied.sum())
        counts['all64_cells'] += fill.size
        counts['full64_cells'] += int((fill == 64).sum())
        if occupied.any():
            occupancies.extend((fill[occupied] / 64).tolist())
        for axis, channel in [('dx', 1), ('dy', 2)]:
            valid = np.zeros_like(s)
            if axis == 'dx':
                valid[:, :-1] = s[:, :-1] & s[:, 1:]
            else:
                valid[:-1] = s[:-1] & s[1:]
            a = axes[axis]
            a['valid_pairs'] += int(valid.sum())
            a['supported_origins'] += n
            a['unknown_pairs'] += int((s & ~valid).sum())
            a['zero_valid_pairs'] += int(((f[channel] == 0) & valid).sum())
            a['zero_supported_origins'] += int(((f[channel] == 0) & s).sum())
            a['clipped_valid_pairs'] += int(((np.abs(f[channel]) >= 5) & valid).sum())
        assert sha(p) == expected
        files[str(p)] = expected
    for a in axes.values():
        a['unknown_fraction_of_supported_origins'] = a['unknown_pairs'] / a['supported_origins']
        a['unknown_fraction_of_zero_supported_origins'] = a['unknown_pairs'] / max(1, a['zero_supported_origins'])
        a['clip_fraction_of_valid_pairs'] = a['clipped_valid_pairs'] / max(1, a['valid_pairs'])
    mean = sums / counts['supported_pixels']
    rms = np.sqrt(squares / counts['supported_pixels'])
    result = dict(complete=True, split='train', images=len(ids), source_sha256=sha(__file__),
        split_sha256=sha(split_path), cache_sha256=files, counts=counts, differences=axes,
        normalized_field_order=['relative_worldZ/100m', 'clip(dx/10m,+-5)', 'clip(dy/10m,+-5)', 'near_surface_views/5'],
        supported_pixel_mean=mean.tolist(), supported_pixel_RMS=rms.tolist(),
        supported_pixel_std=np.sqrt(np.maximum(0, squares/counts['supported_pixels']-mean**2)).tolist(),
        occupied64_raw_support_fraction_quantiles=dict(zip(['min','p25','median','p75','max'],
            np.quantile(occupancies,[0,.25,.5,.75,1]).tolist())),
        limitations='support64 coverage proxy only, not exact feature receptive field or visibility; derivative unknown includes outer image edge; zero-valid may be genuine flatness or float16 rounding; no AP attribution; train-only observation does not modify locked joint training')
    OUT.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'cache_sha256'},indent=2))


if __name__ == '__main__':
    main()
