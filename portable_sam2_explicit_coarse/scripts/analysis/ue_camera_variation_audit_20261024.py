"""Camera variation and identity audit; no predictions or model selection."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('/data/wangcheng/dataset/CitySample_SatUAV_TrainMinimal_v1_20261021'))
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    result, files = {}, {}
    for split in ('train', 'val', 'test'):
        ids = json.loads((args.data / 'splits' / f'{split}.json').read_text())
        intrinsics, relative = {}, {}
        for ident in ids:
            path = args.data / 'points' / f'{ident:06d}' / 'cameras.json'
            files[str(path)] = sha(path)
            c = json.loads(path.read_text())
            assert c['world_units'] == 'cm'
            satellite = np.array(c['satellite']['world_to_camera_cv'], dtype=np.float64)
            for name, camera in [('satellite', c['satellite'])] + list(c['uav'].items()):
                k = camera['intrinsics']
                intrinsics.setdefault(name, []).append([k[x] for x in ('fx', 'fy', 'cx', 'cy')])
                if name != 'satellite':
                    t = satellite @ np.array(camera['camera_to_world_cv'], dtype=np.float64)
                    t[:3, 3] /= 100
                    relative.setdefault(name, []).append(t)
        views = {}
        for name, values in intrinsics.items():
            values = np.array(values)
            row = dict(K_unique_exact=len(np.unique(values, axis=0)),
                       K_fx_fy_cx_cy_min=values.min(0).tolist(),
                       K_fx_fy_cx_cy_max=values.max(0).tolist())
            if name in relative:
                poses = np.array(relative[name])
                row.update(relative_pose_unique_rounded_6=len(np.unique(poses.reshape(len(poses), -1).round(6), axis=0)),
                           relative_translation_m_min=poses[:, :3, 3].min(0).tolist(),
                           relative_translation_m_max=poses[:, :3, 3].max(0).tolist())
            views[name] = row
        result[split] = dict(images=len(ids), views=views)
    output = dict(complete=True, source_sha256=sha(Path(__file__)), split_summary=result,
        camera_file_sha256=files,
        scope='Describe calibration variation only, no AP or necessity proof. Global world translation cancels in UAV-to-satellite relative poses. Test not used to choose model parameters.')
    (args.out / 'complete.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
