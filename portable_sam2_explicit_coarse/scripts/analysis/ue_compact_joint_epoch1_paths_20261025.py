"""Snapshot E1 validation and separate original-path from appended candidates.

Diagnostic only: different jointly trained models, not a fixed-weight modality
intervention; never selects a checkpoint or changes the locked training protocol.
"""
import copy
import json
from pathlib import Path
import shutil
import sys

import numpy as np

WORK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORK / 'scripts/analysis'))
import ue_compact_joint_paired_20261024 as paired


def main():
    root = paired.ROOT
    out = root / 'epoch1_path_diagnostic'
    assert not out.exists(), 'Preserve prior snapshot/analysis'
    out.mkdir()
    snapshot, identities = {}, {}
    arms = ('satellite', 'rgbd', 'constant_rgb', 'reference_plane')
    for arm in arms:
        folder = root / arm
        epoch = paired.read(folder / 'epoch1.json')
        current = folder / 'validation_current'
        assert paired.read(current / 'metrics.json') == epoch['validation']
        complete = paired.read(current / 'complete.json')
        assert complete['split'] == 'val' and complete['purpose'] == 'full_protocol'
        assert complete['images'] == 459 and complete['precision'] == 'FP32'
        dest = out / arm
        dest.mkdir()
        for name in ('records.json', 'metrics.json', 'complete.json'):
            path = current / name
            before = paired.sha(path)
            shutil.copyfile(path, dest / name)
            assert before == paired.sha(path) == paired.sha(dest / name)
        assert paired.read(current / 'metrics.json') == epoch['validation']
        shutil.copyfile(folder / 'epoch1.json', dest / 'epoch1.json')
        snapshot[arm] = paired.read(dest / 'records.json')
        assert len(snapshot[arm]) == len({r['image_id'] for r in snapshot[arm]}) == 459
        identities[arm] = dict(records_sha256=paired.sha(dest / 'records.json'),
            epoch1_checkpoint_sha256=paired.sha(folder / 'epoch1.pth'),
            epoch1_log_sha256=paired.sha(folder / 'epoch1.json'),
            source_hashes=complete['source_hashes'])
    base = snapshot['satellite']
    for rows in snapshot.values():
        assert [r['image_id'] for r in rows] == [r['image_id'] for r in base]
        assert all(b['gt'] == c['gt'] for b, c in zip(base, rows))
    result = dict(complete=False, epoch=1, split='val', images=459, max_dets=250,
        precision='FP32', artifacts=identities, arms={},
        source_sha256=paired.sha(__file__),
        scope='early different-model validation diagnostic, not final AP or fixed-weight causality; original path includes existing satellite candidate supplement; retained-prefix scores copied exactly from combined outputs')
    for arm, rows in snapshot.items():
        original, counts = [], []
        for row in rows:
            dt = copy.deepcopy(row['original_path'])
            n = len(dt['scores'])
            assert row['dt']['bboxes'][:n] == dt['bboxes']
            assert row['dt']['labels'][:n] == dt['labels']
            assert row['dt']['rles'][:n] == dt['rles']
            assert len(row['dt']['scores']) <= 250
            dt['scores'] = row['dt']['scores'][:n]
            assert row['geometry_accepted'] == len(row['dt']['scores']) - n
            counts.append((n, row['geometry_accepted']))
            original.append(dict(image_id=row['image_id'], gt=row['gt'], dt=dt))
        metrics = {}
        for name, kind in [('MaskAP', 'segm'), ('BBoxAP', 'bbox')]:
            eo = paired.fast.original.evaluator(original, kind=kind)
            ec = paired.fast.original.evaluator(rows, kind=kind)
            po, pc = paired.point_metrics(eo), paired.point_metrics(ec)
            key = 'segm/mAP' if kind == 'segm' else 'bbox/mAP'
            assert abs(pc['AP'] - paired.read(out / arm / 'epoch1.json')['validation'][key]) < 1e-12
            metrics[name] = dict(original_path=po, combined=pc,
                appended_candidates_delta=pc['AP'] - po['AP'])
            print(arm, name, metrics[name], flush=True)
        counts = np.asarray(counts)
        result['arms'][arm] = dict(metrics=metrics,
            original_count_mean=float(counts[:, 0].mean()),
            appended_count_mean=float(counts[:, 1].mean()),
            appended_total=int(counts[:, 1].sum()))
    for arm, value in result['arms'].items():
        value['delta_vs_satellite'] = {name: value['metrics'][name]['combined']['AP'] -
            result['arms']['satellite']['metrics'][name]['combined']['AP'] for name in ('MaskAP', 'BBoxAP')}
        assert paired.sha(out / arm / 'records.json') == identities[arm]['records_sha256']
    result['complete'] = True
    (out / 'complete.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
