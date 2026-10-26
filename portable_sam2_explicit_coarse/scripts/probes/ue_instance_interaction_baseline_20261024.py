"""Replay the formal disabled path against all 459 historical strong-SAT rows."""
import argparse
import json

import torch
import ue_instance_interaction_20261024 as run
import ue_instance_interaction_owner_20261024 as owner


def main():
    args = argparse.Namespace(root=run.OUT, seed=44, family='geom', route='internal')
    model, adapter, _, _ = run.build(args)
    adapter.eval()
    adapter.enabled = False
    dataset = run.C.MinimalSatelliteDataset(run.previous.DATA, 'test')
    reference = json.loads(owner.BASE.read_text())
    out = args.root / 'baseline' / 'test'
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    with torch.inference_mode():
        for index in range(len(dataset)):
            sample = run.C.processed(model, run.C.rtmdet_collate_fn([dataset[index]]))
            features = model.extract_feat(sample['inputs'])
            prefix, combined, _ = run.inference_outputs(model, adapter, sample, features, None)
            row = dict(image_id=dataset.samples[index]['image_id'],
                       gt=run.C.serialized(run.C.gt_arrays(sample['data_samples'][0].gt_instances)),
                       dt=run.C.serialized(prefix))
            expected = reference[index]
            assert row['image_id'] == expected['image_id'] and row['gt'] == expected['gt']
            assert row['dt'] == expected['dt'], (index, 'Formal path differs from historical strong baseline')
            rows.append(row)
            if (index + 1) % 25 == 0:
                run.C.dump(out / 'progress.json', dict(images=index + 1))
                print('BASELINE EXACT', index + 1, flush=True)
    assert run.C.state_digest(model) == run.previous.FROZEN_SHA
    run.C.dump(out / 'records.json', rows)
    run.C.dump(out / 'complete.json', dict(complete=True, images=459, historical_exact=True,
        reference_sha256=run.C.file_sha(owner.BASE), source_hashes=run.source_hashes(),
        records_sha256=run.C.file_sha(out / 'records.json')))


if __name__ == '__main__':
    main()
