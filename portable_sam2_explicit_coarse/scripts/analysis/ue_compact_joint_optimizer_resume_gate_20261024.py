"""Disk restore of actual nonempty AdamW/AMP state in the compact runner."""
import argparse
import copy
from pathlib import Path
import sys

WORK = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(WORK), str(WORK / 'scripts/probes'), str(Path(__file__).resolve().parent)]
import ue_compact_joint_run_20261024 as run
from ue_compact_joint_runtime_gate_20261024 import same_rng
import torch


def exact(a, b):
    if torch.is_tensor(a):
        assert torch.is_tensor(b) and torch.equal(a.cpu(), b.cpu())
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            exact(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            exact(x, y)
    else:
        assert a == b, (a, b)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=run.ROOT)
    args = p.parse_args()
    out = args.root / 'optimizer_resume_gate'
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'complete.json').exists()
    source = run.sources()
    gate_sources = {str(Path(__file__)): run.core.C.file_sha(__file__),
        str(Path(__file__).with_name('ue_compact_joint_runtime_gate_20261024.py')):
        run.core.C.file_sha(Path(__file__).with_name('ue_compact_joint_runtime_gate_20261024.py'))}
    runtime = run.Runtime(args.root, 'reference_plane')
    frozen = run.contract(runtime, require_plane_complete=False)
    ds = run.core.C.MinimalSatelliteDataset(run.core.previous.DATA, 'train', augment=True)
    def cycle(step):
        runtime.network.train()
        runtime.optimizer.zero_grad(set_to_none=True)
        ids = []
        for i in range(8):
            ident = ds.samples[i]['image_id']
            sample = run.core.C.processed(runtime.satellite, run.core.C.rtmdet_collate_fn([ds[i]]), True)
            with torch.autocast('cuda', dtype=torch.float16):
                loss = runtime.objective(sample, ident)
            assert torch.isfinite(loss)
            runtime.scaler.scale(loss / 8).backward()
            runtime.clear()
            ids.append(ident)
        runtime.update(step)
        return ids
    ids = cycle(1)
    run.evaluate(runtime, 'val', out / 'engineering_validation', limit=1)
    saved = copy.deepcopy(runtime.state(0, 1, frozen, dict(score=-1., epoch=None)))
    assert saved['optimizer']['state']
    assert any(value['exp_avg'].abs().max() > 0 for value in saved['optimizer']['state'].values())
    checkpoint = out / 'nonempty_optimizer.pth'
    run.core.previous.atomic_save(saved, checkpoint)
    with torch.no_grad():
        next(runtime.network.parameters()).add_(.01)
        for value in runtime.optimizer.state.values():
            value['exp_avg'].fill_(.125)
    runtime.scaler.update(new_scale=128.)
    run.core.C.seed(901)
    runtime.generator.manual_seed(902)
    runtime.restore(checkpoint, frozen)
    exact(saved['model'], runtime.network.state_dict())
    exact(saved['optimizer'], runtime.optimizer.state_dict())
    exact(saved['scaler'], runtime.scaler.state_dict())
    same_rng(saved['rng_state'], run.core.previous.rng_state(runtime.generator))
    assert cycle(2) == ids
    steps = sorted({int(value['step']) for value in runtime.optimizer.state.values()})
    assert max(steps) == 2
    assert run.sources() == source
    assert all(run.core.C.file_sha(path) == sha for path, sha in gate_sources.items())
    run.core.C.dump(out / 'complete.json', dict(complete=True, source_hashes=source,
        gate_source_hashes=gate_sources, actual_saved_update=1, actual_resumed_update=2,
        model_optimizer_scaler_exact_before_resumed_update=True, all_five_rng_exact=True,
        AMP_scale=runtime.scaler.get_scale(), optimizer_step_values_after_resume=steps,
        training_image_ids=ids, accumulation=8, checkpoint_sha256=run.core.C.file_sha(checkpoint),
        scope='Actual nonempty moments restored then a finite accumulation/update executed; no bit-exact CUDA training trajectory or AP claim'))
    print('COMPACT NONEMPTY OPTIMIZER RESTORE PASS', flush=True)


if __name__ == '__main__':
    main()
