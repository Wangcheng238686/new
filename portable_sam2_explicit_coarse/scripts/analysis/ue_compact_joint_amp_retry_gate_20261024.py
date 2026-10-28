"""Force one overflow and verify same-batch AMP retry with one real update."""
import argparse
import hashlib
from pathlib import Path
import sys

WORK = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(WORK), str(WORK / 'scripts/probes')]
import ue_compact_joint_run_20261024 as run
import torch


def digest(values):
    return run.core.previous.tensor_digest({name: value.detach() for name, value in values})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=run.ROOT)
    args = p.parse_args()
    out = args.root / 'amp_retry_gate'
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'complete.json').exists()
    source = run.sources()
    gate_sha = run.core.C.file_sha(__file__)
    runtime = run.Runtime(args.root, 'reference_plane')
    runtime.network.train(); runtime.set_epoch(1)
    ds = run.core.C.MinimalSatelliteDataset(run.core.previous.DATA, 'train', augment=True)
    batches = [(run.core.C.rtmdet_collate_fn([ds[i]]), ds.samples[i]['image_id']) for i in range(8)]
    images = [hashlib.sha256(batch['imgs'].numpy().tobytes()).hexdigest() for batch, _ in batches]
    parameters_before = digest(runtime.network.named_parameters())
    original_step = runtime.optimizer.step
    calls = []
    buffers = []
    def actual_step(*a, **kw):
        calls.append(True)
        return original_step(*a, **kw)
    runtime.optimizer.step = actual_step
    def force(attempt):
        buffers.append(digest(runtime.network.named_buffers()))
        if attempt == 0:
            parameter = next(p for p in runtime.network.parameters() if p.grad is not None)
            parameter.grad.fill_(float('inf'))
        elif attempt == 1:
            assert not calls and not runtime.optimizer.state
            assert digest(runtime.network.named_parameters()) == parameters_before
            assert buffers[0] == buffers[1]
        else:
            raise AssertionError('More than the single injected overflow occurred')
    result = run.accumulation(runtime, batches, 1, before_update=force)
    assert result['retries'] == 1 and len(calls) == 1 and runtime.scaler.get_scale() == 256
    assert all(int(value['step']) == 1 for value in runtime.optimizer.state.values())
    assert [hashlib.sha256(batch['imgs'].numpy().tobytes()).hexdigest() for batch, _ in batches] == images
    difference = max(abs(a - b) for a, b in zip(*result['attempt_losses']))
    assert difference <= 1e-5, difference
    assert run.sources() == source and run.core.C.file_sha(__file__) == gate_sha
    run.core.C.dump(out / 'complete.json', dict(complete=True, source_hashes=source,
        gate_source_sha256=gate_sha, injected_nonfinite_gradient_tensors=1, retries=1,
        optimizer_step_calls=1, actual_updates=1, loss_forward_max_abs_difference=difference,
        same_augmented_batches=True, same_parameters_before_retry=True, same_buffers_after_forward=True,
        final_AMP_scale=runtime.scaler.get_scale(), accumulation=8,
        training_image_ids=[ident for _, ident in batches],
        scope='One injected overflow: AdamW skip then same-batch/RNG/buffer retry succeeds; not every possible numerical failure'))
    print('COMPACT AMP RETRY GATE PASS', flush=True)


if __name__ == '__main__':
    main()
