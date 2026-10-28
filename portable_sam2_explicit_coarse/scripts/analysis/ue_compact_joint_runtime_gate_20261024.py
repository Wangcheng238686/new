"""Actual compact-runner inference, depth isolation and epoch-resume gates."""
import argparse
import copy
import gc
import hashlib
from pathlib import Path
import sys

WORK = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(WORK), str(WORK / 'scripts/probes')]
import ue_compact_joint_run_20261024 as run
import numpy as np
import torch
from torch.utils.data import DataLoader


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def same_rng(a, b):
    assert a['python'] == b['python']
    assert a['numpy'][0] == b['numpy'][0] and np.array_equal(a['numpy'][1], b['numpy'][1])
    assert a['numpy'][2:] == b['numpy'][2:]
    assert torch.equal(a['torch'], b['torch'])
    assert torch.equal(a['loader_generator'], b['loader_generator'])
    assert all(torch.equal(x, y) for x, y in zip(a['cuda'], b['cuda']))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=run.ROOT)
    args = p.parse_args()
    out = args.root / 'runtime_gate'
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'complete.json').exists()
    source = run.sources()
    gate_source = run.core.C.file_sha(__file__)
    forbid = [False]
    attempts = []
    def audit(event, values):
        if event == 'open' and forbid[0] and values and isinstance(values[0], (str, bytes)):
            name = str(values[0])
            if name.endswith('.exr') or any(value in name for value in (
                    '/ue_v1_native_geometry/', '/ue_v1_s2db_geometry_v2/', '/ue_v1_local_adapter_geometry/')):
                attempts.append(name)
                raise AssertionError('Reference-plane path read measured geometry: ' + name)
    sys.addaudithook(audit)
    cases = []
    for arm in run.PROTOCOL['arms']:
        runtime = run.Runtime(args.root, arm)
        forbid[0] = arm in ('satellite', 'reference_plane')
        metrics = run.evaluate(runtime, 'val', out / arm, limit=1)
        forbid[0] = False
        detail = dict(arm=arm, actual_runner_FP32_inference=True, evaluated_images=1,
                      no_measured_file_reads=arm in ('satellite', 'reference_plane'))
        if arm == 'rgbd':
            ds = run.core.C.MinimalSatelliteDataset(run.core.previous.DATA, 'val')
            ident = ds.samples[0]['image_id']
            batch = run.core.C.rtmdet_collate_fn([ds[0]])
            sample = run.core.C.processed(runtime.satellite, batch)
            with torch.inference_mode():
                runtime.joint.prepare_uav(**runtime.payload(ident))
                features = runtime.satellite.extract_feat(sample['inputs'])
                head = runtime.satellite.roi_head
                xx = head._inject_extra_pe(features[0])
                line = torch.arange(250, device='cuda')
                x, y = (line % 16) * 30., (line // 16) * 30.
                boxes = torch.stack((x, y, x + 32, y + 32), 1)
                scores = torch.linspace(.9, .1, 250, device='cuda')
                masks = run.core.consumer.masks(head, xx, features,
                    sample['data_samples'][0].metainfo, boxes, scores)
                assert len(masks['rles']) == 250
                runtime.clear()
                del features, xx
            detail['forced_FP32_masks'] = 250
            detail['peak_reserved_MiB'] = torch.cuda.max_memory_reserved() / 2**20
        cases.append(detail)
        print('RUNTIME INFERENCE GATE', detail, flush=True)
        del runtime
        gc.collect()
        torch.cuda.empty_cache()
    assert not attempts
    # Reference-plane training: target creation and actual loss must not open
    # any measured geometry, not merely hide its values from the forward.
    runtime = run.Runtime(args.root, 'reference_plane')
    frozen = run.contract(runtime, require_plane_complete=False)
    ds = run.core.C.MinimalSatelliteDataset(run.core.previous.DATA, 'train', augment=True)
    # First 16 train samples are predetermined and now cached; no AP selection.
    subset = torch.utils.data.Subset(ds, list(range(16)))
    runtime.network.train()
    state = copy.deepcopy(runtime.state(0, 0, frozen, dict(score=-1., epoch=None)))
    checkpoint = out / 'resume_before_epoch.pth'
    run.core.previous.atomic_save(state, checkpoint)
    def batches():
        loader = DataLoader(subset, batch_size=1, shuffle=True, num_workers=2,
            generator=runtime.generator, collate_fn=run.core.C.rtmdet_collate_fn)
        trace = []
        for i, batch in enumerate(loader):
            if i >= 8:
                break
            ident = int(Path(batch['img_metas'][0]['filename']).parent.parent.name)
            trace.append(dict(image_id=ident, augmented_input_sha256=tensor_sha(batch['img']))
                         if 'img' in batch else dict(image_id=ident, augmented_input_sha256=tensor_sha(batch['imgs'])))
        return trace
    trace_before = batches()
    # Deliberately perturb all five RNG categories, and some weights, before
    # invoking the SAME restore method used by formal training.
    run.core.C.seed(123)
    runtime.generator.manual_seed(456)
    with torch.no_grad():
        next(runtime.network.parameters()).add_(.01)
    restored = runtime.restore(checkpoint, frozen)
    same_rng(state['rng_state'], run.core.previous.rng_state(runtime.generator))
    assert all(torch.equal(value.cpu(), state['model'][name].cpu())
               for name, value in runtime.network.state_dict().items())
    assert runtime.scaler.state_dict() == state['scaler']
    assert runtime.optimizer.state_dict()['state'] == state['optimizer']['state'] == {}
    trace_after = batches()
    assert trace_before == trace_after and len(trace_after) == 8
    # Exercise a complete accumulation/update cycle using the restored runtime.
    runtime.restore(checkpoint, frozen)
    loader = DataLoader(subset, batch_size=1, shuffle=True, num_workers=2,
        generator=runtime.generator, collate_fn=run.core.C.rtmdet_collate_fn)
    runtime.network.train()
    runtime.optimizer.zero_grad(set_to_none=True)
    forbid[0] = True
    trace = []
    for i, batch in enumerate(loader):
        if i >= 8:
            break
        ident = int(Path(batch['img_metas'][0]['filename']).parent.parent.name)
        sample = run.core.C.processed(runtime.satellite, batch, True)
        with torch.autocast('cuda', dtype=torch.float16):
            loss = runtime.objective(sample, ident)
        assert torch.isfinite(loss)
        runtime.scaler.scale(loss / 8).backward()
        runtime.clear()
        trace.append(ident)
    runtime.update(1)
    forbid[0] = False
    assert len(trace) == 8 and trace == [row['image_id'] for row in trace_after]
    assert not attempts
    assert runtime.optimizer.state_dict()['state']
    assert all(int(value['step']) == 1 for value in runtime.optimizer.state_dict()['state'].values())
    assert run.sources() == source and run.core.C.file_sha(__file__) == gate_source
    result = dict(complete=True, source_hashes=source, gate_source_sha256=gate_source,
        inference_cases=cases, reference_plane_measured_file_reads=attempts,
        restore_all_five_rng_exact=True, restored_model_exact=True, loader_and_augmentation_trace=trace_after,
        actual_reference_plane_accumulation=8, actual_optimizer_updates=1,
        optimizer_state_scope='This restore starts before E1 with empty optimizer; nonempty-moment replay remains required',
        scope='4 arms x firstval inference, forced250 RGBD masks, pre-E1 resume and no-depth train cycle; not full training/AP')
    run.core.C.dump(out / 'complete.json', result)
    print('COMPACT RUNTIME GATE PASS', flush=True)


if __name__ == '__main__':
    main()
