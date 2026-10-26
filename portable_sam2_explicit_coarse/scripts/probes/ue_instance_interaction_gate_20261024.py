"""Analytical lattice, actual wiring, and real checkpoint/loader replay gates."""
import argparse
import copy
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import ue_instance_interaction_20261024 as run


def cpu_gate(root):
    y, x = torch.meshgrid(torch.arange(128.), torch.arange(128.), indexing='ij')
    coordinate = torch.stack((4 * x, 4 * y))[None]
    pooled = F.avg_pool2d(coordinate, 2, 2)
    sy, sx = torch.meshgrid(torch.arange(64.), torch.arange(64.), indexing='ij')
    assert torch.equal(pooled[0], torch.stack((8 * sx + 2, 8 * sy + 2)))
    restored = F.interpolate(pooled, size=(128, 128), mode='bilinear', align_corners=False)
    assert torch.equal(restored[:, :, 1:-1, 1:-1], coordinate[:, :, 1:-1, 1:-1])
    assert restored[0, 0, 0, 0] == 2 and restored[0, 0, -1, -1] == 506
    # 3x3 unfold must not wrap the right edge to the next row.
    mask = torch.zeros(1, 1, 64, 64)
    mask[0, 0, 0, -1] = 1
    neighbors = F.unfold(mask, 3, padding=1)[0].T > 0
    assert neighbors[63].sum() == 1 and neighbors[64].sum() == 0
    run.C.dump(root / 'cpu_gate.json', dict(passed=True, target_lattice='U128=4i; pooled64=8j+2',
        interior_ramp_exact=True, edge_extension_centers=[2, 506], no_row_wrap=True,
        geometric_claim='nominal feature lattices and support footprints, not exact 3D visibility'))


def gpu_gate(args):
    model, adapter, _, initial = run.build(args)
    ds = run.C.MinimalSatelliteDataset(run.previous.DATA, 'train')
    i = 0
    sample = run.C.processed(model, run.C.rtmdet_collate_fn([ds[i]]), True)
    x = run.load_geometry(ds.samples[i]['image_id'])
    run.bind_rgb(args, adapter, ds, i)
    targets = run.pilot.Data('train')
    target = {k: v[None].cuda() for k, v in targets[i].items() if k in ('heat', 'reg', 'valid')}
    trunk = next(m for m in model.backbone.modules() if hasattr(m, 'stage_ends') and hasattr(m, 'blocks'))
    received = []
    handle = trunk.blocks[5].register_forward_pre_hook(lambda m, a: received.append(a[0].detach().clone()))
    adapter.eval()
    with torch.no_grad():
        adapter.output.weight.normal_(std=.01)
        adapter.output.bias.fill_(.02)
        adapter.internal = False
        head = run.forward(model, adapter, sample, x)
        center_head = [v.detach().clone() for v in adapter.outputs]
        adapter.internal = True
        internal = run.forward(model, adapter, sample, x)
        assert all(torch.equal(a, b) for a, b in zip(center_head, adapter.outputs))
        assert not torch.equal(received[0], received[1])
        assert torch.equal(head[3][0], internal[3][0])
        assert not torch.equal(head[3][1], internal[3][1])
        assert not torch.equal(head[1], internal[1])
        assert any(not torch.equal(a, b) for a, b in zip(head[0], internal[0]))
    handle.remove()
    adapter.clear()
    del head, internal, received
    initial_cp = torch.load(run.init_path(args), map_location='cpu', weights_only=False)
    adapter.load_state_dict(initial_cp['adapter'], strict=True)
    adapter.internal = args.route == 'internal'
    with torch.inference_mode():
        adapter.enabled = False
        features = model.extract_feat(sample['inputs'])
        original_prefix, original_combined, _ = run.inference_outputs(model, adapter, sample, features, x)
        adapter.enabled = True
        features = run.forward(model, adapter, sample, x)
        prefix, _, _ = run.inference_outputs(model, adapter, sample, features, x)
        assert run.C.serialized(prefix) == run.C.serialized(original_prefix)
        adapter.output.weight.fill_(.01)
        adapter.output.bias.fill_(.02)
        empty = torch.zeros_like(x)
        features = run.forward(model, adapter, sample, empty)
        prefix, combined, accepted = run.inference_outputs(model, adapter, sample, features, empty)
        assert accepted == 0
        assert run.C.serialized(prefix) == run.C.serialized(original_prefix)
        assert run.C.serialized(combined) == run.C.serialized(original_combined)
    adapter.clear()
    adapter.load_state_dict(initial_cp['adapter'], strict=True)
    adapter.train()
    optimizer = torch.optim.AdamW([p for p in adapter.parameters() if p.requires_grad], lr=1e-4, weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda', init_scale=512)
    gen = torch.Generator().manual_seed(args.seed)

    def update():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.float16):
            loss, _ = run.losses(model, adapter, sample, x, target)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), .5)
        assert torch.isfinite(norm)
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        assert scaler.get_scale() >= old_scale
        adapter.clear()
        return float(loss.detach())

    run.C.seed(1044001)
    update()
    saved = dict(adapter=copy.deepcopy(adapter.state_dict()), optimizer=copy.deepcopy(optimizer.state_dict()),
                 scaler=copy.deepcopy(scaler.state_dict()), rng_state=run.previous.rng_state(gen))
    path = args.out / 'resume_gate_snapshot.pth'
    run.previous.atomic_save(saved, path)
    expected_loss = update()
    expected = {k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()}
    restored = torch.load(path, map_location='cpu', weights_only=False)

    def replay(snapshot):
        adapter.load_state_dict(snapshot['adapter'], strict=True)
        # Adam state tensors can otherwise alias the loaded snapshot and
        # mutate it, invalidating the second replay used for calibration.
        optimizer.load_state_dict(copy.deepcopy(snapshot['optimizer']))
        scaler.load_state_dict(copy.deepcopy(snapshot['scaler']))
        run.previous.restore_rng(snapshot['rng_state'], gen)
        value = update()
        return value, {k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()}

    # CUDA RoIAlign/backward can be nondeterministic even with all RNG
    # restored. Establish a same-code, in-memory replay envelope separately
    # from disk restoration; never call this bit-exact weight continuation.
    native_replays = [replay(saved) for _ in range(3)]
    replay_loss, replay_state = replay(restored)
    maximum = max(float((expected[k] - replay_state[k]).abs().max()) for k in expected)
    calibration = max(float((expected[k] - state[k]).abs().max())
                      for _, state in native_replays for k in expected)
    loss_calibration = max(abs(value - expected_loss) for value, _ in native_replays)
    assert maximum <= max(1e-7, calibration * 2), (maximum, calibration)
    assert abs(expected_loss - replay_loss) <= max(1e-6, 2 * loss_calibration)
    # This runner resumes only at epoch boundaries. Recreate the actual
    # next-epoch two-worker loader, including worker augmentation RNG.
    augmented = run.C.MinimalSatelliteDataset(run.previous.DATA, 'train', augment=True)
    def loader_prefix():
        loader = DataLoader(augmented, batch_size=1, shuffle=True, num_workers=2,
                            collate_fn=run.C.rtmdet_collate_fn, generator=gen)
        result = []
        for j, batch in enumerate(loader):
            result.append(dict(file=batch['img_metas'][0]['filename'],
                input_sha256=hashlib.sha256(batch['imgs'].numpy().tobytes()).hexdigest()))
            if j == 3:
                break
        return result
    before = run.previous.rng_state(gen)
    order = loader_prefix()
    run.previous.restore_rng(before, gen)
    assert order == loader_prefix(), 'resumed loader order or augmentation changed'
    assert run.C.state_digest(model) == run.previous.FROZEN_SHA
    run.C.dump(args.out / 'extended_gate.json', dict(passed=True, initial_fingerprint=initial,
        head_internal_center_logits_exact=True, block5_receives_changed_tensor=True,
        formal_inference_zero_write_prefix_exact=True, formal_inference_no_support_combined_exact=True,
        stride4_unchanged_later_consumers_changed=True, resumed_actual_optimizer_update=True,
        resume_parameter_max_abs_error=maximum, replay_noise_calibration=calibration,
        replay_noise_calibration_repeats=3, continuation_bit_exact=False,
        resume_loss_error=abs(expected_loss - replay_loss), five_RNG_restored=True,
        epoch_boundary_loader_order_and_augmentation_exact=True, loader_prefix=order,
        source_hashes=run.source_hashes(), peak_reserved_MiB=torch.cuda.max_memory_reserved() / 2**20))
    print('EXTENDED GATE PASS', args.family, args.route, maximum, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=run.OUT)
    p.add_argument('--out', type=Path)
    p.add_argument('--seed', type=int, default=44)
    p.add_argument('--route', choices=['head', 'internal'], default='internal')
    p.add_argument('--family', choices=['geom', 'rgb'], default='geom')
    p.add_argument('--rgb', default='correct')
    p.add_argument('--cpu', action='store_true')
    a = p.parse_args()
    a.out = a.out or a.root / f'gate_{a.family}_{a.route}'
    a.out.mkdir(parents=True, exist_ok=True)
    if a.cpu:
        cpu_gate(a.root)
    else:
        gpu_gate(a)


if __name__ == '__main__':
    main()
