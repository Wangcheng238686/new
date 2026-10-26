"""Matched query/stride4 geometry interventions; existing live sources unchanged."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time

WORK = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(WORK), str(WORK/'scripts/probes')]
import ue_compact_joint_run_20261024 as base
from data.ue_candidate_evidence import candidate_geometry, load_candidate_geometry
from rsprompter.uav.candidate_detail_interaction import CandidateDetailInteraction, install
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT = Path('/data/wangcheng/checkpoint/ue_candidate_detail_20261026')
CONFIG = WORK/'configs/ablations/ue_candidate_detail_protocol.json'
PROTOCOL = json.loads(CONFIG.read_text())
NEW = ['data/ue_candidate_evidence.py', 'rsprompter/uav/candidate_detail_interaction.py',
       'configs/ablations/ue_candidate_detail_protocol.json', 'scripts/probes/ue_candidate_detail_run_20261026.py']


def hashes():
    return {**base.sources(), **{n:base.core.C.file_sha(WORK/n) for n in NEW}}


def cpu_gate(root):
    adapter = CandidateDetailInteraction()
    # configure only needs the real hook lattice; CPU fake trunk cannot be
    # mistaken for a numerical model gate (actual architecture tested on GPU).
    class Trunk(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.stage_ends = [1, 4, 20, 23]
            self.blocks = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])
            self.blocks[1].dim_out = 112
    fake = torch.nn.Module()
    fake.backbone = Trunk()
    adapter.configure('both', fake)
    gen = torch.Generator().manual_seed(2060)
    selected = torch.randn(16384, 8, 32, generator=gen)
    satellite = torch.randn(16384, 112, generator=gen)
    valid = torch.rand(16384, 8, generator=gen) > .3
    valid[0] = False
    geometry = torch.randn(16384, 8, 7, generator=gen)
    expected = (selected*valid[..., None]).sum(1)/valid.sum(1, keepdim=True).clamp_min(1)
    with torch.no_grad():
        actual = adapter.reduce_rgb(satellite, selected, valid, geometry)
        delta = adapter.geometry_delta(satellite, geometry, valid)
    error = float((actual-expected).abs().max())
    assert error < 1e-6 and delta.count_nonzero() == 0
    with torch.no_grad():
        adapter.detail_output.weight.normal_(std=.02)
        adapter.detail_output.bias.fill_(.1)
    assert adapter.geometry_delta(satellite, geometry, valid)[0].count_nonzero() == 0
    with torch.no_grad():
        adapter.read_key.weight.normal_(std=.1)
        adapter.read_geometry_key.weight.normal_(std=.1)
    changed = adapter.reduce_rgb(satellite, selected, valid, geometry)
    assert (changed-expected).abs().max() > .01
    perm = torch.arange(7, -1, -1)
    permuted = adapter.reduce_rgb(satellite, selected[:, perm], valid[:, perm], geometry[:, perm])
    assert torch.allclose(changed, permuted, atol=2e-6, rtol=1e-5)
    d = adapter.geometry_delta(satellite, geometry, valid)
    d.square().mean().backward()
    assert adapter.detail_value[0].weight.grad.abs().sum() > 0
    base.core.C.dump(root/'cpu_gate.json', dict(passed=True, source_hashes=hashes(),
        uniform_initial_max_error=error, zero_detail_identity=True, no_anchor_zero=True,
        learned_candidate_sensitivity=True, joint_candidate_permutation_invariant=True,
        geometry_value_gradient=True, scope='tensor mechanisms; real model/resource gates separate'))


def runtime(variant, content):
    arm = {'correct':'rgbd', 'constant':'constant_rgb', 'plane':'reference_plane'}[content]
    rt = base.Runtime(base.ROOT, arm)
    install(rt.joint.uav, variant, rt.satellite)
    original = rt.payload
    def payload(ident):
        value = original(ident)
        rt.joint.uav.candidate_geometry = load_candidate_geometry(base.core.previous.DATA,
            ident, value['index'], value['anchor'])
        return value
    rt.payload = payload
    rt.optimizer = torch.optim.AdamW([p for p in rt.network.parameters() if p.requires_grad],
        lr=base.PROTOCOL['lr'], weight_decay=base.PROTOCOL['weight_decay'])
    base.core.C.seed(base.PROTOCOL['seed'])
    return rt


def contract(rt, args):
    return dict(variant=args.variant, content=args.content, protocol=PROTOCOL,
        original_protocol=base.PROTOCOL, source_hashes=hashes(),
        initialization=json.loads((base.ROOT/'initialization.json').read_text()),
        data_identity=base.core.C.archive_identity(base.core.previous.DATA),
        measured_cache_manifest_sha256=base.core.C.file_sha(base.core.OUT/'input_cache_sha256.json'),
        plane_build_sha256=base.core.C.file_sha(base.PLANE/'BUILD_COMPLETE.json'))


def gate(rt, args):
    ds = base.core.C.MinimalSatelliteDataset(base.core.previous.DATA, 'train')
    idx = max(range(len(ds)), key=lambda i:len(ds.samples[i]['annotations']))
    ident = ds.samples[idx]['image_id']
    batch = base.core.C.rtmdet_collate_fn([ds[idx]])
    # Same actual model and weights, before/after enabling the new mechanisms.
    # The reference call explicitly uses the old encoder path; no extra model.
    from rsprompter.uav.compact_joint import CompactInteraction
    from torch.utils._pytree import tree_flatten
    rt.network.eval()
    with torch.inference_mode():
        sample = base.core.C.processed(rt.satellite, batch)
        payload = rt.payload(ident)
        with np.load(Path('/data/wangcheng/cache/ue_v1_s2db_geometry_v2')/f'{int(ident):06d}.npz') as z:
            old = z['geom'].astype(np.float32)
        descriptor = rt.joint.uav.candidate_geometry.cpu().numpy()
        valid = payload['index'].cpu().numpy() >= 0
        original_six = descriptor[..., :6].copy()
        original_six[..., :2] -= .5
        geometry_error = float(np.abs(original_six[valid]-old[valid]).max())
        assert geometry_error < .002, geometry_error
        rt.joint.uav.rgb_context = (payload['rgb'], payload['index'], payload['anchor'])
        CompactInteraction.encode(rt.joint.uav, payload['geometry'])
        reference, _ = tree_flatten(rt.satellite.extract_feat(sample['inputs']))
        reference = [x.detach().cpu().clone() for x in reference if torch.is_tensor(x)]
        reference_head = [x.detach().cpu().clone() for x in rt.joint.uav.outputs]
        rt.clear()
        payload = rt.payload(ident)
        rt.joint.prepare_uav(**payload)
        current, _ = tree_flatten(rt.satellite.extract_feat(sample['inputs']))
        current = [x.detach().cpu() for x in current if torch.is_tensor(x)]
        identity_error = max(float((a-b).abs().max()) for a,b in zip(reference, current))
        head_error = max(float((a-b.cpu()).abs().max()) for a,b in zip(reference_head, rt.joint.uav.outputs))
        assert identity_error < 1e-5 and head_error < 1e-5, (identity_error, head_error)
        rt.clear()
    before = {n:p.detach().cpu().clone() for n,p in rt.network.named_parameters() if not p.requires_grad}
    new_names = ['read_key', 'read_geometry_key'] if args.variant == 'query' else ['detail_output', 'detail_value']
    if args.variant == 'both':
        new_names += ['read_key', 'read_geometry_key']
    changing = {n:p.detach().cpu().clone() for n,p in rt.network.named_parameters()
                if any(k in n for k in new_names) or ('rgb_encoder' in n and 'lora_B' in n)}
    rt.network.train()
    rt.set_epoch(1)
    reports = []
    for step in range(1, 4):
        reports.append(base.accumulation(rt, [(batch, ident)]*8, step))
    params = dict(rt.network.named_parameters())
    assert all(torch.equal(v, params[n].detach().cpu()) for n,v in before.items())
    changed = {n:not torch.equal(v, params[n].detach().cpu()) for n,v in changing.items()}
    for key in new_names+['lora_B']:
        assert any(v for n,v in changed.items() if key in n), key
    rt.network.eval()
    with torch.inference_mode():
        sample = base.core.C.processed(rt.satellite, batch)
        payload = rt.payload(ident)
        rt.joint.prepare_uav(**payload)
        features = rt.satellite.extract_feat(sample['inputs'])
        prefix, combined, accepted = base.core.inference_outputs(rt.satellite, rt.joint.uav,
            sample, features, payload['geometry'])
        assert len(combined['scores']) <= 250 and np.isfinite(combined['scores']).all()
        rt.clear()
        # Exercise the full decoder's maximum permitted RoI count.
        force = torch.tensor([[16.,16.,40.,40.]], device='cuda').repeat(250, 1)
        prediction = base.core.consumer.masks(rt.satellite.roi_head,
            rt.satellite.roi_head._inject_extra_pe(features[0]), features,
            sample['data_samples'][0].metainfo, force, torch.full((250,), .1, device='cuda'))
        assert len(prediction['scores']) == 250
    # Actual nonempty optimizer/scaler and all five RNG restoration.
    frozen = contract(rt, args)
    state = rt.state(0, 3, frozen, dict(score=0., epoch=0))
    generator_expected = rt.generator.get_state().clone()
    restored_path = args.root/'gates'/args.variant/'restore.pth'
    restored_path.parent.mkdir(parents=True, exist_ok=True)
    base.core.previous.atomic_save(state, restored_path)
    torch.rand(3, generator=rt.generator)
    rt.restore(restored_path, frozen, training=True)
    assert torch.equal(rt.generator.get_state(), generator_expected)
    assert rt.optimizer.state
    restored_path.unlink()
    base.core.C.dump(args.root/'gates'/args.variant/'gpu_gate.json', dict(passed=True,
        source_hashes=hashes(), image_id=ident, successful_updates=3, reports=reports,
        changed_parameters=changed, frozen_tensors_exact=len(before), formal_FP32=True,
        max_decoder_instances=250, nonempty_optimizer_restore=True, loader_rng_restored=True,
        initial_features_max_error=identity_error, initial_head_max_error=head_error,
        candidate_geometry_vs_legacy_six_max_error=geometry_error,
        peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20,
        peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20))


def train(rt, args):
    out = args.root/args.variant/args.content
    frozen = contract(rt, args)
    g = json.loads((args.root/'gates'/args.variant/'gpu_gate.json').read_text())
    assert g['passed'] and g['source_hashes'] == hashes()
    if not args.resume:
        assert not (out/'last.pth').exists(), 'Explicit resume required'
    base.core.C.dump(out/'contract.json', frozen)
    epoch = step = 0
    if args.resume:
        cp = rt.restore(args.resume, frozen)
        epoch, step = cp['epoch'], cp['step']
        assert step == epoch*267
    ds = base.core.C.MinimalSatelliteDataset(base.core.previous.DATA, 'train', augment=True)
    for epoch in range(epoch+1, 11):
        rt.network.train()
        rt.set_epoch(epoch)
        loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=2,
            generator=rt.generator, collate_fn=base.core.C.rtmdet_collate_fn)
        order, prepared, total, retries = [], [], 0., 0
        began = time.time()
        for microbatch,batch in enumerate(loader):
            if microbatch >= 2136:
                break
            ident = int(Path(batch['img_metas'][0]['filename']).parent.parent.name)
            order.append(ident)
            prepared.append((batch, ident))
            if len(prepared) == 8:
                row = base.accumulation(rt, prepared, step+1)
                step += 1
                prepared = []
                total += row['loss_sum']
                retries += row['retries']
                if step % 25 == 0:
                    base.core.C.dump(out/'progress.json', dict(epoch=epoch, update=step,
                        last_loss=row['last_loss'], preclip_norm=row['preclip_norm'],
                        AMP_retries_epoch=retries, pid=os.getpid(), variant=args.variant, content=args.content))
                    print('UPDATE', args.variant, args.content, epoch, step, flush=True)
        assert not prepared and len(order) == len(set(order)) == 2136 and step == epoch*267
        reference = json.loads((base.ROOT/rt.arm/f'epoch{epoch}.json').read_text())
        assert order == reference['sample_order']
        metrics = base.evaluate(rt, 'val', out/'validation_current')
        state = rt.state(epoch, step, frozen, dict(score=metrics['segm/mAP'], epoch=epoch))
        base.core.previous.atomic_save(state, out/'last.pth')
        if epoch in (1, 3, 10):
            shutil.copyfile(out/'last.pth', out/f'epoch{epoch}.pth')
        base.core.C.dump(out/f'epoch{epoch}.json', dict(epoch=epoch, step=step, sample_order=order,
            original_prefix_order_exact=True, validation=metrics, mean_microbatch_loss=total/2136,
            AMP_retries_epoch=retries, seconds=time.time()-began,
            peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20))
        assert hashes() == frozen['source_hashes']
    base.core.C.dump(out/'training_complete.json', dict(complete=True, epoch=10, updates=2670,
        checkpoint_sha256=base.core.C.file_sha(out/'epoch10.pth'), contract=frozen))


def evaluate(rt, args):
    folder = args.root/args.variant/args.content
    done = json.loads((folder/'training_complete.json').read_text())
    frozen = contract(rt, args)
    assert done['contract'] == frozen and done['updates'] == 2670
    path = folder/'epoch10.pth'
    assert base.core.C.file_sha(path) == done['checkpoint_sha256']
    if args.split == 'test':
        assert all((args.root/v/c/'training_complete.json').exists() for v,c in PROTOCOL['training_jobs'])
    rt.restore(path, frozen, training=False)
    state_before = base.core.C.state_digest(rt.network)
    donor_map = {}
    if args.mode != 'matched':
        assert args.content == 'correct'
        original = rt.payload
        ds = base.core.C.MinimalSatelliteDataset(base.core.previous.DATA, args.split)
        ids = [s['image_id'] for s in ds.samples]
        def intervened(ident):
            value = original(ident)
            if args.mode == 'constant':
                value['rgb'][:] = rt.joint.uav.rgb_mean
            else:
                donor = ids[(ids.index(ident)+len(ids)//2)%len(ids)]
                donor_map[str(ident)] = donor
                value['rgb'] = rt.uav_inputs.load(donor, geometry_mode='measured', rgb_mode='correct')['rgb']
            return value
        rt.payload = intervened
    out = args.root/'evaluation'/args.split/args.variant/args.content/args.mode
    assert not (out/'complete.json').exists()
    base.evaluate(rt, args.split, out, checkpoint_sha256=done['checkpoint_sha256'])
    assert base.core.C.state_digest(rt.network) == state_before
    identity = json.loads((out/'complete.json').read_text())
    identity.update(variant=args.variant, content=args.content, mode=args.mode,
        source_hashes=hashes(), target_geometry_unchanged=True,
        model_state_unchanged=True, donor_map=donor_map,
        donor_physical_consistency=False if args.mode == 'donor' else None)
    base.core.C.dump(out/'complete.json', identity)


def reference(args):
    arm = {'correct':'rgbd', 'constant':'constant_rgb', 'plane':'reference_plane',
           'satellite':'satellite'}[args.content]
    if args.split == 'test':
        assert all((args.root/v/c/'training_complete.json').exists() for v,c in PROTOCOL['training_jobs'])
    rt = base.Runtime(base.ROOT, arm)
    path = base.ROOT/arm/'epoch10.pth'
    cp = rt.restore(path, base.contract(rt), training=False)
    assert cp['epoch'] == 10 and cp['step'] == 2670
    out = args.root/'reference'/args.split/args.content
    assert not (out/'complete.json').exists()
    base.evaluate(rt, args.split, out, checkpoint_sha256=base.core.C.file_sha(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['cpu_gate', 'gate', 'train', 'eval', 'reference'])
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--variant', choices=PROTOCOL['variants'], default='both')
    parser.add_argument('--content', choices=['correct', 'constant', 'plane', 'satellite'], default='correct')
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--mode', choices=['matched', 'constant', 'donor'], default='matched')
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    if args.action == 'cpu_gate':
        cpu_gate(args.root)
        return
    torch.cuda.set_per_process_memory_fraction(PROTOCOL['allocator_cap_MiB']*2**20/
        torch.cuda.get_device_properties(0).total_memory)
    if args.action == 'reference':
        reference(args)
        return
    assert args.content != 'satellite'
    rt = runtime(args.variant, args.content)
    if args.action == 'gate':
        gate(rt, args)
    elif args.action == 'train':
        train(rt, args)
    else:
        evaluate(rt, args)


if __name__ == '__main__':
    main()
