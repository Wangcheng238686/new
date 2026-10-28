"""Fresh-foundation joint forward/backward gate; not a training experiment."""
import argparse
import gc
from pathlib import Path
import sys
from unittest.mock import patch

WORK = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(WORK), str(WORK / 'scripts/probes')]
import ue_instance_interaction_20261024 as probe
from rsprompter.uav.compact_joint import CompactJoint, build_foundation_satellite
from data.ue_joint_inputs import JointUAVInputs
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    assert not (args.out / 'complete.json').exists()
    foundation = Path('/data/wangcheng/pretrained-models/sam2/sam2_hiera_base_plus.pt')
    foundation_sha = probe.C.file_sha(foundation)
    frozen_sources = probe.source_hashes()
    sources = {name: probe.C.file_sha(WORK / name) for name in (
        'rsprompter/uav/compact_joint.py', 'rsprompter/uav/rgb_p0_encoder.py',
        'data/ue_joint_inputs.py', 'scripts/analysis/ue_compact_joint_foundation_gate_20261024.py',
        'configs/ablations/ue_minimal_single.py', 'configs/ablations/ue_rgbd_single.py',
        'configs/a3sgm_whu1024.py')}
    loads = []
    actual_load = torch.load
    def foundation_only(file, *a, **kw):
        name = file if isinstance(file, (str, Path)) else getattr(file, 'name', None)
        assert name is not None and Path(name).resolve() == foundation.resolve(), name
        loads.append(str(Path(name).resolve()))
        return actual_load(file, *a, **kw)
    probe.C.seed(44)
    with patch('torch.load', side_effect=foundation_only):
        satellite, config = build_foundation_satellite(probe.C._register_and_build,
            WORK / 'configs/ablations/ue_minimal_single.py', foundation)
        common_sha = probe.C.state_digest(satellite)
        # Same fresh satellite state can be used by matched single/dual arms.
        probe.previous.atomic_save(dict(model=satellite.state_dict(), model_config=config,
            seed=44, foundation_sha256=foundation_sha, role='fresh common initialization, no task weights'),
            args.out / 'common_satellite_init.pth')
        joint = CompactJoint(satellite, foundation, source_seed=2045)
    assert loads and all(Path(path) == foundation.resolve() for path in loads)
    assert probe.C.state_digest(joint.satellite) == common_sha
    assert common_sha != probe.previous.FROZEN_SHA, 'Unexpected E73 identity'
    joint.cuda().train()
    names = {name: p.numel() for name, p in joint.named_parameters() if p.requires_grad}
    assert any(name.startswith('satellite.backbone.') and 'lora_' in name for name in names)
    assert any(name.startswith('satellite.roi_head.mask_head.') for name in names)
    assert any(name.startswith('uav.rgb_encoder.') and 'lora_' in name for name in names)
    assert not any(name.startswith('uav.rgb_encoder.') and 'lora_' not in name for name in names)
    frozen = {name: value.detach().cpu().clone() for name, value in joint.named_parameters() if not value.requires_grad}
    ds = probe.C.MinimalSatelliteDataset(probe.previous.DATA, 'train', augment=True)
    targets = probe.pilot.Data('train')
    index = max(range(len(targets)), key=lambda i: len(targets.boxes[i]))
    ident = targets.ids[index]
    ds_index = [row['image_id'] for row in ds.samples].index(ident)
    inputs = JointUAVInputs(probe.previous.DATA, '/data/wangcheng/cache/ue_v1_native_geometry',
        '/data/wangcheng/cache/ue_v1_s2db_geometry_v2', '/data/wangcheng/cache/ue_v1_reference_plane_20261024')
    payload = inputs.load(ident)
    provenance = payload.pop('provenance')
    assert provenance['measured_depth'] and provenance['geometry_mode'] == 'measured'
    target = {key: value[None].cuda() for key, value in targets[index].items() if key in ('heat', 'reg', 'valid')}
    optimizer = torch.optim.AdamW([p for p in joint.parameters() if p.requires_grad], lr=5e-4, weight_decay=.05)
    scaler = torch.amp.GradScaler('cuda', init_scale=512)
    steps = []
    torch.cuda.reset_peak_memory_stats()
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        sample = probe.C.processed(joint.satellite, probe.C.rtmdet_collate_fn([ds[ds_index]]), True)
        with torch.autocast('cuda', dtype=torch.float16):
            losses = joint.loss(sample, **payload, instance_target=target, center_loss_fn=probe.pilot.loss_fn)
            loss = probe.previous.objective(joint.uav, losses)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        groups = {key: 0. for key in ('satellite_lora', 'decoder', 'uav_lora', 'geometry', 'writeback')}
        for name, parameter in joint.named_parameters():
            if parameter.grad is None:
                continue
            assert torch.isfinite(parameter.grad).all(), name
            norm = float(parameter.grad.float().norm())
            if name.startswith('satellite.backbone.') and 'lora_' in name:
                groups['satellite_lora'] += norm
            if name.startswith('satellite.roi_head.mask_head.mask_decoder.'):
                groups['decoder'] += norm
            if name.startswith('uav.rgb_encoder.'):
                groups['uav_lora'] += norm
            if name.startswith(tuple('uav.' + key + '.' for key in ('a', 'b', 'c', 'd', 'up'))):
                groups['geometry'] += norm
            if name.startswith('uav.output.'):
                groups['writeback'] += norm
        clip = torch.nn.utils.clip_grad_norm_(joint.parameters(), .5)
        assert torch.isfinite(clip)
        before_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        assert scaler.get_scale() >= before_scale, 'AMP skipped a real update'
        steps.append(dict(step=step + 1, loss=float(loss.detach()), group_parameter_norm_sums=groups,
                          preclip_norm=float(clip), AMP_scale=scaler.get_scale()))
        print('JOINT GATE', steps[-1], flush=True)
        joint.clear()
    assert all(value > 0 for value in steps[-1]['group_parameter_norm_sums'].values())
    assert all(torch.equal(parameter.detach().cpu(), frozen[name]) for name, parameter in joint.named_parameters() if name in frozen)
    assert probe.C.file_sha(foundation) == foundation_sha and probe.source_hashes() == frozen_sources
    assert all(probe.C.file_sha(WORK / name) == sh for name, sh in sources.items())
    result = dict(complete=True, task_checkpoint_loaded=False, actual_torch_load_paths=loads,
        foundation_sha256=foundation_sha, common_satellite_initial_sha=common_sha,
        satellite_shared_initial_state_exact=True, source_hashes=sources,
        trainable_parameters=names, dense_train_image_id=ident, train_instances=len(targets.boxes[index]),
        input_provenance=provenance, steps=steps, frozen_parameters_exact=True,
        peak_reserved_MiB=torch.cuda.max_memory_reserved() / 2**20,
        scope='3 real AMP joint updates at max-GT train sample; no val/test training, no full run/AP/resume claim')
    probe.C.dump(args.out / 'complete.json', result)
    print('COMPACT JOINT FOUNDATION GATE PASS', flush=True)


if __name__ == '__main__':
    main()
