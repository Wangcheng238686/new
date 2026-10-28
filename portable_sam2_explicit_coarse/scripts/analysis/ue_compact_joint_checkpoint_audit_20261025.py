"""CPU-only checkpoint/optimizer/freeze audit for the compact joint study.

Anchor audits are engineering evidence, not final results. --final requires all
100 epochs and checks both val-best and last; paired prediction audit is separate.
"""
import argparse
from collections import Counter
import gc
import json
from pathlib import Path
import sys
from unittest.mock import patch

WORK = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(WORK), str(WORK / 'scripts/probes')]
import ue_compact_joint_run_20261024 as run
import torch

GATE = Path('/data/wangcheng/checkpoint/ue_instance_interaction_20261024/compact_joint_foundation_gate_v2/complete.json')


def read(path):
    return json.loads(Path(path).read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=run.ROOT)
    p.add_argument('--epoch', type=int, choices=[1, 3, 10, 25, 50, 100], default=1)
    p.add_argument('--final', action='store_true')
    p.add_argument('--arm', choices=run.PROTOCOL['arms'], action='append')
    args = p.parse_args()
    arms = args.arm or run.PROTOCOL['arms']
    assert len(set(arms)) == len(arms)
    if args.final:
        assert set(arms) == set(run.PROTOCOL['arms']), 'Final audit requires all arms'
    root = args.root
    identity = read(root / 'initialization.json')
    source_hashes = run.sources()
    assert identity['source_hashes'] == source_hashes and identity['protocol'] == run.PROTOCOL
    assert identity['task_weights_loaded'] is False
    assert run.core.C.file_sha(root / 'common_init.pth') == identity['common_init_sha256']
    assert run.core.C.file_sha(run.FOUNDATION) == identity['foundation_sha256']
    paths = {}
    for arm in arms:
        if args.final:
            done = read(root / arm / 'training_complete.json')
            assert done['complete'] and done['epoch'] == 100 and done['updates'] == 26700
            paths[arm] = [root / arm / 'best_model.pth', root / arm / 'last.pth']
        else:
            paths[arm] = [root / arm / ('epoch%d.pth' % args.epoch)]
        assert all(path.exists() for path in paths[arm]), arm
    suffix = 'final' if args.final else 'epoch%d_%s' % (args.epoch, '_'.join(arms))
    out = root / 'checkpoint_audit' / (suffix + '.json')
    assert not out.exists(), 'Preserve completed audits'
    common = torch.load(root / 'common_init.pth', map_location='cpu', weights_only=False)
    actual_load = torch.load
    loads = []
    def generic_only(path, *a, **kw):
        assert Path(path).resolve() == run.FOUNDATION.resolve(), path
        loads.append(str(Path(path).resolve()))
        return actual_load(path, *a, **kw)
    run.core.C.seed(run.PROTOCOL['seed'])
    with patch('torch.load', side_effect=generic_only):
        satellite, config = run.build_foundation_satellite(run.core.C._register_and_build,
            WORK / 'configs/ablations/ue_minimal_single.py', run.FOUNDATION)
        joint = run.CompactJoint(satellite, run.FOUNDATION, source_seed=2045)
    assert config == common['model_config']
    joint.satellite.load_state_dict(common['satellite'], strict=True)
    joint.uav.load_state_dict(common['uav'], strict=True)
    assert run.core.C.state_digest(satellite) == identity['satellite_initial_sha']
    assert run.core.previous.tensor_digest(joint.uav.state_dict()) == identity['uav_initial_sha']
    gate = read(GATE)
    trainable = {n: t.numel() for n, t in joint.named_parameters() if t.requires_grad}
    assert trainable == gate['trainable_parameters'], 'Trainable scope drifted from engineering gate'
    metadata = {}
    for arm in arms:
        network = satellite if arm == 'satellite' else joint
        metadata[arm] = dict(
            trainable=[n for n, t in network.named_parameters() if t.requires_grad],
            frozen=[n for n, t in network.named_parameters() if not t.requires_grad],
            buffers=[n for n, _ in network.named_buffers() if n in network.state_dict()],
            nonpersistent_buffers=[n for n, _ in network.named_buffers() if n not in network.state_dict()],
            initial={n: t.detach().clone() for n, t in network.state_dict().items()})
    del joint, satellite, common
    gc.collect()
    results, orders = {}, {}
    for arm in arms:
        info = metadata[arm]
        frozen_contract = read(root / arm / 'contract.json')
        assert frozen_contract['initialization'] == identity
        results[arm] = []
        for path in paths[arm]:
            cp = torch.load(path, map_location='cpu', weights_only=False)
            assert cp['contract'] == frozen_contract
            assert cp['checkpoint_phase'] == 'after_validation'
            assert cp['step'] == cp['epoch'] * 267
            if not args.final:
                assert cp['epoch'] == args.epoch
            else:
                done = read(root / arm / 'training_complete.json')
                if path.name == 'last.pth':
                    assert cp['epoch'] == 100 and cp['step'] == 26700
                    expected_sha = done['last_checkpoint_sha256']
                else:
                    assert cp['epoch'] == done['best']['epoch'] and cp['best'] == done['best']
                    expected_sha = done['best_checkpoint_sha256']
                assert run.core.C.file_sha(path) == expected_sha
            state = cp['model']
            assert state.keys() == info['initial'].keys()
            assert all(torch.equal(state[n], info['initial'][n]) for n in info['frozen'])
            assert all(torch.isfinite(t).all() for t in state.values())
            optimizer = cp['optimizer']
            indices = [i for group in optimizer['param_groups'] for i in group['params']]
            assert len(indices) == len(set(indices)) == len(info['trainable'])
            steps, missing = {}, []
            for name, index in zip(info['trainable'], indices):
                if index not in optimizer['state']:
                    missing.append(name)
                    continue
                slot = optimizer['state'][index]
                assert slot['exp_avg'].shape == slot['exp_avg_sq'].shape == state[name].shape
                assert torch.isfinite(slot['exp_avg']).all() and torch.isfinite(slot['exp_avg_sq']).all()
                steps[name] = int(slot['step'])
                assert 0 < steps[name] <= cp['step']
            canonical = ('satellite.' if arm != 'satellite' else '') + 'rpn_head.rpn_cls.weight'
            assert steps[canonical] == cp['step'], 'Canonical always-used RPN update count differs'
            required_rng = {'python', 'numpy', 'torch', 'cuda', 'loader_generator'}
            assert set(cp['rng_state']) == required_rng
            epoch_row = read(root / arm / ('epoch%d.json' % cp['epoch']))
            assert len(set(epoch_row['sample_order'])) == len(epoch_row['sample_order']) == 2136
            orders[(arm, cp['epoch'])] = epoch_row['sample_order']
            changed = [n for n in info['trainable'] if not torch.equal(state[n], info['initial'][n])]
            changed_buffers = [n for n in info['buffers'] if not torch.equal(state[n], info['initial'][n])]
            # BatchNorm buffers in trainable consumers may change. Frozen source
            # and foundation backbone buffers must remain equal to initialization.
            frozen_buffer_prefixes = ('backbone.',) if arm == 'satellite' else ('satellite.backbone.', 'uav.rgb_encoder.', 'uav.rgb_mean', 'uav.rgb_std')
            assert not any(n.startswith(frozen_buffer_prefixes) for n in changed_buffers)
            results[arm].append(dict(path=str(path), checkpoint_sha256=run.core.C.file_sha(path),
                epoch=cp['epoch'], step=cp['step'], canonical_parameter=canonical,
                optimizer_step_histogram=dict(Counter(steps.values())), optimizer_missing_state=missing,
                frozen_parameter_tensors_exact=len(info['frozen']), trainable_tensors_changed=len(changed),
                changed_buffer_names=changed_buffers,
                nonpersistent_buffers_not_checkpoint_auditable=info['nonpersistent_buffers'],
                AMP_scale=cp['scaler']['scale'],
                RNG_keys=sorted(cp['rng_state']), validation=epoch_row['validation']))
            del cp, state, optimizer
            gc.collect()
    for epoch in {e for _, e in orders}:
        values = [order for (arm, e), order in orders.items() if e == epoch]
        assert all(order == values[0] for order in values)
    assert run.sources() == source_hashes
    run.core.C.dump(out, dict(passed=True, final=args.final, audited_arms=arms,
        scope='actual checkpoint/parameter/buffer/AdamW/RNG audit; not paired AP or content necessity proof',
        foundation_constructor_loads=loads, source_hashes=source_hashes,
        audit_source_sha256=run.core.C.file_sha(Path(__file__)), gate_sha256=run.core.C.file_sha(GATE),
        common_init_sha256=identity['common_init_sha256'], results=results))
    print('CHECKPOINT AUDIT PASS', str(out), flush=True)


if __name__ == '__main__':
    main()
