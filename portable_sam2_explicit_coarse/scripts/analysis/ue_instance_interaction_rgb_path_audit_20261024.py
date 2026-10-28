"""Read actual RGB checkpoints to disclose their effective LoRA update scope.

CPU only. Missing checkpoints are explicitly listed, never treated as passed.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'probes'))
import torch
import ue_instance_interaction_20261024 as run
import ue_instance_interaction_owner_20261024 as owner
from rsprompter.uav.instance_interaction_probe import InstanceInteraction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epoch', type=int, choices=range(1, 11), default=1)
    parser.add_argument('--root', type=Path, default=run.OUT)
    args = parser.parse_args()
    root = args.root
    module = InstanceInteraction()
    module.add_rgb('/data/wangcheng/pretrained-models/sam2/sam2_hiera_base_plus.pt', 2045)
    parameters = list(module.named_parameters())
    initial = torch.load(root / 'init_seed44_rgb.pth', map_location='cpu', weights_only=False)
    declared = sum(p.numel() for name, p in parameters if name.startswith('rgb_encoder.') and p.requires_grad)
    rows = []
    orders = []
    missing = []
    for arm in owner.arms():
        if arm['family'] != 'rgb':
            continue
        folder = root / arm['name']
        cp_file = folder / f'epoch{args.epoch}.pth'
        epoch_file = folder / f'epoch{args.epoch}.json'
        if not epoch_file.exists():
            missing.append(arm['name'])
            continue
        cp = torch.load(cp_file, map_location='cpu', weights_only=False)
        epoch = json.loads(epoch_file.read_text())
        assert cp['epoch'] == args.epoch and cp['step'] == 267 * args.epoch
        assert cp['contract'] == json.loads((folder / 'contract.json').read_text())
        ids = cp['optimizer']['param_groups'][0]['params']
        assert len(ids) == len(parameters)
        state = cp['optimizer']['state']
        active = [(name, p.numel()) for ident, (name, p) in zip(ids, parameters)
                  if ident in state and name.startswith('rgb_encoder.')]
        assert active and all('lora_' in name for name, _ in active)
        blocks = set()
        patch = []
        for name, _ in active:
            parts = name.split('.')
            if 'blocks' in parts:
                blocks.add(int(parts[parts.index('blocks') + 1]))
            else:
                assert 'patch_embed.proj.' in name, ('Unexpected active source module', name)
                patch.append(name)
        assert blocks == {0, 1} and len(patch) == 2
        assert {int(value['step']) for value in state.values()} == {267 * args.epoch}
        for name, tensor in cp['adapter'].items():
            assert torch.isfinite(tensor).all()
            if name.startswith('rgb_encoder.') and 'lora_' not in name:
                assert torch.equal(tensor, initial['adapter'][name]), name
        assert cp['contract']['initial_fingerprint'] == initial['fingerprint']
        assert len(epoch['sample_order']) == len(set(epoch['sample_order'])) == 2136
        assert epoch['frozen_SHA'] == run.previous.FROZEN_SHA
        orders.append(epoch['sample_order'])
        rows.append(dict(arm=arm['name'], actual_updates=267 * args.epoch,
            source_LoRA_parameters_with_optimizer_state=sum(count for _, count in active),
            source_LoRA_tensors_with_optimizer_state=[name for name, _ in active],
            source_blocks_with_optimizer_state=sorted(blocks), patch_embed_LoRA_active=True,
            source_declared_trainable_LoRA_parameters=declared, source_nonLoRA_state_exact=True,
            checkpoint_sha256=run.C.file_sha(cp_file), mean_microbatch_components=epoch['mean_microbatch_components'],
            amp_scale=epoch['amp_scale']))
        del cp
    assert rows, 'No actual checkpoint available'
    assert all(order == orders[0] for order in orders)
    geometry_epoch = root / 'seed44_geom_internal_geometry_correct' / f'epoch{args.epoch}.json'
    cross_family = False
    if geometry_epoch.exists():
        assert orders[0] == json.loads(geometry_epoch.read_text())['sample_order']
        cross_family = True
    result = dict(passed_for_available_checkpoints=True, epoch=args.epoch, actual_checkpoints=len(rows),
        expected_RGB_arms=6, missing_checkpoints=missing, all_RGB_arms_present=not missing,
        paired_order_exact=True, cross_family_order_checked=cross_family, rows=rows,
        audit_source_sha256=run.C.file_sha(__file__),
        scope='Optimizer state proves parameter updates, not useful RGB content or AP. '
              'Only patch embedding and source blocks0/1 participate through P0. '
              'No full-UAV-encoder training claim. Missing arms remain unverified.')
    run.C.dump(root / f'epoch{args.epoch}_rgb_effective_path_audit.json', result)
    print('AVAILABLE RGB PATH AUDIT PASS', len(rows), 'of 6')
    for row in rows:
        print(row['arm'], row['source_LoRA_parameters_with_optimizer_state'], 'of', declared,
              'source LoRA parameters; blocks', row['source_blocks_with_optimizer_state'], '+patch')


if __name__ == '__main__':
    main()
