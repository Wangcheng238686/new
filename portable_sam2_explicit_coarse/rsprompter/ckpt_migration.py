"""Checkpoint migrations used by initialization-only checkpoint loading."""

from typing import Dict, List, Mapping, Tuple

import torch


def migrate_legacy_shape_gate_state_dict(
    source_state: Mapping[str, torch.Tensor],
    target_state: Mapping[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], List[Tuple[str, str]]]:
    """Convert a legacy linear shape gate into the current sigmoid raw gate."""
    migrated = dict(source_state)
    migrations: List[Tuple[str, str]] = []
    old_suffix = "shape_prompt_scale"
    new_suffix = "shape_dense_alpha_raw"

    for old_name, value in tuple(source_state.items()):
        if not old_name.endswith(old_suffix):
            continue
        new_name = old_name[: -len(old_suffix)] + new_suffix
        if new_name not in target_state or new_name in migrated:
            continue
        alpha = value.detach().float().clamp(1e-6, 1.0 - 1e-6)
        migrated[new_name] = torch.logit(alpha).to(target_state[new_name].dtype)
        migrations.append((old_name, new_name))

    return migrated, migrations
