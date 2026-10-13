"""Canonical architecture identity and checkpoint fingerprint helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any, Dict
from pathlib import Path


ARCHITECTURE_SCHEMA_VERSION = 3
_PATH_KEYS = {"checkpoint", "checkpoint_path", "ckpt_path"}


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _plain(value: Any, key: str = "") -> Any:
    if isinstance(value, Mapping):
        return {
            str(name): _plain(item, str(name))
            for name, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if key in _PATH_KEYS and isinstance(value, str):
        return "<EXTERNAL_CHECKPOINT>"
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def normalized_model_config(model_config: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize a resolved cfg.model while ignoring machine-local ckpt paths."""
    return _plain(model_config)


def architecture_fingerprint(model_config: Mapping[str, Any]) -> str:
    payload = json.dumps(
        normalized_model_config(model_config),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def architecture_id(model_config: Mapping[str, Any]) -> str:
    """Map a resolved cfg.model dict to its architecture ID string.

    The ID space is historical: strings must stay byte-stable because
    checkpoints embed the ID computed at training time.  The branches kept
    here are exactly the ones reachable from the a3sgm config family
    (a3sgm_whu1024 / a3sgm_vhr10 / the four ablation deltas); the legacy
    b0/b1/c1-c5/gaussian/roi-sam/c5v2/r3/densecap/CG-tail IDs of removed
    arms were pruned — the fingerprint below still hashes EVERY config key,
    so pruning cannot mask a config drift.
    """
    neck = model_config["neck"]
    head = model_config["roi_head"]["mask_head"]
    if str(neck.get("type", "")) != "RSSAM2PAFPN":
        raise KeyError(
            f"Unsupported neck type {neck.get('type')!r}; only RSSAM2PAFPN "
            "survives the a3sgm-only cleanup."
        )
    stride = int(model_config.get("sam_image_embedding_stride", 32))
    mode = str(head.get("explicit_prompt_mode", "points"))
    if not bool(head.get("shape_prior_cfg", {}).get("enabled", False)):
        raise KeyError(
            "shape_prior_cfg.enabled=False (the raw-MLP prompt route) belongs "
            "to removed baseline arms and has no architecture ID."
        )
    dense_prompt_cfg = head.get("dense_prompt_cfg", {})
    dense_transform = str(dense_prompt_cfg.get("transform", "raw_logits"))
    dense_detach = bool(dense_prompt_cfg.get("detach_input", False))
    if (
        mode == "points_box_dense"
        and stride == 16
        and dense_transform == "raw_logits"
        and dense_detach
    ):
        # The a3sgm base: PBM prompts with sg-detached raw-logit dense canvas.
        base = "r1_c4_rd_pafpn_coarse_points_box_raw_detach_emb64"
    else:
        known = {
            ("points", 16): "pafpn_coarse_points_emb64_p2br0",
            ("box", 16): "pafpn_coarse_box_emb64_p2br0",
            ("points_box", 16): "r1_c3_pafpn_coarse_points_box_emb64",
        }
        base = known.get(
            (mode, stride),
            f"pafpn_coarse_{mode}_emb{1024 // stride}_p2br0",
        )
    tail_cfg = head.get("decoder_tail_refiner_cfg", {})
    if bool(tail_cfg.get("enabled", False)):
        tail_k = int(tail_cfg.get("num_points", 0))
        # Margin/v1 share the udprk suffix by design, preserving the
        # historic semantic ID of the A3 container.
        base = f"{base}_udprk{tail_k}"
    return base


def architecture_contract(model_config: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "schema_version": ARCHITECTURE_SCHEMA_VERSION,
        "architecture_id": architecture_id(model_config),
        "model_fingerprint": architecture_fingerprint(model_config),
    }


def assert_checkpoint_architecture(
    checkpoint: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    operation: str,
    allow_cross_arch: bool = False,
) -> None:
    snapshot = checkpoint.get("config_snapshot", {})
    saved = snapshot.get("architecture_contract", {}) if isinstance(snapshot, Mapping) else {}
    if not saved:
        if allow_cross_arch:
            return
        raise RuntimeError(
            f"{operation} checkpoint lacks architecture_contract; "
            "use explicit cross-architecture opt-in only for intentional legacy migration"
        )
    same = (
        saved.get("architecture_id") == expected.get("architecture_id")
        and saved.get("model_fingerprint") == expected.get("model_fingerprint")
    )
    if not same and not allow_cross_arch:
        raise RuntimeError(
            f"{operation} architecture mismatch: saved={dict(saved)} expected={dict(expected)}"
        )
