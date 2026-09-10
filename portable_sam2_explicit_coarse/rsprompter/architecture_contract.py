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
    neck = model_config["neck"]
    head = model_config["roi_head"]["mask_head"]
    neck_name = (
        "pafpn" if str(neck.get("type", "")) == "RSSAM2PAFPN" else "aggregator"
    )
    stride = int(model_config.get("sam_image_embedding_stride", 32))
    final_mode = str(head.get("final_mask_coordinate_mode", "roi_local"))
    coarse = bool(head.get("shape_prior_cfg", {}).get("enabled", False))
    refiner = bool(head.get("p2_boundary_refiner_cfg", {}).get("enabled", False))
    tail = bool(head.get("decoder_tail_refiner_cfg", {}).get("enabled", False))
    renderer = bool(head.get("canvas_renderer_cfg", {}).get("enabled", False))
    if not coarse:
        known = {
            ("aggregator", "roi_local", 32): "b0_aggregator_mlp",
            ("pafpn", "roi_local", 32): "b1_pafpn_mlp",
            ("aggregator", "full_image", 32): "m0_aggregator_mlp_full_image",
            ("pafpn", "full_image", 32): "m1_pafpn_mlp_full_image",
            ("aggregator", "roi_local", 16): "r0_b0_aggregator_mlp_emb64",
        }
        return known.get(
            (neck_name, final_mode, stride),
            f"{neck_name}_mlp_{final_mode}_emb{1024 // stride}",
        )
    mode = str(head.get("explicit_prompt_mode", "points"))
    dense_prompt_cfg = head.get("dense_prompt_cfg", {})
    dense_transform = str(dense_prompt_cfg.get("transform", "raw_logits"))
    dense_detach = bool(dense_prompt_cfg.get("detach_input", False))
    final_loss_mode = str(
        head.get("final_mask_loss_cfg", {}).get("mode", "standard")
    )
    roi_sam = bool(head.get("roi_sam_cfg", {}).get("enabled", False))
    if (
        neck_name == "pafpn"
        and mode == "points"
        and stride == 32
        and not refiner
        and roi_sam
    ):
        return "c2r_pafpn_coarse_points_roi_sam"
    if (
        neck_name == "pafpn"
        and mode == "points"
        and stride == 32
        and not refiner
        and final_loss_mode == "roi_balanced_dice"
    ):
        return "c2l_pafpn_coarse_points_roi_loss"
    if (
        neck_name == "pafpn"
        and mode == "points_box_dense"
        and stride == 16
        and not refiner
        and dense_transform == "raw_logits"
        and dense_detach
    ):
        return "r1_c4_rd_pafpn_coarse_points_box_raw_detach_emb64"
    if (
        neck_name == "pafpn"
        and mode == "points_box_dense"
        and stride == 16
        and not refiner
        and dense_transform == "gaussian_edt"
        and dense_detach
    ):
        return "r1_c4_g_pafpn_coarse_points_box_gaussian_emb64"
    known = {
        ("aggregator", "points", 32, False): "c1_aggregator_coarse_points",
        ("pafpn", "points", 32, False): "c2_pafpn_coarse_points",
        ("pafpn", "points_box", 32, False): "c3_pafpn_coarse_points_box",
        ("pafpn", "points_box_dense", 32, False): "c4_pafpn_coarse_points_box_dense",
        ("pafpn", "points_box", 16, False): "r1_c3_pafpn_coarse_points_box_emb64",
        ("pafpn", "points_box_dense", 16, False): "r1_c4_pafpn_coarse_points_box_dense_emb64",
        ("pafpn", "points_box_dense", 16, True): "c5v2_pafpn_coarse_p2_boundary_refiner_emb64",
    }
    base = known.get(
        (neck_name, mode, stride, refiner),
        f"{neck_name}_coarse_{mode}_emb{1024 // stride}_p2br{int(refiner)}",
    )
    if renderer:
        base = f"{base}_r3"
    if tail:
        tail_cfg = head["decoder_tail_refiner_cfg"]
        tail_k = int(tail_cfg.get("num_points", 0))
        # A3/v1 has no mode key by design, preserving its historic semantic ID.
        if tail_cfg.get("mode", "residual_v1") == "confidence_gated":
            return f"{base}_udprcgk{tail_k}"
        return f"{base}_udprk{tail_k}"
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
