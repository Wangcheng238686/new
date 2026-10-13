"""Checkpoint-to-model reconstruction: snapshot parsing, contract compat,
registry build and strict state loading.

The two compat variants (``_prebuild_config_compat_variant``,
``_is_legacy_pre_tail_id``) and the contract-matching closure inside
``_resolve_model_config`` are a FROZEN zone: function text is moved verbatim
from the a3sgm cleanup state and must stay byte-identical.
"""

from __future__ import annotations

import copy
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Tuple

import torch


MAINLINE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = MAINLINE_ROOT.parent

logger = logging.getLogger("checkpoint_inference")


def _snapshot(checkpoint: Mapping[str, Any]) -> Dict[str, Any]:
    value = checkpoint.get("config_snapshot", {})
    return dict(value) if isinstance(value, Mapping) else {}


def _prebuild_config_compat_variant(
    model_config: Mapping[str, Any], snapshot: Mapping[str, Any]
) -> Dict[str, Any]:
    """Recover the exact pre-``MODELS.build`` config from an old v2 snapshot.

    Some schema-v2 checkpoints calculated their architecture contract from
    ``cfg.model`` before MMEngine consumed nested fields, then serialized the
    mutated post-build object.  RoI ``train_cfg``/``test_cfg`` are injected by
    the registry, while a newly introduced nested module can be consumed.  In
    particular, A3's UDPR tail lived in ``mask_head_config`` but disappeared
    from the post-build ``model_config``.  This variant is accepted only by
    the normal exact fingerprint check below; it is never a permissive load.
    """
    variant = copy.deepcopy(dict(model_config))
    roi_head = variant.get("roi_head")
    if not isinstance(roi_head, MutableMapping):
        return variant
    roi_head.pop("train_cfg", None)
    roi_head.pop("test_cfg", None)
    mask_head = roi_head.get("mask_head")
    saved_head = snapshot.get("mask_head_config", {})
    if not isinstance(mask_head, MutableMapping) or not isinstance(saved_head, Mapping):
        return variant
    tail_cfg = saved_head.get("decoder_tail_refiner_cfg")
    if isinstance(tail_cfg, Mapping) and bool(tail_cfg.get("enabled", False)):
        mask_head.setdefault("decoder_tail_refiner_cfg", copy.deepcopy(dict(tail_cfg)))
    return variant


def _resolve_sam2_repo(args: argparse.Namespace, snapshot: Mapping[str, Any]) -> Path:
    runtime = snapshot.get("runtime_config", {})
    saved = runtime.get("sam2_repo") if isinstance(runtime, Mapping) else None
    candidate = (
        args.sam2_repo
        or saved
        or os.environ.get("SAM2_REPO")
        or str(REPOSITORY_ROOT / "sam2")
    )
    path = Path(candidate).expanduser().resolve()
    if not (path / "sam2" / "__init__.py").is_file():
        raise FileNotFoundError(
            f"SAM2 repository is invalid: {path}; expected {path / 'sam2/__init__.py'}"
        )
    os.environ["SAM2_REPO"] = str(path)
    sys.path.insert(0, str(path))
    return path


def _targeted_sam2_checkpoint_override(
    model_config: MutableMapping[str, Any],
    checkpoint_path: str,
) -> None:
    backbone = model_config.get("backbone")
    if isinstance(backbone, MutableMapping):
        backbone["checkpoint_path"] = checkpoint_path
        init_cfg = backbone.get("init_cfg")
        if isinstance(init_cfg, MutableMapping):
            init_cfg["checkpoint"] = checkpoint_path

    roi_head = model_config.get("roi_head")
    if not isinstance(roi_head, MutableMapping):
        return
    mask_head = roi_head.get("mask_head")
    if not isinstance(mask_head, MutableMapping):
        return
    decoder_cfg = mask_head.get("sam2_mask_decoder")
    if isinstance(decoder_cfg, MutableMapping):
        decoder_cfg["checkpoint_path"] = checkpoint_path


_TAIL_ID_SUFFIX = re.compile(r"^_udprk\d+$")


def _is_legacy_pre_tail_id(saved_id: Any, resolved: Mapping[str, Any]) -> bool:
    """True when the only architecture-ID difference is the decoder-tail suffix.

    Commit eef70ca (2026-09-18) started appending ``_udprk{K}`` to tail-enabled
    RD/G base IDs so tail models stop sharing an ID with tail-less ones.
    Checkpoints trained before it (e.g. NWPU a3sgm, saved 2026-09-17) carry the
    bare base ID. Only this exact naming gap is tolerated here; content equality
    is still enforced by the fingerprint comparison at the call site, so the
    rule can never mask a real config drift.
    """
    resolved_id = resolved.get("architecture_id")
    if not isinstance(saved_id, str) or not isinstance(resolved_id, str):
        return False
    return bool(
        resolved_id.startswith(saved_id)
        and _TAIL_ID_SUFFIX.match(resolved_id[len(saved_id) :])
    )


def _resolve_model_config(
    args: argparse.Namespace,
    snapshot: Mapping[str, Any],
) -> Tuple[Dict[str, Any], str]:
    saved = snapshot.get("model_config")
    if isinstance(saved, Mapping) and saved:
        model_config = copy.deepcopy(dict(saved))
        source = "checkpoint.config_snapshot.model_config"
    else:
        fallback = args.config or snapshot.get("config_path")
        if not fallback:
            raise RuntimeError(
                "Checkpoint has no embedded model_config. Pass --config for this old checkpoint."
            )
        from mmengine.config import Config

        config_path = Path(str(fallback)).expanduser()
        if not config_path.is_absolute():
            config_path = MAINLINE_ROOT / config_path
        if not config_path.is_file():
            raise FileNotFoundError(f"Fallback config not found: {config_path}")
        cfg = Config.fromfile(str(config_path))
        model_config = cfg.model.to_dict()
        source = str(config_path)

    saved_contract = snapshot.get("architecture_contract", {})
    if saved_contract:
        from mmengine.config import Config
        from rsprompter.architecture_contract import architecture_contract

        def _contract_matches(saved, config):
            """Return the config variant whose contract EQUALS the saved one
            (the input itself, a compat variant, or None).  Returning the
            matched variant keeps downstream config fingerprints identical to
            the training-time contract instead of merely semantically equal."""
            resolved = architecture_contract(config)
            if (
                saved.get("architecture_id") == resolved.get("architecture_id")
                and saved.get("model_fingerprint") == resolved.get("model_fingerprint")
            ):
                return config
            # Naming-only compat for pre-eef70ca checkpoints: identical
            # fingerprint plus the ID gap being exactly the tail suffix.
            if (
                saved.get("model_fingerprint") == resolved.get("model_fingerprint")
                and _is_legacy_pre_tail_id(saved.get("architecture_id"), resolved)
            ):
                logger.warning(
                    "Checkpoint predates the decoder-tail ID suffix: saved=%s "
                    "accepted as %s (fingerprints identical)",
                    saved.get("architecture_id"),
                    resolved.get("architecture_id"),
                )
                return config
            # Compat: checkpoints saved before PROMPT_ENCODER_TRAIN_
            # MASK_DOWNSCALING was plumbed into the main config carry no
            # train_mask_downscaling key; a re-resolution that adds an
            # explicit False is semantically identical (mask-head setdefault)
            # and must not fail the contract.  Only this exact no-op key
            # addition is tolerated, and the KEY-LESS variant is returned so
            # config fingerprints stay bit-identical.
            head_cfg = config.get("roi_head", {}).get("mask_head", {})
            pe_cfg = head_cfg.get("prompt_encoder_cfg")
            if isinstance(pe_cfg, dict) and pe_cfg.get("train_mask_downscaling") is False:
                pe_variant = {k: v for k, v in pe_cfg.items()
                              if k != "train_mask_downscaling"}
                head_variant = dict(head_cfg)
                head_variant["prompt_encoder_cfg"] = pe_variant
                config_variant = dict(config)
                config_variant["roi_head"] = {
                    **config.get("roi_head", {}), "mask_head": head_variant}
                resolved_variant = architecture_contract(config_variant)
                if (
                    saved.get("architecture_id")
                    == resolved_variant.get("architecture_id")
                    and saved.get("model_fingerprint")
                    == resolved_variant.get("model_fingerprint")
                ):
                    return config_variant
            return None

        resolved_contract = architecture_contract(model_config)
        matched_config = _contract_matches(saved_contract, model_config)
        if matched_config is None:
            prebuild_variant = _prebuild_config_compat_variant(model_config, snapshot)
            matched_config = _contract_matches(saved_contract, prebuild_variant)
            if matched_config is not None:
                logger.warning(
                    "Recovered contract-matching pre-build model_config from "
                    "checkpoint snapshot"
                )
        if matched_config is not None and matched_config is not model_config:
            model_config = matched_config
        if matched_config is None:
            # Older schema-v2 training runs calculated the contract before
            # MODELS.build(), but captured cfg.model afterwards. MMEngine
            # registries may consume nested config fields in place, so that
            # embedded post-build copy cannot reproduce the pre-build
            # fingerprint. Recover only from the recorded source config and
            # only when it exactly reproduces the saved contract.
            recorded = args.config or snapshot.get("config_path")
            recorded_path = Path(str(recorded)).expanduser() if recorded else None
            if recorded_path is not None and not recorded_path.is_absolute():
                recorded_path = MAINLINE_ROOT / recorded_path
            if recorded_path is not None and recorded_path.is_file():
                candidate = Config.fromfile(str(recorded_path)).model.to_dict()
                candidate_contract = architecture_contract(candidate)
            else:
                candidate = None
                candidate_contract = {}
            matched_candidate = (
                None if candidate is None
                else _contract_matches(saved_contract, candidate)
            )
            if matched_candidate is None:
                raise RuntimeError(
                    "Embedded model_config does not match its saved architecture "
                    f"contract: saved={dict(saved_contract)} "
                    f"resolved={resolved_contract} "
                    f"recorded_config={candidate_contract}"
                )
            logger.warning(
                "Embedded post-build model_config changed fingerprint; using "
                "contract-matching recorded config %s",
                recorded_path,
            )
            model_config = matched_candidate
            source = str(recorded_path)

    runtime = snapshot.get("runtime_config", {})
    saved_sam2_ckpt = (
        runtime.get("sam2_checkpoint") if isinstance(runtime, Mapping) else None
    )
    sam2_checkpoint = (
        args.sam2_ckpt or saved_sam2_ckpt or os.environ.get("SAM2_CKPT")
    )
    if sam2_checkpoint:
        sam2_checkpoint = str(Path(sam2_checkpoint).expanduser().resolve())
        if not Path(sam2_checkpoint).is_file():
            raise FileNotFoundError(f"SAM2 base checkpoint not found: {sam2_checkpoint}")
        os.environ["SAM2_CKPT"] = sam2_checkpoint
        saved_sha256 = (
            runtime.get("sam2_checkpoint_sha256")
            if isinstance(runtime, Mapping)
            else None
        )
        if saved_sha256:
            from rsprompter.architecture_contract import file_sha256

            actual_sha256 = file_sha256(sam2_checkpoint)
            if actual_sha256 != saved_sha256:
                raise RuntimeError(
                    "SAM2 base checkpoint SHA256 mismatch: "
                    f"saved={saved_sha256} actual={actual_sha256}"
                )
        _targeted_sam2_checkpoint_override(model_config, sam2_checkpoint)
    return model_config, source


def _register_and_build(model_config: Mapping[str, Any]):
    import mmdet.models  # noqa: F401
    from mmengine.config import ConfigDict
    from mmengine.registry import MODELS as MMENGINE_MODELS
    from mmengine.registry import init_default_scope
    from mmdet.models.data_preprocessors import DetDataPreprocessor
    from mmdet.registry import MODELS

    if "DetDataPreprocessor" not in MMENGINE_MODELS:
        MMENGINE_MODELS.register_module(
            name="DetDataPreprocessor",
            module=DetDataPreprocessor,
        )
    import data  # noqa: F401
    import rsprompter  # noqa: F401

    init_default_scope("mmdet")
    return MODELS.build(ConfigDict(copy.deepcopy(dict(model_config))))


def _select_state_dict(
    checkpoint: Mapping[str, Any],
    weights: str,
) -> Mapping[str, torch.Tensor]:
    if weights == "model":
        state = checkpoint["model"]
    else:
        ema = checkpoint.get("ema_state")
        if not isinstance(ema, Mapping) or not isinstance(ema.get("ema_state"), Mapping):
            raise KeyError("--weights ema requested but checkpoint has no EMA state")
        state = ema["ema_state"]
    if not isinstance(state, Mapping):
        raise TypeError(f"Selected {weights} state is not a mapping")
    return state


def _load_model_state(
    model: torch.nn.Module,
    state: Mapping[str, torch.Tensor],
    allow_nonstrict: bool,
) -> Dict[str, List[str]]:
    normalized = dict(state)
    if normalized and all(key.startswith("module.") for key in normalized):
        normalized = {key[len("module.") :]: value for key, value in normalized.items()}
    result = model.load_state_dict(normalized, strict=not allow_nonstrict)
    return {
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
    }

