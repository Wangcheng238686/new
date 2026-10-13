"""Checkpoint lifecycle: verified-best saving, resume/init loading
(EMA + optimizer + scheduler state) and epoch stamping."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.optim as optim

logger = logging.getLogger("portable_sam_fusion")


from optimizer import _assert_no_mask_embedding_contract


def _save_verified_best_checkpoint(
    checkpoint_dir: Path,
    prefix: str,
    score_label: str,
    score_value: float,
    epoch: int,
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    best_metrics: Dict,
    history: Dict,
    config_snapshot: Optional[Dict],
    scaler: Optional[Any],
    ema_state: Optional[Dict] = None,
) -> bool:
    new_best = checkpoint_dir / f"{prefix}_epoch{epoch + 1}.pth"
    _save_checkpoint(
        new_best,
        model,
        optimizer,
        scheduler,
        epoch,
        best_metrics,
        history,
        config_snapshot=config_snapshot,
        scaler=scaler,
        ema_state=ema_state,
    )
    try:
        _verify = torch.load(str(new_best), map_location="cpu", weights_only=False)
        assert "model" in _verify, "Missing 'model' key"
        assert "optimizer" in _verify, "Missing 'optimizer' key"
        assert "scheduler" in _verify, "Missing 'scheduler' key"
        assert "config_snapshot" in _verify, "Missing 'config_snapshot' key"
        logger.info("Saved %s with %s: %.4f", prefix, score_label, score_value)
    except Exception as e:
        logger.error("Checkpoint verification FAILED for %s, keeping old best: %s", prefix, e)
        new_best.unlink(missing_ok=True)
        return False

    best_link = checkpoint_dir / f"{prefix}.pth"
    for old in checkpoint_dir.glob(f"{prefix}_epoch*.pth"):
        if old != new_best:
            old.unlink(missing_ok=True)
    best_link.unlink(missing_ok=True)
    best_link.symlink_to(new_best.name)
    return True


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    epoch: int,
    best_metrics: Dict,
    history: Dict,
    config_snapshot: Optional[Dict] = None,
    scaler: Optional[Any] = None,
    ema_state: Optional[Dict] = None,
):
    """Save checkpoint with model, optimizer, scheduler, config, and training state.

    Saves to a temporary file first, then atomically renames to avoid corruption.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    model_to_save = model.module if hasattr(model, "module") else model
    ckpt = {
        "checkpoint_schema_version": int(
            (config_snapshot or {}).get("checkpoint_schema_version", 0)
        ),
        "model": model_to_save.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "best_metrics": best_metrics,
        "history": history,
        "config_snapshot": config_snapshot or {},
        "architecture_contract": (config_snapshot or {}).get(
            "architecture_contract", {}
        ),
    }
    if scaler is not None:
        ckpt["scaler"] = scaler.state_dict()
    if ema_state is not None:
        ckpt["ema_state"] = ema_state
    # Atomic save: write to .tmp then rename
    tmp_path = path.with_suffix(".pth.tmp")
    torch.save(ckpt, str(tmp_path))
    tmp_path.replace(path)


def _load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
    scaler: Optional[Any] = None,
    expected_architecture: Optional[Dict] = None,
) -> Dict:
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    if expected_architecture is not None:
        from rsprompter.architecture_contract import assert_checkpoint_architecture

        assert_checkpoint_architecture(
            ckpt, expected_architecture, operation="RESUME", allow_cross_arch=False
        )
    model_to_load = model.module if hasattr(model, "module") else model
    # 审查 3.5：RESUME 续训时模型结构应与当前运行完全一致，默认用 strict=True。
    # 若 ckpt 是旧结构（新增/删除了模块），strict=True 会失败——此时设置
    # env RESUME_STRICT=0 回退到宽松模式（仅打印 missing/unexpected 明细）。
    _resume_strict = os.environ.get("RESUME_STRICT", "1") != "0"
    if _resume_strict:
        # strict=True 时 load_state_dict 不返回 msg，missing/unexpected 会直接抛错
        model_to_load.load_state_dict(ckpt["model"], strict=True)
        if int(os.environ.get("RANK", "0")) == 0:
            logger.info("RESUME model load: strict=True (all keys matched)")
    else:
        # 问题 13：宽松模式补打 missing/unexpected keys 明细
        _resume_msg = model_to_load.load_state_dict(ckpt["model"], strict=False)
        if int(os.environ.get("RANK", "0")) == 0:
            logger.info(
                "RESUME model load: strict=False, missing=%d, unexpected=%d",
                len(_resume_msg.missing_keys), len(_resume_msg.unexpected_keys),
            )
            if _resume_msg.missing_keys:
                logger.info("  missing (first 20): %s", _resume_msg.missing_keys[:20])
            if _resume_msg.unexpected_keys:
                logger.info("  unexpected (first 20): %s", _resume_msg.unexpected_keys[:20])
    _assert_no_mask_embedding_contract(model_to_load, "RESUME checkpoint load")
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt





def _load_init_checkpoint(
    path: Path,
    model: torch.nn.Module,
    exclude_prefixes: Tuple[str, ...] = (),
    expected_architecture: Optional[Dict] = None,
    allow_cross_arch: bool = False,
    allowed_missing_prefixes: Tuple[str, ...] = (),
) -> Dict[str, object]:
    """Load matching tensors for initialization, with explicit legacy migration."""
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    if expected_architecture is not None:
        from rsprompter.architecture_contract import assert_checkpoint_architecture

        assert_checkpoint_architecture(
            ckpt,
            expected_architecture,
            operation="INIT_FROM",
            allow_cross_arch=allow_cross_arch,
        )
    source_state = ckpt.get("model", ckpt)
    model_to_load = model.module if hasattr(model, "module") else model
    target_state = model_to_load.state_dict()
    # Historical legacy-key migration (shape_prompt_scale era) was removed
    # with its arms; a3sgm checkpoints load with zero migrations.
    migrations = 0
    migration_pairs: list = []

    compatible = {}
    excluded = 0
    shape_mismatch = 0
    unexpected = 0
    for name, value in source_state.items():
        if any(name.startswith(prefix) for prefix in exclude_prefixes):
            excluded += 1
            continue
        if name not in target_state:
            unexpected += 1
            continue
        if target_state[name].shape != value.shape:
            shape_mismatch += 1
            continue
        compatible[name] = value

    load_result = model_to_load.load_state_dict(compatible, strict=False)
    if allowed_missing_prefixes:
        illegal_missing = [
            key for key in load_result.missing_keys
            if not any(key.startswith(prefix) for prefix in allowed_missing_prefixes)
        ]
        if illegal_missing or shape_mismatch or unexpected:
            raise RuntimeError(
                "cross-architecture init whitelist violated: "
                f"missing={illegal_missing}, shape_mismatch={shape_mismatch}, unexpected={unexpected}"
            )
    _assert_no_mask_embedding_contract(model_to_load, "INIT_FROM checkpoint load")
    return {
        "loaded": len(compatible),
        "excluded": excluded,
        "shape_mismatch": shape_mismatch,
        "unexpected": unexpected,
        "missing": len(load_result.missing_keys),
        "missing_keys": tuple(load_result.missing_keys),
        "migrated": len(migrations),
        "migration_pairs": migrations,
    }


def _set_model_current_epoch(model, epoch_number: int) -> None:
    """Set 1-based epoch on modules with epoch-dependent schedules."""
    core = model.module if hasattr(model, "module") else model
    seen = set()
    for module in core.modules():
        if id(module) in seen:
            continue
        seen.add(id(module))
        setter = getattr(module, "set_current_epoch", None)
        if callable(setter):
            setter(int(epoch_number))
