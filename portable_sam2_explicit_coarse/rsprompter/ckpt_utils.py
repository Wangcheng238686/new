"""Checkpoint 严格加载工具（问题 13 修复）。

之前所有 SAM2 子模块（ImageEncoder / PromptEncoder / MaskDecoder）都用
load_state_dict(strict=False) 且只打印 warning，导致：
  - Base+ 与 Large 权重错配时静默加载错误权重；
  - 新增模块未加载却被误认为加载成功；
  - missing/unexpected key 不区分、不校验。

本模块提供 load_module_state_dict_strict()，按白名单校验 missing/unexpected keys：
非白名单的 key 出现直接 raise，同时打印完整 key 列表。

默认 strict_reproduction=False 保持现有行为（等价 strict=False + 打印），
用户可在 config 中设 checkpoint_load_cfg=dict(strict_reproduction=True) 开启严格模式。
"""
import fnmatch
import logging
from typing import Dict, Iterable, Tuple

import torch
from torch.nn import Module

logger = logging.getLogger(__name__)


def _match_any_pattern(key: str, patterns: Iterable[str]) -> bool:
    """key 是否匹配任意一个 glob 模式（支持 * 通配符）。"""
    return any(fnmatch.fnmatch(key, p) for p in patterns)


def load_module_state_dict_strict(
    module: Module,
    state_dict: Dict[str, torch.Tensor],
    *,
    module_name: str,
    strict_reproduction: bool = False,
    allowed_missing_patterns: Tuple[str, ...] = (),
    allowed_unexpected_patterns: Tuple[str, ...] = (),
    log_prefix: str = "",
) -> Dict[str, int]:
    """加载子模块权重，按白名单严格校验 missing/unexpected keys。

    Args:
        module: 目标子模块（state_dict 的 key 应与 module.state_dict() 一致，
            调用方需先剥离 ckpt 里的前缀如 "image_encoder." / "sam_prompt_encoder."）。
        state_dict: 已剥离前缀的权重 dict。
        module_name: 模块名（仅用于日志与报错信息，如 "ImageEncoder"）。
        strict_reproduction: True 时非白名单 missing/unexpected key 直接 raise。
            False 时等价 strict=False + 打印（保持历史行为）。
        allowed_missing_patterns: 允许 missing 的 key glob 白名单。
        allowed_unexpected_patterns: 允许 unexpected 的 key glob 白名单。

    Returns:
        dict: {"loaded": N, "missing": N, "unexpected": N,
               "missing_violations": N, "unexpected_violations": N}
    """
    msg = module.load_state_dict(state_dict, strict=False)
    loaded = len(state_dict)
    missing = list(msg.missing_keys)
    unexpected = list(msg.unexpected_keys)

    missing_violations = [k for k in missing if not _match_any_pattern(k, allowed_missing_patterns)]
    unexpected_violations = [
        k for k in unexpected if not _match_any_pattern(k, allowed_unexpected_patterns)
    ]

    pref = f"[{log_prefix}{module_name}] " if log_prefix else f"[{module_name}] "
    if is_main_process():
        print(f"{pref}Loaded {loaded} keys. missing={len(missing)}, unexpected={len(unexpected)}")
        if missing:
            sample = missing[:20]
            print(f"{pref}Missing keys ({len(missing)}, first {len(sample)}): {sample}")
        if unexpected:
            sample = unexpected[:20]
            print(f"{pref}Unexpected keys ({len(unexpected)}, first {len(sample)}): {sample}")

    if strict_reproduction:
        if missing_violations:
            raise RuntimeError(
                f"{pref}strict_reproduction=True: {len(missing_violations)} missing keys "
                f"not in allowed_missing_patterns {tuple(allowed_missing_patterns)}. "
                f"First few: {missing_violations[:10]}"
            )
        if unexpected_violations:
            raise RuntimeError(
                f"{pref}strict_reproduction=True: {len(unexpected_violations)} unexpected keys "
                f"not in allowed_unexpected_patterns {tuple(allowed_unexpected_patterns)}. "
                f"First few: {unexpected_violations[:10]}"
            )
    else:
        # 非严格模式：仅警告违规项，不中断（保持历史行为）
        if missing_violations and is_main_process():
            print(f"{pref}WARNING (non-strict): {len(missing_violations)} missing keys "
                  f"not whitelisted. First few: {missing_violations[:10]}")
        if unexpected_violations and is_main_process():
            print(f"{pref}WARNING (non-strict): {len(unexpected_violations)} unexpected keys "
                  f"not whitelisted. First few: {unexpected_violations[:10]}")

    return {
        "loaded": loaded,
        "missing": len(missing),
        "unexpected": len(unexpected),
        "missing_violations": len(missing_violations),
        "unexpected_violations": len(unexpected_violations),
    }


def is_main_process() -> bool:
    """判断是否主进程（避免多卡重复打印）。

    优先用 mmengine.dist.is_main_process（与 models_sam2.py 一致），
    不可用时 fallback 到读 RANK/LOCAL_RANK env。
    """
    try:
        from mmengine.dist import is_main_process as _mm_is_main
        return _mm_is_main()
    except ImportError:
        import os
        rank = os.environ.get("RANK", "0")
        local_rank = os.environ.get("LOCAL_RANK", "0")
        return int(rank) == 0 and int(local_rank) == 0
