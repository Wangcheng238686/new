"""SAM2 model size / checkpoint 唯一配置来源（问题 2 修复）。

之前 SAM2 的 model_size、checkpoint 路径、neck in_channels 散落在多个 config
与子配置中各自硬编码（典型 bug：文件名为 baseplus 实际跑 large）。本模块建立
唯一变量 sam2_model_size + sam2_checkpoint，自动派生所有相关组件，避免错配。

用法（在任何 config 顶部）：
    from configs._sam2_registry import build_sam2_cfg
    sam2 = build_sam2_cfg()            # 读取 env SAM2_MODEL_SIZE / SAM2_CKPT
    # sam2.model_size / sam2.checkpoint / sam2.neck_in_channels / sam2.yaml

权威来源优先级：
    1. 环境变量 SAM2_MODEL_SIZE / SAM2_CKPT（外部传入，最高优先级）
    2. 本文件 _DEFAULT_MODEL_SIZE（默认 "large"，保持历史复现行为）
"""
import os
import hashlib
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# 历史默认：保持 large 不变，不破坏现有复现
_DEFAULT_MODEL_SIZE = "large"

# 已知 SAM2 Hiera 规格。neck_in_channels 是 RSFeatureAggregatorSAM2 / RSSAM2PAFPN
# 用来从 backbone 各 stage 通道数的硬编码 key。
_KNOWN_SIZES = {
    "tiny": dict(yaml="sam2_hiera_t.yaml"),
    "small": dict(yaml="sam2_hiera_s.yaml"),
    "base_plus": dict(yaml="sam2_hiera_b+.yaml"),
    "large": dict(yaml="sam2_hiera_l.yaml"),
}


@dataclass(frozen=True)
class SAM2ModelConfig:
    """SAM2 模型规格的单一真相源。"""
    model_size: str            # "large" / "base_plus" / "small" / "tiny"
    checkpoint: str            # 绝对路径，预训练 .pt
    yaml: str                  # 绝对路径，模型结构 .yaml（如 sam2_hiera_l.yaml）
    neck_in_channels: str      # neck in_channels 硬编码 key（与 backbone 通道数对应）

    @property
    def model_size_is_large(self) -> bool:
        return self.model_size == "large"


def _resolve_checkpoint(model_size: str, env_ckpt: Optional[str]) -> str:
    """优先 env SAM2_CKPT，其次按 model_size 推断默认文件名。"""
    if env_ckpt:
        return os.path.abspath(env_ckpt)
    # 与历史默认对齐：checkpoints/ 目录下的同名文件
    fname = {
        "large": "sam2_hiera_large.pt",
        "base_plus": "sam2_hiera_base_plus.pt",
        "small": "sam2_hiera_small.pt",
        "tiny": "sam2_hiera_tiny.pt",
    }.get(model_size, "sam2_hiera_large.pt")
    return os.path.abspath(os.path.join("checkpoints", fname))


def _resolve_yaml(model_size: str, env_yaml: Optional[str]) -> str:
    if env_yaml:
        return os.path.abspath(env_yaml)
    known = _KNOWN_SIZES.get(model_size)
    fname = known["yaml"] if known else "sam2_hiera_l.yaml"
    return os.path.abspath(os.path.join("checkpoints", fname))


def _neck_in_channels(model_size: str) -> str:
    """neck in_channels 与 SAM2 backbone 各 stage 通道数绑定，用字符串 key 表达。"""
    if model_size in _KNOWN_SIZES:
        return f"sam2_hiera_{model_size}"
    # 未知 size 走 large 通道（保守）
    return "sam2_hiera_large"


def build_sam2_cfg(
    model_size: Optional[str] = None,
    checkpoint: Optional[str] = None,
    yaml_path: Optional[str] = None,
) -> SAM2ModelConfig:
    """构建 SAM2 模型规格。

    Args:
        model_size: 显式指定；None 时读 env SAM2_MODEL_SIZE，再 fallback 默认。
        checkpoint: 显式指定 ckpt 绝对路径；None 时读 env SAM2_CKPT，再按 size 推断。
        yaml_path: 显式指定 yaml；None 时读 env SAM2_CONFIG，再按 size 推断。
    """
    size = (model_size or os.environ.get("SAM2_MODEL_SIZE") or _DEFAULT_MODEL_SIZE).lower()
    if size not in _KNOWN_SIZES:
        raise ValueError(
            f"Unknown SAM2_MODEL_SIZE={size!r}; expected one of {sorted(_KNOWN_SIZES)}"
        )
    ckpt = _resolve_checkpoint(size, checkpoint or os.environ.get("SAM2_CKPT"))
    yaml_ = _resolve_yaml(size, yaml_path or os.environ.get("SAM2_CONFIG"))
    return SAM2ModelConfig(
        model_size=size,
        checkpoint=ckpt,
        yaml=yaml_,
        neck_in_channels=_neck_in_channels(size),
    )


def sha256_of_file(path: str, chunk: int = 1 << 20) -> str:
    """计算 checkpoint 的 SHA256，供启动 manifest 用。文件不存在返回 '<missing>'。"""
    if not os.path.exists(path):
        return "<missing>"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()
