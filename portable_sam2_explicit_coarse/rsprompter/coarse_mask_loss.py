"""CoarseMaskLoss — ROI-local coarse mask 的独立强监督（批次2 子任务1）。

替代 models.py 里旧的 _shape_prior_aux_loss（dice+bce，target 用 pred 尺寸）。
本模块在预测原生分辨率上计算 loss（二值 target 用 nearest 对齐到 pred）。

组成：BCE + Dice + min-pool boundary（可微）+ distance（本批次 NotImplementedError）。
诊断：返回 dice/iou/boundary_tp/fp/fn（metric 在 no_grad 下，DDP 聚合由上层做）。

关键设计（审查意见 5.x）：
  - boundary erosion 用 min-pool（-max_pool(-x)），不是 avg-pool（非形态学 erosion）；
  - 空 ROI 返回 pred.sum()*0.0（连接计算图），不创建无关零 tensor；
  - boundary metric 只返回 tp/fp/fn，上层 all-reduce 后算 P/R/F1；
  - distance_weight>0 raise NotImplementedError（本批次不实现距离变换 loss）。
"""
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CoarseMaskLoss(nn.Module):
    """ROI-local coarse mask 的 BCE+Dice+boundary 组合 loss。

    Args:
        weight: 总 loss 权重（乘到最终标量上，对应旧 shape_prior_loss_weight）。
        bce_weight / dice_weight / boundary_weight: 各分项权重。
        distance_weight: >0 时 raise NotImplementedError（本批次不实现）。
        bce_reduction: BCE reduction（mean 对应旧实现）。
        dice_smooth: Dice 平滑项（旧实现用 1e-6 在 union，这里在 inter/union 各加 smooth）。
        boundary_tolerance: boundary metric 的容忍像素数（1-2px），仅 metric 用。
    """

    def __init__(
        self,
        weight: float = 0.05,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        boundary_weight: float = 0.0,
        distance_weight: float = 0.0,
        bce_reduction: str = "mean",
        dice_smooth: float = 1e-6,
        boundary_tolerance: int = 2,
        compute_metrics: bool = False,
        schedule_mode: Optional[str] = None,
        weight_schedule: Optional[List[Dict]] = None,
    ):
        super().__init__()
        self.weight = float(weight)
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.boundary_weight = float(boundary_weight)
        self.distance_weight = float(distance_weight)
        self.bce_reduction = str(bce_reduction)
        self.dice_smooth = float(dice_smooth)
        self.boundary_tolerance = int(boundary_tolerance)
        self.compute_metrics = bool(compute_metrics)
        self.weight_schedule = [dict(item) for item in (weight_schedule or [])]
        # Backward compatibility: older self-describing coarse checkpoints
        # stored weight_schedule but not schedule_mode.
        self.schedule_mode = (
            "two_stage" if self.weight_schedule else "fixed"
        ) if schedule_mode is None else str(schedule_mode)
        self.current_epoch = 1
        self._validate_weight_schedule()
        if self.distance_weight > 0:
            raise NotImplementedError(
                "coarse_mask_loss_cfg.distance_weight > 0 is not implemented in batch2. "
                "Use boundary_weight for boundary supervision, or set distance_weight=0."
            )


    def _validate_weight_schedule(self) -> None:
        if self.schedule_mode not in {"fixed", "two_stage"}:
            raise ValueError(
                "coarse loss schedule_mode must be fixed or two_stage, "
                f"got {self.schedule_mode!r}"
            )
        if self.schedule_mode == "fixed" and self.weight_schedule:
            raise ValueError("fixed coarse loss schedule must not define weight_schedule")
        if self.schedule_mode == "two_stage" and len(self.weight_schedule) != 2:
            raise ValueError("two_stage coarse loss schedule requires exactly two ranges")
        previous_end = 0
        for item in self.weight_schedule:
            start = int(item["start_epoch"])
            end = int(item["end_epoch"])
            value = float(item["weight"])
            if start <= 0 or end < start or value < 0:
                raise ValueError(f"Invalid coarse loss schedule item: {item}")
            if start <= previous_end:
                raise ValueError("coarse loss schedule ranges must be ordered and non-overlapping")
            previous_end = end

    def set_current_epoch(self, epoch_number: int) -> None:
        self.current_epoch = max(1, int(epoch_number))

    def effective_weight(self) -> float:
        for item in self.weight_schedule:
            if int(item["start_epoch"]) <= self.current_epoch <= int(item["end_epoch"]):
                return float(item["weight"])
        return self.weight

    def forward(self, pred_logits: Tensor, target_mask: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        """计算 coarse mask loss。

        Args:
            pred_logits: [N,1,H,W] 或 [N,H,W] coarse mask logits（ROI-local）。
            target_mask: [N,H,W] 或 [N,1,H,W] 二值 GT（已 resize 到 pred 尺寸，值 [0,1]）。

        Returns:
            loss: 标量（已乘 weight）。
            stats: dict，含 dice(1-dice,越小越好) / iou / boundary_tp / boundary_fp / boundary_fn。
                   metric 项在 no_grad 下计算，上层 DDP all-reduce 后算 P/R/F1。
        """
        # 统一 shape：pred [N,1,H,W], target [N,H,W]
        if pred_logits.dim() == 3:
            pred_logits = pred_logits.unsqueeze(1)
        assert pred_logits.dim() == 4 and pred_logits.shape[1] == 1, (
            f"pred_logits must be [N,1,H,W], got {tuple(pred_logits.shape)}"
        )
        if target_mask.dim() == 4:
            target_mask = target_mask.squeeze(1)
        assert target_mask.dim() == 3, f"target_mask must be [N,H,W], got {tuple(target_mask.shape)}"

        n = pred_logits.shape[0]
        # 空 ROI：返回连接计算图的零 loss（审查 5.x）
        if n == 0:
            zero = pred_logits.sum() * 0.0
            effective_weight = self.effective_weight()
            return zero, {
                "dice": zero.detach(),
                "iou": zero.detach(),
                "bce_raw": zero.detach(),
                "dice_loss_raw": zero.detach(),
                "loss_weight": pred_logits.new_tensor(effective_weight).detach(),
                "weighted_loss": zero.detach(),
                "boundary_tp": zero.detach(),
                "boundary_fp": zero.detach(),
                "boundary_fn": zero.detach(),
            }

        # pred 与 target 尺寸必须一致（由调用方保证 target 已 resize 到 pred 尺寸）
        if pred_logits.shape[-2:] != target_mask.shape[-2:]:
            raise ValueError(
                f"pred {tuple(pred_logits.shape)} and target {tuple(target_mask.shape)} "
                "must share H,W; caller should resize target to pred size first."
            )

        pred_prob = torch.sigmoid(pred_logits.squeeze(1))  # [N,H,W]

        # ---- BCE ----
        bce = F.binary_cross_entropy_with_logits(
            pred_logits.squeeze(1), target_mask, reduction=self.bce_reduction
        )

        # ---- Dice（标准平滑 Dice，审查第四节：保证非负）----
        # dice_score = (2*inter + smooth) / (union + smooth)，空 target+空 pred 时 = 0（非负）
        inter = (pred_prob * target_mask).flatten(1).sum(dim=1)  # [N]
        union = pred_prob.flatten(1).sum(dim=1) + target_mask.flatten(1).sum(dim=1)
        dice_score = (2.0 * inter + self.dice_smooth) / (union + self.dice_smooth)
        dice_per = 1.0 - dice_score
        dice = dice_per.mean()

        # ---- IoU（诊断）----
        iou_per = (inter + self.dice_smooth) / (union - inter + self.dice_smooth)
        iou = iou_per.mean()

        total = self.bce_weight * bce + self.dice_weight * dice

        # ---- Boundary loss（min-pool erosion，可微）----
        if self.boundary_weight > 0:
            boundary_loss = self._boundary_loss(pred_prob, target_mask)
            total = total + self.boundary_weight * boundary_loss
        else:
            boundary_loss = torch.zeros_like(total)

        # ---- Boundary metric（按需计算；默认关闭以避免无效 max-pool）----
        if self.compute_metrics:
            with torch.no_grad():
                tp, fp, fn = self._boundary_metric(pred_prob, target_mask)
        else:
            tp = fp = fn = pred_logits.new_zeros(()).detach()

        effective_weight = self.effective_weight()
        loss = effective_weight * total
        stats = {
            "dice": dice.detach(),
            "iou": iou.detach(),
            "bce_raw": bce.detach(),
            "dice_loss_raw": dice.detach(),
            "loss_weight": pred_logits.new_tensor(effective_weight).detach(),
            "weighted_loss": loss.detach(),
            "boundary_tp": tp,
            "boundary_fp": fp,
            "boundary_fn": fn,
        }
        return loss, stats

    @staticmethod
    def _morphological_boundary(prob: Tensor) -> Tensor:
        """可微形态学边界：dilation - erosion，用 min-pool（-max_pool(-x)）。

        prob: [N,H,W] ∈ [0,1]。返回 [N,H,W] ∈ [0,1] 的边界带。
        """
        p = prob.unsqueeze(1)  # [N,1,H,W]
        # Explicit zero padding makes the ROI exterior background for both
        # dilation and erosion.  In particular, an all-foreground ROI now has
        # a valid boundary along the ROI canvas edge.
        padded = F.pad(p, (1, 1, 1, 1), mode="constant", value=0.0)
        dilation = F.max_pool2d(padded, kernel_size=3, stride=1, padding=0)
        erosion = -F.max_pool2d(-padded, kernel_size=3, stride=1, padding=0)
        return (dilation - erosion).squeeze(1).clamp(0.0, 1.0)

    def _boundary_loss(self, pred_prob: Tensor, target_mask: Tensor) -> Tensor:
        """可微 boundary loss：pred_boundary 与 target_boundary 的 BCE。"""
        pred_b = self._morphological_boundary(pred_prob)
        with torch.no_grad():
            target_b = self._morphological_boundary(target_mask)
        return F.binary_cross_entropy(pred_b.clamp(0, 1), target_b.clamp(0, 1), reduction="mean")

    def _boundary_metric(self, pred_prob: Tensor, target_mask: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """boundary F-score 的 tp/fp/fn（tolerance 容忍，no_grad）。

        tolerance 用 max_pool 膨胀 target_boundary 来近似"距离容忍"：
        pred_boundary 像素落在膨胀后的 target_boundary 内 → TP，否则 FP；
        target_boundary 像素未被任何 pred_boundary 覆盖 → FN。
        """
        pred_bin = (pred_prob > 0.5).float()
        target_bin = (target_mask > 0.5).float()
        pred_b = self._morphological_boundary(pred_bin)
        target_b = self._morphological_boundary(target_bin)
        # tolerance 膨胀（用 max_pool 多次迭代近似 dilate N 像素）
        tol = max(1, self.boundary_tolerance)
        target_b_dilated = target_b.unsqueeze(1)
        for _ in range(tol):
            target_b_dilated = F.max_pool2d(target_b_dilated, kernel_size=3, stride=1, padding=1)
        target_b_dilated = target_b_dilated.squeeze(1)
        pred_b_dilated = pred_b.unsqueeze(1)
        for _ in range(tol):
            pred_b_dilated = F.max_pool2d(pred_b_dilated, kernel_size=3, stride=1, padding=1)
        pred_b_dilated = pred_b_dilated.squeeze(1)

        tp = (pred_b * target_b_dilated).flatten(1).sum(dim=1)  # [N]
        fp = (pred_b * (1.0 - target_b_dilated)).flatten(1).sum(dim=1)
        fn = (target_b * (1.0 - pred_b_dilated)).flatten(1).sum(dim=1)
        return tp.sum(), fp.sum(), fn.sum()
