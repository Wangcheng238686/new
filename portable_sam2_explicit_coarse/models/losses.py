import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.registry import MODELS
from typing import Optional


class CrossViewContrastiveLoss(nn.Module):
    def __init__(self, bev_dim: int = 128, sat_dim: int = 256, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.bev_proj = nn.Linear(bev_dim, sat_dim)

    def forward(self, bev_feat: torch.Tensor, sat_feat: torch.Tensor) -> torch.Tensor:
        B = bev_feat.shape[0]

        bev_global = F.adaptive_avg_pool2d(bev_feat, 1).flatten(1)
        sat_global = F.adaptive_avg_pool2d(sat_feat, 1).flatten(1)

        bev_global = self.bev_proj(bev_global)
        bev_global = F.normalize(bev_global, dim=1)
        sat_global = F.normalize(sat_global, dim=1)

        logits = torch.matmul(bev_global, sat_global.T) / self.temperature

        labels = torch.arange(B, device=bev_feat.device)

        loss = F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
        return loss / 2


class FeatureConsistencyLoss(nn.Module):
    def __init__(self, bev_dim: int = 128, sat_dim: int = 256, hidden_dim: int = 256):
        super().__init__()
        self.sat_projector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(sat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.bev_projector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(bev_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        sat_feat: torch.Tensor,
        bev_feat: torch.Tensor,
    ) -> torch.Tensor:
        sat_embed = self.sat_projector(sat_feat)
        bev_embed = self.bev_projector(bev_feat)

        sat_embed = F.normalize(sat_embed, dim=1)
        bev_embed = F.normalize(bev_embed, dim=1)

        similarity = torch.sum(sat_embed * bev_embed, dim=1)
        loss = 1 - similarity.mean()

        return loss


class GeometricConsistencyLoss(nn.Module):
    def __init__(self, translation_weight: float = 1.0, rotation_weight: float = 1.0):
        super().__init__()
        self.translation_weight = translation_weight
        self.rotation_weight = rotation_weight

    def forward(self, alignment_matrix: torch.Tensor) -> torch.Tensor:
        R = alignment_matrix[:, :3, :3]
        t = alignment_matrix[:, :3, 3]
        
        RtR = torch.bmm(R.transpose(1, 2), R)
        eye = torch.eye(3, device=alignment_matrix.device).unsqueeze(0).expand_as(RtR)
        ortho_loss = torch.norm(RtR - eye, dim=(1, 2)).mean()
        
        det_R = torch.linalg.det(R)
        det_loss = torch.mean((det_R - 1.0) ** 2)
        
        trans_reg = torch.mean(t ** 2)
        
        total_loss = (
            self.rotation_weight * (ortho_loss + det_loss) +
            self.translation_weight * trans_reg
        )
        
        return total_loss


class SpatialSmoothnessLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, bev_feat: torch.Tensor) -> torch.Tensor:
        dx = bev_feat[:, :, :, 1:] - bev_feat[:, :, :, :-1]
        dy = bev_feat[:, :, 1:, :] - bev_feat[:, :, :-1, :]
        
        loss_x = torch.mean(torch.abs(dx))
        loss_y = torch.mean(torch.abs(dy))
        
        return loss_x + loss_y


class SoftClDiceLoss(nn.Module):
    """
    Soft clDice Loss for topology-preserving segmentation.
    
    This loss encourages the skeleton of the predicted mask to match
    the skeleton of the ground truth, preserving topological connectivity.
    
    Reference: https://arxiv.org/abs/2003.07311
    """
    
    def __init__(self, kernel_size: int = 3, max_iterations: int = 10):
        super().__init__()
        self.kernel_size = kernel_size
        self.max_iterations = max_iterations
        self.padding = kernel_size // 2
    
    def _soft_skeletonize(self, mask: torch.Tensor) -> torch.Tensor:
        """Soft skeletonization using iterative thinning."""
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        
        skeleton = mask.clone()
        
        for _ in range(self.max_iterations):
            eroded = 1.0 - F.avg_pool2d(
                1.0 - skeleton, kernel_size=self.kernel_size, stride=1, padding=self.padding
            )
            dilated = F.max_pool2d(
                eroded, kernel_size=self.kernel_size, stride=1, padding=self.padding
            )
            skeleton = torch.clamp(skeleton - dilated + eroded, min=0, max=1)
        
        return skeleton
    
    def _soft_dice(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute soft Dice coefficient."""
        smooth = 1e-6
        intersection = (pred * target).sum(dim=(1, 2, 3))
        union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return dice
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Soft clDice Loss.
        
        Args:
            pred: Predicted mask [B, H, W] or [B, 1, H, W], values in [0, 1]
            target: Target mask [B, H, W] or [B, 1, H, W], values in [0, 1]
        
        Returns:
            loss: Scalar loss value
        """
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        if target.dim() == 3:
            target = target.unsqueeze(1)
        
        pred_skeleton = self._soft_skeletonize(pred)
        target_skeleton = self._soft_skeletonize(target)
        
        tprec = self._soft_dice(pred_skeleton, target)
        tsens = self._soft_dice(target_skeleton, pred)
        
        cldice = (2.0 * tprec * tsens) / (tprec + tsens + 1e-6)
        
        loss = 1.0 - cldice.mean()
        
        return loss

@MODELS.register_module()
class BCEDiceMaskLoss(nn.Module):
    """Binary mask loss with BCE and Dice terms.

    The call signature matches MMDetection mask losses. For multi-class mask
    heads, the foreground channel indexed by ``labels`` is selected before
    computing the binary loss. That lets us optimize the current single-class
    setup now while keeping a usable path for future multi-class datasets.
    """

    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        pos_weight: float = 1.0,
        eps: float = 1e-6,
        loss_weight: float = 1.0,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.pos_weight = pos_weight
        self.eps = eps
        self.loss_weight = loss_weight

    def _select_logits(self, pred: torch.Tensor, labels: Optional[torch.Tensor]) -> torch.Tensor:
        if pred.dim() == 3:
            return pred.unsqueeze(1)
        if pred.dim() != 4:
            raise ValueError(f"Expected pred with 3 or 4 dims, got {tuple(pred.shape)}")
        if pred.size(1) == 1:
            return pred
        if labels is None:
            raise ValueError("labels are required when pred has multiple mask channels")
        batch_inds = torch.arange(pred.size(0), device=pred.device)
        return pred[batch_inds, labels].unsqueeze(1)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        logits = self._select_logits(pred, labels)
        if target.dim() == 3:
            target = target.unsqueeze(1)
        target = target.to(dtype=logits.dtype)

        bce = F.binary_cross_entropy_with_logits(
            logits,
            target,
            pos_weight=logits.new_tensor(self.pos_weight),
        )

        probs = torch.sigmoid(logits)
        intersection = (probs * target).sum(dim=(1, 2, 3))
        denom = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = 1.0 - ((2.0 * intersection + self.eps) / (denom + self.eps))
        dice = dice.mean()

        return self.loss_weight * (self.bce_weight * bce + self.dice_weight * dice)


@MODELS.register_module()
class WeightedMaskCrossEntropyLoss(nn.Module):
    """Binary mask BCEWithLogits loss with configurable foreground weight.

    This mirrors the current class-aware mask-head API while staying simple
    enough for the single-class setting we are optimizing now.
    """

    def __init__(
        self,
        pos_weight: float = 1.0,
        loss_weight: float = 1.0,
    ):
        super().__init__()
        self.pos_weight = pos_weight
        self.loss_weight = loss_weight

    def _select_logits(self, pred: torch.Tensor, labels: Optional[torch.Tensor]) -> torch.Tensor:
        if pred.dim() == 3:
            return pred.unsqueeze(1)
        if pred.dim() != 4:
            raise ValueError(f"Expected pred with 3 or 4 dims, got {tuple(pred.shape)}")
        if pred.size(1) == 1:
            return pred
        if labels is None:
            raise ValueError("labels are required when pred has multiple mask channels")
        batch_inds = torch.arange(pred.size(0), device=pred.device)
        return pred[batch_inds, labels].unsqueeze(1)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        logits = self._select_logits(pred, labels)
        if target.dim() == 3:
            target = target.unsqueeze(1)
        target = target.to(dtype=logits.dtype)

        loss = F.binary_cross_entropy_with_logits(
            logits,
            target,
            pos_weight=logits.new_tensor(self.pos_weight),
        )
        return self.loss_weight * loss
