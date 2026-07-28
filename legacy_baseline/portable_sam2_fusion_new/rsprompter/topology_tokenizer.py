"""
Topology Tokenizer for IIMR (Iterative Instance Memory Reasoning)

This module implements differentiable topology extraction operations:
1. Soft morphological operations (erosion, dilation)
2. Boundary extraction
3. Token generation for memory attention

Note: Skeleton extraction (SoftSkeletonExtractor) was removed because the
morphological skeletonization approach is designed for tubular/connected
structures (blood vessels, roads) and does not suit building instance
segmentation in remote sensing imagery.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SoftMorphology(nn.Module):
    """
    Differentiable morphological operations using pooling.

    Soft Erosion: Using AvgPool to approximate erosion
    Soft Dilation: Using MaxPool to approximate dilation
    """

    def __init__(
        self,
        kernel_size: int = 3,
        iterations: int = 1,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.iterations = iterations
        self.padding = kernel_size // 2

    def soft_erosion(self, x: Tensor) -> Tensor:
        """
        Soft erosion using average pooling.

        For binary masks, this approximates erosion by taking local minimum.
        Using (1 - AvgPool(1 - x)) gives a differentiable approximation.

        Args:
            x: Input tensor [B, 1, H, W] or [B, H, W]

        Returns:
            Eroded tensor
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)

        for _ in range(self.iterations):
            x = 1.0 - F.avg_pool2d(
                1.0 - x,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.padding,
                count_include_pad=False
            )

        return x

    def soft_dilation(self, x: Tensor) -> Tensor:
        """
        Soft dilation using max pooling.

        For binary masks, this approximates dilation by taking local maximum.

        Args:
            x: Input tensor [B, 1, H, W] or [B, H, W]

        Returns:
            Dilated tensor
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)

        for _ in range(self.iterations):
            x = F.max_pool2d(
                x,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.padding
            )

        return x

    def forward(self, x: Tensor, operation: str = 'dilation') -> Tensor:
        """
        Apply morphological operation.

        Args:
            x: Input tensor
            operation: 'erosion' or 'dilation'

        Returns:
            Result tensor
        """
        if operation == 'erosion':
            return self.soft_erosion(x)
        elif operation == 'dilation':
            return self.soft_dilation(x)
        else:
            raise ValueError(f"Unknown operation: {operation}")


class BoundaryExtractor(nn.Module):
    """
    Extract boundary from mask using differentiable operations.

    Boundary = Dilation - Erosion (morphological gradient)
    """

    def __init__(
        self,
        kernel_size: int = 3,
        erosion_iterations: int = 1,
        dilation_iterations: int = 1,
    ):
        super().__init__()
        self.morphology = SoftMorphology(
            kernel_size=kernel_size,
            iterations=1,
        )
        self.erosion_iterations = erosion_iterations
        self.dilation_iterations = dilation_iterations

    def forward(self, mask: Tensor) -> Tensor:
        """
        Extract boundary from mask.

        Args:
            mask: Input mask [B, H, W] or [B, 1, H, W]
                  Values should be in [0, 1] range

        Returns:
            boundary: Boundary mask [B, H, W]
        """
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)

        dilated = mask
        for _ in range(self.dilation_iterations):
            dilated = self.morphology.soft_dilation(dilated)

        eroded = mask
        for _ in range(self.erosion_iterations):
            eroded = self.morphology.soft_erosion(eroded)

        boundary = dilated - eroded

        return boundary.squeeze(1)


class TopologyTokenizer(nn.Module):
    """
    Topology Tokenizer for IIMR

    Extracts boundary features from masks and converts them to tokens
    for memory attention.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        kernel_size: int = 3,
        boundary_erosion_iter: int = 1,
        boundary_dilation_iter: int = 1,
        num_boundary_tokens: int = 64,
        token_encoder_type: str = 'linear',
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_boundary_tokens = num_boundary_tokens
        self.boundary_pool_size = self._get_pool_size(num_boundary_tokens)

        self.boundary_extractor = BoundaryExtractor(
            kernel_size=kernel_size,
            erosion_iterations=boundary_erosion_iter,
            dilation_iterations=boundary_dilation_iter,
        )

        if token_encoder_type == 'linear':
            self.boundary_encoder = nn.Sequential(
                nn.Conv2d(1, embed_dim // 2, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(embed_dim // 2, embed_dim, 1),
            )
        elif token_encoder_type == 'conv':
            self.boundary_encoder = nn.Sequential(
                nn.Conv2d(1, embed_dim // 2, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(embed_dim // 2, embed_dim, 3, padding=1),
            )
        else:
            raise ValueError(f"Unknown token_encoder_type: {token_encoder_type}")

        self.boundary_pos_embed = nn.Parameter(
            torch.randn(1, num_boundary_tokens, embed_dim) * 0.02
        )

        self.boundary_pool = nn.AdaptiveAvgPool2d(self.boundary_pool_size)

    @staticmethod
    def _get_pool_size(num_tokens: int) -> Tuple[int, int]:
        if num_tokens <= 0:
            return (1, 1)
        h = int(num_tokens ** 0.5)
        while h > 1 and num_tokens % h != 0:
            h -= 1
        w = max(1, num_tokens // h)
        return (h, w)

    def forward(self, mask: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Extract boundary tokens from mask.

        Args:
            mask: Input mask [B, 1, H, W] or [B, H, W]

        Returns:
            boundary_tokens: [B, N_b, C]
            boundary_mask: [B, H, W]
        """
        boundary_mask = self.boundary_extractor(mask)
        boundary_encoded = self.boundary_encoder(boundary_mask.unsqueeze(1))
        boundary_pooled = self.boundary_pool(boundary_encoded)
        boundary_tokens = boundary_pooled.flatten(2).transpose(1, 2)
        boundary_tokens = boundary_tokens + self.boundary_pos_embed[:, :boundary_tokens.shape[1]]
        return boundary_tokens, boundary_mask

    def get_boundary_tokens(self, mask: Tensor) -> Tuple[Tensor, Tensor]:
        """Alias for forward()."""
        return self.forward(mask)
