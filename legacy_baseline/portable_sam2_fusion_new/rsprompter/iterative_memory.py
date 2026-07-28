"""
Iterative Instance Memory Reasoning (IIMR) Module

This module implements the core iterative reasoning loop for single-image instance segmentation
using SAM2's memory mechanism as a pseudo-temporal reasoning engine.

Key Components:
1. IterativeMemoryReasoner: Main class that wraps the iterative inference loop
2. SimplifiedMemoryBank: Simplified memory bank for storing mask and topology features
3. UncertaintyPointGenerator: Generates dynamic prompts based on prediction uncertainty
"""

import sys
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SimplifiedMemoryBank:
    """
    Simplified Memory Bank for storing mask features across iterations.
    
    Unlike SAM2's video memory which stores features across frames,
    this memory bank stores features across iteration steps for a single image.
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        max_iterations: int = 5,
        device: Optional[torch.device] = None,
    ):
        self.embed_dim = embed_dim
        self.max_iterations = max_iterations
        self.device = device
        
        self.mask_features: Optional[Tensor] = None
        self.object_features: Optional[Tensor] = None
        self.topology_features: Optional[Tensor] = None
        self.mask_logits: Optional[Tensor] = None
        self.iteration_indices: List[int] = []
        self.current_iteration: int = 0
    
    def reset(self, batch_size: int = 1, spatial_size: Tuple[int, int] = (32, 32), in_channels: Optional[int] = None):
        """Reset memory bank for a new image."""
        self.current_iteration = 0
        self.iteration_indices = []
        self.mask_logits = None
        actual_channels = in_channels if in_channels is not None else self.embed_dim
        self.mask_features = torch.zeros(
            batch_size, self.max_iterations, actual_channels, *spatial_size,
            device=self.device
        )
        self.object_features = torch.zeros(
            batch_size, self.max_iterations, self.embed_dim,
            device=self.device
        )
        self.topology_features = torch.zeros(
            batch_size, self.max_iterations, 128, self.embed_dim,
            device=self.device
        )
    
    def update(
        self,
        iteration_idx: int,
        mask_features: Tensor,
        mask_logits: Optional[Tensor] = None,
        object_features: Optional[Tensor] = None,
        topology_features: Optional[Tensor] = None,
    ):
        """
        Update memory bank with features from current iteration.
        
        Args:
            iteration_idx: Current iteration index
            mask_features: Mask features [B, C, H, W] or [B, C, H*W]
            mask_logits: Geometric mask predictions (e.g. [B, 1, 256, 256])
            object_features: Object-level features [B, C]
            topology_features: Topology tokens [B, N, C]
        """
        if self.mask_features is None:
            batch_size = mask_features.shape[0]
            if mask_features.dim() == 4:
                spatial_size = mask_features.shape[-2:]
            else:
                spatial_size = (1, mask_features.shape[-1])
            self.reset(batch_size, spatial_size)
        
        if iteration_idx < self.max_iterations:
            if mask_features.dim() == 4:
                self.mask_features[:, iteration_idx] = mask_features
            else:
                h = w = int(mask_features.shape[-1] ** 0.5)
                self.mask_features[:, iteration_idx] = mask_features.reshape(
                    mask_features.shape[0], -1, h, w
                )
            
            if object_features is not None:
                self.object_features[:, iteration_idx] = object_features
            
            if mask_logits is not None:
                # Lazy initialization for mask_logits to adapt to actual prediction resolution (e.g. 256x256)
                if self.mask_logits is None:
                    _, num_classes, H, W = mask_logits.shape
                    self.mask_logits = torch.zeros(
                        mask_features.shape[0], self.max_iterations, num_classes, H, W,
                        device=self.device
                    )
                
                # Check shape matching (for unexpected cases like SAM2 decoder fallback low_res_masks)
                if mask_logits.shape[-2:] != self.mask_logits.shape[-2:]:
                    mask_logits = F.interpolate(mask_logits, size=self.mask_logits.shape[-2:], mode="bilinear", align_corners=False)
                    
                self.mask_logits[:, iteration_idx] = mask_logits
            
            if topology_features is not None:
                num_tokens = topology_features.shape[1]
                if num_tokens <= self.topology_features.shape[2]:
                    self.topology_features[:, iteration_idx, :num_tokens] = topology_features
            
            if iteration_idx not in self.iteration_indices:
                self.iteration_indices.append(iteration_idx)
            
            self.current_iteration = iteration_idx + 1
    
    def get_memory(
        self,
        up_to_iteration: Optional[int] = None
    ) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor]]:
        """
        Get memory features up to a certain iteration.
        
        Args:
            up_to_iteration: Get memory up to this iteration (exclusive)
                            If None, returns all stored memory
        
        Returns:
            mask_memory: [B, T, C, H, W]
            object_memory: [B, T, C]
            topology_memory: [B, T, N, C]
            mask_logits: [B, T, C_m, H_m, W_m] or None
        """
        if up_to_iteration is None:
            up_to_iteration = self.current_iteration
        
        valid_iterations = [i for i in self.iteration_indices if i < up_to_iteration]
        
        if not valid_iterations:
            return (
                self.mask_features[:, :0].clone(),
                self.object_features[:, :0].clone(),
                self.topology_features[:, :0].clone(),
                self.mask_logits[:, :0].clone() if self.mask_logits is not None else None,
            )
        
        # Return memory snapshots decoupled from bank storage to avoid autograd
        # version mismatch when bank is updated in-place in later iterations.
        mask_memory = self.mask_features[:, :len(valid_iterations)].clone()
        object_memory = self.object_features[:, :len(valid_iterations)].clone()
        topology_memory = self.topology_features[:, :len(valid_iterations)].clone()
        mask_logits_memory = (
            self.mask_logits[:, :len(valid_iterations)].clone()
            if self.mask_logits is not None
            else None
        )
        
        return mask_memory, object_memory, topology_memory, mask_logits_memory
    
    def get_attention_mask(
        self,
        up_to_iteration: Optional[int] = None
    ) -> Tensor:
        """
        Get attention mask for valid memory entries.
        
        Returns:
            attention_mask: [B, T] where 1 indicates valid memory
        """
        if up_to_iteration is None:
            up_to_iteration = self.current_iteration
        
        valid_iterations = [i for i in self.iteration_indices if i < up_to_iteration]
        mask = torch.zeros(self.mask_features.shape[0], self.max_iterations, device=self.device)
        
        for i in valid_iterations:
            mask[:, i] = 1.0
        
        return mask[:, :len(valid_iterations)]


class UncertaintyPointGenerator(nn.Module):
    """
    Generates dynamic prompt points based on prediction uncertainty.
    
    Identifies pixels where the model is most uncertain (logits close to 0)
    and generates positive/negative points for the next iteration.
    """
    
    def __init__(
        self,
        num_points: int = 8,
        balance_ratio: float = 0.5,
        min_point_distance: int = 0,
        use_boundary_uncertainty: bool = False,
        boundary_kernel_size: int = 3,
        boundary_mix_weight: float = 0.1,
        adaptive_balance: bool = False,
    ):
        super().__init__()
        self.num_points = num_points
        self.balance_ratio = balance_ratio
        self.min_point_distance = min_point_distance
        self.use_boundary_uncertainty = use_boundary_uncertainty
        self.boundary_kernel_size = boundary_kernel_size
        self.boundary_mix_weight = boundary_mix_weight
        self.adaptive_balance = adaptive_balance

    def _compute_adaptive_balance(self, mask_logits: Tensor) -> float:
        pos_region = mask_logits >= 0
        neg_region = mask_logits < 0
        pos_uncertainty = ((1.0 / (torch.abs(mask_logits) + 1e-6)) * pos_region.float()).sum()
        neg_uncertainty = ((1.0 / (torch.abs(mask_logits) + 1e-6)) * neg_region.float()).sum()
        total = pos_uncertainty + neg_uncertainty + 1e-8
        raw_ratio = (pos_uncertainty / total).item()
        return max(0.15, min(0.5, raw_ratio))

    def _select_diverse_indices(
        self,
        flat_scores: Tensor,
        mask: Tensor,
        num_select: int,
        width: int,
    ) -> List[int]:
        """Greedily pick low-score candidates while enforcing spatial diversity."""
        if num_select <= 0:
            return []

        valid_scores = flat_scores.clone()
        valid_scores[~mask.view(-1)] = float("inf")
        candidate_count = min(valid_scores.shape[0], max(num_select * 8, num_select))
        _, candidate_indices = torch.topk(valid_scores, candidate_count, largest=False)

        selected: List[int] = []
        min_dist_sq = float(self.min_point_distance * self.min_point_distance)

        for idx in candidate_indices.tolist():
            if not torch.isfinite(valid_scores[idx]):
                continue
            if self.min_point_distance > 0:
                y, x = divmod(idx, width)
                keep = True
                for prev_idx in selected:
                    py, px = divmod(prev_idx, width)
                    if float((x - px) ** 2 + (y - py) ** 2) < min_dist_sq:
                        keep = False
                        break
                if not keep:
                    continue
            selected.append(idx)
            if len(selected) >= num_select:
                break

        if len(selected) < num_select:
            for idx in candidate_indices.tolist():
                if not torch.isfinite(valid_scores[idx]) or idx in selected:
                    continue
                selected.append(idx)
                if len(selected) >= num_select:
                    break

        return selected[:num_select]
    
    def forward(
        self,
        mask_logits: Tensor,
        gt_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Generate uncertainty-based points.
        
        Args:
            mask_logits: Raw mask logits [B, H, W] or [B, 1, H, W]
                        (before sigmoid)
            gt_mask: Ground truth mask for determining point type [B, H, W]
                    If None, uses logits sign to determine type
        
        Returns:
            points: [B, N, 2] - (x, y) coordinates
            point_labels: [B, N] - 1 for positive, 0 for negative
        """
        if mask_logits.dim() == 4:
            mask_logits = mask_logits.squeeze(1)
        
        B, H, W = mask_logits.shape
        device = mask_logits.device
        
        if self.use_boundary_uncertainty:
            pred_mask_prob = torch.sigmoid(mask_logits)
            padding = self.boundary_kernel_size // 2
            dilated = F.max_pool2d(
                pred_mask_prob.unsqueeze(1),
                self.boundary_kernel_size, stride=1, padding=padding
            ).squeeze(1)
            eroded = -F.max_pool2d(
                -pred_mask_prob.unsqueeze(1),
                self.boundary_kernel_size, stride=1, padding=padding
            ).squeeze(1)
            boundary = dilated - eroded
            boundary_uncertainty = torch.abs(mask_logits) / (boundary + 1e-6)
            uncertainty = boundary_uncertainty + self.boundary_mix_weight * torch.abs(mask_logits)
        else:
            uncertainty = torch.abs(mask_logits)
        
        if gt_mask is not None:
            positive_mask = gt_mask > 0.5
            negative_mask = gt_mask <= 0.5
        else:
            positive_mask = mask_logits >= 0
            negative_mask = mask_logits < 0

        if self.adaptive_balance:
            balance_ratio = self._compute_adaptive_balance(mask_logits)
        else:
            balance_ratio = self.balance_ratio
        num_pos = max(1, int(self.num_points * balance_ratio))
        num_neg = self.num_points - num_pos
        
        points_list = []
        labels_list = []
        
        for b in range(B):
            batch_points = []
            batch_labels = []
            
            pos_indices = self._select_diverse_indices(
                flat_scores=uncertainty[b].view(-1),
                mask=positive_mask[b],
                num_select=num_pos,
                width=W,
            )

            for idx in pos_indices:
                y, x = divmod(idx, W)
                batch_points.append([x, y])
                batch_labels.append(1)

            neg_indices = self._select_diverse_indices(
                flat_scores=uncertainty[b].view(-1),
                mask=negative_mask[b],
                num_select=num_neg,
                width=W,
            )

            for idx in neg_indices:
                y, x = divmod(idx, W)
                batch_points.append([x, y])
                batch_labels.append(0)
            
            while len(batch_points) < self.num_points:
                batch_points.append([0, 0])
                batch_labels.append(-1)
            
            points_list.append(batch_points[:self.num_points])
            labels_list.append(batch_labels[:self.num_points])
        
        points = torch.tensor(points_list, device=device, dtype=torch.float32)
        point_labels = torch.tensor(labels_list, device=device, dtype=torch.long)
        
        return points, point_labels


class IterativeMemoryReasoner(nn.Module):
    """
    Iterative Instance Memory Reasoner (IIMR)
    
    This module implements the core iterative reasoning loop that:
    1. Takes initial features and prompts
    2. Iteratively refines predictions using memory
    3. Optionally uses topology tokens for structure-aware reasoning
    
    The key insight is that SAM2's memory mechanism, designed for video,
    can be repurposed for iterative refinement on a single image by treating
    each iteration as a "pseudo-frame".
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_iterations: int = 3,
        use_topology_memory: bool = False,
        topology_to_decoder: bool = False,
        use_dynamic_prompting: bool = True,
        num_uncertainty_points: int = 8,
        dynamic_balance_ratio: float = 0.5,
        dynamic_min_point_distance: int = 0,
        dynamic_start_iteration: int = 1,
        memory_attention_heads: int = 8,
        memory_attention_chunk_size: int = 128,
        min_topology_memory_iteration: int = 1,
        topology_memory_conf_threshold: float = 0.65,
        use_roi_pos_encoding: bool = True,
        use_geometry_aware_kv: bool = False,
        in_channels: int = 512,
        use_boundary_uncertainty: bool = False,
        boundary_kernel_size: int = 3,
        boundary_mix_weight: float = 0.1,
        adaptive_balance: bool = False,
        use_learnable_residual: bool = False,
        residual_weight_init: float = 0.02,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_iterations = num_iterations
        self.use_topology_memory = use_topology_memory
        self.topology_to_decoder = topology_to_decoder
        self.use_geometry_aware_kv = use_geometry_aware_kv
        self.use_dynamic_prompting = use_dynamic_prompting
        self.dynamic_balance_ratio = dynamic_balance_ratio
        dynamic_min_point_distance = dynamic_min_point_distance
        self.dynamic_start_iteration = dynamic_start_iteration
        self.min_topology_memory_iteration = min_topology_memory_iteration
        self.topology_memory_conf_threshold = topology_memory_conf_threshold
        self.memory_attention_chunk_size = max(1, int(memory_attention_chunk_size))
        self.use_learnable_residual = use_learnable_residual
        self.residual_weight_init = float(
            min(max(residual_weight_init, 1e-4), 1.0 - 1e-4)
        )

        # Memory Bank
        self.memory_bank = SimplifiedMemoryBank(
            embed_dim=embed_dim,
            max_iterations=num_iterations + 1,
        )

        # Dynamic Prompting
        if use_dynamic_prompting:
            self.uncertainty_generator = UncertaintyPointGenerator(
                num_points=num_uncertainty_points,
                balance_ratio=dynamic_balance_ratio,
                min_point_distance=dynamic_min_point_distance,
                use_boundary_uncertainty=use_boundary_uncertainty,
                boundary_kernel_size=boundary_kernel_size,
                boundary_mix_weight=boundary_mix_weight,
                adaptive_balance=adaptive_balance,
            )

        # Topology Tokenizer (only when geometry-aware KV injection is enabled)
        if self.use_geometry_aware_kv or self.use_topology_memory:
            from .topology_tokenizer import TopologyTokenizer
            self.topology_tokenizer = TopologyTokenizer(
                embed_dim=embed_dim,
                num_boundary_tokens=8,
            )

        # Memory Attention (简化版，使用 PyTorch 原生 MultiheadAttention)
        self.memory_attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=memory_attention_heads,
            batch_first=True,
        )

        if self.use_learnable_residual:
            init_logit = math.log(self.residual_weight_init / (1.0 - self.residual_weight_init))
            self.residual_weight = nn.Parameter(torch.tensor(init_logit, dtype=torch.float32))

        # ROI Spatial Position Encoder (ReZero-style gating: MLP正常初始化, scale=0保证warm start)
        self.use_roi_pos_encoding = use_roi_pos_encoding
        # Always initialize cache so reset_memory is safe even when ROI PE is disabled.
        self._roi_pos_cache: Dict[Tuple[int, int, str], Tensor] = {}
        if self.use_roi_pos_encoding:
            self.roi_pos_encoder = nn.Sequential(
                nn.Linear(2, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, self.embed_dim),
            )
            # ReZero scale: init=1e-2 to avoid cold-start (grad through attention softmax is weak)
            self.roi_pos_scale = nn.Parameter(torch.full((1,), 1e-2))
            # Cache raw normalized coordinates per (H, W, device).
            # We do not cache encoded outputs so gradients flow through the MLP.

        # Iteration weights (可选，用于加权融合不同迭代的结果)
        self.iteration_weights = nn.Parameter(
            torch.ones(num_iterations) / num_iterations
        )

        # Projection layers for channel dimension mismatch
        if in_channels != embed_dim:
            self.input_proj = nn.Linear(in_channels, embed_dim)
            self.output_proj = nn.Linear(embed_dim, in_channels)
        else:
            self.input_proj = None
            self.output_proj = None
        self._memory_attention_oom_warned = False

    def expand_pseudo_temporal_sequence(
        self,
        features: Tensor,
        num_steps: int = 1,
    ) -> Tensor:
        """Backward-compatible helper for legacy call-sites.

        Some training branches still call this method before iterative memory
        attention. We keep a lightweight implementation so older code paths do
        not crash while preserving current single-step behavior.

        Args:
            features: Feature tensor in shape [B, C, H, W] or [B, T, C, H, W].
            num_steps: Target pseudo-temporal length when input is 4D.

        Returns:
            Tensor shaped [B, T, C, H, W].
        """
        if features.dim() == 5:
            return features

        if features.dim() != 4:
            raise ValueError(
                f"expand_pseudo_temporal_sequence expects 4D/5D input, got {features.shape}"
            )

        steps = max(1, int(num_steps))
        if steps == 1:
            return features.unsqueeze(1)

        return features.unsqueeze(1).repeat(1, steps, 1, 1, 1)

    def reset_memory(self, batch_size: int, spatial_size: Tuple[int, int], device: torch.device, in_channels: Optional[int] = None):
        """Reset memory bank for new inference."""
        self.memory_bank.device = device
        self.memory_bank.reset(batch_size, spatial_size, in_channels)

        self._roi_pos_cache.clear()

    def _get_roi_pos_embed(self, H: int, W: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Generate ROI-relative position embeddings in embed_dim space."""
        cache_key = (H, W, str(device))
        if cache_key not in self._roi_pos_cache:
            coords_h = torch.linspace(0, 1, H, device=device, dtype=torch.float32)
            coords_w = torch.linspace(0, 1, W, device=device, dtype=torch.float32)
            grid_y, grid_x = torch.meshgrid(coords_h, coords_w, indexing='ij')
            roi_coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
            self._roi_pos_cache[cache_key] = roi_coords

        roi_coords = self._roi_pos_cache[cache_key].to(device=device, dtype=dtype)
        return self.roi_pos_encoder(roi_coords)

    def forward_iteration(
        self,
        current_features: Tensor,
        iteration_idx: int,
        prompt_embeddings: Optional[Tensor] = None,
        positional_embeddings: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Execute one iteration of memory-enhanced reasoning.
        
        Args:
            current_features: Current frame features [B, C, H, W]
            iteration_idx: Current iteration index
            prompt_embeddings: Optional prompt embeddings
            positional_embeddings: Positional embeddings for attention
        
        Returns:
            refined_features: Memory-enhanced features [B, C, H, W]
        """
        if iteration_idx == 0:
            return current_features

        mask_memory, object_memory, topology_memory, mask_logits_memory = self.memory_bank.get_memory(
            up_to_iteration=iteration_idx
        )
        
        if mask_memory.shape[1] == 0:
            return current_features
        
        B, C, H, W = current_features.shape
        T = mask_memory.shape[1]
        _, _, mem_C, mem_H, mem_W = mask_memory.shape
        
        current_flat = current_features.flatten(2).transpose(1, 2)
        memory_flat = mask_memory.flatten(3)
        memory_flat = memory_flat.permute(0, 1, 3, 2).reshape(B, T * mem_H * mem_W, mem_C)
        
        # Geometry-Aware Memory: Extract boundary tokens from historical mask_logits
        if self.use_geometry_aware_kv and mask_logits_memory is not None and getattr(self, 'topology_tokenizer', None) is not None:
            # Flatten T dimension into B dimension for topology extraction
            # mask_logits_memory: [B, T, C_m, H_m, W_m] -> [(B*T), C_m, H_m, W_m]
            _, _, C_m, H_m, W_m = mask_logits_memory.shape
            mask_logits_flat = mask_logits_memory.reshape(-1, C_m, H_m, W_m)

            # Convert to prob before tokenizer
            mask_probs = torch.sigmoid(mask_logits_flat)
            # Take mean across classes/channels. Usually C_m = 1 for binary instance segmentation mask
            mask_probs_single = mask_probs.mean(dim=1, keepdim=True)

            # Fetch boundary tokens: [B*T, N_b, embed_dim]
            bound_toks, _ = self.topology_tokenizer(mask_probs_single)

            if bound_toks is not None:
                combined_topo_len = bound_toks.shape[1]
                bound_toks = bound_toks.reshape(B, T * combined_topo_len, -1)

                # Align embed_dim if necessary
                topo_C = bound_toks.shape[-1]
                if topo_C != mem_C:
                    bound_toks = F.pad(bound_toks, (0, mem_C - topo_C))

                # Inject boundary tokens into attention Key/Value stream
                memory_flat = torch.cat([memory_flat, bound_toks], dim=1)

        if self.use_topology_memory and topology_memory.shape[1] > 0:
            topo_C = topology_memory.shape[-1]
            topology_flat = topology_memory.reshape(B, T * topology_memory.shape[2], topo_C)
            if topo_C != mem_C:
                topology_flat = F.pad(topology_flat, (0, mem_C - topo_C))
            memory_flat = torch.cat([memory_flat, topology_flat], dim=1)
        
        if self.input_proj is not None:
            current_flat = self.input_proj(current_flat)
            memory_flat = self.input_proj(memory_flat)

        # --- ROI Spatial Position Encoding (ReZero gated) ---
        if self.use_roi_pos_encoding:
            pos_scale = self.roi_pos_scale  # learnable PE gate (warm-start controlled by init)

            current_pos_embed = self._get_roi_pos_embed(
                H, W, current_flat.device, current_flat.dtype
            )
            current_flat = current_flat + pos_scale * current_pos_embed.unsqueeze(0)

            # Apply PE only to spatially aligned memory tokens.
            # Topology tokens appended later in memory_flat are intentionally left unchanged.
            spatial_len = T * mem_H * mem_W
            memory_pos_embed = self._get_roi_pos_embed(
                mem_H, mem_W, memory_flat.device, memory_flat.dtype
            )
            memory_pos_embed = memory_pos_embed.repeat(T, 1)

            memory_spatial = memory_flat[:, :spatial_len, :] + pos_scale * memory_pos_embed.unsqueeze(0)
            if spatial_len < memory_flat.shape[1]:
                memory_flat = torch.cat([memory_spatial, memory_flat[:, spatial_len:, :]], dim=1)
            else:
                memory_flat = memory_spatial

        # Memory attention using PyTorch native MultiheadAttention.
        # Chunk over ROI batch to avoid OOM when proposal count is large.
        try:
            if B <= self.memory_attention_chunk_size:
                attended, _ = self.memory_attention(current_flat, memory_flat, memory_flat)
            else:
                attended_chunks: List[Tensor] = []
                for start in range(0, B, self.memory_attention_chunk_size):
                    end = min(start + self.memory_attention_chunk_size, B)
                    chunk_attended, _ = self.memory_attention(
                        current_flat[start:end],
                        memory_flat[start:end],
                        memory_flat[start:end],
                    )
                    attended_chunks.append(chunk_attended)
                attended = torch.cat(attended_chunks, dim=0)
        except torch.OutOfMemoryError:
            # Keep training alive if memory attention becomes too expensive.
            if not self._memory_attention_oom_warned:
                print("[IIMR] memory attention OOM, fallback to identity features for this step")
                self._memory_attention_oom_warned = True
            return current_features

        if self.output_proj is not None:
            attended = self.output_proj(attended)
        
        refined_features = attended.transpose(1, 2).view(B, C, H, W)
        
        return refined_features
    
    def update_memory(
        self,
        iteration_idx: int,
        mask_features: Tensor,
        mask_logits: Optional[Tensor] = None,
        object_features: Optional[Tensor] = None,
        topology_features: Optional[Tensor] = None,
    ):
        """Update memory bank with current iteration's features."""
        allow_topology_write = (
            self.use_topology_memory
            and topology_features is not None
            and iteration_idx >= self.min_topology_memory_iteration
        )
        if allow_topology_write and mask_logits is not None:
            mask_probs = torch.sigmoid(mask_logits)
            per_roi_conf = torch.maximum(mask_probs, 1.0 - mask_probs).mean(dim=(-1, -2, -3))
            allow_topology_write = bool(
                (per_roi_conf >= self.topology_memory_conf_threshold).any().item()
            )

        stored_topology = topology_features if allow_topology_write else None
        self.memory_bank.update(iteration_idx, mask_features, mask_logits, object_features, stored_topology)
    
    def generate_dynamic_prompts(
        self,
        mask_logits: Tensor,
        gt_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Generate dynamic prompts based on uncertainty."""
        if not self.use_dynamic_prompting:
            return None, None
        
        return self.uncertainty_generator(mask_logits, gt_mask)
    
    def extract_topology_tokens(self, mask: Tensor) -> Tensor:
        """Extract boundary tokens from mask for topology memory."""
        if not self.use_topology_memory:
            return None

        tokens, _ = self.topology_tokenizer(mask)
        return tokens
    
    def forward(
        self,
        initial_features: Tensor,
        mask_decoder: nn.Module,
        prompt_encoder: nn.Module,
        initial_prompts: Optional[Dict] = None,
        positional_embeddings: Optional[Tensor] = None,
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """
        Execute the full iterative reasoning loop.
        
        Args:
            initial_features: Initial image features [B, C, H, W]
            mask_decoder: SAM mask decoder module
            prompt_encoder: SAM prompt encoder module
            initial_prompts: Initial prompts (bboxes, points, etc.)
            positional_embeddings: Positional embeddings
        
        Returns:
            all_masks: List of mask predictions from each iteration
            all_iou_preds: List of IoU predictions from each iteration
        """
        B, C, H, W = initial_features.shape
        device = initial_features.device
        
        self.reset_memory(B, (H, W), device)
        
        all_masks = []
        all_iou_preds = []
        
        current_features = initial_features
        
        for t in range(self.num_iterations):
            if t > 0:
                current_features = self.forward_iteration(
                    current_features, t, positional_embeddings=positional_embeddings
                )
            
            if t == 0 and initial_prompts is not None:
                sparse_embeddings = initial_prompts.get('sparse_embeddings')
                dense_embeddings = initial_prompts.get('dense_embeddings')
            else:
                if (
                    self.use_dynamic_prompting
                    and len(all_masks) > 0
                    and t >= self.dynamic_start_iteration
                ):
                    prev_mask = all_masks[-1]
                    points, labels = self.generate_dynamic_prompts(prev_mask)
                    sparse_embeddings = prompt_encoder(points=points, labels=labels)
                    dense_embeddings = None
                else:
                    sparse_embeddings = None
                    dense_embeddings = None
            
            low_res_masks, iou_predictions = mask_decoder(
                current_features,
                positional_embeddings,
                sparse_embeddings,
                dense_embeddings,
            )
            
            all_masks.append(low_res_masks)
            all_iou_preds.append(iou_predictions)
            
            topology_tokens = None
            if self.use_topology_memory:
                mask_probs = torch.sigmoid(low_res_masks)
                topology_tokens = self.extract_topology_tokens(mask_probs.squeeze(1))
            
            self.update_memory(t, low_res_masks, topology_features=topology_tokens)
        
        return all_masks, all_iou_preds
