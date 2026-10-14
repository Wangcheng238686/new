"""Explicit RGB-D payload flow through the existing two-stage detector."""

import copy

import torch
import torch.nn.functional as F
from torch import nn
from mmdet.registry import MODELS

from .anchor_detector import RSPrompterAnchor
from ..uav.cvt import DepthCVT
from ..uav.fusion import ConfidenceFusion
from ..uav.projection import BEVToSatellite


@MODELS.register_module()
class RSPrompterUAVDual(RSPrompterAnchor):
    def __init__(
        self,
        uav_backbone,
        uav_bev,
        fpn_fusion=None,
        sam_fusion=None,
        uav_encoder_chunk=5,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if fpn_fusion is None and sam_fusion is None:
            raise ValueError("A dual detector requires a fusion route")
        self.uav_backbone = MODELS.build(copy.deepcopy(uav_backbone))
        self.uav_bev = DepthCVT(**uav_bev)
        self.bev_projection = BEVToSatellite(self.uav_bev.geometry["image_size"])
        self.uav_encoder_chunk = int(uav_encoder_chunk)
        if self.uav_encoder_chunk < 1:
            raise ValueError("UAV encoder chunk must be positive")
        self.fpn_fusion = None
        self.fpn_levels = ()
        if fpn_fusion is not None:
            cfg = dict(fpn_fusion)
            self.fpn_levels = tuple(cfg.pop("levels", (1, 2)))
            self.fpn_fusion = nn.ModuleList(
                [
                    ConfidenceFusion(satellite_channels=512, **cfg)
                    for _ in self.fpn_levels
                ]
            )
        self.sam_fusion = (
            ConfidenceFusion(satellite_channels=256, **sam_fusion)
            if sam_fusion is not None
            else None
        )
        self.register_buffer(
            "uav_mean",
            torch.tensor([123.675, 116.28, 103.53]).reshape(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "uav_std",
            torch.tensor([58.395, 57.12, 57.375]).reshape(1, 3, 1, 1),
            persistent=False,
        )
        self.uav_intervention = "correct"
        self._uav_telemetry = {}

    def _extract_dual_feat(self, batch_inputs, samples):
        x, embedding, pe, high = super().extract_feat(batch_inputs)
        payloads = []
        for sample in samples:
            if not hasattr(sample, "uav"):
                raise ValueError(
                    "Dual prediction/training requires the UAV RGB-D payload"
                )
            payloads.append(sample.uav)
        if self.uav_intervention == "zero":
            self._uav_telemetry = {}
            return x, embedding, pe, high
        rgb = torch.stack([p.rgb for p in payloads]).flatten(0, 1)
        if rgb.shape[-2:] != (self.uav_bev.geometry["image_size"],) * 2:
            raise ValueError("Unexpected UAV RGB input resolution")
        rgb = (rgb - self.uav_mean) / self.uav_std
        features = []
        for start in range(0, len(rgb), self.uav_encoder_chunk):
            _, pyramid = self.uav_backbone(rgb[start : start + self.uav_encoder_chunk])
            detailed = pyramid[1]
            semantic = F.interpolate(
                pyramid[2],
                size=detailed.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            features.append(detailed + semantic)
        feature = torch.cat(features).reshape(
            len(payloads), 5, 256, *features[0].shape[-2:]
        )
        bev = self.uav_bev(feature, payloads)
        if self.uav_intervention == "shift":
            bev = bev.roll(shifts=(16, 16), dims=(-2, -1))
        elif self.uav_intervention != "correct":
            raise ValueError(f"Unknown intervention: {self.uav_intervention}")
        stats = {}
        x = list(x)
        if self.fpn_fusion is not None:
            for level, fusion in zip(self.fpn_levels, self.fpn_fusion):
                uav, confidence, coverage = self.bev_projection(
                    bev, payloads, x[level].shape[-2:]
                )
                x[level], local = fusion(x[level], uav, confidence, coverage)
                stats.update(
                    {f"UAV/fpn_s{2**(level+2)}/{k}": v for k, v in local.items()}
                )
        if self.sam_fusion is not None:
            uav, confidence, coverage = self.bev_projection(
                bev, payloads, embedding.shape[-2:]
            )
            embedding, local = self.sam_fusion(embedding, uav, confidence, coverage)
            stats.update({f"UAV/sam/{k}": v for k, v in local.items()})
        for key in ("observation_support", "height_valid"):
            stats[f"UAV/{key}"] = (
                torch.stack([getattr(p, key).float().mean() for p in payloads])
                .mean()
                .detach()
            )
        stats["UAV/candidate_available"] = (
            torch.stack(
                [(p.candidate_index >= 0).any(dim=1).float().mean() for p in payloads]
            )
            .mean()
            .detach()
        )
        stats.update(
            {
                f"UAV/attention/{key}": value
                for key, value in self.uav_bev.diagnostics.items()
            }
        )
        confidence = torch.cat([p.bev_confidence[p.height_valid] for p in payloads])
        if len(confidence):
            for key, column, scale in [
                ("upper_fraction", 1, 1.0),
                ("selected_bucket_sigma_m", 2, self.uav_bev.geometry["height_bin_m"]),
                ("nearest_xy_m", 3, self.uav_bev.geometry["radius_m"]),
            ]:
                stats[f"UAV/geometry/{key}"] = (
                    confidence[:, column].mean() * scale
                ).detach()
        self._uav_telemetry = stats
        return tuple(x), embedding, pe, high

    def loss(self, batch_inputs, batch_data_samples):
        x, embedding, pe, high = self._extract_dual_feat(
            batch_inputs, batch_data_samples
        )
        rpn_samples = copy.deepcopy(batch_data_samples)
        for sample in rpn_samples:
            sample.gt_instances.labels = torch.zeros_like(sample.gt_instances.labels)
        proposal_cfg = self.train_cfg.get("rpn_proposal", self.test_cfg.rpn)
        rpn_loss, proposals = self.rpn_head.loss_and_predict(
            x, rpn_samples, proposal_cfg=proposal_cfg
        )
        losses = {
            f"rpn_{key}" if "loss" in key and "rpn" not in key else key: value
            for key, value in rpn_loss.items()
        }
        losses.update(
            self.roi_head.loss(
                x,
                proposals,
                batch_data_samples,
                image_embeddings=embedding,
                image_positional_embeddings=pe,
                high_res_features=high,
            )
        )
        return losses

    def predict(self, batch_inputs, batch_data_samples, rescale=True):
        x, embedding, pe, high = self._extract_dual_feat(
            batch_inputs, batch_data_samples
        )
        if batch_data_samples[0].get("proposals", None) is None:
            proposals = self.rpn_head.predict(x, batch_data_samples, rescale=False)
        else:
            proposals = [s.proposals for s in batch_data_samples]
        results = self.roi_head.predict(
            x,
            proposals,
            batch_data_samples,
            rescale=rescale,
            image_embeddings=embedding,
            image_positional_embeddings=pe,
            high_res_features=high,
        )
        return self.add_pred_to_datasample(batch_data_samples, results)

    def get_telemetry(self):
        return {**super().get_telemetry(), **self._uav_telemetry}
