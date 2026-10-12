"""RoI head: sampling, the mask-forward path, loss wiring, and prediction.

Owns everything that happens around the mask head during training and
inference: assigner/sampler, bbox loss (optional fp32), the prompt-box
contract (proposal boxes only), the per-RoI mask forward, and aggregation of
the three a3sgm training losses (final mask, decoder-tail margin, coarse
auxiliary) plus their telemetry.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from torch import Tensor

from mmdet.models import StandardRoIHead
from mmdet.models.task_modules import SamplingResult
from mmdet.models.utils import empty_instances, unpack_gt_instances
from mmdet.registry import MODELS
from mmdet.structures import DetDataSample, SampleList
from mmdet.structures.bbox import bbox2roi
from mmdet.utils import InstanceList


@MODELS.register_module()
class RSPrompterAnchorRoIPromptHead(StandardRoIHead):
    def __init__(
        self,
        with_extra_pe=False,
        bbox_train_fp32=False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.bbox_train_fp32 = bool(bbox_train_fp32)
        if with_extra_pe:
            # Parameter-free spatial sine pattern added to every feature level
            # before assigner/RPN/RoI consumption. Both paper checkpoints were
            # trained with it on; it has no state-dict keys, so only forward
            # equivalence can guard it.
            out_channels = self.bbox_roi_extractor.out_channels
            positional_encoding = dict(num_feats=out_channels // 2, normalize=True)
            from mmdet.models import SinePositionalEncoding

            self.extra_pe = SinePositionalEncoding(**positional_encoding)

    def _inject_extra_pe(self, x: Tuple[Tensor]) -> Tuple[Tensor]:
        if not hasattr(self, "extra_pe"):
            return x
        bs, _, h, w = x[0].shape
        mask_pe = torch.zeros((bs, h, w), device=x[0].device, dtype=torch.bool)
        img_feats_pe = self.extra_pe(mask_pe)
        return tuple(
            feat + F.interpolate(
                img_feats_pe, size=feat.shape[-2:], mode="bilinear",
                align_corners=False,
            )
            for feat in x
        )

    def get_telemetry(self) -> Dict[str, object]:
        stats: Dict[str, object] = {}
        mask_head = getattr(self, "mask_head", None)
        if mask_head is not None and hasattr(mask_head, "get_forward_telemetry"):
            stats.update(mask_head.get_forward_telemetry())
        for key, value in (getattr(self, "_last_coarse_stats", {}) or {}).items():
            stats[f"COARSE/{key}"] = value
        stats.update(getattr(self, "_last_box_jitter_stats", {}) or {})
        stats.update(self._last_prompt_box_stats or {})

        # Convert count/sum diagnostics into interpretable local-rank ratios.
        def _ratio(out_key: str, num_key: str, den_key: str):
            if num_key not in stats or den_key not in stats:
                return
            num, den = stats[num_key], stats[den_key]
            if torch.is_tensor(num) and torch.is_tensor(den):
                stats[out_key] = num.detach().float() / den.detach().float().clamp_min(1.0)

        _ratio("SP/pos_in_pred_fg", "SP/pos_in_pred_fg_count", "SP/valid_pos_count")
        _ratio("SP/neg_in_pred_bg", "SP/neg_in_pred_bg_count", "SP/valid_neg_count")
        _ratio("SP/mean_point_distance", "SP/point_distance_sum", "SP/point_pair_count")
        _ratio("SP/point_overlap_ratio", "SP/point_overlap_count", "SP/point_pair_count")
        _ratio("JITTER/area_ratio_mean", "JITTER/area_ratio_sum", "JITTER/area_ratio_count")
        _ratio("SP/p1_valid_ratio", "SP/p1_valid_count", "SP/roi_count")
        _ratio("SP/p2_valid_ratio", "SP/p2_valid_count", "SP/roi_count")
        _ratio("SP/n1_valid_ratio", "SP/n1_valid_count", "SP/roi_count")
        _ratio("SP/n2_valid_ratio", "SP/n2_valid_count", "SP/roi_count")
        _ratio("SP/empty_ratio", "SP/empty_count", "SP/roi_count")
        _ratio("SP/full_ratio", "SP/full_count", "SP/roi_count")
        _ratio("SP/all_invalid_ratio", "SP/all_invalid_count", "SP/roi_count")
        _ratio("SP/p1_in_gt_fg", "SP/p1_in_gt_fg_count", "SP/p1_gt_valid_count")
        _ratio("SP/p2_in_gt_fg", "SP/p2_in_gt_fg_count", "SP/p2_gt_valid_count")
        _ratio("SP/n1_in_gt_bg", "SP/n1_in_gt_bg_count", "SP/n1_gt_valid_count")
        _ratio("SP/n2_in_gt_bg", "SP/n2_in_gt_bg_count", "SP/n2_gt_valid_count")
        return stats

    def bbox_loss(
        self,
        x: Tuple[Tensor],
        sampling_results: List[SamplingResult],
    ) -> dict:
        if not self.bbox_train_fp32:
            return super().bbox_loss(x, sampling_results)

        rois = bbox2roi([res.priors for res in sampling_results])
        with torch.autocast(device_type=x[0].device.type, enabled=False):
            fp32_x = tuple(feat.float() for feat in x)
            bbox_results = self._bbox_forward(fp32_x, rois.float())
            bbox_loss_and_target = self.bbox_head.loss_and_target(
                cls_score=bbox_results["cls_score"],
                bbox_pred=bbox_results["bbox_pred"],
                rois=rois.float(),
                sampling_results=sampling_results,
                rcnn_train_cfg=self.train_cfg,
            )
        bbox_results.update(loss_bbox=bbox_loss_and_target["loss_bbox"])
        return bbox_results

    def _mask_forward(
        self,
        x: Tuple[Tensor],
        rois: Tensor,
        image_embeddings=None,
        image_positional_embeddings=None,
        high_res_features=None,
    ) -> dict:
        # A dedicated mask_roi_extractor is part of the architecture contract;
        # the shared-extractor variant belonged to removed baseline arms.
        mask_feats = self.mask_roi_extractor(x[: self.mask_roi_extractor.num_inputs], rois)
        head_out = self.mask_head(
            mask_feats,
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
            roi_img_ids=rois[:, 0],
            high_res_features=high_res_features,
            boxes=rois[:, 1:],
            p2_feature=x[0],
            prompt_rois=rois,
            renderer_features=x,
        )
        return dict(
            mask_preds=head_out.mask_logits,
            coarse_outputs=head_out.coarse_outputs,
            tail_outputs=head_out.tail_outputs,
        )

    def mask_loss(
        self,
        x: Tuple[Tensor],
        sampling_results: List[SamplingResult],
        batch_gt_instances: InstanceList,
        image_embeddings=None,
        image_positional_embeddings=None,
        high_res_features=None,
        batch_img_metas: Optional[List[dict]] = None,
    ) -> dict:
        original_pos_rois = bbox2roi([res.pos_priors for res in sampling_results])
        if len(original_pos_rois) == 0:
            return self._empty_rank_losses(x)

        jitter_cfg = dict(getattr(self.mask_head, "box_prompt_cfg", {}) or {})
        if float(jitter_cfg.get("jitter_prob", 0.0)) > 0.0:
            raise ValueError(
                "box prompt jitter belonged to a removed arm and cannot be "
                "enabled in the a3sgm-only codebase"
            )
        # Box/prompt telemetry: proposal boxes are the only prompt source, so
        # only the RoI count is real; the fallback/IoU placeholders of the
        # removed refined-box machinery are gone.
        self._last_prompt_box_stats = {
            "BOX/num_rois": original_pos_rois.new_tensor(float(len(original_pos_rois))),
        }
        zero = original_pos_rois.new_zeros(())
        self._last_box_jitter_stats = {
            "JITTER/num_rois": original_pos_rois.new_tensor(float(len(original_pos_rois))),
            "JITTER/num_jittered": zero,
            "JITTER/mean_iou": original_pos_rois.new_tensor(1.0),
            "JITTER/fallback_count": zero,
            "JITTER/area_ratio_sum": original_pos_rois.new_tensor(float(len(original_pos_rois))),
            "JITTER/area_ratio_count": original_pos_rois.new_tensor(float(len(original_pos_rois))),
        }

        pos_rois = original_pos_rois
        pos_counts = [res.pos_priors.shape[0] for res in sampling_results]
        prompt_pos_priors = list(pos_rois[:, 1:].split(pos_counts, dim=0))
        # From this point onward every mask/prompt operation uses pos_rois.
        # Sampling assignments and bbox losses remain tied to sampling_results.
        mask_results = self._mask_forward(
            x,
            pos_rois,
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
            high_res_features=high_res_features,
        )

        mask_loss_and_target = self.mask_head.loss_and_target(
            mask_preds=mask_results["mask_preds"],
            sampling_results=sampling_results,
            batch_gt_instances=batch_gt_instances,
            rcnn_train_cfg=self.train_cfg,
            final_mask_pos_priors=prompt_pos_priors,
        )
        losses = dict(mask_loss_and_target["loss_mask"])
        telemetry = dict(mask_loss_and_target.get("debug_stats", {}))
        self._apply_tail_loss(
            losses, telemetry, mask_results, mask_loss_and_target,
            sampling_results, batch_gt_instances, prompt_pos_priors,
        )
        self._apply_coarse_loss(
            losses, telemetry, mask_results,
            sampling_results, batch_gt_instances, prompt_pos_priors,
        )
        # Loss-vs-telemetry channel split: the returned dict contains ONLY
        # backward-able loss tensors; every diagnostic tensor rides the
        # telemetry channel so callers never filter by key-name substrings.
        return losses, telemetry

    def _empty_rank_losses(self, x: Tuple[Tensor]) -> dict:
        """Loss bookkeeping for a rank with no positive proposals.

        Do not leak diagnostics from the previous batch, and match the UDPR
        DDP collectives taken on non-empty ranks — without them a sparse rank
        deadlocks its peers while they normalize the margin point loss.
        """
        zero = x[0].new_zeros(())
        self._last_coarse_stats = {
            "dice": zero,
            "iou": zero,
            "boundary_tp": zero,
            "boundary_fp": zero,
            "boundary_fn": zero,
        }
        self._last_prompt_box_stats = {"BOX/num_rois": zero}
        self._last_box_jitter_stats = {
            "JITTER/num_rois": zero,
            "JITTER/num_jittered": zero,
            "JITTER/mean_iou": zero,
            "JITTER/fallback_count": zero,
            "JITTER/area_ratio_sum": zero,
            "JITTER/area_ratio_count": zero,
        }
        losses = dict(loss_mask=0 * x[0].sum())
        telemetry: Dict[str, object] = {}
        if getattr(self.mask_head, "decoder_tail_enabled", False):
            # Mirror the non-empty rank's two _global_tail_bce collectives so
            # a sparse rank neither deadlocks its peers nor misreports fields.
            tail_loss, tail_stats = self.mask_head.decoder_tail_refiner.empty_point_loss(zero)
            losses["loss_decoder_tail"] = tail_loss
            telemetry.update(tail_stats)
            self.mask_head._forward_telemetry = dict(tail_stats)
        return losses, telemetry

    def _apply_tail_loss(
        self,
        losses: dict,
        telemetry: dict,
        mask_results: dict,
        mask_loss_and_target: dict,
        sampling_results: List[SamplingResult],
        batch_gt_instances: InstanceList,
        prompt_pos_priors: List[Tensor],
    ) -> None:
        tail_outputs = mask_results.get("tail_outputs")
        if tail_outputs is None:
            return
        if self.mask_head.final_mask_coordinate_mode == "full_image":
            tail_targets = mask_loss_and_target["mask_targets"]
        else:
            # roi_local keeps the historical final-mask targets; the tail's
            # selected-point indices live on the decoder's native ROI grid, so
            # its point loss needs per-RoI GT crops at that native resolution.
            tail_targets = self.mask_head.get_coarse_targets(
                sampling_results,
                batch_gt_instances,
                mask_size=tail_outputs["base_logits_detached"].shape[-2:],
                prompt_pos_priors=prompt_pos_priors,
            )
        tail_loss, tail_stats = self.mask_head.decoder_tail_refiner.point_loss(
            tail_outputs, tail_targets
        )
        losses["loss_decoder_tail"] = tail_loss
        telemetry.update(tail_stats)
        # Append target-aware tail telemetry after the standard target is
        # available; the forward snapshot is logged later in this iteration.
        current_debug = getattr(self.mask_head, "_forward_telemetry", {}) or {}
        current_debug.update({
            key: value.detach() if torch.is_tensor(value) else value
            for key, value in tail_stats.items()
        })
        self.mask_head._forward_telemetry = current_debug

    def _apply_coarse_loss(
        self,
        losses: dict,
        telemetry: dict,
        mask_results: dict,
        sampling_results: List[SamplingResult],
        batch_gt_instances: InstanceList,
        prompt_pos_priors: List[Tensor],
    ) -> None:
        """ROI-local coarse-mask supervision (the coarse head's only teacher).

        GT crops are resized with nearest-neighbour to the coarse resolution,
        using the same prompt priors the mask branch saw.
        """
        coarse_outputs = mask_results.get("coarse_outputs")
        if coarse_outputs is None:
            return
        sp_weight = getattr(self.mask_head, "shape_prior_loss_weight", 0.0)
        if not (sp_weight > 0 and hasattr(self.mask_head, "coarse_mask_loss")):
            return
        coarse_loss, coarse_stats = self.mask_head.coarse_mask_loss(
            coarse_outputs["raw_logits"], sampling_results, batch_gt_instances,
            prompt_pos_priors=prompt_pos_priors,
        )
        losses["loss_shape_prior"] = coarse_loss
        if not hasattr(self, "_last_coarse_stats") or self._last_coarse_stats is None:
            self._last_coarse_stats = {}
        raw_stats = {key: value.detach() for key, value in coarse_stats.items()}
        self._last_coarse_stats.update(raw_stats)
        self._last_coarse_stats.update({
            "raw_dice_score": 1.0 - raw_stats["dice"],
            "raw_iou": raw_stats["iou"],
            "raw_boundary_tp": raw_stats["boundary_tp"],
            "raw_boundary_fp": raw_stats["boundary_fp"],
            "raw_boundary_fn": raw_stats["boundary_fn"],
        })

    def loss(
        self,
        x: Tuple[Tensor],
        rpn_results_list: InstanceList,
        batch_data_samples: List[DetDataSample],
        image_embeddings=None,
        image_positional_embeddings=None,
        high_res_features=None,
    ) -> dict:
        assert len(rpn_results_list) == len(batch_data_samples)
        outputs = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, _ = outputs
        x = self._inject_extra_pe(x)

        num_imgs = len(batch_data_samples)
        sampling_results = []
        for i in range(num_imgs):
            rpn_results = rpn_results_list[i]
            rpn_results.priors = rpn_results.pop("bboxes")

            assign_result = self.bbox_assigner.assign(rpn_results, batch_gt_instances[i], batch_gt_instances_ignore[i])
            sampling_result = self.bbox_sampler.sample(
                assign_result,
                rpn_results,
                batch_gt_instances[i],
                feats=[lvl_feat[i][None] for lvl_feat in x],
            )
            sampling_results.append(sampling_result)

        losses: Dict[str, object] = {}
        # Per-iteration telemetry stash: every diagnostic tensor produced on
        # this rank's loss path (detection metrics, TAIL/*, COARSE/*-derived
        # values) lands here, never inside the returned loss dict.
        self._loss_telemetry: Dict[str, object] = {}
        if self.with_bbox:
            bbox_results = self.bbox_loss(x, sampling_results)
            for key, value in bbox_results["loss_bbox"].items():
                # mmdet mixes metric tensors (e.g. "acc") into the loss dict;
                # only genuine losses belong in the backward channel.
                if key.startswith("loss_"):
                    losses[key] = value
                else:
                    self._loss_telemetry[f"DET/{key}"] = value

        if self.with_mask:
            mask_losses, mask_telemetry = self.mask_loss(
                x,
                sampling_results,
                batch_gt_instances,
                image_embeddings=image_embeddings,
                image_positional_embeddings=image_positional_embeddings,
                high_res_features=high_res_features,
                batch_img_metas=[sample.metainfo for sample in batch_data_samples],
            )
            losses.update(mask_losses)
            self._loss_telemetry.update(mask_telemetry)
        return losses

    def predict_mask(
        self,
        x: Tuple[Tensor],
        batch_img_metas: List[dict],
        results_list: InstanceList,
        rescale: bool = False,
        image_embeddings=None,
        image_positional_embeddings=None,
        high_res_features=None,
    ) -> InstanceList:
        bboxes = [res.bboxes for res in results_list]
        mask_rois = bbox2roi(bboxes)
        if mask_rois.shape[0] == 0:
            return empty_instances(
                batch_img_metas,
                mask_rois.device,
                task_type="mask",
                instance_results=results_list,
                mask_thr_binary=self.test_cfg.mask_thr_binary,
            )

        # Dense images can keep ~100 ROIs after NMS; feeding them through the
        # frozen SAM2 decoder at once spikes activation memory (high-res
        # features are per-ROI). Chunking is numerically identical and caps
        # the peak. Set mask_roi_chunk_size=0 on the roi head to disable.
        chunk_size = int(getattr(self, "mask_roi_chunk_size", 32) or 0)
        if chunk_size > 0 and mask_rois.shape[0] > chunk_size:
            mask_preds_parts = []
            for start in range(0, mask_rois.shape[0], chunk_size):
                chunk_results = self._mask_forward(
                    x,
                    mask_rois[start : start + chunk_size],
                    image_embeddings=image_embeddings,
                    image_positional_embeddings=image_positional_embeddings,
                    high_res_features=high_res_features,
                )
                mask_preds_parts.append(chunk_results["mask_preds"])
            mask_preds = torch.cat(mask_preds_parts, dim=0)
        else:
            mask_results = self._mask_forward(
                x,
                mask_rois,
                image_embeddings=image_embeddings,
                image_positional_embeddings=image_positional_embeddings,
                high_res_features=high_res_features,
            )
            mask_preds = mask_results["mask_preds"]
        num_mask_rois_per_img = [len(res) for res in results_list]
        mask_preds = mask_preds.split(num_mask_rois_per_img, 0)

        return self.mask_head.predict_by_feat(
            mask_preds=mask_preds,
            results_list=results_list,
            batch_img_metas=batch_img_metas,
            rcnn_test_cfg=self.test_cfg,
            rescale=rescale,
        )

    def predict(
        self,
        x: Tuple[Tensor],
        rpn_results_list: InstanceList,
        batch_data_samples: SampleList,
        rescale: bool = False,
        image_embeddings=None,
        image_positional_embeddings=None,
        high_res_features=None,
    ) -> InstanceList:
        batch_img_metas = [data_samples.metainfo for data_samples in batch_data_samples]
        x = self._inject_extra_pe(x)

        bbox_rescale = rescale if not self.with_mask else False
        results_list = self.predict_bbox(
            x,
            batch_img_metas,
            rpn_results_list,
            rcnn_test_cfg=self.test_cfg,
            rescale=bbox_rescale,
        )

        if self.with_mask:
            results_list = self.predict_mask(
                x,
                batch_img_metas,
                results_list,
                rescale=rescale,
                image_embeddings=image_embeddings,
                image_positional_embeddings=image_positional_embeddings,
                high_res_features=high_res_features,
            )
        return results_list
