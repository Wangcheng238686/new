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

    def get_prompt_debug_stats(self) -> Dict[str, object]:
        stats: Dict[str, object] = {}
        mask_head = getattr(self, "mask_head", None)
        if mask_head is not None and hasattr(mask_head, "get_uav_debug_stats"):
            stats.update(mask_head.get_uav_debug_stats())
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
        _ratio("BOX/fallback_ratio", "BOX/fallback_count", "BOX/num_rois")
        _ratio("BOX/refined_proposal_iou", "BOX/refined_proposal_iou_sum", "BOX/refined_valid_count")
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
        rois: Tensor = None,
        pos_inds: Optional[Tensor] = None,
        bbox_feats: Optional[Tensor] = None,
        image_embeddings=None,
        image_positional_embeddings=None,
        high_res_features=None,
        roi_img_ids_override: Optional[Tensor] = None,
        boxes_override: Optional[Tensor] = None,
    ) -> dict:
        assert (rois is not None) ^ (pos_inds is not None and bbox_feats is not None)
        if rois is not None:
            mask_feats = self.mask_roi_extractor(x[: self.mask_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
        else:
            assert bbox_feats is not None
            mask_feats = bbox_feats[pos_inds]

        if rois is not None:
            prompt_rois = rois
        elif roi_img_ids_override is not None and boxes_override is not None:
            prompt_rois = torch.cat(
                [roi_img_ids_override[:, None].to(boxes_override.dtype), boxes_override],
                dim=1,
            )
        else:
            prompt_rois = None

        mask_head_out = self.mask_head(
            mask_feats,
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
            roi_img_ids=(
                rois[:, 0]
                if rois is not None
                else roi_img_ids_override
            ),
            high_res_features=high_res_features,
            boxes=rois[:, 1:] if rois is not None else boxes_override,
            p2_feature=x[0],
            prompt_rois=prompt_rois,
            renderer_features=x,
        )
        if len(mask_head_out) != 6:
            raise RuntimeError(
                "Canonical mask head must return 6 values, got "
                f"{len(mask_head_out)}"
            )
        (
            mask_preds,
            iou_predictions,
            coarse_outputs,
            _uav_aux_losses,  # always {} since the UAV branch was removed
            quality_predictions,
            mask_tokens,
        ) = mask_head_out
        return dict(
            mask_preds=mask_preds,
            mask_feats=mask_feats,
            iou_predictions=iou_predictions,
            quality_predictions=quality_predictions,
            mask_tokens=mask_tokens,
            coarse_outputs=coarse_outputs,
            tail_outputs=getattr(self.mask_head, "_last_tail_outputs", None),
        )

    def mask_loss(
        self,
        x: Tuple[Tensor],
        sampling_results: List[SamplingResult],
        bbox_feats: Tensor,
        bbox_results: Optional[dict],
        batch_gt_instances: InstanceList,
        image_embeddings=None,
        image_positional_embeddings=None,
        high_res_features=None,
        batch_img_metas: Optional[List[dict]] = None,
    ) -> dict:
        original_pos_rois = bbox2roi([res.pos_priors for res in sampling_results])
        if len(original_pos_rois) == 0:
            return self._empty_rank_losses(x)

        prompt_box_cfg = dict(
            getattr(self.mask_head, "mask_prompt_box_cfg", {}) or {}
        )
        train_box_source = str(
            prompt_box_cfg.get("train_box_source", "proposal")
        )
        # Proposal boxes are the only supported prompt source: no canonical
        # config uses a SABL bbox head anymore.
        if train_box_source != "proposal":
            raise ValueError(
                f"Unsupported train_box_source={train_box_source!r}; "
                "only 'proposal' is supported"
            )
        jitter_cfg = dict(getattr(self.mask_head, "box_prompt_cfg", {}) or {})
        if float(jitter_cfg.get("jitter_prob", 0.0)) > 0.0:
            raise ValueError(
                "box prompt jitter belonged to a removed arm and cannot be "
                "enabled in the a3sgm-only codebase"
            )
        zero = original_pos_rois.new_zeros(())
        self._last_prompt_box_stats = {
            "BOX/num_rois": original_pos_rois.new_tensor(float(len(original_pos_rois))),
            "BOX/fallback_count": zero,
            "BOX/refined_proposal_iou_sum": zero,
            "BOX/refined_valid_count": zero,
        }
        self._last_box_jitter_stats = {
            "JITTER/num_rois": original_pos_rois.new_tensor(float(len(original_pos_rois))),
            "JITTER/num_jittered": zero,
            "JITTER/mean_iou": original_pos_rois.new_tensor(1.0),
            "JITTER/fallback_count": zero,
            "JITTER/area_ratio_sum": original_pos_rois.new_tensor(float(len(original_pos_rois))),
            "JITTER/area_ratio_count": original_pos_rois.new_tensor(float(len(original_pos_rois))),
        }

        prompt_rois = original_pos_rois
        pos_counts = [res.pos_priors.shape[0] for res in sampling_results]
        prompt_pos_priors = list(prompt_rois[:, 1:].split(pos_counts, dim=0))
        # From this point onward every mask/prompt operation uses prompt_rois.
        # Sampling assignments and bbox losses remain tied to sampling_results.
        pos_rois = prompt_rois

        if not self.share_roi_extractor:
            mask_results = self._mask_forward(
                x,
                pos_rois,
                image_embeddings=image_embeddings,
                image_positional_embeddings=image_positional_embeddings,
                high_res_features=high_res_features,
            )
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(torch.ones(res.pos_priors.shape[0], device=device, dtype=torch.uint8))
                pos_inds.append(torch.zeros(res.neg_priors.shape[0], device=device, dtype=torch.uint8))
            pos_inds = torch.cat(pos_inds)
            mask_results = self._mask_forward(
                x,
                pos_inds=pos_inds,
                bbox_feats=bbox_feats,
                high_res_features=high_res_features,
                roi_img_ids_override=pos_rois[:, 0] if len(pos_rois) > 0 else None,
                boxes_override=pos_rois[:, 1:] if len(pos_rois) > 0 else None,
            )

        mask_loss_and_target = self.mask_head.loss_and_target(
            mask_preds=mask_results["mask_preds"],
            sampling_results=sampling_results,
            batch_gt_instances=batch_gt_instances,
            rcnn_train_cfg=self.train_cfg,
            final_mask_pos_priors=prompt_pos_priors,
        )
        mask_results.update(loss_mask=mask_loss_and_target["loss_mask"])
        mask_results["loss_mask"].update(
            mask_loss_and_target.get("debug_stats", {})
        )
        self._apply_tail_loss(
            mask_results, mask_loss_and_target,
            sampling_results, batch_gt_instances, prompt_pos_priors,
        )
        self._apply_coarse_loss(
            mask_results,
            sampling_results, batch_gt_instances, prompt_pos_priors,
        )
        return mask_results

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
        self._last_prompt_box_stats = {
            "BOX/num_rois": zero,
            "BOX/fallback_count": zero,
            "BOX/refined_proposal_iou_sum": zero,
            "BOX/refined_valid_count": zero,
        }
        self._last_box_jitter_stats = {
            "JITTER/num_rois": zero,
            "JITTER/num_jittered": zero,
            "JITTER/mean_iou": zero,
            "JITTER/fallback_count": zero,
            "JITTER/area_ratio_sum": zero,
            "JITTER/area_ratio_count": zero,
        }
        losses = dict(loss_mask=0 * x[0].sum())
        if getattr(self.mask_head, "decoder_tail_enabled", False):
            tail_loss, tail_stats = self.mask_head.decoder_tail_refiner.empty_point_loss(zero)
            losses["loss_decoder_tail"] = tail_loss
            losses.update(tail_stats)
            self.mask_head._last_uav_debug_stats = dict(tail_stats)
        return dict(loss_mask=losses)

    def _apply_tail_loss(
        self,
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
        mask_results["loss_mask"]["loss_decoder_tail"] = tail_loss
        mask_results["loss_mask"].update(tail_stats)
        # Append target-aware tail telemetry after the standard target is
        # available; the forward snapshot is logged later in this iteration.
        current_debug = getattr(self.mask_head, "_last_uav_debug_stats", {}) or {}
        current_debug.update({
            key: value.detach() if torch.is_tensor(value) else value
            for key, value in tail_stats.items()
        })
        self.mask_head._last_uav_debug_stats = current_debug

    def _apply_coarse_loss(
        self,
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
        mask_results["loss_shape_prior"] = {"loss_shape_prior": coarse_loss}
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

        losses = dict()
        if self.with_bbox:
            bbox_results = self.bbox_loss(x, sampling_results)
            losses.update(bbox_results["loss_bbox"])

        if self.with_mask:
            mask_results = self.mask_loss(
                x,
                sampling_results,
                bbox_results["bbox_feats"],
                bbox_results,
                batch_gt_instances,
                image_embeddings=image_embeddings,
                image_positional_embeddings=image_positional_embeddings,
                high_res_features=high_res_features,
                batch_img_metas=[sample.metainfo for sample in batch_data_samples],
            )
            losses.update(mask_results["loss_mask"])
            if "loss_shape_prior" in mask_results:
                losses.update(mask_results["loss_shape_prior"])
            for loss_name, loss_value in mask_results.items():
                # The full_image roi-balanced final-mask loss reports its
                # components as loss_roi_* dicts.
                if (
                    loss_name.startswith("loss_roi_")
                    and isinstance(loss_value, dict)
                ):
                    losses.update(loss_value)
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
