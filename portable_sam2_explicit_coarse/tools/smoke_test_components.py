#!/usr/bin/env python
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("RSPROMPTER_LIGHT_IMPORT", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from mmengine import ConfigDict
from mmengine.structures import InstanceData
from mmdet.structures.mask import BitmapMasks

from rsprompter.coarse_mask_loss import CoarseMaskLoss
from rsprompter.dense_prompt_utils import paste_roi_to_full_canvas
from rsprompter.p2_boundary_refiner import P2BoundaryRefiner
from rsprompter.models_sam2 import (
    RSSAM2PAFPN,
    RSSAM2PositionalEmbedding,
    RSPrompterAnchorMaskHeadSAM2,
    _load_pretrained_no_mask_embedding,
)
from rsprompter.models import RSPrompterAnchor
import rsprompter.models_sam2 as models_sam2
from rsprompter.shape_prior import ShapePointMiner, ShapePriorInjector
from inference.infer_from_checkpoint import (
    _restore_embedded_architecture_environment,
)
from utils.coco_eval_utils import build_coco_gt_and_dt


def main():
    torch.manual_seed(44)

    # Early schema-v2 checkpoints keep the P2 switch only inside the embedded
    # mask-head config.  Checkpoint inference must restore it before falling
    # back to the recorded config factory for architecture verification.
    for enabled, expected in ((True, "1"), (False, "0")):
        embedded = {
            "roi_head": {
                "mask_head": {
                    "p2_boundary_refiner_cfg": {"enabled": enabled},
                }
            }
        }
        with patch.dict(
            os.environ, {"P2_BOUNDARY_REFINER_ENABLED": "stale"}
        ):
            _restore_embedded_architecture_environment(embedded)
            assert os.environ["P2_BOUNDARY_REFINER_ENABLED"] == expected

    # C5-v2 construction is RNG-isolated: all shared modules created before or
    # after the optional refiner must be byte-exact with the R1-C4 control.
    torch.manual_seed(44)
    shared_control = torch.nn.Linear(7, 7)
    after_control = torch.nn.Linear(7, 7)
    torch.manual_seed(44)
    shared_treatment = torch.nn.Linear(7, 7)
    with torch.random.fork_rng(devices=[]):
        _ = P2BoundaryRefiner(
            p2_in_channels=8, projected_channels=8, mid_channels=8
        )
    after_treatment = torch.nn.Linear(7, 7)
    for control, treatment in (
        (shared_control, shared_treatment),
        (after_control, after_treatment),
    ):
        for left, right in zip(control.state_dict().values(), treatment.state_dict().values()):
            assert torch.equal(left, right), "optional refiner perturbed shared RNG state"

    # Evaluation must never silently replace a configured quality score with
    # detector confidence. This is the final fail-fast guard shared by
    # standalone inference and any caller of the common COCO helper.
    empty_gt = {
        "bboxes": np.zeros((0, 4), dtype=np.float32),
        "labels": np.zeros((0,), dtype=np.int64),
        "masks": np.zeros((0, 8, 8), dtype=np.uint8),
    }
    detector_only_dt = {
        "bboxes": np.zeros((1, 4), dtype=np.float32),
        "scores": np.ones((1,), dtype=np.float32),
        "masks": np.zeros((1, 8, 8), dtype=np.uint8),
    }
    try:
        build_coco_gt_and_dt(
            [empty_gt],
            [detector_only_dt],
            [{"img_shape": (8, 8, 3)}],
            score_key="mask_scores",
        )
    except KeyError as error:
        assert "silent score fallback" in str(error)
    else:
        raise AssertionError("missing mask quality scores must fail fast")

    pyramid = (
        torch.zeros(1, 1, 256, 256),
        torch.zeros(1, 1, 128, 128),
        torch.zeros(1, 1, 64, 64),
        torch.zeros(1, 1, 32, 32),
    )
    vision_outputs = (pyramid[-1], pyramid)
    input_stub = torch.zeros(1, 3, 1024, 1024)
    official_selector = SimpleNamespace(sam_image_embedding_stride=16)
    legacy_selector = SimpleNamespace(sam_image_embedding_stride=32)
    official_embedding = RSPrompterAnchor._select_sam_image_embeddings(
        official_selector, vision_outputs, input_stub
    )
    legacy_embedding = RSPrompterAnchor._select_sam_image_embeddings(
        legacy_selector, vision_outputs, input_stub
    )
    assert official_embedding.shape[-2:] == (64, 64)
    assert legacy_embedding.shape[-2:] == (32, 32)
    dtype_probe = (torch.zeros(1, 256, 2, 2, dtype=torch.float64),)
    assert RSSAM2PAFPN._parse_inputs(dtype_probe)[0].dtype == torch.float64

    legacy_pe = RSSAM2PositionalEmbedding(
        image_size=1024,
        patch_size=16,
        embedding_stride=32,
        embed_dim=256,
    )
    raw_legacy_pe = legacy_pe.get_dense_pe()
    assert raw_legacy_pe.shape == (1, 256, 64, 64)
    assert legacy_pe.pos_embed.requires_grad
    assert torch.count_nonzero(raw_legacy_pe).item() == 0
    effective_legacy_pe = RSPrompterAnchor.get_image_wide_positional_embeddings(
        SimpleNamespace(shared_image_embedding=legacy_pe), size=32
    )
    assert effective_legacy_pe.shape == (1, 256, 32, 32)
    effective_legacy_pe.sum().backward()
    assert legacy_pe.pos_embed.grad is not None

    try:
        _load_pretrained_no_mask_embedding(
            "/definitely/missing/sam2_no_mask_checkpoint.pt"
        )
    except RuntimeError as error:
        assert "does not exist" in str(error)
    else:
        raise AssertionError(
            "requested pretrained no-mask embedding must never fall back to zeros"
        )
    with patch.object(models_sam2.os.path, "isfile", return_value=True), patch.object(
        models_sam2, "_load_sam2_checkpoint", return_value={}
    ):
        try:
            _load_pretrained_no_mask_embedding("missing-key.pt")
        except RuntimeError as error:
            assert "is missing" in str(error)
        else:
            raise AssertionError(
                "missing pretrained no-mask key must never fall back to zeros"
            )
    drifted_no_mask = SimpleNamespace(
        load_no_mask_pretrained=True,
        no_mask_pretrained_loaded=True,
        no_mask_embed=torch.nn.Parameter(
            torch.zeros(1, 256, 1, 1), requires_grad=False
        ),
        _pretrained_no_mask_reference=torch.ones(1, 256, 1, 1),
        freeze_no_mask_embed=True,
    )
    try:
        RSPrompterAnchorMaskHeadSAM2.assert_no_mask_embedding_contract(
            drifted_no_mask, "smoke drift"
        )
    except RuntimeError as error:
        assert "no longer matches" in str(error)
    else:
        raise AssertionError(
            "checkpoint overwrite of pretrained no-mask embedding must fail"
        )

    # Regression contract for the final instance mask coordinate system.  The
    # positive proposal lies entirely inside a larger GT mask, so an ROI crop
    # must be all foreground.  Resizing the uncropped full-image GT would have
    # a much lower mean and fail this assertion.
    gt_bitmap = np.zeros((1, 16, 16), dtype=np.uint8)
    gt_bitmap[0, 2:14, 2:14] = 1
    sampling_result = SimpleNamespace(
        pos_priors=torch.tensor([[4.0, 4.0, 12.0, 12.0]]),
        pos_assigned_gt_inds=torch.tensor([0], dtype=torch.long),
    )
    gt_instances = InstanceData(
        masks=BitmapMasks(gt_bitmap, height=16, width=16)
    )
    final_mask_target = RSPrompterAnchorMaskHeadSAM2.get_targets(
        None,
        [sampling_result],
        [gt_instances],
        ConfigDict(mask_size=(8, 8)),
    )
    assert final_mask_target.shape == (1, 8, 8)
    assert float(final_mask_target.mean()) > 0.99, (
        "final mask target must be cropped by the positive proposal (ROI-local)"
    )
    full_mask_target = RSPrompterAnchorMaskHeadSAM2.get_full_image_targets(
        None,
        [sampling_result],
        [gt_instances],
        target_size=(8, 8),
        device=torch.device("cpu"),
    )
    assert full_mask_target.shape == (1, 8, 8)
    assert 0.25 < float(full_mask_target.mean()) < 0.75, (
        "full-image target must preserve global foreground occupancy"
    )
    full_logits = torch.full((1, 1, 8, 8), -10.0)
    full_logits[:, :, :4, :4] = 10.0
    full_result = RSPrompterAnchorMaskHeadSAM2._predict_by_feat_single(
        SimpleNamespace(
            final_mask_coordinate_mode="full_image", class_agnostic=True
        ),
        mask_preds=full_logits,
        bboxes=torch.tensor([[8.0, 8.0, 16.0, 16.0]]),
        labels=torch.zeros(1, dtype=torch.long),
        img_meta={"img_shape": (16, 16, 3), "ori_shape": (16, 16, 3)},
        rcnn_test_cfg=ConfigDict(mask_thr_binary=0.5),
    )
    assert bool(full_result[0, 2, 2]) and not bool(full_result[0, 12, 12]), (
        "full-image SAM2 masks must be resized once, not pasted through the bbox"
    )

    n = 3
    roi = torch.randn(n, 512, 14, 14, requires_grad=True)
    image = torch.randn(2, 256, 32, 32, requires_grad=True)
    boxes = torch.tensor(
        [[20.0, 30.0, 300.0, 350.0], [400.0, 100.0, 800.0, 600.0],
         [50.0, 500.0, 250.0, 900.0]]
    )
    roi_img_ids = torch.tensor([0, 1, 1])

    shape = ShapePriorInjector(
        img_size=1024,
        visual_context_dim=256,
        roi_feat_channels=512,
        coarse_mask_output_size=64,
    )
    ctx = shape.forward_context_visual(image)
    assert ctx is None
    assert shape.fusion_type == "roi_only"
    assert shape.visual_context_proj is None
    assert shape.spatial_cross_attn is None
    assert shape.gamma_head is None and shape.beta_head is None
    raw, _, _, _ = shape.forward_roi(ctx, roi, boxes, roi_img_ids)
    assert raw.shape == (n, 1, 64, 64)
    assert float(shape.get_debug_stats()["CONTEXT/enabled"].item()) == 0.0

    # FiLM remains constructible only as an explicit later ablation, and its
    # zero-initialized output heads preserve the local path initially.
    film_shape = ShapePriorInjector(
        img_size=1024,
        visual_context_dim=256,
        roi_feat_channels=512,
        fusion_type="gated_spatial_film",
        coarse_mask_output_size=64,
    )
    film_ctx = film_shape.forward_context_visual(image)
    assert film_ctx.shape == (2, 64, 512)
    film_raw, _, _, _ = film_shape.forward_roi(
        film_ctx, roi, boxes, roi_img_ids
    )
    assert torch.equal(film_raw, film_shape.mask_decoder(roi))

    p2 = torch.randn(2, 512, 64, 64, requires_grad=True)
    prompt_rois = torch.cat(
        [
            roi_img_ids[:, None].float(),
            torch.tensor(
                [[5.0, 8.0, 55.0, 58.0], [10.0, 4.0, 60.0, 50.0],
                 [3.0, 16.0, 45.0, 62.0]]
            ),
        ],
        dim=1,
    )
    refiner = P2BoundaryRefiner(
        projected_channels=8,
        mid_channels=8,
        spatial_scale=1.0,
    )
    refiner_outputs = refiner(p2, prompt_rois, raw)
    refined = refiner_outputs["refined_logits"]
    assert refined.shape == raw.shape
    assert torch.equal(refined, raw), (
        "zero-init P2BoundaryRefiner must exactly reproduce R1-C4"
    )
    assert refiner_outputs["delta_logits"].abs().max() <= 0.400001
    boundary_target = torch.sigmoid(raw.detach()).ge(0.5).float()
    boundary_sum, boundary_count, _ = refiner.boundary_loss(
        raw,
        refiner_outputs["delta_logits"],
        refiner_outputs["raw_boundary_band"],
        refiner_outputs["search_support"],
        boundary_target,
    )
    (boundary_sum / boundary_count.clamp_min(1.0)).backward(retain_graph=True)
    assert p2.grad is None, "P2 must be detached from refiner gradients"
    assert refiner.residual_head.weight.grad is not None
    assert roi.grad is None, "boundary auxiliary loss must detach raw coarse logits"
    canvas = paste_roi_to_full_canvas(refined, boxes, 128, 128, 1024)
    assert canvas.shape == (n, 1, 128, 128)
    canvas.mean().backward()
    assert roi.grad is not None
    assert image.grad is None, "roi_only coarse route must not consume image context"
    with torch.no_grad():
        refiner.residual_head.weight.fill_(100.0)
        refiner.residual_head.bias.fill_(100.0)
    bounded_outputs = refiner(p2, prompt_rois, raw)
    assert float(bounded_outputs["delta_logits"].detach().abs().max()) <= 0.400001
    assert torch.count_nonzero(
        bounded_outputs["delta_logits"]
        * (1.0 - bounded_outputs["search_support"])
    ) == 0
    checkerboard = (
        (torch.arange(64)[:, None] + torch.arange(64)[None, :]) % 2
    ).float()
    checker_logits = (checkerboard * 2.0 - 1.0)[None, None].repeat(n, 1, 1, 1) * 20.0
    rejected_outputs = refiner(p2, prompt_rois, checker_logits)
    assert not bool(rejected_outputs["support_valid"].any())
    assert torch.count_nonzero(rejected_outputs["delta_logits"]) == 0

    miner = ShapePointMiner(
        point_warmup_cfg=dict(
            enabled=False,
            no_point_epochs=0,
            one_pair_epochs=0,
            full_2p2n_start_epoch=1,
        )
    )
    coords, labels, _ = miner(refined, boxes, 1024)
    assert coords.shape == (n, 4, 2)
    assert labels.shape == (n, 4)

    ambiguous = torch.zeros(2, 1, 16, 16)
    miner_adaptive = ShapePointMiner(
        adaptive_validity=True,
        point_warmup_cfg=dict(
            enabled=False,
            no_point_epochs=0,
            one_pair_epochs=0,
            full_2p2n_start_epoch=1,
        ),
    )
    _, adaptive_labels, adaptive_stats = miner_adaptive(
        ambiguous, boxes[:2], 1024
    )
    assert (adaptive_labels == -1).all()
    assert int(adaptive_stats["all_invalid_count"].item()) == 2
    points, _, _ = RSPrompterAnchorMaskHeadSAM2._select_explicit_prompt_inputs(
        SimpleNamespace(
            prompt_sparse_mode="shape_point",
            explicit_use_point_prompt=True,
            explicit_use_box_prompt=False,
            explicit_use_dense_prompt=False,
        ),
        torch.zeros(2, 4, 2),
        adaptive_labels,
        boxes[:2],
        None,
    )
    assert points is None, "all-invalid points must become a true no-point prompt"
    fake_sparse = torch.ones(2, 6, 256)
    neutralized = RSPrompterAnchorMaskHeadSAM2._neutralize_invalid_point_tokens(
        fake_sparse,
        torch.tensor([[-1, -1, -1, -1], [1, -1, 0, -1]]),
    )
    assert torch.count_nonzero(neutralized[0, :4]).item() == 0
    assert torch.count_nonzero(neutralized[0, 4:]).item() > 0
    assert torch.count_nonzero(neutralized[1, [1, 3]]).item() == 0

    miner_fixed = ShapePointMiner(
        adaptive_validity=False,
        point_warmup_cfg=dict(
            enabled=False,
            no_point_epochs=0,
            one_pair_epochs=0,
            full_2p2n_start_epoch=1,
        ),
    )
    _, fixed_labels, fixed_stats = miner_fixed(ambiguous, boxes[:2], 1024)
    assert torch.equal(
        fixed_labels,
        torch.tensor([[1, 1, 0, 0], [1, 1, 0, 0]]),
    )
    assert int(fixed_stats["all_invalid_count"].item()) == 0
    fixed_yx, _ = miner_fixed.get_last_local_points()
    for row in fixed_yx:
        assert len({tuple(point.tolist()) for point in row}) == 4
    assert int(fixed_stats["point_overlap_count"].item()) == 0

    staged_logits = torch.full((2, 1, 16, 16), -10.0)
    staged_logits[:, :, 4:12, 4:12] = 10.0
    miner_warmup = ShapePointMiner(
        point_warmup_cfg=dict(
            enabled=True,
            no_point_epochs=2,
            one_pair_epochs=2,
            full_2p2n_start_epoch=5,
        )
    )
    miner_warmup.set_current_epoch(1)
    miner_warmup.train()
    _, train_warmup_labels, _ = miner_warmup(staged_logits, boxes[:2], 1024)
    miner_warmup.eval()
    _, eval_warmup_labels, _ = miner_warmup(staged_logits, boxes[:2], 1024)
    assert torch.equal(train_warmup_labels, eval_warmup_labels)
    assert (eval_warmup_labels == -1).all()

    try:
        ShapePriorInjector(typo_context_tokens=64)
    except TypeError:
        pass
    else:
        raise AssertionError("unknown ShapePrior config keys must fail fast")

    target = torch.zeros(2, 16, 16)
    fixed_loss = CoarseMaskLoss(weight=0.10, schedule_mode="fixed")
    _, fixed_loss_stats = fixed_loss(ambiguous, target)
    assert abs(float(fixed_loss_stats["loss_weight"]) - 0.10) < 1e-6
    scheduled_loss = CoarseMaskLoss(
        weight=0.10,
        schedule_mode="two_stage",
        weight_schedule=[
            dict(start_epoch=1, end_epoch=5, weight=0.20),
            dict(start_epoch=6, end_epoch=10, weight=0.10),
        ],
    )
    scheduled_loss.set_current_epoch(5)
    _, stage1_stats = scheduled_loss(ambiguous, target)
    scheduled_loss.set_current_epoch(6)
    _, stage2_stats = scheduled_loss(ambiguous, target)
    assert abs(float(stage1_stats["loss_weight"]) - 0.20) < 1e-6
    assert abs(float(stage2_stats["loss_weight"]) - 0.10) < 1e-6
    legacy_scheduled_loss = CoarseMaskLoss(
        weight=0.10,
        weight_schedule=[
            dict(start_epoch=1, end_epoch=5, weight=0.20),
            dict(start_epoch=6, end_epoch=10, weight=0.10),
        ],
    )
    assert legacy_scheduled_loss.schedule_mode == "two_stage"

    support = RSPrompterAnchorMaskHeadSAM2._dense_embedding_box_support(
        SimpleNamespace(prompt_encoder_image_size=1024),
        boxes[:1],
        (32, 32),
        torch.float32,
    )
    assert support.shape == (1, 1, 32, 32)
    assert float(support[:, :, -1, -1].item()) == 0.0
    print("component smoke test: OK")


if __name__ == "__main__":
    main()
