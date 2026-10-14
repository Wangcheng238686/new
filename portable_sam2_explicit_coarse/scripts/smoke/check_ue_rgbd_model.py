#!/usr/bin/env python
"""Real-data gradients, one update, empty-rank DDP and 300-RoI resource checks.

Run through the project's existing cvt2 environment. This is an engineering
check, not an AP experiment; it always starts from generic SAM2 weights.
"""
import argparse
import copy
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "train")]
os.environ["RSPROMPTER_LIGHT_IMPORT"] = "0"
os.environ.setdefault("SAM2_REPO", str(ROOT.parent / "sam2"))
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import numpy as np
import torch
import torch.distributed as dist
from mmengine import Config
from mmengine.registry import init_default_scope
from mmengine.structures import InstanceData
from mmdet.registry import MODELS
import rsprompter
from data.ue_rgbd_dataset import UERGBDInstanceDataset, ue_collate
from data.ue_payload import attach_uav_payload
from gt_samples import _build_gt_data_samples
from optimizer import _build_optimizer, _partition_aux_clip_params


def sum_losses(values):
    parts = []
    for value in values:
        parts.extend(value if isinstance(value, (list, tuple)) else [value])
    return sum(parts)


def gradient_report(loss, model, require_nonzero=True):
    result = {}
    for prefix in ("uav_backbone.", "uav_bev."):
        params = [
            p
            for name, p in model.named_parameters()
            if name.startswith(prefix) and p.requires_grad
        ]
        # Use the runner's loss scale: tiny individual FP16 gradients can
        # otherwise disappear before they reach float32 LoRA parameters.
        gradients = torch.autograd.grad(
            loss * 512.0, params, retain_graph=True, allow_unused=True
        )
        gradients = [g.detach().float() / 512.0 for g in gradients if g is not None]
        norm = (
            float(torch.stack([g.square().sum() for g in gradients]).sum().sqrt())
            if gradients
            else 0.0
        )
        result[prefix] = dict(
            norm=norm,
            used=len(gradients),
            nonzero=sum(bool(g.count_nonzero()) for g in gradients),
        )
        assert np.isfinite(norm) and (norm > 0 or not require_nonzero), (prefix, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cache", default="/tmp/uav_dual_dev_data/cache")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="For per-loss checks choose real GT reaching the fused P3/P4 RoI levels",
    )
    parser.add_argument("--ddp-empty-rank", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    if args.ddp_empty_rank:
        dist.init_process_group("nccl")
    torch.cuda.set_device(rank)
    torch.manual_seed(42)
    np.random.seed(42)
    init_default_scope("mmdet")
    config = Config.fromfile(str(ROOT / "configs/ablations/ue_rgbd_dual.py"))
    model = MODELS.build(copy.deepcopy(config.model)).cuda(rank)
    dataset = UERGBDInstanceDataset(
        args.manifest, "train", config.ue_geometry, cache_root=args.cache
    )
    batch = ue_collate([dataset[args.sample_index]])
    samples = _build_gt_data_samples(
        batch["imgs"],
        batch["img_metas"],
        batch["gt_bboxes"],
        batch["gt_labels"],
        batch["gt_masks"],
        f"cuda:{rank}",
    )
    attach_uav_payload(samples, batch)
    if args.ddp_empty_rank and rank == 1:
        payload = samples[0].uav
        payload.candidate_index.fill_(-1)
        payload.observation_support.zero_()
        payload.height_valid.zero_()
        for name in (
            "project_source",
            "project_pixel",
            "project_mass",
            "project_quality",
        ):
            setattr(payload, name, getattr(payload, name)[:0])
    processed = model.data_preprocessor(
        dict(inputs=batch["imgs"], data_samples=samples), training=True
    )
    samples = processed["data_samples"]
    images = processed["inputs"]
    optimizer = _build_optimizer(
        model,
        lr=2e-4,
        backbone_lr_mult=0.1,
        other_lr_mult=1.0,
        bbox_head_lr_mult=1.0,
        uav_lr_mult=0.1,
    )
    ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
    assert len(ids) == len(set(ids)) == sum(p.requires_grad for p in model.parameters())
    before = {
        name: p.detach().clone()
        for name, p in model.named_parameters()
        if p.requires_grad
        and name.startswith(("uav_backbone.", "uav_bev.", "fpn_fusion.", "sam_fusion."))
    }
    wrapped = (
        torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[rank], find_unused_parameters=True
        )
        if args.ddp_empty_rank
        else model
    )
    torch.cuda.reset_peak_memory_stats()
    model.train()
    roi_levels = []

    def record_levels(extractor, inputs):
        levels = extractor.map_roi_levels(inputs[1], len(extractor.featmap_strides))
        roi_levels.extend(levels.detach().cpu().tolist())

    handle = model.roi_head.mask_roi_extractor.register_forward_pre_hook(record_levels)
    with torch.amp.autocast("cuda"):
        losses = wrapped(images, samples, mode="loss")
        total = sum_losses(losses.values())
    handle.remove()
    assert torch.isfinite(total)
    result = dict(loss=float(total.detach()), rank=rank)
    result["coarse_roi_level_counts"] = {str(i): roi_levels.count(i) for i in range(4)}
    if not args.ddp_empty_rank:
        result["loss_paths"] = {
            "detector": gradient_report(
                sum_losses(
                    v
                    for k, v in losses.items()
                    if k in ("loss_rpn_cls", "loss_rpn_bbox", "loss_cls", "loss_bbox")
                ),
                model,
            ),
            "coarse": gradient_report(
                losses["loss_shape_prior"], model, require_nonzero=False
            ),
            "final_mask": gradient_report(losses["loss_mask"], model),
            "udpr": gradient_report(losses["loss_decoder_tail"], model),
        }
    scaler = torch.amp.GradScaler("cuda", init_scale=512.0)
    if not args.ddp_empty_rank and any(level in (1, 2) for level in roi_levels):
        assert all(
            report["norm"] > 0 for report in result["loss_paths"]["coarse"].values()
        )
    scaler.scale(total).backward()
    scaler.unscale_(optimizer)
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )
    base, auxiliaries = _partition_aux_clip_params(model)
    for params in [base, *auxiliaries]:
        torch.nn.utils.clip_grad_norm_(params, 0.5)
    scaler.step(optimizer)
    scaler.update()
    changed = {
        prefix: sum(
            not torch.equal(p.detach(), before[name])
            for name, p in model.named_parameters()
            if name in before and name.startswith(prefix)
        )
        for prefix in ("uav_backbone.", "uav_bev.", "fpn_fusion.", "sam_fusion.")
    }
    assert all(changed.values()), changed
    result["changed_parameter_tensors"] = changed
    result["training_peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
    result["training_peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
    del before, losses, total
    optimizer.zero_grad(set_to_none=True)
    if args.ddp_empty_rank:
        for parameter in model.parameters():
            expected = parameter.detach().clone()
            dist.broadcast(expected, src=0)
            assert torch.equal(parameter.detach(), expected)
        result["ddp_parameters_identical"] = True
        args.output = args.output.with_name(args.output.stem + f"_rank{rank}.json")
    else:
        model.eval()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad(), torch.amp.autocast("cuda"):
            features = model._extract_dual_feat(images, samples)
            result["telemetry"] = {
                key: float(value)
                for key, value in model.get_telemetry().items()
                if torch.is_tensor(value) and value.numel() == 1
            }
            bboxes = samples[0].gt_instances.bboxes
            indices = torch.arange(300, device=images.device) % len(bboxes)
            instances = InstanceData(
                bboxes=bboxes[indices],
                labels=torch.zeros(300, dtype=torch.long, device=images.device),
                scores=torch.ones(300, device=images.device),
            )
            masks = model.roi_head.predict_mask(
                features[0],
                [samples[0].metainfo],
                [instances],
                image_embeddings=features[1],
                image_positional_embeddings=features[2],
                high_res_features=features[3],
            )
            assert len(masks[0]) == 300 and masks[0].masks.shape == (300, 512, 512)
            result["mask_rois"] = 300
            result["inference_peak_allocated_mib"] = (
                torch.cuda.max_memory_allocated() / 2**20
            )
            result["inference_peak_reserved_mib"] = (
                torch.cuda.max_memory_reserved() / 2**20
            )
            del masks, features
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if args.ddp_empty_rank:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
