"""Optimizer assembly: parameter groups, LR multipliers, norm freezing
and the decoupled aux-head grad-clip partition (root-cause fix for
aux heads stealing the base clip budget)."""

import logging
import os

import torch
import torch.optim as optim

logger = logging.getLogger("portable_sam_fusion")


from distributed import _get_model_core


def _build_optimizer(
    model: torch.nn.Module,
    lr: float,
    backbone_lr_mult: float,
    other_lr_mult: float,
    bbox_head_lr_mult: float,
    mask_decoder_lr_mult: float = 0.05,
    no_mask_lr_mult: float = 0.1,
    prompt_encoder_lr_mult: float = 0.1,
    shape_prior_lr_mult: float = 1.0,
    decoder_tail_lr_mult: float = 1.0,
    weight_decay: float = 0.05,
    fusion_no_wd: bool = False,
) -> optim.Optimizer:
    params_backbone = []
    names_backbone = []
    params_mask_decoder = []
    names_mask_decoder = []
    params_no_mask = []
    names_no_mask = []
    params_prompt_encoder = []
    names_prompt_encoder = []
    params_shape_prior = []
    names_shape_prior = []
    params_iou_head = []
    names_iou_head = []
    params_decoder_tail = []
    names_decoder_tail = []
    params_bbox_head = []
    names_bbox_head = []
    params_other = []
    names_other = []
    params_fusion = []
    names_fusion = []
    # EXP-0015/0016: the DUAL-WHY-NULL diagnosis names wd 0.05 as one of the
    # forces holding the zero-init fusion gates shut. Matched attrs are the
    # RSPrompterAnchorDual fusion modules only (top-level gate/proj/gate_e/
    # proj_e, incl. ModuleDict children); nested backbone blocks never start
    # with these tokens after prefix stripping. WARM-R1B adds the bev_head
    # buildingness gate; WARM-R1C adds the bev_attn descriptor pathway (same
    # fusion-side family).
    FUSION_ROOTS = ("gate", "proj", "gate_e", "proj_e", "bev_head", "bev_attn")

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        match_name = name.replace("_fsdp_wrapped_module.", "").replace("module.", "")
        if fusion_no_wd and match_name.split(".")[0] in FUSION_ROOTS:
            params_fusion.append(param)
            names_fusion.append(name)
            continue
        if "roi_head.mask_head.decoder_tail_refiner" in match_name:
            params_decoder_tail.append(param)
            names_decoder_tail.append(name)
        elif match_name.startswith("backbone.") or match_name.startswith("shared_image_embedding."):
            params_backbone.append(param)
            names_backbone.append(name)
        elif "roi_head.mask_head.mask_decoder.decoder.iou_prediction_head" in match_name:
            # The IoU head keeps its own group (historical C2-era contract)
            # even though the quality-rerank module is gone.
            params_iou_head.append(param)
            names_iou_head.append(name)
        elif "roi_head.mask_head.mask_decoder" in match_name:
            params_mask_decoder.append(param)
            names_mask_decoder.append(name)
        elif "roi_head.mask_head.no_mask_embed" in match_name:
            params_no_mask.append(param)
            names_no_mask.append(name)
        elif "roi_head.mask_head.prompt_encoder" in match_name:
            params_prompt_encoder.append(param)
            names_prompt_encoder.append(name)
        elif (
            "roi_head.mask_head.shape_injector" in match_name
            or "roi_head.mask_head.shape_prompt_scale" in match_name
            or "roi_head.mask_head.roi_dense" in match_name
        ):
            params_shape_prior.append(param)
            names_shape_prior.append(name)
        elif (
            bbox_head_lr_mult > 0
            and match_name.startswith("roi_head.bbox_head.")
        ):
            params_bbox_head.append(param)
            names_bbox_head.append(name)
        else:
            params_other.append(param)
            names_other.append(name)

    param_groups = []
    audit_groups = []
    if params_fusion:
        # same LR the fusion params would have received in 'other'; only the
        # decay differs (the single preregistered lever of EXP-0015/0016)
        param_groups.append(
            {
                "params": params_fusion,
                "lr": lr * other_lr_mult,
                "weight_decay": 0.0,
                "name": "fusion",
            }
        )
        audit_groups.append(("fusion", params_fusion, names_fusion, lr * other_lr_mult, 0.0))
    if params_backbone:
        param_groups.append(
            {
                "params": params_backbone,
                "lr": lr * backbone_lr_mult,
                "name": "backbone",
            }
        )
        audit_groups.append(("backbone", params_backbone, names_backbone, lr * backbone_lr_mult, weight_decay))
    if params_mask_decoder:
        param_groups.append(
            {
                "params": params_mask_decoder,
                "lr": lr * mask_decoder_lr_mult,
                "weight_decay": weight_decay,
                "name": "mask_decoder",
            }
        )
        audit_groups.append(("mask_decoder", params_mask_decoder, names_mask_decoder, lr * mask_decoder_lr_mult, weight_decay))
    if params_no_mask:
        param_groups.append(
            {
                "params": params_no_mask,
                "lr": lr * no_mask_lr_mult,
                "weight_decay": weight_decay,
                "name": "no_mask_embed",
            }
        )
        audit_groups.append(("no_mask_embed", params_no_mask, names_no_mask, lr * no_mask_lr_mult, weight_decay))
    if params_prompt_encoder:
        param_groups.append(
            {
                "params": params_prompt_encoder,
                "lr": lr * prompt_encoder_lr_mult,
                "weight_decay": 0.0,
                "name": "prompt_encoder",
            }
        )
        audit_groups.append(("prompt_encoder", params_prompt_encoder, names_prompt_encoder, lr * prompt_encoder_lr_mult, 0.0))
    if params_shape_prior:
        param_groups.append(
            {
                "params": params_shape_prior,
                "lr": lr * shape_prior_lr_mult,
                "name": "shape_prior",
            }
        )
        audit_groups.append(("shape_prior", params_shape_prior, names_shape_prior, lr * shape_prior_lr_mult, weight_decay))
    if params_iou_head:
        param_groups.append({"params": params_iou_head, "lr": lr,
                             "name": "iou_head"})
        audit_groups.append(("iou_head", params_iou_head, names_iou_head, lr, weight_decay))
    if params_decoder_tail:
        param_groups.append({"params": params_decoder_tail, "lr": lr * decoder_tail_lr_mult,
                             "weight_decay": weight_decay, "name": "decoder_tail"})
        audit_groups.append(("decoder_tail", params_decoder_tail, names_decoder_tail,
                             lr * decoder_tail_lr_mult, weight_decay))
    if params_bbox_head:
        param_groups.append(
            {
                "params": params_bbox_head,
                "lr": lr * bbox_head_lr_mult,
                "name": "bbox_head",
            }
        )
        audit_groups.append(("bbox_head", params_bbox_head, names_bbox_head, lr * bbox_head_lr_mult, weight_decay))
    if params_other:
        param_groups.append(
            {
                "params": params_other,
                "lr": lr * other_lr_mult,
                "name": "other",
            }
        )
        audit_groups.append(("other", params_other, names_other, lr * other_lr_mult, weight_decay))
    if not param_groups:
        raise RuntimeError(
            "No trainable parameters found. Check freezing / requires_grad."
        )

    if int(os.environ.get("RANK", "0")) == 0:
        for group_name, params, names, group_lr, group_wd in audit_groups:
            count = sum(p.numel() for p in params)
            first = names[0] if names else "<none>"
            last = names[-1] if names else "<none>"
            logger.info(
                "Optimizer group %-15s params=%d tensors=%d lr=%.3e wd=%.3e first=%s last=%s",
                group_name,
                count,
                len(params),
                group_lr,
                group_wd,
                first,
                last,
            )

    return optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay, eps=1e-6)


def _set_norm_eval(model: torch.nn.Module):
    for m in model.modules():
        if isinstance(
            m, (torch.nn.BatchNorm2d, torch.nn.SyncBatchNorm, torch.nn.GroupNorm)
        ):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False


# Zero-init residual auxiliary heads whose gradients were side-loaded into
# the single global grad-norm clip.  Audit 2026-09-14: with the legacy clip
# these heads usurp the base's budget (tail EMA clipped-grad L2 0.158 of the
# 0.5 norm at E220 while a0_sg saturates it base-only at 0.458), imposing a
# one-directional, time-varying downscale of base gradients from step 1 --
# the mechanism behind the persistent bbox deficits of every auxiliary-head
# arm (a3sg/a3sgt/a3sgtm/densecapres256) while the coupling-free tail-only
# heat-start passes both gates.  P2/renderer lineages keep the legacy clip.
_AUX_CLIP_MODULE_PREFIXES = (
    "roi_head.mask_head.decoder_tail_refiner.",
)


def _partition_aux_clip_params(model: torch.nn.Module):
    """Split parameters into (base, aux_groups) for decoupled grad clipping.

    aux_groups has one entry per aux module actually present.  Params whose
    grad is None (e.g. the frozen base in tail-only mode) are skipped by
    clip_grad_norm_ itself, matching legacy semantics of clipping
    ``model.parameters()`` over whichever params carry grads.
    """
    named = list(_get_model_core(model).named_parameters())
    groups = [
        [param for name, param in named if name.startswith(prefix)]
        for prefix in _AUX_CLIP_MODULE_PREFIXES
    ]
    groups = [group for group in groups if group]
    aux_ids = {id(param) for group in groups for param in group}
    base = [param for _, param in named if id(param) not in aux_ids]
    return base, groups


def _assert_no_mask_embedding_contract(
    model: torch.nn.Module, context: str
) -> None:
    model_core = _get_model_core(model)
    mask_head = getattr(getattr(model_core, "roi_head", None), "mask_head", None)
    validator = getattr(mask_head, "assert_no_mask_embedding_contract", None)
    if validator is not None:
        validator(context)
