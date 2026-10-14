"""UE route adapters for the shared training runner."""

from data.ue_rgbd_dataset import UERGBDInstanceDataset, ue_loader


def validate_resume_snapshot(checkpoint, current):
    """Reject changed UE experiment identities before loading any state."""
    import os

    if os.environ.get("RESUME_STRICT", "1") != "1":
        raise ValueError("UE resume requires strict model loading")
    saved = checkpoint.get("config_snapshot", {})
    keys = (
        "manifest_sha256",
        "geometry",
        "geometry_signature",
        "split_floor_m",
        "load_uav",
        "single_class",
        "num_classes",
        "image_size",
        "dataset_format",
    )
    for key in keys:
        expected = current["data_config"].get(key)
        actual = saved.get("data_config", {}).get(key)
        if expected is None or actual != expected:
            raise ValueError(f"UE resume data identity mismatch: {key}")
    for key in (
        "evaluation_contract",
        "train_subset_ratio",
        "val_subset_ratio",
        "eval_bbox",
        "save_bbox_best_metric",
        "save_composite_best",
        "composite_metric_spec",
        "early_stopping_metric_spec",
        "early_stopping_patience",
        "early_stopping_start_epoch",
        "early_stopping_min_delta",
        "early_stopping_smooth_window",
        "effective_batch_size",
        "optimizer_steps_per_epoch",
    ):
        if saved.get(key) != current.get(key):
            raise ValueError(f"UE resume experiment identity mismatch: {key}")
    semantic_args = (
        "batch_size",
        "val_batch_size",
        "grad_accum_steps",
        "seed",
        "decoupled_aux_clip",
        "freeze_bn",
        "amp",
        "ema_enabled",
        "ema_decay",
        "ema_update_every",
        "ema_eval",
        "ema_eval_start_epoch",
        "val_every_n_epochs",
        "max_train_batches",
        "max_val_batches",
        "lr",
        "weight_decay",
        "backbone_lr_mult",
        "mask_decoder_lr_mult",
        "other_lr_mult",
        "bbox_head_lr_mult",
        "uav_lr_mult",
        "cvt_lr_mult",
        "fusion_lr_mult",
        "warmup_iters", "warmup_epochs", "no_mask_lr_mult", "prompt_encoder_lr_mult",
        "shape_prior_lr_mult", "decoder_tail_lr_mult",
    )
    for key in semantic_args:
        if saved.get("training_args", {}).get(key) != current["training_args"].get(key):
            raise ValueError(f"UE resume training identity mismatch: {key}")
    expected_hash = current["runtime_config"].get("sam2_checkpoint_sha256")
    if (
        not expected_hash
        or saved.get("runtime_config", {}).get("sam2_checkpoint_sha256")
        != expected_hash
    ):
        raise ValueError("UE resume generic SAM2 identity mismatch")


def create_ue_train_loader(
    *,
    manifest,
    geometry,
    load_uav,
    cache_root,
    satellite_root,
    uav_root,
    data_root,
    batch_size,
    val_batch_size,
    distributed,
    rank,
    world_size,
    seed,
    train_subset_ratio,
    val_subset_ratio,
    image_size,
    **legacy,
):
    if tuple(image_size) != (geometry["image_size"],) * 2:
        raise ValueError("UE runner/image geometry size mismatch")
    for key in (
        "flip_prob",
        "vflip_prob",
        "multi_scale_resize_prob",
        "random_erasing_prob",
    ):
        if legacy.get(key, 0):
            raise ValueError("UE geometry requires disabled spatial augmentation")
    kwargs = dict(
        manifest=manifest,
        geometry=geometry,
        load_uav=load_uav,
        cache_root=cache_root,
        satellite_root=satellite_root,
        uav_root=uav_root,
    )
    train = UERGBDInstanceDataset(
        split="train", subset_ratio=train_subset_ratio, **kwargs
    )
    val = UERGBDInstanceDataset(
        split="validation", subset_ratio=val_subset_ratio, **kwargs
    )
    return (
        ue_loader(
            train,
            batch_size,
            train=True,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
            seed=seed,
        ),
        ue_loader(val, val_batch_size, seed=seed),
        train,
    )
