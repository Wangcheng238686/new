"""
RSPrompter with SAM2-Hiera-Large - 卫星-无人机融合版本

整合:
1. 卫星分支: SAM2 Large + LoRA + 高分辨率特征
2. 无人机分支: 跨视角注意力 + 深度感知 + 高度引导融合

主要改进:
1. ✅ SAM2 Hiera Large 作为卫星和无人机的 backbone
2. ✅ LoRA 高效微调 (r=16, alpha=32, dropout=0.05)
3. ✅ 高度引导的空间融合
4. ✅ 深度感知的 BEV 生成
5. ✅ 512 通道维度
"""
import os

work_dir = os.environ.get("WORK_DIR", "./work_dirs/portable_sam2_fusion_large_window/rsprompter_sat_drone_sam2_large_v1")

num_classes = 1
prompt_shape = (100, 5)

sam2_ckpt_path = os.environ.get(
    "SAM2_CKPT",
    os.path.abspath(os.path.join("checkpoints", "sam2_hiera_large.pt")),
)

model = dict(
    type="RSPrompterAnchorDroneGuidance",
    
    enable_drone_branch=True,
    share_sam2_backbone=False,
    use_height_guided_fusion=True,
    
    iimr=dict(
        enabled=True,
        num_iterations=3,
        use_topology_memory=True,
        use_dynamic_prompting=True,
    ),
    
    data_preprocessor=dict(
        type="DetDataPreprocessor",
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        bgr_to_rgb=True,
        pad_mask=True,
        pad_size_divisor=32,
        batch_augments=[
            dict(
                type="BatchFixedSizePad",
                size=(512, 512),
                img_pad_value=0,
                pad_mask=True,
                mask_pad_value=0,
                pad_seg=False,
            )
        ],
    ),
    decoder_freeze=False,
    shared_image_embedding=dict(
        type="RSSAM2PositionalEmbedding",
        image_size=512,
        patch_size=16,
        embed_dim=256,
    ),
    
    backbone=dict(
        type="RSSAM2VisionEncoder",
        checkpoint_path=sam2_ckpt_path,
        init_cfg=dict(type="Pretrained", checkpoint=sam2_ckpt_path),
        use_gradient_checkpointing=True,
        use_lora=True,
        lora_config=dict(
            r=16,
            lora_alpha=32,
            target_modules=["qkv", "proj"],
            lora_dropout=0.05,
            bias="none",
            modules_to_save=["neck"],
        ),
        freeze_backbone=False,
        freeze_backbone_stages=-1,
        model_size="large",
    ),
    
    neck=dict(
        type="RSFeatureAggregatorSAM2",
        in_channels="sam2_hiera_large",
        out_channels=512,
        hidden_channels=64,
        select_layers=[0, 1, 2, 3],
        fpn_type="fused",
    ),
    
    rpn_head=dict(
        type="RPNHead",
        in_channels=512,
        feat_channels=512,
        anchor_generator=dict(
            type="AnchorGenerator",
            scales=[4, 8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64],
        ),
        bbox_coder=dict(
            type="DeltaXYWHBBoxCoder",
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0],
        ),
        loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type="SmoothL1Loss", loss_weight=1.0),
    ),
    
    roi_head=dict(
        type="RSPrompterAnchorRoIPromptHead",
        with_extra_pe=True,
        iimr=dict(
            enabled=True,
            num_iterations=3,
            use_topology_memory=True,
            use_dynamic_prompting=True,
        ),
        bbox_roi_extractor=dict(
            type="SingleRoIExtractor",
            roi_layer=dict(type="RoIAlign", output_size=7, sampling_ratio=0),
            out_channels=512,
            featmap_strides=[4, 8, 16, 32],
        ),
        bbox_head=dict(
            type="Shared2FCBBoxHead",
            in_channels=512,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=num_classes,
            bbox_coder=dict(
                type="DeltaXYWHBBoxCoder",
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2],
            ),
            reg_class_agnostic=False,
            loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type="SmoothL1Loss", loss_weight=1.0),
        ),
        mask_roi_extractor=dict(
            type="SingleRoIExtractor",
            roi_layer=dict(type="RoIAlign", output_size=14, sampling_ratio=0),
            out_channels=512,
            featmap_strides=[4, 8, 16, 32],
        ),
        mask_head=dict(
            type="RSPrompterAnchorMaskHeadSAM2",
            sam2_mask_decoder=dict(
                checkpoint_path=sam2_ckpt_path,
                use_high_res_features=True,
            ),
            in_channels=512,
            roi_feat_size=14,
            per_pointset_point=prompt_shape[1],
            with_sincos=True,
            multimask_output=False,
            class_agnostic=True,
            loss_mask=dict(type="CrossEntropyLoss", use_mask=True, loss_weight=1.0),
        ),
    ),
    
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type="MaxIoUAssigner",
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1,
            ),
            sampler=dict(type="RandomSampler", num=256, pos_fraction=0.5, neg_pos_ub=-1, add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False,
        ),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type="nms", iou_threshold=0.7),
            min_bbox_size=0,
        ),
        rcnn=dict(
            assigner=dict(
                type="MaxIoUAssigner",
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=True,
                ignore_iof_thr=-1,
            ),
            sampler=dict(type="RandomSampler", num=256, pos_fraction=0.25, neg_pos_ub=-1, add_gt_as_proposals=True),
            mask_size=(512, 512),
            pos_weight=-1,
            debug=False,
        ),
    ),
    test_cfg=dict(
        rpn=dict(nms_pre=1000, max_per_img=1000, nms=dict(type="nms", iou_threshold=0.7), min_bbox_size=0),
        rcnn=dict(score_thr=0.05, nms=dict(type="nms", iou_threshold=0.5), max_per_img=100, mask_thr_binary=0.4),
    ),
    
    drone_branch=dict(
        use_sam2_backbone=True,
        freeze=False,
        dim=128,
        input_size=512,
        middle=[2, 2, 2, 2],
        scale=1.0,
        use_checkpoint=True,
        cross_view=dict(
            image_height=512,
            image_width=512,
            qkv_bias=True,
            heads=4,
            dim_head=32,
            no_image_features=False,
            skip=True,
        ),
        bev_embedding=dict(
            bev_height=128,# 改进: 80 → 128 (提升小目标表征能力)
            bev_width=128,   
            sigma=128,
        ),
        scene_alignment=dict(
            max_scenes=1000,
            embed_dim=32,
            init_scale=0.01,
        ),
        depth_encoder=dict(
            enabled=True,
            encoder="vits",
            freeze=True,
            use_depth_features=True,
            depth_dim=128, # 改进: 64 → 128 (增强深度特征表征)
            use_multiview_consistency=False,
            consistency_weight=0.1,
        ),
        model_size="large",
    ),
    
    guidance=dict(
        bev_dim=128,
        level_channels=[512, 512, 512, 512, 512],
    ),
    
    spatial_fusion_cfg=dict(
        num_heads=4,
        temperature=0.1,
        align_loss_weight=0.1,
        downsample_factor=1,
        max_attn_size=64,
        use_checkpoint=True,
        height_dim=128, # 改进: 64 → 128 (与depth_dim保持一致)
        use_height_gate=True,
        height_loss_weight=0.01,
        use_variance_loss=True,
        use_sparsity_loss=False,
        use_smoothness_loss=False,
        variance_weight=0.05,
        sparsity_weight=0.0,
        smoothness_weight=0.0,
    ),
    
    cross_view_loss=None,
    max_scenes=1000,
    sam2_ckpt_path=sam2_ckpt_path,
)

dataset_type = "SatDataset"
data_root = os.environ.get("SAT_DATA_ROOT", "data/portable_sat_fusion")

backend_args = None
train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True, poly2mask=False),
    dict(type="Resize", scale=(512, 512), keep_ratio=False),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PackDetInputs"),
]

test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="Resize", scale=(512, 512), keep_ratio=False),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True, poly2mask=False),
    dict(
        type="PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
    ),
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    batch_sampler=dict(type="AspectRatioBatchSampler"),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="annotations/instances_train.json",
        data_prefix=dict(img="images/train"),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args,
    ),
)

val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="annotations/instances_val.json",
        data_prefix=dict(img="images/val"),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args,
    ),
)

test_dataloader = val_dataloader

val_evaluator = dict(
    type="CocoMetric",
    ann_file=os.path.join(data_root, "annotations/instances_val.json"),
    metric=["bbox", "segm"],
    format_only=False,
    backend_args=backend_args,
)

test_evaluator = val_evaluator

train_cfg = dict(
    type="EpochBasedTrainLoop",
    max_epochs=100,
    val_interval=10,
)

val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(
        type="AdamW",
        lr=0.00005,
        weight_decay=0.01,
    ),
    paramwise_cfg=dict(
        custom_keys={
            "backbone": dict(lr_mult=0.1),
            "neck": dict(lr_mult=1.0),
            "roi_head": dict(lr_mult=1.0),
            "rpn_head": dict(lr_mult=1.0),
            "drone_encoder": dict(lr_mult=1.0),
            "guidance": dict(lr_mult=1.0),
        }
    ),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)

param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=0.1,
        by_epoch=True,
        begin=0,
        end=10,
        convert_to_iter_based=True,
    ),
    dict(
        type="CosineAnnealingLR",
        eta_min=0.000005,
        by_epoch=True,
        begin=10,
        end=100,
        convert_to_iter_based=True,
    ),
]

default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(type="CheckpointHook", interval=10, max_keep_ckpts=5),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="DetVisualizationHook"),
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)

vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(
    type="DetLocalVisualizer",
    vis_backends=vis_backends,
    name="visualizer",
)

log_processor = dict(type="LogProcessor", window_size=50, by_epoch=True)
log_level = "INFO"
load_from = None
resume = False
