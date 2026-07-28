"""
RSPrompter with SAM2-Hiera-Large - 完整改进版本
主要改进:
1. ✅ 启用 use_high_res_features (高分辨率特征增强mask decoder)
2. ✅ 改进 LoRA 配置 (r=16, alpha=32, dropout=0.05)
3. ✅ 修复所有已知的维度匹配问题
4. ✅ 优化学习率策略
"""
import os
import importlib.util

work_dir = os.environ.get("WORK_DIR", "./work_dirs/portable_sam_fusion/rsprompter_anchor_satS_v11_sam2_large_full")

num_classes = 1
prompt_shape = (100, 5)  # 与 SAM1 基线一致

# --- 问题 2 修复：SAM2 model_size / checkpoint / neck_in_channels 由单一 registry 统一解析 ---
# 之前 model_size="large"、neck in_channels="sam2_hiera_large"、sam2_ckpt_path 三处各自硬编码，
# 导致 baseplus config 名实不符（文件名 baseplus 实际跑 large）。现在统一从 env SAM2_MODEL_SIZE
# / SAM2_CKPT 解析，默认 "large" 保持历史复现行为。
#
# 注意：mmengine Config.fromfile 用 eval 执行本文件，执行环境里没有 __file__，
# 因此不能用 os.path.dirname(__file__)，改用 cwd + 已知相对路径查找 registry。
def _load_sam2_registry():
    _candidates = [
        os.path.join(os.getcwd(), "configs", "_sam2_registry.py"),       # 从项目根运行（start 脚本 cd 到此）
        os.path.join(
            os.getcwd(),
            "portable_sam2_explicit_coarse",
            "configs",
            "_sam2_registry.py",
        ),  # 从派生项目仓库根运行
        os.path.join(os.getcwd(), "_sam2_registry.py"),                  # cwd 恰好在 configs/
        os.path.join(os.path.dirname(os.getcwd()), "configs", "_sam2_registry.py"),  # cwd 在项目子目录
    ]
    _reg_path = next((p for p in _candidates if os.path.isfile(p)), None)
    if _reg_path is None:
        raise FileNotFoundError(
            "Cannot find configs/_sam2_registry.py. Run from project root. "
            f"Tried: {_candidates}"
        )
    _spec = importlib.util.spec_from_file_location("_sam2_registry", _reg_path)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    return _mod

_sam2_reg = _load_sam2_registry()
_sam2 = _sam2_reg.build_sam2_cfg()

sam2_ckpt_path = _sam2.checkpoint
sam2_config_path = _sam2.yaml
sam2_model_size = _sam2.model_size  # 供启动 manifest 与下游断言使用

neck_type = os.environ.get("NECK_TYPE", "pafpn").lower()
if neck_type == "aggregator":
    neck_cfg = dict(
        type="RSFeatureAggregatorSAM2",
        in_channels=_sam2.neck_in_channels,
        out_channels=512,
        hidden_channels=64,
        select_layers=[0, 1, 2, 3],
    )
elif neck_type == "pafpn":
    # mid_channels controls internal FPN/PAN width; out_channels=512 is the
    # detector interface contract. mid=256 (== backbone output width) keeps
    # full information while cutting N×N conv cost 4×, and a per-level 1×1
    # projection lifts mid→out at the end so downstream in_channels stays 512.
    # PAFPN_MID_CHANNELS=512 reproduces the legacy all-512 behavior.
    neck_cfg = dict(
        type="RSSAM2PAFPN",
        in_channels=_sam2.neck_in_channels,
        out_channels=512,
        mid_channels=int(os.environ.get("PAFPN_MID_CHANNELS", "256")),
        num_levels=4,
        norm_groups=32,
    )
else:
    raise ValueError(f"Unsupported NECK_TYPE={neck_type!r}; expected 'pafpn' or 'aggregator'")

model = dict(
    type="RSPrompterAnchor",
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
        # 改进的 LoRA 配置
        use_lora=True,
        lora_config=dict(
            r=16,  # 增加秩以获得更强的适应能力
            lora_alpha=32,  # 增加alpha以增强LoRA效果
            target_modules=["qkv", "proj"],  # 只使用支持的模块
            lora_dropout=0.05,  # 降低dropout以获得更好的收敛
            bias="none",
            modules_to_save=["neck"],  # neck完全可训练
        ),
        freeze_backbone=False,
        freeze_backbone_stages=-1,
        model_size=sam2_model_size,  # 由 _sam2_registry 统一解析（问题 2）
    ),
    neck=neck_cfg,
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
                use_high_res_features=True,  # ✅ 启用高分辨率特征
            ),
            in_channels=512,
            roi_feat_size=14,
            per_pointset_point=prompt_shape[1],
            with_sincos=True,
            multimask_output=False,
            class_agnostic=True,
            dense_prompt_mode=os.environ.get("PROMPT_DENSE_MODE", "none"),
            freeze_mask_decoder=os.environ.get("FREEZE_DECODER", "1") != "0",
            freeze_no_mask_embed=os.environ.get("FREEZE_NO_MASK", "1") != "0",
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
)

# 数据集配置
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

# 训练 schedule - 优化学习率
train_cfg = dict(
    type="EpochBasedTrainLoop",
    max_epochs=100,
    val_interval=10,
)

val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

# 优化器 - 改进学习率
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(
        type="AdamW",
        lr=0.00005,  # 稍高的基础学习率
        weight_decay=0.01,
    ),
    paramwise_cfg=dict(
        custom_keys={
            "backbone": dict(lr_mult=0.1),  # LoRA backbone 低学习率,保护预训练特征
            "neck": dict(lr_mult=1.0),
            "roi_head": dict(lr_mult=1.0),
            "rpn_head": dict(lr_mult=1.0),
        }
    ),
)

# 学习率调度 - 使用更平滑的衰减
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

# 日志和检查点
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
