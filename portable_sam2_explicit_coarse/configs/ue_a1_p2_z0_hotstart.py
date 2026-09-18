"""A1 hot-start: frozen UAV encoder, z=0 geometry, P2-only residual."""
import os

_base_ = ["./whu1024_baseplus_explicit_coarse.py"]

sam2_ckpt_path = os.environ.get(
    "SAM2_CKPT", "/data1/wangcheng/pretrained/sam2_hiera_base_plus.pt"
)

model = dict(
    type="A1P2RSPrompterAnchor",
    drone_backbone=dict(
        type="RSSAM2VisionEncoder",
        checkpoint_path=sam2_ckpt_path,
        init_cfg=dict(type="Pretrained", checkpoint=sam2_ckpt_path),
        model_size="base_plus",
        freeze_backbone=True,
        use_gradient_checkpointing=False,
    ),
    # Only these new parameters are trainable during A1 hot-start.  P2 is
    # 256x256 / z=0, with at least two valid UAV views required per query.
    a1_cfg=dict(channels=512, uav_channels=256, query_chunk=2048, target_meters=360.0),
)
