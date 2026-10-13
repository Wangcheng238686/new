"""Frozen independent UAV encoder; same settings as the archived UE route."""
import os

checkpoint = os.environ["SAM2_CKPT"]
uav_encoder = dict(
    type="RSSAM2VisionEncoder", checkpoint_path=checkpoint,
    init_cfg=dict(type="Pretrained", checkpoint=checkpoint), model_size="base_plus",
    freeze_backbone=True, use_gradient_checkpointing=False,
)
