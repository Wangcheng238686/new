"""Stage-1 UE dual stream: frozen UAV SAM2 + sparse geometric BEV -> P2/P3."""
import os

_base_ = ["./whu1024_baseplus_explicit_coarse.py"]

sam2_ckpt_path = os.environ.get(
    "SAM2_CKPT", "/data/wangcheng/pretrained-models/sam2/sam2_hiera_base_plus.pt"
)

model = dict(
    type="DualStreamRSPrompterAnchor",
    # This is deliberately another encoder instance.  It has the same original
    # Base+ checkpoint as satellite, but no shared trainable parameters.
    drone_backbone=dict(
        type="RSSAM2VisionEncoder",
        checkpoint_path=sam2_ckpt_path,
        init_cfg=dict(type="Pretrained", checkpoint=sam2_ckpt_path),
        model_size="base_plus",
        freeze_backbone=True,
        use_gradient_checkpointing=False,
    ),
    fusion_cfg=dict(
        channels=512,
        bev_dim=128,
        heads=4,
        query_chunk=256,
        windows=(7, 5, 3, 3),
        target_meters=360.0,
        # Also inject the BEV into the SAM2 stride-16 image embedding canvas
        # (the only feature the mask decoder attends to directly).  Toggled by
        # env so the P2P3-only control config stays untouched.
        inject_embedding=os.environ.get("UE_DUAL_INJECT_EMBEDDING", "0") == "1",
        embed_channels=256,
    ),
)
