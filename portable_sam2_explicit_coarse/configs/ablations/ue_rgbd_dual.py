"""Independent depth-CVT BEV with both satellite fusion routes."""
_base_ = 'ue_rgbd_single.py'

model = dict(
    type='RSPrompterUAVDual',
    uav_backbone=dict(type='RSSAM2VisionEncoder',
                      checkpoint_path='/data/wangcheng/pretrained-models/sam2/sam2_hiera_base_plus.pt',
                      use_gradient_checkpointing=True,use_lora=True,freeze_backbone=False,
                      model_size='base_plus',
                      lora_config=dict(r=16,lora_alpha=32,target_modules=['qkv','proj'],
                                       lora_dropout=.05,bias='none',modules_to_save=['neck'])),
    uav_bev=dict(geometry=_base_.ue_geometry,channels=256,layers=2,heads=8,
                 ffn_dim=512,query_chunk=1024,checkpoint_attention=True),
    fpn_fusion=dict(levels=(1,2),uav_channels=256,alpha_init=.05),
    sam_fusion=dict(uav_channels=256,alpha_init=.05),
    uav_encoder_chunk=5,
)
