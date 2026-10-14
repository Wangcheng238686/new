"""512 satellite-only control, trained independently on the paired UE split."""
_base_ = '../a3sgm_whu1024.py'

model = dict(
    data_preprocessor=dict(batch_augments=[dict(type='BatchFixedSizePad',size=(512,512),
                                                img_pad_value=0,pad_mask=True,mask_pad_value=0,pad_seg=False)]),
    shared_image_embedding=dict(image_size=512),
    roi_head=dict(mask_head=dict(prompt_encoder_image_size=512,prompt_encoder_embed_size=32,
                                 shape_prior_cfg=dict(img_size=512))),
    train_cfg=dict(rcnn=dict(mask_size=(512,512))),
    test_cfg=dict(rcnn=dict(max_per_img=300)),
)
ue_geometry = dict(side_m=512.,grid_size=128,image_size=512,source_stride=8,
                   radius_m=8.,candidates_per_view=8,depth_sample_stride=4,
                   height_bin_m=2.,min_height_points=3,collision_band_m=2.)
