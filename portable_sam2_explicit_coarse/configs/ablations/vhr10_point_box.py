"""论文消融表 P+B 行：点+框双离散提示（无 dense/无尾）。"""
_base_ = ['../a3sgm_vhr10.py']

model = dict(
    roi_head=dict(
        mask_head=dict(
            explicit_prompt_mode='points_box',
            shape_prior_cfg=dict(use_shape_dense=False),
            dense_prompt_cfg=dict(detach_input=False),
            prompt_encoder_cfg=dict(train_mask_downscaling=False),
            decoder_tail_refiner_cfg=dict(_delete_=True),
        ),
    ),
)
