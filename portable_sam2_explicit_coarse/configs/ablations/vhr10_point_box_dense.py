"""论文消融表 P+B+M 行（a0_sg 底座）：PBM 全提示，仅去掉 UDPR 尾。"""
_base_ = ['../a3sgm_vhr10.py']

model = dict(
    roi_head=dict(
        mask_head=dict(
            decoder_tail_refiner_cfg=dict(_delete_=True),
        ),
    ),
)
