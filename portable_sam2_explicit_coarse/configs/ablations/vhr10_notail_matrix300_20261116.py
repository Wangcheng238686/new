"""NWPU VHR-10 matrix300 UBR 模块消融（EXP-0050）：无尾形态（UDPR 关停且不装 UBR）。

与 EXP-0047（vhr10_ubr_matrix300_20261114.py）的唯一变量 = 不安装 native UBR
（也不构造 UDPR）——尾模块整体缺席，模型为论文 P/B 系同款 Tail-less prompt-path
形态（mask_head 无尾 else 分支，decoder 三元组契约）。其余逐键一致：分辨率族
1024、full_image + roi_balanced_dice 坐标/监督契约、PE downscaling 可训、
num_classes 10。SAM2 底座三处路径按 SAM2_CKPT 环境解析（指纹 schema v3 归一化，
不影响 parity 论证）。协议预登记：docs/evidence/whu_nwpu_ubr_rerun_protocol_20261114.md
"UBR 模块消融"节。
"""
import os

_base_ = '../a3sgm_vhr10.py'

_SAM2 = os.environ.get(
    'SAM2_CKPT', '/data1/wangcheng/pretrained/sam2_hiera_base_plus.pt'
)

model = dict(
    backbone=dict(
        checkpoint_path=_SAM2,
        init_cfg=dict(checkpoint=_SAM2),
    ),
    roi_head=dict(
        mask_head=dict(
            decoder_tail_refiner_cfg=dict(_delete_=True, enabled=False),
            sam2_mask_decoder=dict(checkpoint_path=_SAM2),
        ),
    ),
)
