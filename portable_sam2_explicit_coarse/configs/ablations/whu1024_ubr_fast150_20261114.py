"""WHU-1024 fast150 协议重跑（EXP-0046）：UDPR 关停，native UBR 尾经 --native-ubr 安装。

与 a3sgm_whu1024.py 的架构差异唯一 = decoder_tail_refiner 不构造
（native UBR 残差头由训练器 --native-ubr 旗标在 build 后安装，见
rsprompter/native_ubr_integration.py）。分辨率族与 roi_local 坐标契约不变。

SAM2 底座三处 checkpoint 路径改按 SAM2_CKPT 环境变量解析（基 config 内是
lthpc 绝对路径，本机不存在会直接 build 失败）。架构指纹 schema v3 将
checkpoint/checkpoint_path 键归一化为 <EXTERNAL_CHECKPOINT>，本覆盖不改变
指纹，与论文行 parity 可证。协议预登记：
docs/evidence/whu_nwpu_ubr_rerun_protocol_20261114.md。
"""
import os

_base_ = '../a3sgm_whu1024.py'

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
