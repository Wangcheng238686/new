"""论文消融表 P+B 行·matrix300 重跑（queued-prepared，无尾形态镜像论文）。

= 论文 vhr10_point_box.py（prompt 组成消融 delta：explicit_prompt_mode /
use_shape_dense=False / dense detach_input=False / PE downscaling 关 / UDPR 关）
+ SAM2 底座三处路径本机化（指纹 schema v3 归一化）。**不装任何尾模块**——
与论文消融表行同构（prompt 轴在无尾形态下隔离）。判读：本臂 vs 论文 P+B 行
（同口径对照）+ 消融表内四行（P/B/P+B/P+B+M=EXP-0050）+ 尾轴（0050 vs 0047）。
协议：matrix300 冻结合同。预登记：whu_nwpu_ubr_rerun_protocol_20261114.md。
"""
import os

_base_ = './vhr10_point_box.py'

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
            sam2_mask_decoder=dict(checkpoint_path=_SAM2),
        ),
    ),
)
