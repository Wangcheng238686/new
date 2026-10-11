# SURFBEV-B0 1024 全臂 harness（训练前）

在 wrapper 固定的 `EXPLICIT_PROMPT_MODE=points_box_dense`、`P2_BOUNDARY_REFINER_ENABLED=1`、
`SAM_IMAGE_EMBED_STRIDE=16`、`SHAPE_DENSE_TRANSFORM=raw_logits`、`SHAPE_DENSE_DETACH=1`、
`FINAL_MASK_COORDINATE_MODE=roi_local` 和同一 SAM2 ckpt 环境下解析 `Config.fromfile`，用
`architecture_contract` 得到：

| 臂 config | architecture ID | model fingerprint |
| --- | --- | --- |
| `whu1024_baseplus_explicit_coarse.py` | `c5v2_pafpn_coarse_p2_boundary_refiner_emb64` | `36549a00a8282b2b9903d8c07c7cc9ce6f6e8fba3e93d58212de1999b1aa3dab` |
| `ue_a1_p2_z0_hotstart.py` | `ue_a1_p2_z0__c5v2_pafpn_coarse_p2_boundary_refiner_emb64` | `6f6f10dedf0de9347bb8ec22fadacdf461c5440beb10bdc7a6337ebb043f5538` |
| `ue_dual_stream_p2p3_sparse.py` | `ue_dual_p2p3_sparse__c5v2_pafpn_coarse_p2_boundary_refiner_emb64` | `7e3ce8fa3466ac6936b8a6f270ac7be022ccf4605bc0e9e79351f43956816f90` |
| `ue_raw_hero_b0.py` | `c5v2_pafpn_coarse_p2_boundary_refiner_emb64` | `36549a00a8282b2b9903d8c07c7cc9ce6f6e8fba3e93d58212de1999b1aa3dab` |

`B0.model.to_dict() == WHU1024.model.to_dict()` 为真，模型键集差为空；B0 唯一变更是
数据、空间 split、训练 run 身份。旧三个 config 及它们依赖的模型实现未在本轮修改，
与 `f4f31a6` 相比解析输入不变，因此旧臂 fingerprint 不变。

验证：raw train/val/test DataLoader 数量 949/100/151，1024 图像与 mask、两倍 bbox；
GPU2 实际 1024 前向得到 P2 256²、P3 128²、SAM 64²；同一场景 loss=4.77194 且
`backward()` 成功，峰值 reserved 10.81 GiB。以上只证明接线与资源可行，不能证明
训练收敛或 AP 增益。
