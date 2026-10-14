# EXP-0007 全量拉起前审计（节点 1，独立审计，2026-10-14）

审计对象：EXP-0007 启动准备（预注册 docs/EXP-0007_preregistration.md、
configs/ue_whu_a3sgm_512.py、data/ue_raw_pair_dataset.py 512 原生路径、
ue/contract.py 双基线身份注册表、train/train_rsprompter_fusion.py 传参、
smoke 工件 /data1/.../probes/EXP0007-smoke-20261014/）。

## 通过项

- **A 尺寸一致性**：递归语义 diff 确认两配置差异恰为七个声明尺寸字段
  （BatchFixedSizePad、shared_image_embedding.image_size、
  prompt_encoder_image_size、prompt_encoder_embed_size 64→32、shape_prior
  img_size、mask_size、coarse_mask_output_size 64→32），通道维与结构逐字
  一致；512 尺寸链自洽（embedding 32²=image/16、P3 64²=image/8、
  coarse=32=embed_size；anchor_detector 对 expected_hw 硬校验兜底）。
- **B 1024 封存不动**：a3sgm_whu1024.py 零改动，指纹复算
  `5d5f78ec…` 与封存声明一致；ue_whu_a3sgm.py（1024 UE 现役）不受影响；
  contract.py 新注册表对 1024 判定与旧逻辑等价；512 指纹
  `5305b3155914a2b2e7ecaf24b54957aee86fd6f47db93358daead2cbac5be36c`。
- **C 数据路径等价**：全部 manifest/split/计数契约校验位于尺寸分支之前、
  两路径逐字一致；512 路径仅去 resize、scale 1.0、GT 原生；UAV 不涉。
- **E（实现部分）**：smoke checkpoint 封印自验（baseline_id/指纹/512/
  validation_only/SAM2 SHA/dirty echo）齐备，eval 有限非 NaN。

## 阻断项（已处置）

- **D 启动协议缺口**：考古 EXP-0006 真实旗标（epochs 100、early-stopping
  0、compute-val-loss 0、amp 1、ema 0/0、SAVE_LAST_MODEL=1）；原拟命令的
  默认值（80 epoch、EMA 开、早停武装、无 last 落盘）违反预注册对齐条款。
  → 处置：专用 wrapper `scripts/ue/run_exp0007_ddp4.sh` 固化全部旗标与
  SHA/指纹/干净树守卫，不依赖手工追加；启动 commit `87864eb` + tag
  `EXP-0007`。

## 登记待办（开封前，非启动阻塞）

- `inference/eval_records.py` UE 推理路径需按 checkpoint
  data_config.image_size 参数化（当前钉 1024），EXP-0007 test 开封事件前
  必须完成。
