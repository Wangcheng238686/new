# EXP-0007 预注册：封存 a3sgm @ 512 原生 UE 锚点（ue-whu-a3sgm-512-v1）

## 路由与动机

基座级 run（EXP-ID 发号 EXP-0007，tag 打在启动 commit）。动机链条：DUAL-R0
（R0_MIXED）证明 1024 上采样锚点上 UAV 可兑现 bbox 空间 ≈ +1.8 点且 62% 漏检
UAV 不可见；用户决议建立**原生 512 锚点**重开空间（卫星原生 512、UAV 原生
512 低空——信息不对称按物理 GSD 定义，不用插值补贴卫星侧），并一并清偿
"封存基座从未在 UE 训练"与"身份链载不进历史锚点"两笔债。

## 协议声明（claim 口径，用户已定方向）

- **主协议 = 原生 GSD 评测**：输入与 GT 均为磁盘原生 512 栅格，无任何插值
  放大。EXP-0006（1024 上采样，c5v2 系）保留为**算力对照参考**，不作为被比
  基线；双流增益主张一律以本协议表述。
- 架构：封存 WHU a3sgm（与 95026ae 字节一致移植），输入尺寸协议 512。因
  architecture fingerprint 哈希全部模型配置键，512 尺寸字段构成指纹差异，
  故立**新基线身份 `ue-whu-a3sgm-512-v1`**（1024 封存身份
  `ue-whu-a3sgm-v1` 原样保留）；契约链扩展为双身份注册表，新 checkpoint 按
  新身份封印，修复后 build_baseline 可载入两代锚点。
- 数据：`ue_raw_hero_uav_pairset_20261011` + `surfbev_b0_split_v2.json`
  （SHA `17f49430…`，v2 为最新防泄露 split），694/100/151，无随机增强，
  seed 44，从预训练 SAM2 全新开始——除"输入尺寸 512 + 架构换封存系"外协议
  与 EXP-0006 对齐。变量混淆（架构×分辨率）声明接受：架构切换是既定迁移
  方向，分辨率效应如需归因再补对照 run。

## 四要素

1. 主指标：**test COCO segm AP@[.50:.95]** 与 **bbox AP@[.50:.95]**
   （151 场 test；开封为独立事件——last checkpoint 产生后按冻结协议推理一次，
   此前任何人不得读 test）。伴随：AP50/AP75、面积三分位分层 AP、按
   EXP-0006@1024 的算力对照描述（非配对主张）。
2. 选择口径：每 seed 固定 **last epoch**（100 epoch，同 EXP-0006）。
3. 显著性：本 run 为锚点建立 run，不做增益主张；其上的 DUAL 探针按各自
   预注册的 scene-paired bootstrap。
4. 门：训练本身无判定门（失败=训练事故，按早停纪律处理）；后续方向门在新
   锚点上的 DUAL-R0 重跑预注册中定义（漏检质量 × 可查询率的可攻击空间）。

## 启动与遥测

- 启动：`scripts/ue/run_ue_whu_b0.sh`（RUN_TAG=EXP-0007，新 WORK_DIR），四卡
  DDP、契约 echo（config dump、git commit/dirty、SAM2 SHA、种子、超参全量
  落盘）；启动前 commit + 打 tag `EXP-0007`。
- 先决 smoke：单卡 1-epoch 限量（若干 iter 前反向 + val 1 场 + checkpoint
  存取往返 + 身份封印自验）通过并经独立审计后启动四卡。
- 遥测三件套：训练日志标准字段；512 路径的 P3 形状（512 输入 → P3 64²）与
  SAM2 位置编码插值入 smoke 校验；机制遥测不替代结局。

## 工件

checkpoint/日志：`/data1/wangcheng/checkpoint/surfbev_b0/surfacebev_v2_512_sealed_ddp4_s44_EXP-0007/`
（目录名启动时定稿登记）；结论 MD 于训练完成 + test 开封后另立。

## 事故追加（2026-10-14）

训练窗口内生产数据集被 componentclean_v2 修复管线实时改写（49 train/15 val
GT），本 run 作废为污染记录，由 EXP-0009（冻结快照、协议逐项一致）替代。
