# EXP-0015 / EXP-0016 预注册：路线内双臂（2026-10-16，启动前）

PI 决议（2026-10-16）：**不换路线**——继续在双流门控注入路线上寻找双指标增益；
批准并行两臂，均含 **门控正则豁免**（DUAL-WHY-NULL 诊断点名的关闭力之一：
wd 0.05 持续把零初始化门拉回零）。诊断依据：0012 门控扫描 k=2 出现 +0.40 点
峰值（门欠开）、BEV 96% 卫星独特但任务对齐弱（fg/bg 1.19）、优化动力学把通路
推向惰性/代偿。

## 共同杠杆：fusion 免 weight decay（trainer 侧单变量）

- 实现：`train/optimizer.py` 新增 `fusion_no_wd` 开关——匹配 RSPrompterAnchorDual
  顶层融合参数（`gate`/`proj`/`gate_e`/`proj_e` 含 ModuleDict 子项）单独成组，
  `weight_decay=0.0`，**学习率与其原所在 other 组完全一致**（lr×other_lr_mult）。
  不含任何其他改动（无门控学习率加成——那是另一个变量，未批准）。
- 口径：包装器契约 echo `wd=0.05 … fusion_no_wd=1`，trainer 参数
  `--fusion-no-wd 1`；全局 wd 与既有臂一致，唯一差异是融合参数组。
- 关闭条件下（不传参）行为与既有臂逐位一致（参数组划分不变）。

## EXP-0015 = 1+2：wd 豁免 + LoRA + 仅P2

- 配置 `configs/ue_whu_a3sgm_512_dualv2_l0_lora.py`（身份
  `ue-whu-a3sgm-512-dualv2-l0-lora-v1`，指纹 `5c7e1afc…`）= EXP-0013 配置 +
  uav_encoder `use_lora: True`（编码器内置 r=16/α=32/qkv+proj/neck 配方，
  与 EXP-0012 同款）。
- 变量结构（组合臂，如实登记）：vs 0013 = +LoRA +wd豁免；vs 0012 = +wd豁免
  −L1 分支。归因依赖既有阶梯（0012−0011 已隔离 LoRA、0013−0011 已隔离 L0），
  本臂主 A/B 仍对 EXP-0009。
- 预期机制：LoRA 是已证唯一显著正向单变量（+0.56/+0.49），0012 表型为
  "依赖但门欠开"——wd 豁免直接攻其欠开机制。

## EXP-0016 = 1+4：wd 豁免 + image_embeddings 注入点

- 配置 `configs/ue_whu_a3sgm_512_dualembed.py`（身份
  `ue-whu-a3sgm-512-dualembed-v1`，指纹 `12a63439…`）：`target_levels: []`
  （无金字塔注入）+ `embed_fusion: True, embed_channels: 256`——UAV BEV
  （262ch，光栅化到 32×32=stride-16 网格）经 1×1 投影 + 零初始化逐通道门控
  加到 **SAM2 掩码解码器的稠密条件场 image_embeddings** 上；冻结 UAV 编码器。
- 注入点依据：拉通分析确认 BEV 现在到不了掩码合成主通路（image_embeddings/
  high_res 均绕过 PAFN）；本臂把注入移到合成点。**单注入位**：不动
  high_res_features（留作后续消融）。
- 变量结构（组合臂）：vs 0010 = 注入点（P3→embedding）+ wd豁免；主 A/B 对
  EXP-0009。
- 遥测：`_E` 后缀键（gate_mean_E 等）；步骤0等价（gate_e=0 → 与单流严格
  等价，eval 口径逐位，卫星主干 LoRA dropout 判读边界同 0013 预注册）。
- 风险登记：扰动预训练 embedding 可能破坏 prompt-encoder 对齐（prompt
  encoder 冻结、其先验按预训练分布设计）；零初始化门控缓解（开度由训练
  决定）。32×32 分辨率较粗，UAV 细节被压缩——若本臂惰性，后续消融应先试
  high_res 注入位而非直接判路线。

## 四要素（两臂共同继承）

1. 主指标：val segm/bbox AP@[.50:.95] 各臂 − EXP-0009 差；伴随：AP50/AP75、
   UAV-off 消融差、门控遥测全套（0015 单层无后缀键；0016 `_E` 键）。
2. 选择口径：last epoch 100（协议与全阶梯逐项一致：v2 快照、seed 44、
   DDP4 batch1×accum2、lr 5e-4、wd 0.05（融合组 0）、100 epoch、无增强、
   禁早停、test 不开封）。
3. 显著性：val 100 场 scene-paired percentile bootstrap（10,000 draws、
   seed 20261014，拉通工具同口径）。
4. 判定门：DUAL_GAIN / DUAL_ENGAGE_NULL / DUAL_USED_BUT_NULL / DUAL_HARM
   沿用；参与度算子用 v2.3 有效开度（Σ|Δ|/Σ|proj|，无量纲；legacy δ_L 已
   证伪）。0016 的开度算子在 embedding 位（open_E）。

## 预注册解读条款

- 0015 若 DUAL_GAIN 或两指标 CI 下界 >0 → wd 豁免+LoRA 组合成立，成为
  新基准臂；若仍持平/负 → "门欠开"不是唯一压制机制，下一变量指向
  训练动力学（延迟解冻）。
- 0016 若 segm 增益 > bbox（预期非对称）→ 与注入点机制一致（embedding 只
  喂掩码合成）；若惰性（开度 < 0.20 且 on−off 不显著）→ 注入位仍太粗，
  high_res 消融优先。
- 两臂同批并行共租四卡；A/B 与落判并入拉通工具（ladder 需为 `_E` 键与
  空 target_levels 补 delta_operator 分支——结论期工具改动，届时登记）。

## 工件与审计

wrapper `scripts/ue/run_exp0015_ddp4.sh`、`scripts/ue/run_exp0016_ddp4.sh`
（守卫族同 0011/0013 + fusion_no_wd 契约）；启动前独立审计（实施+接线）；
首次 epoch 遥测核验；结论前结果审计。冒烟须覆盖：0015 LoRA 端到端梯度 +
optimizer fusion 组 wd=0；0016 embed 前向 + 步骤0等价（eval 逐位）+
`gate_e` 梯度非零 + `_E` 遥测键。

### 启动前冒烟结果（2026-10-16，真实冻结快照 val 场景，全通过）

- 0015：LoRA 可训 1,492,528（与 0012 登记数一致）；gate=0 初始门控梯度>0
  （防驻点性质保持），gate=0.01 开启后 LoRA 端到端梯度和>0（with_grad
  光栅链在 L0/128² 复验）；`fusion_no_wd=1` 时 gate/proj 三张量入 fusion
  组（wd=0.0、lr=other 同款），`fusion_no_wd=0` 时回到 other 组（关闭
  条件行为逐位一致）。
- 0016：`target_levels=[]`+embed 单点结构正确；训练态 `_E` 遥测键五项、
  32² 网格 coverage 0.9707；gate_e 梯度>0；eval 口径 gate_e=0 全输出逐位
  等价；gate_e=0.01 扰动只改 image_embeddings、金字塔四层逐位不变；
  gate_e/proj_e 入 fusion 组 wd=0。
- 判读边界重申（同 0013 预注册）：步骤0逐位等价只在 eval 口径下成立
  （卫星主干 LoRA dropout 0.05 训练态激活）。
