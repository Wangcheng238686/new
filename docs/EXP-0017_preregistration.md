# EXP-0017 预注册：0015 配方 + 融合延迟解冻（2026-10-16，启动前）

PI 决议（2026-10-16）：继续 0015 谱系。EXP-0015 预注册解读条款兑现——
"若仍持平/负 → 门欠开不是唯一压制机制，下一变量指向训练动力学（融合
延迟解冻）"。0015 的机制分解：UAV 收益 on−off = +0.38/+0.34（双显著），
但训练期代价 off−single = −0.35/−0.58 几乎恰好抵消。

## 单变量

**vs EXP-0015 唯一差异 = `--fusion-unlock-epoch 50`**：融合门控在第 1–50
epoch（含）被值钉零（每个 optimizer step 后重置为 0），第 51 epoch 起自由
学习。配置、身份（`ue-whu-a3sgm-512-dualv2-l0-lora-v1`）、协议（含
`--fusion-no-wd 1`）与 0015 逐字一致。身份复用遵循 EXP-0007/0009 先例，
由 wrapper 契约行（`fusion_unlock_epoch=50`）、RUN_TAG（`…fk50…`）、tag
与登记表区分。

机制：锁定期 gate=0 → 整条 UAV 支路（含 LoRA）梯度恒零（∂x/∂bev=gate=0），
卫星侧在纯净的单流等价目标下收敛 50 epoch；解锁后 gate 与 UAV-LoRA 只在
成熟基座上学习。实现为值钉零（不改 requires_grad），DDP 桶与优化器成员
资格与 0015 完全一致；关闭开关（默认 0）行为与 0015 逐位一致。

解锁点 50 的依据：0015 轨迹 ep50 时 segm ≈ 0.74+（基座成熟），cosine
剩余学习率 = 峰值一半（2.5e-4），给 gate+UAV-LoRA 留 50 epoch 学习窗。

## 四要素（继承 EXP-0008 谱系 + 0015-0016 预注册）

1. 主指标：val segm/bbox AP@[.50:.95] − EXP-0009 差（同表对照 0015）；
   伴随：UAV-off 消融差、开度遥测全套。
2. 选择口径：last epoch 100。
3. 显著性：val 100 场 scene-paired percentile bootstrap（10,000 draws、
   seed 20261014，拉通工具同口径）。
4. 判定门：DUAL_GAIN / DUAL_ENGAGE_NULL / DUAL_USED_BUT_NULL / DUAL_HARM
   沿用；参与度用 v2.3 有效开度 + on−off（legacy δ_L 已证伪）。

## 机械核验线（锁定期完整性）

- epoch 1–50 的遥测 `delta_p3_ratio` 必须恒为 0.000000（gate 钉零的逐
  epoch 证据）；第 51 epoch 起 >0。
- 训练日志须含 "fusion gates pinned to zero through epoch 50" 与
  "fusion gate unlock takes effect at epoch 51" 两行（rank0）。

## 预注册解读条款

- **主预期**：off−single 代价项向 0 收缩 → 净差 ≈ on−off 收益
  （0015 为 +0.38/+0.34）→ **DUAL_GAIN 首次成为现实可能**；
- 若解锁后门控不再开启（注入比停在 0，惰性复发）→ 成熟基座上训练拒绝
  开门，延迟解冻反例成立，如实登记；
- 若净差仍持平且 off−single 仍 ≈ −0.4 → 代价非早期扰动而是持续存在
  （如梯度竞争/容量挤占），转向正则或容量侧变量；
- 若 DUAL_HARM → 该谱系停止叠加，复盘。

## 风险与边界

- 锁定期的基座轨迹与 0015 前半程同分布但不完全同轨迹（随机数消费次序
  不同）；本臂不声称与 0015 前半程逐位一致。
- 算力：100 epoch 全程（与单臂同），无早停。
- 机制遥测不替代结局验证；test 不开封。
