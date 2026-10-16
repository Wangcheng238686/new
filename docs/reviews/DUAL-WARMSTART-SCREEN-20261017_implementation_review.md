# DUAL-WARMSTART-SCREEN 实施审计（2026-10-17）

审计对象：`a8dff49`（warm-screen 实现）+ `9f9e6df`（launch 修复）；
独立子代理审计于 2026-10-17 01:24 完成（Explore，只读），全文结论如下；
审计发现的阻断级偏差已当日修复并复冒烟（见文末闭环记录）。

## 审计结论（子代理原文要点）

**VERDICT: PASS**（8 个领域逐项 OK，无阻断问题），两条判读级 caveat：

1. **冻结语义偏差（已修复）**：`--freeze-base` 原实现对 allow-prefix 内参数
   无差别 `requires_grad_(True)`，重新启训了 peft-LoRA 冻结的 UAV 编码器基座
   （70.6M 全量，而非 0015 的 1.49M LoRA 集合）——违反"EXP-0015 配方逐字"
   意图，破坏 on−off 与 0015（+0.38）的可比性。预注册 optimizer 审计行
   本身披露了该参数量。
2. 探针读出训练长度（4000 步无早停 vs 预注册"≤200 epoch 早停"）与
   IoU(const) 操作化（逐场全 1 = 覆盖率，非训练均值常数图）偏离预注册字面，
   方向均为**不利于阳性结论**（保守）；sham 门用固定 −0.10 边距（预注册
   未指明显著性口径）。

其余领域（dual.py 门/头/辅助损失、freeze 时序与前缀安全、eval 重钉、
optimizer 分组、checkpointing 白名单语义、cli/脚本/配置单变量性、
eval/parity 脚本协议复用）均核验 OK。

## 修复闭环（2026-10-17 01:27）

- 偏差 1 修复：`train_rsprompter_fusion.py` 冻结规则改为
  **trainable = 前缀匹配 ∩ 构建时已可训**（保留 LoRA adapters 与
  modules_to_save neck 的 0015 精确可训集）。复冒烟证据：
  `other` 组 = 1,492,528 参数（=0015 LoRA 集），frozen 975 张量，
  fusion 组 135,168（r1）/323,265（r1b）。提交 `cdc9e822`。
- R1 首跑（epoch 5 内）按 PID 树终止、workdir 删除，未产生任何被引用的
  数字；重启后契约行 `WARMSCREEN_CONTRACT arm=r1 commit=cdc9e822 dirty=0`。
- 评估时序采纳审计建议：先跑 OFF 遍再跑 ON 遍（summary 的 on−off 块
  依赖 OFF rows 先存在）。
- 偏差 2 不修改（保守方向，不 Inflate 阳性），在结论文档判读边界中登记。

审计工件：子代理会话输出（sess_463660c0，agent_21f9245a）；
冒烟日志 `/tmp/warm_smoke_r1b2.log`（修复后）、`/tmp/warm_smoke_r1b.log`
（修复前，含 70.6M other 组证据行）。
