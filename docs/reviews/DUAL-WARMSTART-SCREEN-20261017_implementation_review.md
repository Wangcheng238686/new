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

## 追加：WARM-R1C 模块审计（2026-10-17 17:05）

首轮 VERDICT: **FAIL**（4 项阻断：锚点 denom clamp_min 符号翻转、旋转方向
反用 R_w2c、查询/键网格未换算 512 内参帧、wrapper init 白名单漏 bev_attn）
——全部当日修复：符号保持 denom、R_c2w 行形式、×512/N 网格换算、白名单
补齐；数值自检（锚点跨度 35,438cm≈360m tile、z≡0）+ 训练路径重冒烟通过。
增量复核 VERDICT: **PASS**（4/4 修复确认）。复核注记的 512 参数差异系
proj Conv2d 的 bias（审计计数 164,608 漏加，实为 165,120）。

## 追加 2：WARM-R1Q 模块审计（2026-10-18 01:00，三轮闭环）

首轮 **FAIL**（B1 sat_feat 前向实现整体缺失——并发编辑使两处 replace 静默
未命中；B2 sat_feat 关键字回归 R1C/R1CH；B3 阶梯 wrapper 签名不兼容）。
修复后复核 **FAIL**（残留：调用点传 4D 批维 neck 输出而前向只支持 3D；
const_* 仍吞 sat_feat）。二轮修复（4D squeeze + 显式 assert + 双透传，
经真实调用点形状验证）后 **VERDICT: PASS**。修复链提交：
548d84e → c1cd545。教训登记：对并发编辑过的文件，replace 必须带 assert。

## 追加 3：WARM-R1R 审计（2026-10-19 10:20，首轮 FAIL→修复→重启）

首轮 **FAIL** 两阻断：(1) zmean 查表帧 bug——512 帧 sat 像素按 128 宽度
归一（u/W*2-1），≥15/16 图幅 z 读数落 OOB→0 或错位 4×；(2) bbox 适配器
结构性死路——delta 加在 bbox_head 已消费的 bbox_feats 上，无下游读者，
bbox 指标恒等于基线。ep11 的带 bug 筛查已终止删除（无工件引用）。
修复（当前 commit）：z 查表按 512 帧归一；_bbox_forward 复刻父类并在
bbox_head 之前注入 delta。复冒烟绿（loss 有限、train 0.154）。
次要登记未改（半像素约定差 ~1-2px，量级可忽略）。
