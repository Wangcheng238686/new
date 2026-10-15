# EXP-0014 预注册：BEV 逐通道定标融合臂（dualnorm）

临时号 SURFBEV-snap512dualnorm-sealed-DDP4-v2-noaug；EXP-ID 于全量启动
commit 登记发放（本文件即 EXP-0014）。立项目期 2026-10-16。

## 背景与假设（证据先行）

DUAL-DELTA-DECOMP-20261016 机制探针（docs/DUAL-DELTA-DECOMP-20261016_conclusion.md）
量化了 EXP-0010 融合通路的量纲缺陷：单 bin BEV 262 通道中 zmean（≈4297 cm）与
zres（≈576 cm）比 256 维 UAV 语义描述子（≈0.018）大 5 个数量级；1×1 conv
默认初始化下投影范数为卫星 P3 的 62 倍、35.5% 由这两个几何通道构成，
语义描述子贡献仅 0.04%——**语义内容在数值上从未参与注入**；训练把门关至
0.44% 有效开度（预注册算子 δ_L≈9e-5），泄漏成分类偏置（46% 空间常数能量），
AP 持平。

**本臂假设**：修复量纲通路（BEV 逐通道定标至可比量级）后，(i) 语义描述子
获得数值参与资格，投影由全通道构成而非两个 cm 通道；(ii) 若训练仍不开门
（有效开度 <0.20）或开门后指标仍无增益，则归因从"数值通路缺陷"转移到
"任务层不需要密集注入"，与 P1–P4/SZAB 先验一致，密集注入路线最终关账；
(iii) 若 DUAL_GAIN 达成，则推翻四连探针先验，立即升级两 seed 复验。

## 路由判定

EXP-0008 谱系（从头联合训练双臂，A/B 对 EXP-0009 冻结快照单流锚点）。
热启动筛查豁免继承 EXP-0008 预注册的 PI 决议并记录：联合训练能学到的
利用方式是冻结读出原理上测不到的（且本臂唯一的变量是输入定标，不新增
可训模块）。基座动力学型风险（训练动力学改变）如实标注：定标改变
gate/proj 的梯度尺度与平衡点，这正是本臂的观测对象。

## 唯一变量声明

对 EXP-0010 的唯一差异：BEV 张量在进入 proj 前乘以**冻结的逐通道定标
向量** `s ∈ R^262`（buffer，不参与训练）。融合位置（P3 单层）、门控零
初始化、proj 默认初始化、第 0 步与单流逐位等价性、遥测键名、训练协议
（v2 split、seed 44、100 epoch last、无增强、DDP4 batch1/累积2、lr 5e-4、
wd 0.05、SAVE_LAST_MODEL=1）全部不变。

定标向量定义（离线 harvest，一次性，确定性）：

- 数据：冻结快照 train split **全部 694 场**（不得使用 val/test 任何信息）；
- 前向：与训练完全同构的冻结路径（RSSAM2VisionEncoder base_plus 预训练权
  重 + `_k_and_w2c` 相机契约 + `rasterize_surfacebev` 单 bin
  height_edges=[-1e9,1e9]，输出 hw=(64,64)，fp32）；
- 统计：`rms_c = sqrt( Σ_scenes Σ_cells bev[c]^2 / (694×4096) )`（含未
  覆盖零单元，确定性）；
- 定标：`s_c = 1 / max(rms_c, 0.05)`；floor=0.05 的理由：rms 统计含零
  单元，近死通道（如 zstd≈0.0024）若被提升到单位量级会把数值噪声放大
  到与真实信号同阶；floor 限制最大放大 20×，使死通道定标后仍 ≤0.05。
- 产物：JSON（schema/rms/scale/n_scenes/hw/floor），路径
  `/data2/wangcheng/checkpoint/surfbev_b0/bev_stats/exp0014_bev_scale_p3.json`，
  sha256 于 harvest 后登记在本文件附录（启动前闭合）。
- 模型侧：`RSPrompterAnchorDual` 新增可选参量 `bev_stats`（路径或 None）；
  None（默认）时行为与现有全部臂逐位一致；提供时注册 buffer `bev_scale`
  并在 `_uav_bev` 返回前乘入（fp32 域，autograd 安全）。新身份
  `ue-whu-a3sgm-512-dualnorm-v1`（config `configs/ue_whu_a3sgm_512_dualnorm.py`，
  相对 dual config 的唯一键差异 = `bev_stats`）。

## 遥测与判定算子（v2.3 勘误，先于本臂启动）

v2.1 漂移算子 δ_L = mean|gate_L| / mean|proj_L(bev)| 被 DUAL-DELTA-DECOMP
证伪"无尺度歧义"声明（分母被 cm 通道污染）。本臂起替换为**尺度不变的
有效开度**：

- `open_L = mean( |gate_L ⊙ proj_L(bev')| ) / mean( |proj_L(bev')| )`
  （逐元素绝对值之商；两项均取末 epoch checkpoint 权重与 val 首 batch
  前向，结论期工具一次性产出；bev' 为定标后张量）；
- 层 L"参与" ⇔ open_L ≥ 0.20；legacy δ_L 同步报告供对照（勘误登记于
  DUAL-DELTA-DECOMP-20261016_conclusion.md，不追溯已启动臂的判定文本）。
- 训练期逐 epoch 遥测沿用四件套（gate mean/std/absmax、delta_p3_ratio、
  bev_coverage）＋结论期补 open_L。

## 四要素

1. **主指标**：val 100 场 segm AP@[.50:.95] 与 bbox AP@[.50:.95] 的
   EXP-0014 − EXP-0009 差。机制对照：EXP-0014 − EXP-0010 差（定标的
   净效应）。伴随指标（≥2 维度）：AP50/AP75、面积三分位分层、UAV-off
   消融差（`UE_UAV_EVAL_OFF=1`）、有效开度/legacy δ/delta_p3_ratio/
   bev_coverage 全套。
2. **选择口径**：last epoch（第 100），与全阶梯一致；best 只作参考不进判定。
3. **显著性**：val 100 场逐场 scene-paired percentile bootstrap（10,000
   draws、seed 20261014，沿 0008 谱系）；单 seed 结论边界如实标注。
4. **判定门（预注册）**：
   - `DUAL_GAIN` = 两主指标 bootstrap CI 下界均 >0 且幅度 ≥ +0.5 点；
   - `DUAL_ENGAGE_NULL` = 指标无增益且 open_P3 < 0.20（归一化后训练仍未
     选用 → 密集注入路线按四连探针+本臂证据最终关账）；
   - `DUAL_USED_BUT_NULL` = open_P3 ≥ 0.20 且 UAV-off 消融差显著但终指标
     无增益（用了但没用对地方 → 归因任务层，转 RoI 级/训练期监督用法）；
   - `DUAL_HARM` = 任一主指标显著为负。
   预测登记（非门）：主预期 ENGAGE_NULL 或 USED_BUT_NULL。

## 工件与审计

- wrapper `scripts/ue/run_exp0014_ddp4.sh`（守卫族同 EXP-0010：干净树、
  manifest/split 指纹、快照 payload 指纹、架构指纹、身份、输出目录不
  存在、stats 文件 sha256 钉死；契约 echo EXP14_CONTRACT）；
- 启动前独立子代理实施+接线审计（VERDICT 落盘 docs/reviews/）；
- 冒烟：子集 2 epoch 于 scratch 目录（非正式产物，voided）；
- 全臂 harness：六条已登记 baseline 指纹逐一不变 + 新旧 config diff 恰为
  `{bev_stats}` 键集；
- test 开封为独立事件（与 0009–0013 同批）。

## 启动前附录（harvest 与冒烟后、审计前补登）

- stats 文件：`/data2/wangcheng/checkpoint/surfbev_b0/bev_stats/exp0014_bev_scale_p3.json`
  sha256 `f7c3c5fba3ff0ea67ad26dde44660180dcf436a56062149ef0bdd907f9a1ad3b`
  （694/694 train 场景，2026-10-16）；
- harvest 日志：`/data2/wangcheng/checkpoint/surfbev_b0/probes/EXP0014-bevstats-harvest.log`；
- 关键统计（train 全量）：desc[0:256] rms 中位 0.01422 / 最大 0.1373；
  zmean 4309、zstd 0.07531、vcount 0.978、zres 1005。定标后各通道 rms
  落入 (0,1]：描述子多数被 floor 截断在 ×20（定标后 ≈0.28），几何/计数
  通道 ≈1.0——原五个数量级的失衡收敛到一个数量级内，floor 同时保证
  近死通道不被放大越级；
- 冒烟证据（voided，非正式产物）：
  `probes/EXP0014-smoke.log`（DDP4、子集 1%、2 epoch：配置/身份/遥测键/
  第 0 步等价性/optimizer 分组含 gate+proj 全部验证；epoch 内门=0 与
  EXP-0012 已放行冒烟同签名——warmup+AMP 首步跳步伪影）；
  `probes/EXP0014-smoke2.log`（单 rank、warmup 10、3 epoch：门 absmax
  3.5e-4→2.6e-3→4.4e-3，delta_p3_ratio 8e-5→8e-4 单调上升——归一化
  通路梯度存活，直接数值验证）。


