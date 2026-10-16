# EXP-0014 结论：DUAL_HARM（定标修复成功于机制层，训练仍拒用；密集注入路线三级关账）

2026-10-16。预注册 `docs/EXP-0014_preregistration.md`（判定门原文见该文件）。
臂：SURFBEV-snap512dualnorm-sealed-DDP4-v2-noaug，身份
`ue-whu-a3sgm-512-dualnorm-v1`，commit `e193045`（tag `EXP-0014`），审计
PASS（`docs/reviews/EXP-0014_implementation_review.md`）。100 epoch 完整
完成（2026-10-16 01:41–08:51）。

## 判定（按预注册四要素）

主指标（val 100 场逐场 scene-paired percentile bootstrap，10,000 draws，
seed 20261014，last-epoch checkpoint，与阶梯 A/B 同工具同基准）：

| 对照 | segm AP 差 | bbox AP 差 |
| --- | --- | --- |
| EXP-0014 − EXP-0009 | **−0.0091** CI[−0.0157, −0.0029] | **−0.0087** CI[−0.0148, −0.0026] |
| EXP-0014(off) − EXP-0009 | −0.0086 CI[−0.0152, −0.0025] | −0.0084 CI[−0.0142, −0.0027] |

- **判定 = DUAL_HARM**（两主指标 CI 上界均 < 0，显著为负；预注册门
  "任一主指标显著为负"即触发，本臂两个均触发）。
- 参与度（v2.3 有效开度，末代权重 + val 前向，探针
  `probes/DUAL-DELTA-DECOMP-0014-20261016/`）：**open_P3 = 0.035 < 0.20**
  （legacy δ_L = 0.119 亦 < 0.20）——未参与条件同时成立，机制归因：
  注入通路整体是轻微噪声负担而非信息源。UAV-off 消融差 ≈ +0.05 点
  （off 略好，惰性签名，同 EXP-0010）。
- 预注册预测登记（ENGAGE_NULL/USED_BUT_NULL 主预期）方向正确（无增益）
  但正式落判为 HARM——负 CI 强于预期，如实登记。

## 机制读数（修复是否生效：生效）

| 量 | EXP-0010（未定标） | EXP-0014（定标） |
| --- | --- | --- |
| \|\|proj(bev)\|\| / \|\|base\|\| | 62× | **0.48×** |
| BEV zmean/desc post-scale 量级 | 4297 vs 0.018（5 个数量级） | 1.0 vs 0.32（同阶） |
| gate std / absmax（末代） | 0.0033 / 0.015 | **0.038 / 0.118**（×8–11 动态范围） |
| 有效开度 Σ\|Δ\|/Σ\|proj\| | 0.0044 | 0.035 |
| delta/base 范数比 | 0.378 | **0.023** |
| delta 偏置能量占比 | 46% | 32%（细结构 64%，更结构化） |

定标在工程上完全成功：量纲通路修复、门获得真实动态范围、注入由全通道
构成且更结构化。但训练在修复后仍把aggregate 注入压到基座范数的 2.3%、
可用投影信号的 3.5%，且指标显著为负。

## 谱系决议：密集注入路线三级关账

1. 冻结读出级：P1-V2→P4 四连探针（cell 级描述子读出无净增益）；
2. 未定标训练级：EXP-0010/0011/0012/0013（阶梯 A/B 全部 ≤ 单流；
   `probes/DUAL-AB-LADDER-20261016/`，single 0.7764/0.7533 为全阶梯最高）；
3. 定标训练级：EXP-0010 的唯一缺陷（量纲）被本臂修复后仍无增益且显著负。

三层证据收敛：**P3/P2 cell 级门控残差注入这一用法则对 UE 512 双 AP 无
提升路径，最终关账**。归因落在任务层（密集 BEV 不携带卫星流不能自行提取
的实例级 AP 相关边际信息，同 SZAB 结论），不再是表示/量纲/门控初始化等
工程层。下一步用法候选（RoI 级条件化 / 训练期几何蒸馏 / 掩码精化通道）
见 DUAL-DELTA-DECOMP-20261016_conclusion.md 方向决议节，留 PI 裁决。

## 工件

- 训练日志 `/data2/wangcheng/checkpoint/surfbev_b0/launcher_logs/EXP-0014-ddp4.log`
  （last: segm 0.7507 / bbox 0.7298，trainer 基准；best_segm 0.7565@79 仅参考）
- checkpoint `/data2/wangcheng/checkpoint/surfbev_b0/surfacebev_v2_512dualnorm_sealed_ddp4_s44_EXP-0014/`
- 逐场 A/B（on/off + bootstrap）：`probes/DUAL-AB-LADDER-20261016/rows_dual_norm*.jsonl`、
  `eval_dual_norm.log`（驱动脚本一次性复用 probe_dual_ab_ladder 机制）
- 开度分解：`probes/DUAL-DELTA-DECOMP-0014-20261016/decomp.json`
- 结果子代理复核：`docs/reviews/DUAL-AB-LADDER-20261016_result_audit.md`
  加遗节（dual_norm 并入六臂拉通后独立复算，VERDICT PASS；本臂逐场数字与
  上表逐一核对一致）
