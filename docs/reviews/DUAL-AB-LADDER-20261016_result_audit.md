# DUAL-AB-LADDER-20261016 结果审计（独立复核）

- 审计对象：`docs/DUAL-AB-LADDER-20261016_conclusion.md`、
  `/data2/wangcheng/checkpoint/surfbev_b0/probes/DUAL-AB-LADDER-20261016/`（summary.json、
  rows_*.jsonl、identity_*.json）、工具
  `portable_sam2_explicit_coarse/scripts/ue/probe_dual_ab_ladder.py`，
  对照 `docs/EXP-0008_preregistration.md` v2/v2.1/v2.2/v2.3。
- 审计方式：全部数字用 python 从 rows/summary/日志/检查点独立重算，不采信结论文档
  自述。环境：`/data2/wangcheng/envs/cvt2/bin/python`（torch 2.8.0、numpy 1.26.4，
  与产物 pycache 的 cpython-39 谱系一致）。
- 日期：2026-10-16。

## 1. 工件完整性 — 通过

- 9 个阶梯 rows 文件（single、dual_p3{,_off}、dual_p2p3{,_off}、
  dual_p2p3_lora{,_off}、dual_p2{,_off}）`wc -l` 均为 100；11 个 rows 文件
  （含 dual_norm 两个，见注 2）scene_id 逐位同序（python 比对 `ids == ref_ids`
  全 True），首场 `Coord_002670_...`、末场 `Coord_003163_...`。字段
  {scene_id,n_gt,n_pred,bbox,segm}，无 NaN，n_gt 合计 2792。
- identity 映射与预期一致：single=`ue-whu-a3sgm-512-v1`、
  dual_p3=`...-dual-v1`、dual_p2p3=`...-dualv2-v1`、
  dual_p2p3_lora=`...-duallora-v1`、dual_p2=`...-dualv2-l0-v1`；
  summary.json identities 块同值。
- 五个 identity 的 `checkpoint_sha256` 与 `sha256sum` 各 workdir
  `last_model_epoch100.pth` 逐字节相等：
  single `75bfea31…`、dual_p3 `66245eb9…`、dual_p2p3 `237a8335…`、
  dual_p2p3_lora `821d1f9e…`、dual_p2 `fff6e46e…`。
- 附带核对：single 模型指纹 `5305b315…`、dual_p2 指纹 `277a95dd…` 与预注册
  引文一致；split_manifest_sha256 `17f49430…` = v2 split 登记 SHA。

## 2. 数字 — 通过（一处 2 位小数显示误差，见注 1）

- rows 重算均值 vs summary.json means：9 键 × 2 指标最大绝对差 = 0.000e+00。
- 终值对照表（4 位小数）与 summary.json 全部一致（single 0.7764/0.7533 …
  dual_p2 off 0.7610/0.7391）。
- off−single 分解（点）= 100*(mean(off)−mean(single)) 重算：
  0010 −0.8327/−0.6452；0011 −1.4812/−1.6167；0012 −1.3289/−1.6524；
  0013 −1.5436/−1.4190。文档 −0.83/−0.64、−1.48/−1.62、−1.33/−1.65、
  −1.54/−1.42：除 0010 bbox（−0.6452 按 2 位小数应舍为 −0.65，文档截断成
  −0.64，偏差 0.005 点）外全部吻合。
- 训练日志终值：grep 各日志 `Epoch 100/100` —
  0010 `segm/mAP=0.7550 bbox/mAP=0.7317`、0011 `0.7550/0.7327`、
  0012 `0.7555/0.7349`、0013 `0.7581/0.7326`，与文档一致。
- EXP-0009 日志确为 1 行中止重启残根（"output directory already exists"，
  0 条 Epoch 行）。按指定方法 `torch.load(..., weights_only=False)` 读封存检查点
  `history.val_metrics[-1]`（epoch=99 即第 100 轮）：
  segm/mAP=0.7592601、bbox/mAP=0.7374422 → 0.7593/0.7374，与文档一致。
  方法交叉验证：EXP-0013 检查点 history 0.7580643/0.7325605 vs 完整日志
  0.7581/0.7326，4 位小数吻合，证明该字段即日志终值来源。

## 3. Bootstrap 与判定逻辑 — 通过

- `boot()`（工具 131–134 行）：每次调用新建 `default_rng(20261014)`，
  `integers(0,100,(10000,100))` 索引矩阵、逐行均值、2.5/97.5 分位。同种子
  每次调用复用同一重采样索引（n 恒为 100）——配对口径下合理且完全确定。
- 按工具调用图独立重跑全部 22 个对比（4 臂 × 2 指标 × {vs_single, vs_off} +
  3 组臂间 × 2 指标）：与 summary.json 的 mean 及 CI 两端最大绝对差均为
  0.000e+00（逐位复现）。
- 结论文档判定表 8 组 mean/CI（点）与 summary.json 全部按 2 位小数吻合；
  on−off 收益行（−0.11/−0.10、+0.86/+1.00、+1.19/+1.59、+0.83/+0.63）吻合。
- 指定命题逐条复核（点）：
  - 0012 两 CI 含 0：segm [−0.80,+0.48]、bbox [−0.73,+0.58] ✓；
  - 0010 两指标显著负（CI 上界 −0.37/−0.16 均 <0）✓；0011 segm 显著负
    （上界 −0.02 <0，bbox 含 0）✓；0013 bbox 显著负（上界 −0.10 <0，
    segm 含 0）✓；
  - on−off CI 下界 >0：0011 [+0.51,+1.21]/[+0.64,+1.35]、0012
    [+0.75,+1.64]/[+1.13,+2.07]、0013 [+0.46,+1.19]/[+0.32,+0.95] 均 ✓；
    0010 [−0.25,+0.03]/[−0.24,+0.04] 均含 0 ✓；
  - 0012−0011 bbox [+0.0057,+1.1185] 排除 0 ✓、segm [−0.036,+1.031]
    含 0（"边缘"表述恰当）✓；
  - 0013−0011 两 CI 均含 0（[−0.75,+0.41]/[−0.71,+0.50]）✓。
- 判定代码忠实性（工具 283–289 行）：`gain = all(ci_lower>0 and mean>=0.005)`
  双指标 = 预注册 DUAL_GAIN（CI 下界均 >0 且 ≥+0.5 点，0.005 即 0.5 点）✓；
  `harm = any(ci_upper<0)` = "任一主指标显著为负" ✓；USED_BUT_NULL 需
  engaged(δ_L≥0.20) 且 off 显著，优先级 GAIN→HARM→USED→ENGAGE_NULL 与
  v2/v2.1 一致。以同逻辑重算四臂判定与 summary.json verdicts 完全一致。
- 0012 字母口径 vs 实质：字母 DUAL_ENGAGE_NULL 完全源于 δ 算子失效导致
  `engaged_any_level=False`（δ_L≈1e-4 < 0.20）；按预注册本意（参与 +
  UAV-off 显著 + 无增益）其实质对应 DUAL_USED_BUT_NULL。诚实性评估：文档
  在判定表内标注"字母口径…见 δ 勘误"、勘误节明说"受影响的仅是 NULL 子类
  标签（0012 的字母口径）"，且机制节明示该臂 UAV 依赖显著——事实无隐瞒，
  判读路径透明；唯"实质为持平"未把更正后的子类标签（USED_BUT_NULL）写出，
  留给读者推导，属轻微不精确而非误导（见注 3）。

## 4. δ 算子伪影 — 通过

- summary.json delta_operator 六层：δ_L ∈ [9.059e-05, 1.014e-04]（全部 ≈1e-4）✓；
  分母 proj_bev_mean_abs ∈ [21.27, 50.48]，与文档"≈21–50"及 cm 量纲放大
  （gate_mean_abs 0.002–0.005 ÷ 21–50 → 1e-4）解释自洽 ✓。
- 结论引用 delta_p3_ratio 0.30–0.49 来自 gate_last_epoch 遥测块：实测
  [0.2973, 0.4898]，端点按 2 位小数即 0.30/0.49 ✓。

## 5. 封存纪律 — 通过

- `grep -n "test" probe_dual_ab_ladder.py` 零命中；两处数据集构造均为
  split `'val'` + SHA 登记的 v2 split 清单（数据集类强制校验清单哈希与
  空间划分规则）。
- probes 目录 4 个 eval 日志 grep "test" 零命中。
- 结论文档唯一 "test" 出现为"test 全程未开封"——未开封声明，非测试结果
  声明；全文无任何 test 数字。

## 6. 因子矩阵一致性 — 通过

- 0013−0011 两指标 CI 均含 0 → 预注册 v2.3 解读第一支（P2 单层充分）成立，
  与文档落判一致。
- 两单层臂对单流：0010 两指标点估计 −0.94/−0.75 且均显著负；0013
  −0.72/−0.79（bbox 显著负、segm CI 含 0）。文档"对单流均为负"为方向性
  陈述，四个点估计确全为负，显著性已由判定表逐格标注，无过度声明。

## 非阻断备注

1. （显示误差）0010 off−single bbox 文档写 −0.64，正确 2 位小数舍入为
   −0.65（精确 −0.6452；系截断）。量级 0.005 点，不影响任何结论。
2. （工件卫生）probes 目录内混入 EXP-0014（dualnorm 谱系）产物：
   `rows_dual_norm{,_off}.jsonl`、`identity_dual_norm.json`、
   `eval_dual_norm.log`（09:55 写入，晚于 summary.json 09:54）。该臂不在本
   工具 ARMS 映射内，未进入 summary.json 合并，对本次拉通数字零影响；建议
   后续为 EXP-0014 另立目录，避免多实验共享同一落判工件目录。
3. （表述建议）0012 行可将"实质为持平"补一句"按算子本意应归
   DUAL_USED_BUT_NULL"，把更正后子类标签显式写出；现有披露已足以推出，
   仅余精确度问题。

## 结论

六项全部独立复核通过：数字逐位复现（rows→means→bootstrap 22 对比零差异）、
检查点身份与哈希闭合、判定门实现忠实于预注册、δ 算子失效已如实登记且仅
影响 0012 的 NULL 子类字母标签、test 未开封。上述三条备注均不构成阻断。

VERDICT: PASS

---

# 加遗复核（2026-10-16，dual_norm 并入后的六臂版本）

范围：PASS 后的三处变更——(1) 工具 ARMS 注册表新增 `dual_norm`
（EXP-0014，ckpt `surfacebev_v2_512dualnorm_sealed_ddp4_s44_EXP-0014/last_model_epoch100.pth`、
日志 `EXP-0014-ddp4.log`）及臂间对比 (dual_norm, dual_p3)；(2) `--summarize`
重跑，summary.json 并入 dual_norm；(3) 结论文档更新为六臂版。本节同时作为
`docs/EXP-0014_conclusion.md` "结果子代理复核：待归档" 的归档复核。

## (a) dual_norm 工件完整性 — 通过

- `rows_dual_norm.jsonl` 与 `rows_dual_norm_off.jsonl` 各 100 行，scene_id
  顺序与 `rows_single.jsonl` 逐位一致（python 比对 True/True）。
- `identity_dual_norm.json`：baseline_id = `ue-whu-a3sgm-512-dualnorm-v1`；
  checkpoint_sha256 `04ae28b1…` 与 `sha256sum` EXP-0014
  `last_model_epoch100.pth` 逐字节相等；指纹 `473d1ede…` 与注册表行一致。

## (b) summary.json dual_norm 数字 — 通过

- 我的独立 bootstrap 重跑（同 seed/口径，含新调用图全部 28 个对比）与新
  summary.json 的 mean 及 CI 两端最大绝对差 = 0.000e+00；且旧五臂全部数值
  与首审逐位不变（回归核对 5 个抽样键全 True）——重跑无扰动。
- means：on 0.7673601/0.7446467 → 0.7674/0.7446 ✓；off
  0.7678700/0.7449612 → 0.7679/0.7450 ✓。
- vs_single：segm −0.009073 CI[−0.015739, −0.002940] → −0.0091
  CI[−0.0157,−0.0029] ✓；bbox −0.008670 CI[−0.014806, −0.002635] →
  −0.0087 CI[−0.0148,−0.0026] ✓（两上界 <0，DUAL_HARM 成立，
  verdicts.dual_norm 一致）。
- on−off：segm CI[−0.00178, +0.00082]、bbox CI[−0.00165, +0.00104] 均
  含 0 ✓；dual_norm−dual_p3：segm CI[−0.00516, +0.00571]、bbox
  CI[−0.00639, +0.00394] 均含 0 ✓。
- 与 `docs/EXP-0014_conclusion.md` 独立数字交叉：其主表两行即本 summary
  同源数（同工具同 seed），其中 (off)−single 行不在 summary 内，我独立
  重算 segm −0.008564 CI[−0.015167, −0.002520]、bbox −0.008355
  CI[−0.014202, −0.002728]，与其引用的 −0.0086 CI[−0.0152,−0.0025] /
  −0.0084 CI[−0.0142,−0.0027] 吻合 ✓。

## (c) 六臂结论文档一致性 — 通过（一处 2 位小数显示误差，见备注 A4）

- 六臂表：EXP-0014 全局 0.7507/0.7298 = EXP-0014-ddp4.log `Epoch 100/100`
  行 ✓；on/off 逐场 0.7674/0.7446、0.7679/0.7450 = summary ✓。
  0009–0013 各行与首审一致（未变动）。
- 0010 off−single bbox 已更正为 −0.65（精确 −0.6452）✓——首审备注 1 落实。
- 0012 判定行现为"字母口径 DUAL_ENGAGE_NULL（δ_L<0.20）；按参与实质
  （on−off 显著、delta_p3_ratio 0.45–0.49）为 USED_BUT_NULL 语义"：
  on−off 两 CI 下界 >0 ✓，lora 双层 delta_p3_ratio 0.4898/0.4531 →
  0.45–0.49 ✓——首审备注 3 落实，表述与 summary 数值一致。
- EXP-0014 判定行 −0.91 [−1.57,−0.29] / −0.87 [−1.48,−0.26] = summary
  （双上界 <0）✓。
- δ 勘误更新：定标后 δ_L=0.11726、分母 proj_bev_mean_abs=0.26403 → 文档
  0.117/0.26 ✓（summary delta_operator.dual_norm.L1）；有效开度 0.035 与
  EXP-0014 结论 open_P3=0.035（decomp 探针）一致——两探针的 legacy δ_L
  0.117（本工具）vs 0.119（decomp 探针）系不同实例化，均 <0.20，无矛盾。
- 三级证据关账陈述（P1–P4 冻结读出 / 未定标阶梯 / 定标臂）与
  EXP-0014_conclusion.md 谱系决议逐条对应 ✓；下一步替换为 DECOMP 方向
  决议三项（高度蒸馏 / 边界带精化攻 R0 segm +11.94 oracle / RoI 交叉注意
  力 bbox 上限 +1.8），均见于 DUAL-DELTA-DECOMP-20261016_conclusion.md
  与 DUAL-R0-DECOMP-20261014_conclusion.md ✓。臂间 0014−0010
  +0.03/−0.12 双 CI 含 0 ✓；0014 on−off −0.05/−0.03 双 CI 含 0 ✓。
- 封存复检：更新后工具 grep "test" 仍零命中（ARMS 增改不涉数据集路径）。

## (d) EXP_REGISTRY.md — 通过（0014 行有同步滞后，见备注 A5）

- 0010–0013 四行均含完成时刻 + last 值 + 拉通判定：0.7550/0.7317
  DUAL_HARM、0.7550/0.7327 DUAL_HARM（segm 显著负）、0.7555/0.7349
  净差不显著（USED_BUT_NULL 语义）、0.7581/0.7326 DUAL_HARM（bbox 显著
  负）+ P2≈P2+P3——与日志终值及六臂结论逐项一致 ✓。

## 加遗备注（均不阻断）

- A4（显示误差，同首审注 1 类型）：代价成分行 "0014 −0.85/−0.83" 精确值
  −0.8564/−0.8355，2 位小数应为 −0.86/−0.84（截断所致），且与
  EXP-0014_conclusion.md 自身正确舍入的 −0.0086/−0.0084 不一致；偏差
  0.005–0.006 点。另"代价（约 −0.8~−1.7）"为宽松概括（实际 −0.65~−1.65），
  已带"约"字，无实质影响。
- A5（登记同步）：EXP-0014 注册表行尚停在启动信息，未写入
  "完成 100 epoch + DUAL_HARM" 终态；EXP-0014_conclusion.md 的
  "结果子代理复核：待归档" 行在写回论文口径前应指向本加遗。均为登记
  动作滞后，不影响本次数字与判定。

## 加遗结论

六臂合并重算逐位复现、旧值零扰动、EXP-0014 独立数字闭合、文档三处更新
（−0.65、USED_BUT_NULL 语义、δ 重标定）与工件一致、注册表 0010–0013
同步到位。备注 A4/A5 为显示舍入与登记滞后，不构成阻断。

VERDICT: PASS
