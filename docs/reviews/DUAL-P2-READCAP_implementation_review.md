# DUAL-P2-READCAP-20261014 首次定版实现审计（独立复核人，2026-10-14）

审计对象：`docs/DUAL-P2-READCAP-20261014_preregistration.md`（预注册）、
`portable_sam2_explicit_coarse/scripts/ue/probe_dual_p2_readcap.py`（实现）。
口径基准：`probe_surfacebev_p1.py`（P1）。实测抽查：P1-V2 缓存 npz（只读）。
复核人不参与被审材料的设计与开发。

## A. 预注册-实现一致性

**A1 划分【无发现】** fit 555/select 139、seed 20261014、按缓存文件序置换；
实测重现划分确定、fit∩select=∅；val 首次读取在全部模型与阈值冻结之后，仅评一次。

**A2 训练协议【无发现】** 网格、Adam lr 1e-3、batch 4096、max 50 epoch、
patience 5 逐项一致；BCEWithLogits/MSE；早停语义正确，恢复 best-select 权重。

**A3 公共配置选择【无发现】** mean(SAT, correct) select 损失取 min，sham90
不参与选择；三臂同结构同种子；逐目标独立。

**A4 阈值规则【无发现】** correct 臂、患病率校正 F1 公式与预注册逐字一致；
π=select 各场 yb 均值（用全 16384 cell 的 `yb` 键，实测 train npz 含该键）；
THRS 0.1–0.9 九档；三臂共用阈值；两 reader 各用各的最优阈值，公平。

**A5 平局取小阈值【无发现，重点核验】** THRS 升序遍历，`> best_f + 1e-12`
严格更新，相等保留先到达的较小阈值；best_f 初值 −1.0 保证首阈值必命中。

**A6 torch 种子【建议】** 实现为 `SEED + hi*10 + wi`，实测 6 配置种子无碰撞；
同配置三臂同种子满足配对要求；属预注册未声明偏差，应在结论 MD 登记。

**A7 parity 锚与 DEGRADED【无发现】** Logistic/Ridge 口径与 P1 fit() 一致；
DEGRADED 哨兵与预注册一致且优先于四判。

## B. 泄漏路径

**B1 拟合边界【无发现】** scaler/模型只用 fit；选择只碰 select；val 只在最终
一次性评估出现。**B2 select 适应性【无发现】** 早停对 select 的轻度适应性已
由预注册明文声明；select 不承担判定。**B3 seed 污染【无发现】** 无碰撞；三臂
共享初始化/shuffle 是配对控制变量；boot() 六组差值共用同一重采样索引，配对
一致，与 P1 相同。

## C. 与 P1-V2 口径可比性

**C1 三指标【无发现】** f1/auc/rho 逐场计算与 P1 evaluate() 逐字等价（含
单类回退 0.5、rho 回退 0）；boundary F1 用全 16384 cell。
**C2 池化【无发现】** pool_feats 与 P1 p3mask+pool 等价（maxpool8、掩码均值、
空掩码零向量）；池化在原始臂特征上进行后再过对应臂 scaler，与 P1 相同。
**C3 空 pred 防护【建议】** P1 有 `len(pred)` 分支，P2 原缺；实测 val 100 场
pred 17–98 全非空、当前数据不触发；建议补齐以保持逐字等价。
**C4 阈值共用【无发现】。**

## D. 数据处理正确性

**D1 偏移拼接【无发现】** rows_for 正确处理每场行数不等（实测 boundary/error
每场 512 行、iou 每场 20–100 行）。**D2 键名【无发现】** 与实测 npz 键完全对应。
**D3 标签边界【无发现】** uint8→float32 BCE 目标。**D4 惰性加载【无发现】**
train npz 的 `pred` 大数组从未被访问；每场 close()。

## E. 统计与判定

**E1 bootstrap【无发现】** 与 P1 逐字等价，seed 为本探针预注册值。
**E2 六门与 verdict【无发现】**（在预注册定义无歧义的组合上）。
**E3 判定完备性缺口【建议（预注册缺陷）】** 四判未穷尽 8 种组合（如仅 auc
两门过时无法落判；两门过的组合亦未定义）；实现兜底 else→NULL，但 NULL 语义
与"auc 两门过"矛盾。建议正式 run 前补判定勘误。——已处置：预注册勘误第 1
条补全判定格（P2_PARTIAL(<通过门列表>)）。

## F. 工程卫生

**F1 缓存只读【无发现】。** **F2 工件齐全先写后用【无发现】。**
**F3 split.json 三集断言【建议】** 原只写 fit/select、只断言两集；建议补 val
名单与三集断言。——已处置。**F4 smoke 不误产 decision【无发现】。**
**F5 fail-closed【无发现】。**
**F6 其他【建议】** (a) 正式 run 应把 stderr 一并落盘；(b) 死代码清理；——已
处置（含 out≠cache 拒绝、线性锚全网格轨迹落盘）。(c)(d) 一并处置。

## 总评

核心条款（划分、网格、选择规则、阈值规则、parity 锚、DEGRADED、bootstrap、
六门判定）逐条一致；泄漏路径干净；三指标与池化口径与 P1-V2 逐字等价；缓存
严格只读、工件先写后用、异常 fail-closed。发现均为建议级，已在正式 run 前
处置（见预注册勘误第 3 条）。

VERDICT: APPROVED
