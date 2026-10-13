# DUAL-P3-SEPSTR-20261014：旁路表示读出探针（预注册）

## 路由与谱系

A 路线冻结诊断探针，DUAL-P2（P2_NULL，见其结论）判定表的直接下一步；不训练
任何模型、不重新 harvest、不读 test。P2 结论定位的瓶颈是融合臂的二次随机
投影破坏 SAT 非线性结构；本探针在**同一冻结缓存**上把读出输入改为旁路拼接，
检验"无损保留 SAT 通道时 BEV 信息能否让非线性读出全面获益"。

## 冻结输入与臂组装

输入与 DUAL-P2 完全相同（P1-V2 R1a 缓存，contract SHA `af04fc44…`，只读；
fit 555/select 139/val 100 划分与种子 20261014 复用 P2 的 split 规则与
`split.json`）。臂定义改为读出侧组装：

- `SAT`：x_SAT（64 维）；
- `correct`：[x_SAT ‖ x_correct_fused]（128 维）；
- `sham90`：[x_SAT ‖ x_sham90_fused]（128 维）。

correct/sham 两臂共享完全相同的 SAT 半区；唯一下变量是融合半区的几何正确性。
按 P2 判定表，此为"分通道"的读出侧实现；**若 P3 亦 NULL，读出侧动作用尽，
必须重新 harvest（升维/真分通道），不得再在此缓存上调读出**。

## 协议（与 DUAL-P2 逐字一致，仅臂输入维度不同）

MLP 网格（hidden {(64,),(128,),(64,64)} × wd {1e-4,1e-3}）、Adam lr 1e-3、
batch 4096、max 50 epoch、patience 5、torch 种子 20261014+10·h+w；每目标三臂
共用配置（mean(SAT,correct) select 损失最小）；阈值：select 上 correct 臂
患病率校正 F1（π=select 各场 yb 均值），三臂共用；线性 parity 锚
（Logistic C 网格/Ridge α=1/C，同选择规则）；DEGRADED 哨兵（MLP SAT val F1
< 线性 SAT val F1 − 0.005）；iou 实例池化 = 拼接特征上均值池化（与分别池化
后拼接数学等价）。三指标、bootstrap（10,000 draws、seed 20261014）、六门、
勘误判定格与 P2 预注册逐字一致。

## 假设与判定

H-P3：旁路表示下 `correct−SAT` 与 `correct−sham90` 六门在 val 全部 CI 下界
>0（P3_PASS）→ 放行 DUAL-H1 型热启动筛查：learned 双流融合模块（保留流
身份、独立流编码器）+ 强制参与度遥测。其余组合按 P2 勘误判定格落判；
P3_NULL 的后续动作见上节（重新 harvest，读出侧禁手）。

## test 政策与遥测

不读 test；val 100 场仍为一次性判定集（历史：P1-V2 选参 + P2 判定，本次为
第三次使用，如实登记；本探针选参仍全在 train 内部）。遥测三件套与 P2 同：
三项为冻结特征可读性度量非 AP；parity 锚校准；机制不替代结局验证。

## 工件

`/data1/wangcheng/checkpoint/surfbev_b0/probes/DUAL-P3-SEPSTR-20261014/`
（同 P2 六件套）；结论 MD 与增量实现审计落 docs/、docs/reviews/。
