# DUAL-P4-CHANS loader 修复增量审计（独立复核人，2026-10-14）

审计对象：commit `6474cb6`（loader 修复）及其重跑结果。复核人不参与被审材料开发。

## 1. 提交范围

仅改动 `probe_dual_p4_channels.py`（+16/−14）：load_train 改为按目标分栈
（tr[t]['x']/{sat,bevc,bevs}、tr[t]['y']），run_readout 5 处引用替换；
fit/阈值/bootstrap/判定等数值路径零改动（共享模块自 8707db3 未动，
无残留旧引用）。**无额外改动。**

## 2. 行对齐（独立复算）

- 代码推演：拼接顺序与 lens 记录严格同序，rows_for 的 cumsum 块与 per-target
  拼接精确匹配。
- 真实数据执行：694 场 cache 上三目标 x/y 按场分块与 npz 原始数组逐一
  array_equal；555/139 切片抽查一致；总行数 355324/355328/36031 与 lens 和一致。
- 缺陷复现：父提交的交错 loader 下 rows_for 取行错位（复现 DEGRADED 根因），
  旧 run select loss ≈ 0.693（=ln2，标签随机化不可学习），与哨兵吻合。

## 3. 处置合规

invalid_loader_run/ 六件原样封存（READCAP_DEGRADED 与 sat_mlp_f1=0.190 未改写）；
cache/ 零修改；新旧 split.json 逐字节相同（种子确定性），loader 修复是两次
运行间唯一差异。符合"实现缺陷→修复→重跑，无效结果封存不改写"。

## 4. 修复后 summary

parity sat_mlp_f1=0.82733 > sat_lin_f1=0.74412（裕度 +0.083）→ degraded=false；
f1/auc/rho 三对门均未成对通过 → decision=P4_NULL 一致；
boundary_closed=false 与预注册条款（correct 仍胜 sham90 F1）一致。

备注（不阻塞）：重跑日志原存于 probes/ 父目录，已补拷为工件目录
run_readout.stderr.log；fit_mlp 的 overflow warning 为 P2 时代既有行为，
log_loss 对饱和 sigmoid 数值安全。

VERDICT: APPROVED
