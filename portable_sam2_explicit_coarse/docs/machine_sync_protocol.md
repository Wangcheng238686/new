# 双机同步协议（lthpc × machine2）

日期：2026-09-09。适用分支：`machine2/fast-whu150`（两机共用的唯一工作分支）。
背景：两机**无共享文件系统**（lthpc `/data`，machine2 `/data1`），checkpoint 与
原始日志互不可见；所有跨机信息只能经 git 分支流动。本协议经 2026-09-09
dev100 双机交叉验证的经验固化（机器偏移 +0.011、三轮脚本合并冲突、
A2/R1 三抽样结案），修订需双方 commit 留痕。

## 1. 推送节奏

- **谁产出结论谁推送，小步快推**：每完成一个实验层（如 dev100 四臂、
  matrix300 一批）→ 本机回填结果文档 → `fetch` → 有分叉先 `rebase` → `push`。
- 提交粒度 = 一个可独立阅读的结论单元（文档回填 + 配套脚本改动同仓提交）。
- commit message 写明证据来源（日志文件名含 PID）与结论文句话。

## 2. 热点文件纪律（冲突 treadmill 的根治）

改动以下文件前**必须先 fetch 并处理完远端新提交**：

- `portable_sam2_explicit_coarse/scripts/_run_vhr10.sh`
- `portable_sam2_explicit_coarse/scripts/ablations/vhr10_p2v2_dev.sh`
- `portable_sam2_explicit_coarse/configs/` 配置链（`whu1024_baseplus_*` 等）

2026-09-09 起以下已入主线为 canonical，**请勿回退为硬编码**：
`BATCH_SIZE` / `GRAD_ACCUM_STEPS` / `CUDA_VISIBLE_DEVICES` / `NPROC_PER_NODE`
的 `:-` 参数化、`SEED` 覆盖与非 44 seed 的 RUN_TAG `_seed<n>` 后缀。
单卡臂等价启动：`CUDA_VISIBLE_DEVICES=<g> NPROC_PER_NODE=1 GRAD_ACCUM_STEPS=8`
（有效 batch 恒 8，与四卡 4×1×2 梯度数学等价，已双机验证）。

## 3. Evidence 约定（跨机可复查的唯一通道）

- **原始训练日志不入库**（维持既有政策）；数字走结果文档，文档内
  日志指向带 PID 全名。
- **关键臂完赛后必须跑 `scripts/ablations/vhr10_p2v2_eval.sh <arm> <best|last>`**
  导出 `metrics.json / dt_records.json / gt_records.json / run_manifest.json`，
  以 evidence 身份入库——这是对方机器复查数字、以及配对 bootstrap CI
  （矩阵 D5 强制项）的唯一输入。eval 需空卡，安排在臂完赛释放 GPU 后。
- 结论的效力排序：同机同 seed 配对 > 同机异 seed > 跨机直比（不作数，
  已知机器偏移 +0.011，见 `p2v2_dev100_a0_a1_a2e_results.md` §3b.3）。
- 单 seed 单次的臂间差低于 0.008（A0 双 seed 实测噪声标尺）不宣称效应。

## 4. 资产边界

- 各机未跟踪文件（tracking 笔记、viz 产物、环境补丁、临时脚本）默认
  留本地；认为对对方有复用价值的，经路径清理后入库并在 commit message
  注明来源机器。
- 机器本地环境配置：`configs/environment.local.sh`（不跟踪）；跨机共享
  的新环境变量需同步进 `configs/environment2.sh`（lthpc）与 local 文件
  并在文档登记。

## 5. 协议传播

本文件是协议唯一权威版本，随分支传播；对协议的任何修订以 commit 形式
修改本文件并在 commit message 中声明"协议修订"。新机器加入时先读本文件。
