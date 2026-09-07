# WHU P2 边界细化消融矩阵

当前主线只保留一个可比较的四行矩阵。四行均使用 PAFPN、SAM2 Base+、stride-16
image embedding、ROI-local final-mask、相同的数据增强、AMP、4 卡、batch=1 ×
accumulation=2、150 epoch、WHU 全量 train/validation 和 seed 44。

| 脚本 | Prompt/P2 变量 |
|---|---|
| `whu_p2_matrix_point.sh` | 2P2N points |
| `whu_p2_matrix_point_box.sh` | points + box |
| `whu_p2_matrix_point_box_mask.sh` | points + box + raw-logit dense mask |
| `whu_p2_matrix_full.sh` | points + box + dense mask + P2BoundaryRefiner |

dense mask 与 P2 Full 默认 `SHAPE_DENSE_DETACH=0`：最终 mask loss 可沿现有
PromptEncoder dense-mask 路径回传至 coarse head；在 Full 行也会回传至 P2BoundaryRefiner。
P2 feature 本身仍保持设计上的 detach。四行只改变表中明确列出的 prompt/P2 变量。

从项目主目录执行：

```bash
cd /home/wangcheng2021/project/new/portable_sam2_explicit_coarse

RUN_IN_BACKGROUND=0 bash scripts/ablations/whu_p2_matrix_point.sh
RUN_IN_BACKGROUND=0 bash scripts/ablations/whu_p2_matrix_point_box.sh
RUN_IN_BACKGROUND=0 bash scripts/ablations/whu_p2_matrix_point_box_mask.sh
RUN_IN_BACKGROUND=0 bash scripts/ablations/whu_p2_matrix_full.sh
```

正式矩阵前的短程开发只运行下面两个固定的两臂系列。每个系列均为两张卡
`DEV_GPU_LIST=1,2`、每卡 batch=1、accumulation=4，因此有效全局 batch 仍为 8；其余训练
超参沿用正式矩阵。固定 WHU 10% train / 10% validation、seed 44、15 epoch、每 3 epoch
验证、EMA-off，两个候选臂串行执行。

```bash
# D1：Point+Box control → Point+Box+Mask，候选 dense alpha 初值 0.10。
bash scripts/ablations/whu_d1_dense_dev_10p_2gpu.sh

# D2：D1 参数冻结后运行。Mask control → Full；Full 使用 beta=0.10、
# delta_logit_max=0.50 的保守既有 P2BR。
bash scripts/ablations/whu_d2_p2_dev_10p_2gpu.sh
```

默认使用 GPU `1,2`；可用恰含两个物理卡号的 `DEV_GPU_LIST` 覆盖，例如
`DEV_GPU_LIST=0,3 bash scripts/ablations/whu_d1_dense_dev_10p_2gpu.sh`。D2 中的 dense
参数必须替换为 D1 实际获胜且冻结的值后才能启动；脚本中的 `0.10/1.0` 是第一候选，不得
把 D1 未通过的设置带入 D2 或正式矩阵。

所有入口经 `_run_ablation.sh` 校验解析后的 architecture ID、数据集与模型契约。
`DRY_RUN=1 CHECK_DATA=1 PREFLIGHT_MODEL=1` 可完成不训练的预检。

## full_image 空间契约协议（whu_fi_*，2026-09）

roi_local 矩阵的全图特征/提示输入与 ROI-local 监督/粘贴之间存在未显式实现
的空间重映射（诊断见会话记录）。`whu_fi_*` 系列把最终 mask 契约切到
`full_image`（整图 target 监督 + 推理只 resize 一次），并配 `roi_balanced_dice`
loss（稀疏前景下全画布标准 BCE 会塌缩到全背景捷径，git 6dc6aa4——该 loss 是
协议组成部分而非扫参）：

```bash
# 四行矩阵（Point → +Box → +Mask → Full），协议=common 默认（4 卡/150ep/全量）
bash scripts/ablations/whu_fi_matrix_full_data.sh

# D1/D2 的 full_image 版：除契约两字段外与 roi_local D 系列逐项一致
bash scripts/ablations/whu_fi_d1_dense_10p_2gpu.sh
bash scripts/ablations/whu_fi_d2_p2_10p_2gpu.sh
```

run tag（`whu_fi_matrix_*` / `whu_d1fi_*` / `whu_d2fi_*`）与 roi_local 结果
完全隔离；跨协议（新旧坐标契约之间）的 mAP 对比**不可**直接当作同协议比较。
注意：full_image 协议的 architecture ID 与 roi_local 同名字符串相同（历史
checkpoint 兼容优先），区分靠 checkpoint 内嵌 model_config/fingerprint 与
run tag。串行约定：`whu_fi_*` 全部强制 `RUN_IN_BACKGROUND=0`（公共 runner
默认是后台启动即返回，四行矩阵会并发抢卡），上一臂成功结束才启动下一臂；
需要整系列后台时对整个 wrapper 用 `nohup`，不要依赖 runner 的后台模式。

## D5：dense 门控初值 + PE 读取端适配（2026-09-06，冻结权重诊断后）

设计依据：`docs/p2_dense_frozen_diagnostic_report.md`（实例准确画布 +
门开大的全量配对增益 +0.0154）与 `docs/d5_review_handoff.md`（Codex
定稿方案）。两臂均为 Mask 行（P2 关闭）、full_image 契约、D 系列 dev
协议；判读：D5-A vs 历史 Mask（0.6090/0.6133）筛门控初值作用，D5-B vs
D5-A 筛 PE 适配作用，候选须超 Point+Box 0.6144 并配对复跑确认。

```bash
# D5-A：门控初值 0.5（可学习 global_sigmoid 门，非 fixed），PE 全冻结
DEV_GPU_LIST=0,1 bash scripts/ablations/whu_fi_d5a_mask_gate050_10p_2gpu.sh

# D5-B：D5-A + 解冻 mask_downscaling（4,684 参数）+ PE 组 LR 5e-5
DEV_GPU_LIST=2,3 bash scripts/ablations/whu_fi_d5b_mask_gate050_pe010_10p_2gpu.sh
```

验收遥测：`Epoch N prompt pathway` 行的 `pe_mask_downscaling` 分支
（梯度/更新 RMS）与 `Optimizer group prompt_encoder` 审计行（参数数
4684、LR 5e-5）；详见根目录 `DEBUG_FIELDS.md`。注意 wrapper 的协议
变量（RUN_TAG/MAX_EPOCHS/子集比例/DEV_GPU_LIST 等）为无条件 export，
做 smoke 时需用独立副本或临时改写，避免污染正式 run 目录。

## VHR-10 入口（共享 runner：`scripts/_run_vhr10.sh`）

5 个薄入口只声明变体名；协议与架构 exports 块（C4 事故教训）单一存在于
`scripts/_run_vhr10.sh`。变体：`fast400`（RUN_TAG vhr10_c5v2_400）、`jitter400`
（预注册未跑）、`large400`/`large600`（hiera-large 全链路，600 支持
`RESUME_FROM` 断点续训）、`ft200`（c5v2-400 获胜者 lr 1e-4 续训 200）。

```bash
RUN_IN_BACKGROUND=1 bash scripts/ablations/vhr10_large600.sh
```

### NWPU P2-v2 dev（已实现，未授权启动）

三臂使用同一 100ep / 520 train + 130 validation / 4 GPU / fi / raw-selection
协议，前台串行运行并保存真正的 E100 末轮权重。A0 是严格的 D5-B PBM 底座
（alpha=0.5、dense 不 detach、解冻 mask_downscaling、PE LR multiplier=0.1）；
A1 只加原 P2，A2 只将 P2 auxiliary loss 改为 correction-keep R1。

```bash
DRY_RUN=1 bash scripts/ablations/vhr10_p2v2_dev.sh a0
DRY_RUN=1 bash scripts/ablations/vhr10_p2v2_dev.sh a1
DRY_RUN=1 bash scripts/ablations/vhr10_p2v2_dev.sh a2
```

三臂须严格串行（每臂占满四卡，任一失败即停止），可用统一入口：

```bash
bash scripts/ablations/vhr10_p2v2_dev_series.sh
```

正式训练必须在 VHR-10 推理/manifest/bootstrap/p2_off smoke 均通过、并得到用户
启动授权后执行；100ep 是机制筛选，不是对 600ep 最终协议的否定性结论。

每臂结束后按实际权重口径重放 checkpoint 内嵌的 VHR-10 数据契约，并导出模型
1024-space 的 `gt_records.json`、`dt_records.json`、`images.json`；这三者与
`run_manifest.json`（含 130 个 `processed_image_ids`）共同构成 paired bootstrap 输入：

```bash
bash scripts/ablations/vhr10_p2v2_eval.sh a0 best
bash scripts/ablations/vhr10_p2v2_eval.sh a0 last
```

## 目录地图（2026-09 重构后）

```text
scripts/
├── load_environment.sh            # 环境加载（19+ 脚本 source）
├── _run_ablation.sh               # WHU 共享 runner（仅 coarse 路由）
├── _run_vhr10.sh                  # VHR-10 共享 runner（5 变体）
├── infer_whu_checkpoint.sh / visualize_p2_checkpoint.sh
├── ablations/                     # 薄入口与 eval wrapper（本目录）
└── smoke/                         # 组件自检、DDP 控制面单测、契约校验
tools/                             # eval_boundary_ap / eval_vhr10_original_scale /
                                   # mask_to_coco_whu512 / visualize_instances /
                                   # smoke_test_components
inference/probes/                  # 审计探针（手动调用）
```

历史代际（b0/b1/c1/c2/m0/m1/r0、coarse_strategy、prompt_content_p2_matrix、
ablations_uecoco 等 31 文件）已删除，结论固化在 `logs/test_eval/*/REPORT.md`
与 git 历史（commit 9f5af76 之前）。
