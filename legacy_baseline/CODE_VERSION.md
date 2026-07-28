# 代码版本与实验溯源

## 1. 代码基线

| 项 | 值 |
|---|---|
| Git 仓库 | `portable_sam2_fusion_new` |
| 分支 | `master` |
| Commit | `eee7517` (`eee751738d0c4e9f5fa7fdc017cec56bafe11584`) |
| Commit 时间 | 2026-06-14 12:37:30 +0800 |
| Commit message | `fix: wire DDP timeout to TORCH_DDP_TIMEOUT_SECONDS / NCCL_TIMEOUT` |
| 本包源码来源 | `git archive eee7517 portable_sam2_fusion_new` 导出，源码层面零改动 |

> 备注：两个实验的训练日志虽然保存在 `worktrees/ue-coco/` 工作树（分支 `whu1024-ue-coco-line`）的
> `logs/` 目录下，但 `.params` 快照明确记录 `git_commit=eee7517 git_branch=master`，即**运行时的代码
> 状态是 master 的 eee7517**，而非 ue-coco 工作树的提交。本包即以 eee7517 为准。

## 2. 两个最强实验

两者代码完全相同（eee7517），区别仅在训练超参。

### 2.1 whu1024_baseline_noms（bbox 最强）

| 项 | 值 |
|---|---|
| RUN_TAG | `baseline_noms_4gpu` |
| 运行时间 | 2026-06-14 16:18 启动 |
| 日志 | `logs/ablation_hyperparam_whu1024_baseplus/whu1024_baseline_noms_4gpu_baseplus_ddp4_20260614_161842.log` |
| 数据 | WHU, `--dataset-format whu_coco`, 单类 building |
| 分辨率 | `image_size=1024 1024` |
| 有效 batch | 8（bs=2/gpu × accum=1 × 4gpu） |
| LR | 5e-4, backbone_mult=1.0 |
| Epochs | 80（训满） |
| IIMR | **disabled** |
| Multi-scale 增强 | **OFF** (`multi_scale_resize_prob=0`)（"noms" = no multi-scale） |
| FPN | fused |
| EMA | enabled, decay=0.999, eval/save_best=1 |
| 早停 | metric=segm_map, patience=10, min_epochs=20 |
| seed | 44, deterministic=1, amp=0 |

### 2.2 whu1024_bs1a2_fixddp（segm 最强）

| 项 | 值 |
|---|---|
| RUN_TAG | `bs1a2_fixddp` |
| 运行时间 | 2026-06-06 00:35 启动 |
| 日志 | `logs/ablation_hyperparam_whu512_baseplus/whu1024_bs1a2_fixddp_003534.log` |
| 有效 batch | 8（**bs=1/gpu × accum=2** × 4gpu） |
| IIMR | **disabled** |
| Multi-scale 增强 | OFF |
| 其余 | 同 baseline_noms |
| 训练长度 | 早停于 epoch 27（segm 峰值在 epoch 17） |

> 注：该实验无 `.params` 快照（非标准脚本启动），参数从日志内容确认。
> "fixddp" 指修复了 DDP 相关问题；与 baseline_noms 的唯一实质差异是 batch size / 梯度累积方式。

## 3. COCO 指标明细（WHU validation, IoU=0.50:0.95, maxDets=100）

### 3.1 baseline_noms 峰值（Epoch 78）

**bbox：**

| 指标 | 值 |
|---|---|
| **AP @[IoU=0.50:0.95 \| area=all \| maxDets=100]** | **0.788** |
| AP @[IoU=0.50] | 0.914 |
| AP @[IoU=0.75] | 0.867 |
| AP @[small / medium / large] | 0.434 / 0.819 / 0.889 |
| AR @[maxDets=100] | 0.821 |
| AR @[small / medium] | 0.522 / 0.853 |

**segm（同次评估）：**

| 指标 | 值 |
|---|---|
| **AP @[IoU=0.50:0.95 \| area=all \| maxDets=100]** | **0.748** |
| AP @[IoU=0.50] | 0.915 |
| AP @[IoU=0.75] | 0.867 |
| AP @[small / medium / large] | 0.426 / 0.786 / 0.818 |
| AR @[maxDets=100] | 0.780 |
| AR @[small / medium] | 0.507 / 0.818 |

### 3.2 bs1a2_fixddp 峰值（Epoch 17）

**bbox：**

| 指标 | 值 |
|---|---|
| AP @[IoU=0.50:0.95 \| area=all] | 0.767 |
| AP @[IoU=0.50] | 0.946 |
| AP @[IoU=0.75] | 0.870 |
| AP @[small / medium / large] | 0.383 / 0.747 / 0.858 |
| AR @[maxDets=100] | 0.806 |

**segm：**

| 指标 | 值 |
|---|---|
| **AP @[IoU=0.50:0.95 \| area=all \| maxDets=100]** | **0.754** |
| AP @[IoU=0.50] | 0.947 |
| AP @[IoU=0.75] | 0.863 |
| AP @[small / medium / large] | 0.354 / 0.733 / 0.854 |
| AR @[maxDets=100] | 0.785 |

## 4. Config 继承链

```
configs/rsprompter_anchor_whu1024_base_plus_singlecls.py   (本包复现入口)
  └── _base_ = ./rsprompter_anchor_satS_v11_sam2_large_full.py   (根 config，无 _base_)
```

- `whu1024_base_plus_singlecls.py` 覆盖：backbone `model_size="base_plus"`、neck
  `in_channels="sam2_hiera_base"`、`num_classes=1`、`dataset_type="WHUCocoSingleClass"`、
  SAM2 ckpt 指向 base_plus、data_root 指向 WHU。
- 最终 `model.type = "RSPrompterAnchorDroneGuidance"`（enable_drone_branch=False，单流）。

## 5. 本包相对 eee7517 的改动（仅可移植性）

为提升可移植性，本包对 eee7517 源码做了以下**最小修改**，均保留原默认值以兼容当前机器：

1. `rsprompter/models_sam2.py`：新增 `_ensure_sam2_on_path()` helper，将原硬编码
   `sys.path.insert(0, '/home/wangcheng/project/sam2-main')`（两处）改为读
   `SAM2_SOURCE_PATH` 环境变量（默认 `/data/wangcheng/myproject/sam2-main`），目录不存在时静默跳过。
2. `uav/sam2_backbone.py`：同上，硬编码 sam2-main 路径改为环境变量化。
3. `scripts/run_hparam_whu1024_baseplus_4gpu.sh`：`PYTHON`/`DATA`/`CKPT_BASE` 改为 `${VAR:-default}` 形式。
4. 新增 `scripts/run_baseline_noms.sh`、`scripts/run_bs1a2_fixddp.sh` 两个封装脚本（非原项目文件）。
5. 新增 `scripts/inference_whu1024_baseplus.sh`（原项目该 commit 无此文件，从 ue-coco 工作树引入，兼容 eee7517 config）。

模型结构、训练逻辑、损失函数等核心代码**完全未改**。
