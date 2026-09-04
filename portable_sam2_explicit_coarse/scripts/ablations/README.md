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
