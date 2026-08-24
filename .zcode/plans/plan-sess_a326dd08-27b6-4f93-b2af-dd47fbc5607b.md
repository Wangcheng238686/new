## 目标
为 ue_coco 数据集创建一个 R1-C4-RD 单流实验（pafpn + coarse points_box_dense + stride16 + detached raw_logits dense + resize 到 1024），使用修复后的无泄露划分（clean.json + images/）。

## 背景约束（已核实）
- 训练器 `train_rsprompter_fusion.py` **只读 config 的 `model` 字段**，数据完全靠 CLI（`--data-root`、`--use-whu-coco`、`--image-size`）+ `create_train_loader` 工厂传入。
- **障碍**：训练器 line 1831-1834 把 WHU 的 ann_file/img_subdir 写成硬编码字面量（`2.4 annotation/annotation/train.json` 等），工厂本身支持参数但训练器没暴露。
- ue_coco 的 clean.json 在 `annotations/{train,val,test}.clean.json`，图片在统一目录 `images/`。
- ue_coco 原生 512×512，你选择 resize 到 1024（与 WHU 对齐，stride16 下 embedding 仍 64×64）。
- R1-C4-RD 的 model config 是 `configs/whu1024_baseplus_explicit_coarse.py`，model 架构与数据集无关，可直接复用。

## 用户决策
1. 独立目录（`scripts/ablations_uecoco/`，不依赖 WHU 的 `_run_ablation.sh`）
2. 复制改写新脚本（不扩展公共脚本）
3. image-size = 1024×1024

## 实施步骤

### 1. 训练器加 env-var 覆载（最小侵入，默认行为不变）
`train/train_rsprompter_fusion.py` line 1831-1834，把硬编码字面量改成读 env-var（带 WHU 默认值）：
```python
whu_train_ann_file=os.environ.get("WHU_TRAIN_ANN_FILE", "2.4 annotation/annotation/train.json"),
whu_val_ann_file=os.environ.get("WHU_VAL_ANN_FILE", "2.4 annotation/annotation/validation.json"),
whu_train_img_subdir=os.environ.get("WHU_TRAIN_IMG_SUBDIR", "2.1 train/train"),
whu_val_img_subdir=os.environ.get("WHU_VAL_IMG_SUBDIR", "2.3 valid/validation"),
```
不设 env-var 时完全等价于现有 WHU 行为（所有正在跑的 WHU 实验不受影响）。

### 2. 新建 ue_coco wrapper（独立、自包含）
`scripts/ablations_uecoco/r1_c4_rd_pafpn_coarse_points_box_raw_detach_emb64.sh`：
- 不调用 `_run_ablation.sh`，直接构造 `torchrun` + 训练器命令
- export ue_coco 专用的环境变量（数据根、ann_file 路径指向 clean.json、img_subdir=images、image-size 1024、PROMPT_ROUTE 等架构开关）
- 数据根 `/data1/wangcheng/dataset/ue_coco`
- ann_file: `annotations/train.clean.json` / `val.clean.json`
- img_subdir: `images`（统一目录）
- 复用 model config `configs/whu1024_baseplus_explicit_coarse.py`（model 架构不变）
- checkpoint/log 目录独立到 `ue_coco` 子目录，避免与 WHU 混淆

### 3. 校验
- python 语法检查训练器改动
- 用 create_train_loader 实测 ue_coco 数据能加载（clean.json + images/ + 1024 resize）
- dry-run 确认命令拼接正确

### 不做的事
- 不改 `_run_ablation.sh`（WHU 实验零影响）
- 不新建 config 文件（model 架构复用现有）
- 不动 WHU 数据集/实验

## 待确认的开放点
启动后会在你指定的 GPU 上跑（需要你确认哪组卡空闲，当前 WHU roi_local 实验可能还在占卡）。