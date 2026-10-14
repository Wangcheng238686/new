# 双流实现与实验入口

工作树：`/home/wangcheng2021/project/new-uav-dual`；分支：`codex/uav-bev-dual`。
模型目录为其中的 `portable_sam2_explicit_coarse`。所有命令在模型目录运行。

## 实现结构

| 文件/目录 | 职责 |
|---|---|
| `data/ue_rgbd_dataset.py`、`ue_payload.py` | 卫星可见实例 local ID 标签、五视图 RGB 和辅助 tensor 的统一输入 |
| `data/ue_geometry_cache.py` | camera-Z EXR R通道读取、下界校验、不可训练几何缓存 |
| `tools/prepare_ue_rgbd.py` | 完整配对清单、含相机顶点的保守视锥包络、跨集合建筑排查 |
| `rsprompter/uav/cameras.py` | CV column 相机合同、厘米→米、局部坐标、投影/反投影 |
| `rsprompter/uav/surfaces.py` | 原始深度观测、上表面高度及跨视角支持统计 |
| `rsprompter/uav/candidates.py` | 半径内高度相容候选、同 token 去重、每视角候选预算 |
| `rsprompter/uav/cvt.py` | 独立世界网格 query、深度位置条件稀疏 cross-view attention |
| `rsprompter/uav/footprints.py`、`projection.py` | 细卫星像素足迹竞争、稀疏质量加权特征汇总 |
| `rsprompter/uav/fusion.py` | 小尺度门控残差、零支持回退及遥测 |
| `rsprompter/uav/diagnostics.py` | 最多256个空间分布query的detached注意力诊断；禁用共享权重autocast缓存 |
| `tools/visualize_ue_alignment.py` | 不使用标签的BEV/卫星对齐检查图与观测cell CSV |
| `rsprompter/detection/dual_detector.py` | 独立 UAV LoRA 编码器、P3/P4 与完整 SAM embedding 两入口 |
| `train/ue_support.py`、`ue_telemetry.py` | UE loader、严格续训身份检查、AMP unscale 后梯度遥测 |
| `inference/ue_rebuild.py` | 完整 UE snapshot 严格重建、预训练/数据身份、路径迁移 |
| `utils/ue_evaluation.py` | AP300 诊断命名，主 AP100 保留 |
| `configs/ablations/ue_rgbd_*.py` | single/dual/fpn/sam 四臂；禁用入口的键不存在 |

物理范围由 UAV 原点和米制 BEV 定义，不要求与卫星视野相等。通过相机与代表高度向完整卫星特征网格投影，卫星不裁剪到 BEV 范围。最终 mask 的 `roi_local` 目标和 bbox paste 沿用单流；SAM 输入仍是全图 embedding。

## 统一入口

`bash scripts/ue_rgbd.sh <single|dual|fpn|sam> <prepare|verify|train|infer|dump-env>`。
默认 Python 环境为 `/data/wangcheng/envs/cvt2/bin/python`；可显式设置 `PYTHON_BIN`。
每卡 batch1、AMP、独立 UAV LoRA、梯度累积2。正式模型从通用 SAM2 联合初始化，入口拒绝 `--init-from`。

默认每50个mini-batch记录支持/高度/几何置信、各入口残差、注意力局部/高度相容mass和加权偏差、unscale后的新组梯度及裁剪比例。注意力统计是256 query以内的FP32近似，不是几何真值或全网注意力精确统计。

### 1. 构建清单

```bash
bash scripts/ue_rgbd.sh dual prepare \
  --satellite-root /path/to/satellite-production \
  --uav-root /path/to/uav-production \
  --output-dir /path/to/frozen-split
```

输出 `manifest.json`。默认 train/validation；需要独立 test 时明确提供 `--test-boundary-m`，也可调整 `--axis` 和 `--train-fraction`。划分失败时会报空集合，不静默回退。训练期间固定清单与其对应已验收文件；继续增加数据应新建清单和实验。当前开发用清单不作为最终数据版本裁决。

### 2. 独立训练 A/D

```bash
NPROC_PER_NODE=4 bash scripts/ue_rgbd.sh single train \
  --data-root /path/to/frozen-split \
  --ue-manifest /path/to/frozen-split/manifest.json \
  --ue-geometry-cache /path/to/geometry-cache \
  --checkpoint-dir /path/to/checkpoints/single

NPROC_PER_NODE=4 bash scripts/ue_rgbd.sh dual train \
  --data-root /path/to/frozen-split \
  --ue-manifest /path/to/frozen-split/manifest.json \
  --ue-geometry-cache /path/to/geometry-cache \
  --checkpoint-dir /path/to/checkpoints/dual
```

两臂使用同清单、种子、训练步数、有效 batch 和选模规则。B/C 单入口通过 `fpn`、`sam` 后续运行。验证选模看完整卫星 mask AP100，模型输出上限300；`*_max300` 仅为另报诊断。训练包含 EMA；UE 检查点保存 raw 训练权重与对应 optimizer，并单独保存 EMA。默认推理 `--weights auto` 使用该检查点记录的验证权重；`--weights model` 指定 raw，`--weights ema` 指定 EMA。

续训沿用同参数，增加 `--resume-from /path/to/last_model_epochN.pth`。允许延长 epochs，调度器延续checkpoint原状态，不重建新cosine周期；snapshot记录当前目标与父checkpoint。数据、几何、指标身份及核心训练语义变更会拒绝，不能以旧最优阈值继续比较新指标。

### 3. 推理与输入依赖诊断

```bash
bash scripts/ue_rgbd.sh dual infer \
  --checkpoint /path/to/checkpoints/dual/best_model.pth \
  --split test --output-dir /path/to/evidence/correct
```

默认从检查点恢复清单和数据根目录；新机器可用 `--ue-satellite-root`、`--ue-uav-root`、`--sam2-ckpt` 迁移，身份仍核验。明确评估新清单用 `--ue-manifest`，其新 hash 写入 run manifest。不存在 test split 会报错。

`--uav-mode zero` 移除 UAV 残差；`--uav-mode shift` 将 BEV 特征平移16格（64m），保留原几何投影作为错位诊断。它们用于依赖检查，不能替代独立 A/D 训练或证明净贡献。跨场景错配与更细归因实验可后续扩展，首版不声称已支持全部可变视角场景。

## 开发验收入口

```bash
bash scripts/ue_rgbd.sh dual verify --output /tmp/ue-geometry.json

PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /data/wangcheng/envs/cvt2/bin/python scripts/smoke/check_ue_rgbd_arms.py \
  --output /tmp/ue-arms.json

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /data/wangcheng/envs/cvt2/bin/python scripts/smoke/check_ue_rgbd_model.py \
  --manifest /path/to/frozen-split/manifest.json \
  --sample-index 11 --output /tmp/ue-model.json
```

逐损失检查需真实 RoI 到达融合层；小建筑或采样没选中大建筑时，coarse 的 P2 路径不会给 UAV 梯度。检测、final-mask 与 UDPR 仍提供监督，不据此增加 UAV 标签。脚本默认从 generic SAM2 起点，做一次优化器更新并压测300个掩码 RoI，不是训练效果测试。

```bash
CUDA_VISIBLE_DEVICES=0 /data/wangcheng/envs/cvt2/bin/python \
  scripts/smoke/check_ue_attention_diagnostics.py

/data/wangcheng/envs/cvt2/bin/python tools/visualize_ue_alignment.py \
  --manifest /path/to/frozen-split/manifest.json \
  --sample-index 11 --output-dir /path/to/alignment-check
```

前一项比较AMP/checkpoint下启停诊断的输出与梯度；后一项输出PNG/PDF、CSV及统计JSON。投影修正幅度不等于对齐误差。稀疏feature采用稳定排序/分段归约，避免CUDA原子汇总造成重复推理的微漂移。

空支持多卡：在上一条模型检查中加 `--ddp-empty-rank`，并经 `python -m torch.distributed.run --standalone --nproc_per_node=2` 启动。rank1 清空几何支持，仍进入训练 loss，检查两 rank 参数同步。

## 当前证据与限制

工程检查不证明 AP 收益。单层代表高度、4m 整格足迹与缺少卫星真实遮挡观测是已接受近似；正式性能结论需空间独立 test、A/D 同预算对照及不确定性检查。

开发证据临时目录：`/tmp/uav_dual_dev_preflight`；检查点与预测：`/tmp/uav_dual_dev_data`。原始日志/权重不入 git。独立审核的发现、修复与最终结论应在同目录设计记录中保留，验收结束后汇总可版本化的小型报告。
