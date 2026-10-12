# PromptMiner-SAM2（a3sgm）

ICASSP 2026 论文 *PromptMiner-SAM2: Prompt Mining for Remote Sensing Instance
Segmentation* 的模型代码与实验证据仓库。2026-10 完成 a3sgm-only 清理：只保留论文
报告的模型臂，行为与清理前逐位等价（验证记录见
`portable_sam2_explicit_coarse/docs/cleanup_rounds.md`）。

## 仓库布局

| 目录 | 内容 |
|---|---|
| `icassp2026_promptminer/` | 论文手稿、图件、参考论文 |
| `portable_sam2_explicit_coarse/` | 模型代码（rsprompter 包、训练器、推理、静态配置） |
| `portable_sam2_explicit_coarse/docs/` | 全部文档：指南、遥测字典、治理协议、追踪表、证据（见其 `README.md` 索引） |
| `sam2/` | vendored 官方 SAM2 仓库（原样不动） |
| `fig1_components/`、`fig1_rows.png`、`viz_pick/`、`paper-reviews/` | 论文图件资产与审稿记录 |

## 模型一句话

RSPrompter-anchor 单流两阶段检测器（SAM2 Hiera-B+ LoRA 编码器 → PAFPN → RPN/RoI）
+ **CSPM** 粗形状提示挖掘（coarse 64×64 → 2P2N 点 + 框 + dense canvas 三模态提示）
+ **UDPR** margin 尾（K=64 不确定点选点、有界残差、部署阈值锚定损失）。
WHU-1024 mask AP 73.4 档（fast150 基线行）、NWPU VHR-10 mask AP 68.3（a3sgm bestC）。

## 快速上手

```bash
source portable_sam2_explicit_coarse/configs/environment2.sh   # 机器路径（lthpc）

# 训练（两个正式入口；静态配置 = 论文 ckpt 的训练期契约）
bash portable_sam2_explicit_coarse/scripts/ablations/whu1024_a3sgm_roilocal_full_fast.sh
bash portable_sam2_explicit_coarse/scripts/ablations/vhr10_a3sgm_matrix300.sh

# 论文消融臂（P / B / P+B / P+B+M）：ABLATION_CONFIG 指向 delta 配置
ABLATION_CONFIG=configs/ablations/vhr10_point_box_dense.py \
  RUN_TAG=vhr10_matrix300_pbm_notail bash .../vhr10_a3sgm_matrix300.sh

# 推理 / 评测（自描述 checkpoint，无需重设环境）
bash portable_sam2_explicit_coarse/scripts/infer_whu_checkpoint.sh <ckpt> [ema]
bash portable_sam2_explicit_coarse/scripts/infer_nwpu_checkpoint.sh <ckpt>

# 等价性回归门（改代码后必跑）
python3 portable_sam2_explicit_coarse/scripts/smoke/check_a3sgm_checkpoint_load.py
```

## 阅读代码

按 `portable_sam2_explicit_coarse/docs/codebase_guide.md` 的九步阅读路线；
机制全景见 `docs/a3sgm_architecture_guide.md`（行号对应清理前提交 add7482）。
