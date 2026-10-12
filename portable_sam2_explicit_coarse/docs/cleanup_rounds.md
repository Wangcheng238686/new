# a3sgm 代码清理轮次日志（cleanup/a3sgm-only 分支）

目标：只保留论文报告的 a3sgm 臂，行为绝对不变，模型包 ~4,400 行。
本文件记录每轮的变更清单与关键教训，随最终单次 commit 交付。

## 执行环境（任何轮次续跑前先读这里）

- 临时 worktree：`/home/wangcheng2021/project/new-cleanup`（分支 `cleanup/a3sgm-only`，基线 add7482）
- 主工作区：`/home/wangcheng2021/project/new`（分支 machine2/fast-whu150，**最终只在此 commit 一次**，不 push）
- python：`/data/wangcheng/envs/cvt2/bin/python`；每轮改完跑两门：
  1. `python3 scripts/smoke/check_a3sgm_checkpoint_load.py`（3 权重重建+strict load）
  2. `/data/wangcheng/envs/cvt2/bin/python scripts/smoke/check_a3sgm_forward_equivalence.py`
     （固定 2 图 CPU 前向 sha256 对 `scripts/smoke/a3sgm_forward_golden.json`；`--update` 重录）
- 论文权重：NWPU=`vhr10_p2v2_matrix300_a3sgm_pbm_sg_udprk64tm_.../best_composite_model_epoch270.pth`（论文 NWPU 行 68.3）、
  WHU=`whu1024_a3sgm_roilocal_full_fast_.../best_model_epoch118.pth`（raw+EMA）
- 终局门：两 ckpt 全量推理 diff 磁盘金样（`inference_test_best_e118{,_ema}/`、`inference_*_vhr10/`）

## 硬性禁区（已实证）

- `rsprompter/architecture_contract.py` 冻结（ID/指纹算法一字不改）
- `inference/infer_from_checkpoint.py` 的两个 compat 变体（:241-267 prebuild、:372-389 md 无键）
- nn.Module 属性名 = state_dict 671 键；registry 名；mask head `__init__` 全部具名参数（死臂参数保留签名只删分支）
- fi（full_image）+ roi_local 双坐标契约都在用（NWPU/WHU 各一）
- margin 锚定校验（boundary_logit=log(0.4/0.6)，容差 1e-4）

## 轮次记录

### R0-R1 金样与 NWPU 拒载修复
- NWPU ckpt 9/17 存的旧 ID 无 `_udprk64` 尾缀；9/18 `eef70ca` 改了 `architecture_id()`。
  指纹本身经 prebuild 变体（pop roi train/test cfg）可精确复现 → 修复=在
  `infer_from_checkpoint._contract_matches` 加"指纹相等 + ID 差恰好是 `_udprk\d+` 后缀"放行规则（`_is_legacy_pre_tail_id`）。
- 新增两回归脚本（见上）。3/3 加载 PASS，前向哈希金样可复现。

### R2 批量删除（72 文件，40,149→35,144 行）
死臂入口脚本 44 个（vhr10 历史/ksweep/full600/fi-dev 矩阵/旧论文行/nwpu10/jitter/large/c5v2/udpr_k64/
WHU matrix 四件套/test_only_from_ckpt 等）、根脚本 3、configs 5（nwpu10/jitter/large/densefix）、
rsprompter 整文件 4（p2_boundary_refiner/canvas_renderer/box_jitter/ckpt_migration）、
tools/smoke_test_components.py、死臂 smoke 测试 18。

### R3 sam2_mask_head.py 1587→1063
删：ROI-SAM/R3 渲染器/DenseCap/P2BRR/quality头(G1/G2)/诊断干预开关(no_sparse/zero_image_pe/
zero_high_res/zero_base_dense/frequency/spatial)/MLP point_emb 路线/shape_scale_mode 非 global_sigmoid 档/
explicit_prompt_mode "mask" 值/PE 稀疏解冻分支/prune_unused 分支/tailstop 训练分支。
重构：__init__ 拆 `_reject_removed_arms/_init_final_mask_loss_cfg/_init_no_mask_embedding/
_init_prompt_encoder/_init_shape_prior/_init_point_miner/_init_decoder_tail`；predict 双路径的
margin 锚定校验合并为 `_check_tail_anchor_once`。
保留：全部遥测（SP/DENSE/GATE/TAIL/PROMPT/GAUSSIAN 前缀）、`_last_uav_debug_stats` 遥测总线、
`_last_shape_point_*`/`_last_tail_outputs`、无尾消融分支（论文 P/B/P+B/P+M 行）、
`init_weights` 覆写（跳过 FCNMaskHead 父类 conv 初始化，承重）。
踩坑：辅助方法泄漏 __init__ 自由变量（shape_prior_loss_weight 等）×3；class_agnostic 赋值顺序。

### R4 anchor_roi_head.py 748→552
删：P2BR 边界损失接线/R3 canvas 损失/DenseCap 空秩触碰与辅助损失/quality 损失与重排序/
box jitter 分支（box_jitter.py 已删，>0 显式报错）/uav_aux_losses 下游收集。
重构：mask_loss 拆 `_empty_rank_losses/_apply_tail_loss/_apply_coarse_loss`（含 DDP 空秩双 collective 保命注释）。
**重大教训：`with_extra_pe` 不是死臂**——两个论文 ckpt 嵌入配置实测 `with_extra_pe=True`（无参数
SinePositionalEncoding，不进 state_dict，架构指南也漏记）；R4 初稿删它后被前向门当场拦截，已全量恢复
（构造 + loss/predict 注入，收敛为 `_inject_extra_pe` 单一函数）。教训：无参数模块只能靠前向等价门守护。

## 待办轮次
R5 decoder_tail_refiner 只留 margin → R6 injector/dense_utils/helpers → R7 miner/neck/vision →
R8 detector/targets/coarse_loss/ckpt_utils/facades → R9 训练器瘦身（50 步 loss 对比验收）→
R10 配置坍缩 _base_ 静态配置（指纹验证）→ R11 infer_from_checkpoint 清理 →
R12 仓库级删除（probes/baselines wrapper/refactor_golden/legacy_baseline/顶层旧tex/.worktrees gitlink/
197 skip-worktree 幽灵索引）→ R13 终文档 → 终局全量推理 diff + 主工作区单 commit。

### R9 训练器瘦身（4540→2960）
外科手术式删除（非重写——50 步门只覆盖早期训练，resume/选模/早停等深路径逐字保留）：
UAV 参数族与镜像键、三个 module-only 训练域、quality/densecap/p2br 监视器函数族与
epoch 日志、mask-loss ramp、_materialize_p2、ckpt_migration 调用、--test-only 评估链
（入口脚本 R2 已删）、FSDP。iou_prediction_head 独立参数组保留（改名 iou_head，历史契约）。
**等价验收（三跑校准）**：ref/旧码×2 + new/新码各 50 优化步（WHU 子集 2%、1024²、AMP、
RCNN_SAMPLER_NUM=32、seed44）：全部 27 个 loss/遥测指标的 |new−ref| ≤ |ref−ref2| 同代码
自噪声量级（GPU scatter 原子操作噪声），恒等量（TAIL/selected_count=416、changed_fraction、
boundary_logit、det_weight）逐位一致；参数组表仅 quality_head→iou_head 改名（同参同 LR）。
教训：bit 级对比在 GPU 原子操作下不可达，必须用同代码双跑标定噪声带。

### R5 decoder_tail_refiner 575→228
只留耦合 margin 容器：删 v1 BCE 损失、tailstop 隔离分支、confidence_gated*（A4/A5）
全部构造/前向/损失/遥测与 DDP 空秩镜像；`mlp.{0,1,3}` 参数名逐字节保留（ckpt 兼容）；
`_stop_margin_point_loss` 更名 `point_loss`（oracle 证据注释保留）。两 ckpt 尾配置实测
`residual_v1_margin/K64/margin0.10/boundary−0.4054651` 完全一致。单测裁剪为 margin 契约
（test_decoder_tail_stop_margin.py → test_decoder_tail_margin.py）。

### R6 injector/dense_utils/helpers 992→432
injector 351→127（只留 roi_only，删 RoIBoxEncoding/上下文投影/FiLM 族）；dense_prompt_utils
350→123（删 signed/gaussian 变换与 scipy 依赖）；helpers 291→182（删 ROI-SAM 辅助、
signed/gaussian 分支、gate 死档位、死诊断）。facade 同步（shape_prior 去 RoIBoxEncoding）。

### R7 miner/neck/vision 1489→1142
miner 593→473（删 point warmup 分级、固定对照 2P2N、点位 jitter；遥测键 warmup_stage=2/
pjitter_count=0 常量保留）；neck 321→151（删 RSFeatureAggregatorSAM2 与 DEBUG 打印）；
vision 545→518（删两个 prune 占位类）。

### R8 targets/coarse_loss/detector
targets 删 roi_sam 分支——**修复训练期隐患**：R3 已删 `roi_sam_enabled` 属性，loss_and_target
的 elif 引用会在 roi_local 训练时 AttributeError（推理不经过 loss 路径，双门测不到）。
coarse_loss 247→241（删 boundary_weight>0 损失分支；boundary METRIC 遥测保留——两 ckpt
compute_metrics=True）。

### R10 配置坍缩 + 指纹机器验证
用"假 python"拦截器从真实启动链捕获最终 cfg.model 与命令行，生成静态配置：
- `configs/a3sgm_whu1024.py` / `a3sgm_vhr10.py`：**architecture_fingerprint 与两论文 ckpt
  训练期契约逐字节相等**（NWPU ID 差仅 `_udprk64` 历史尾缀，R1 规则覆盖）；
- `configs/ablations/vhr10_{point_only,box_only,point_box,point_box_dense}.py`：论文消融
  P/B/P+B/P+B+M 的 `_base_` delta（加减键；`_delete_=True` 使尾 cfg 解析为 `{}`=禁用，
  与"键缺失"模型行为等价）；逐键与 env 管线 dump 相等。
瘦启动器 `whu1024_a3sgm_roilocal_full_fast.sh`/`vhr10_a3sgm_matrix300.sh`（ABLATION_CONFIG
可指 delta）；删 env 三层管线与解析 configs。静态冒烟（无模型 env）：28/30 指标落
ref-ref2 噪声带，两个越界者（rpn_bbox/loss_bbox）在四跑全距内且 R9 new 跑同样越界。

### R11 infer_from_checkpoint 1060→1023
删 --disable-p2/--p2-beta/--disable-decoder-tail 探针旗标、DenseBR 拒载/迁移。
**契约 compat 区（两个变体 + R1 旧 ID 规则）逐字节未动。**

### R12 仓库级删除
inference/probes 24 个、scripts/probes、baselines 外部基线 wrapper 47 个（图件脚本
make_teaser/make_fig1_rows/layout_grid/pick_viz/dump_coarse/crop/scan 保留）、tools 灰件 3、
legacy_baseline/ 整目录、顶层旧 tex/pdf/中文叙事 PDF/teaser 候选图/viz 中间产物、
.worktrees 双 gitlink、refactor_golden（磁盘）。portable py+sh 40,149→14,807。

## 最终行数（portable_sam2_explicit_coarse，py+sh）
rsprompter 4,866（含全量遥测）｜trainer 2,960｜infer 1,023｜data 695｜tools/utils ~1,550｜
smoke ~1,200｜configs+ablations ~1,400｜图件脚本 ~1,600｜入口/评测脚本 ~300。

## 终局等价门（2026-10-09，主工作区落盘前）

清理后代码对两份论文权重全量推理，与磁盘金样逐条对账：

| 组 | 规模 | 结果 |
|---|---|---|
| WHU test · raw e118 | 前 600 图 25,667 条预测 | **逐位一致**（Δscore/Δbbox=0，RLE 全同） |
| WHU test · EMA e118 | 前 600 图 25,843 条预测 | **逐位一致** |
| NWPU val · bestC e270 | 130 图 772 条预测 + 24 项 COCO 指标 | **预测逐位一致，指标精确相等（max|Δ|=0.00e+00）** |

说明：WHU 全量 2220 图推理在当前主机被 OOM killer 截杀（掩膜累积 ~120GB，他人占 62GB），
改用前 600 图前缀等价（27% 测试集、5.15 万条预测全对）；NWPU 为完整验证集。
模型行为与清理前**逐位等价**由此关闭。

---

# 第二轮激进重构（cleanup2/radical-refactor，2026-10-12 起）

判据：数学逐位不变 / 7 权重可加载 / loss 数值不变 / 契约 compat 区逐字节不动；
允许变化：日志键名与排版（命名归正）、内部函数签名。

## 新增回归门（R1）
- `check_a3sgm_ddp_launch.py`：2-rank torchrun × 3 步——**上线即捕获 R1 修复的
  use_fsdp 孤儿引用**（上轮删 FSDP 参数时 DDP 条件没同步，torchrun 一发即
  AttributeError；单进程冒烟测不出此类缺陷）。
- `check_a3sgm_resume.py`：2ep 训练存 last_checkpoint → RESUME 续 1ep（保存/恢复/
  snapshot 合并链）。注意 last_checkpoint 仅非最终 epoch 写（最终 epoch 写 last_model）。
- 加载门扩至 7 权重（+消融四臂 P/B/PB/PBM，护栏 ID 修剪后的 r1_c3/pafpn_coarse_* 分支）。

## R2 死码剥除（审计实证后删）
infer 两个 env 恢复函数（17 env 键全库零消费者）；roi_head share_extractor 双路径/
mask_prompt_box_cfg 4 死键/BOX 写死零占位/loss_roi_* 死转发/bbox_feats 死参；
detector transformers 依赖（全库唯一 import 服务两处永假 isinstance）；
mask head 11 墓碑参数（6 个 ckpt 不带+5 个 kwargs 吞）+不可能状态验证+roi_mask_size；
契约 architecture_id 修剪 9 处不可达分支（指纹函数未动；6 配置 ID+2 指纹硬编码比对通过）；
trainer 4 组死常量+snapshot 死镜像键；decoder ALLOW_DECODER_FALLBACK 死开关。

## R3 losses/遥测分通道（核心设计变更）
roi_head.mask_loss 返回 (losses, telemetry)；loss() 只返回纯损失——mmdet 混入的
`acc` 指标键剥入 `DET/*` 遥测；trainer 对损失通道**无过滤全量求和**（"loss"子串
过滤与 train/val 大小写不一致这类 bug 从设计上灭绝）；遥测经 `roi_head._loss_telemetry`
stash 计量进 epoch 日志（键序与数值不变）；det-loss 三段 staging（恒 1.0）全家清除。

## R4 MaskHeadOutput + 命名归正
mask head 返回值 6 元组→dataclass（mask_logits/coarse_outputs/tail_outputs 三个
有消费者的字段；消灭恒空 {} 槽、iou 复制槽、_last_tail_outputs 侧信道）。
遥测总线 uav*→telemetry 家族；--sat-*-lr-mult→--backbone/other-lr-mult（优化器组名
同改）；--use-*-coco→--dataset {whu,vhr10}；删 --gpu-id/--ema-save-best/--subset-ratio
别名与陈旧 data-root 默认。

## R6 训练器结构化
_parse_args（139 行 argparse+校验）与 _build_config_snapshot（188 行快照组装，5 个
选择旋钮显式返回）抽为函数；训练/验证两段语义相同的 GT 构建块合一为
`_build_gt_data_samples`。main() 2959→2630 行内聚（保留 epoch 主循环在 main 内——
其与 DDP 顺序契约强耦合，拆出风险大于收益）。
