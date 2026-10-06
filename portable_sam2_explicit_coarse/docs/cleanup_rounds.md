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
