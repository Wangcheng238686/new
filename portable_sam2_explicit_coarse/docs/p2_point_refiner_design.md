# P2 Point Refiner — 最终设计（基于 2026-08-23/24 全部测量）

## 一句话定义

用 stride-4 高频特征把挖点器（ShapePointMiner）输出的 4 个 prompt 点做有界位置校准。
角色：**提议由 miner 出、校准由 P2 出、语言保持 miner 语义不变**。
旧 P2BoundaryRefiner（logit 带内残差）退役：实测天花板 +0.003（含作弊版完美修正），
β 符号翻转换 AP 不变，路线确认死亡。

## 证据基础（全部为配对测量，logs/eval_p2_ab/）

| 测量 | 结果 | 设计含义 |
|---|---|---|
| 粗掩码带内修正 oracle（±8 全力） | +0.0027 mAP | prompt-logit 路线封死，dense prompt 吃 raw 即可 |
| miner-on-GT oracle（2P2N 同语义完美点） | +0.0082~+0.0111 | 唯一有余量的通道 = 点位置精度 |
| 同增益跨训练阶段（ep49/57/62） | +0.0088/+0.0082/+0.0111 | 非收敛补偿，是能力缺口 |
| 自制 GT 点采样器 2/4/8 点 | −0.012/−0.009/−0.038 | decoder 与 miner 点语义共适应；**不许换点风格、不许冻 decoder 加点** |
| 最终掩码误差结构 | 仅 22.8% 在边界 3px 带内；FP 过分割占 51-69%，小物体最差 | 负点校准（贴真边界外侧砍 FP）价值最大 |
| 挖点粗指标随训练 | 正点 ep5 起 100% 在 GT 内，粗掩码 IoU 自 ep30 平台 ~0.89 | 残余差距是结构性的（64 格量化+平台中心启发式） |

## 模块规格

```
P2PointRefiner(nn.Module)
输入:
  p2_feature  [B,512,H/4,W/4]     # PAFPN P2，同现在
  prompt_rois [N,5]               # 同现在
  coarse_logits [N,1,64,64]       # raw（不再有 refined）
  local_yx [N,4,2], labels [N,4]  # miner.get_last_local_points()
输出:
  offsets [N,4,2]                 # 图像像素，有界

结构（前半段直接复用现 refiner 组件）:
  p2_projection : 1×1 Conv 512→64 + GN + GELU          # 复用
  roi_align     : RoIAlign(32×32, scale=0.25)          # 复用
  feat_pt       : grid_sample(roi_p2, (local_yx+0.5)/64*2-1)   # [N,4,64]
  ctx_pt        : [该点粗掩码σ值, 到预测边界距离, label嵌入, box_w/h] (+ ROI池化全局向量)
  head          : MLP(64+ctx → 128 → 2)，末层零初始化（起点=不动点）
  bound         : Δ = tanh(head) × clip(0.25·min(bw,bh), 2, 16) px
  正/负点分头（或 label 条件化）：正点学"真实形体内鲁棒平台"，负点学"贴真边界外"
```

## 集成点（4 处代码改动）

1. `rsprompter/models_sam2.py:~2300`：miner 调用后加 3 行——取 local_yx、算 offsets、
   `coords = where(labels>=0, coords+offsets, coords)` 后 clamp 到图像；
   同步更新 `_last_shape_point_local_yx`（训练统计用）。
2. `rsprompter/models_sam2.py:2133-2166`：旧 P2BoundaryRefiner 调用块删除；
   dense prompt 分支改吃 raw logits；`p2_boundary_refiner_cfg` 换 `p2_point_refiner_cfg`
   （新 env 开关 `P2_POINT_REFINER_ENABLED` + 尺寸 env，沿用现有模式）。
3. `rsprompter/p2_point_refiner.py`：新模块 + 蒸馏损失
   `loss_p2_point`：训练时对 `get_coarse_targets` 的 GT@64×64 取 `gt_logits = t*16-8`，
   `no_grad` 跑同一个 miner 得 `gt_coords`；slot 规范化（p1/p2、n1/n2 各自按 (y,x) 排序配对）；
   `SmoothL1(beta=2px)`，负点权重 ×1.5（FP 砍除价值大）；建议总权重 0.2。
   端到端信号顺带免费：coords 可微，最终 mask loss 梯度可流过 offsets。
4. `train/train_rsprompter_fusion.py`：仿 `--train-quality-head-only`（:1458/:1734）加
   `--train-p2-point-refiner-only` scope（冻结全部、只解冻新模块，~15 行）；
   每 epoch 日志加 mean|offset|、点误差(px)；每 5 epoch 附一次 offset 置零配对 val（防白跑教训）。

## 训练与验收（两段式，先便宜后昂贵）

**Stage 1（数小时，refiner-only 热启动验证）**
```
从当前 run best ckpt --init-from 热启动；--train-p2-point-refiner-only；
--train-subset-ratio 0.5；--max-train-batches 截断；6-8 epochs
```
验收（事先定死，不达标即杀）：
- 点校准误差（vs gt-mined 点，px）较未校准基线降 ≥30%；
- 终态配对探针（offsets on/off）≥ +0.004 mAP（oracle 天花板的一半）。

**Stage 2（仅 Stage 1 通过后，论文级全量）**
与 `P2_BOUNDARY_REFINER_ENABLED=0` 的无 P2 对照等预算配对训练；
比较 best-epoch 指标 + 终态 checkpoint 上 offset 置零配对探针；
当前正在跑的 run 保留为"logit 版 P2"历史对照臂（免费）。

## 明确不做（证据否决）

- ❌ 任何形式的粗掩码 logit 残差 / dense prompt 精修（天花板 +0.003）
- ❌ 冻结 decoder 下改点数或换点风格（2/4/8 点实测全负收益）
- ❌ 扩点数（2P2N → 4P4N 等）——除非将来愿意为它做联合重训，且目前无证据支持
- ❌ 跳过 Stage 1 直接全量训练

## 预期与风险

- 诚实预期：学到的模块通常拿到 oracle 的 3~6 成 → +0.004~0.007 mAP，mAP_s 获益最大；
- 主要风险：P2 特征对"真实边缘在哪"的判别力不足（旧头 |δ| 与需求零相关的教训）——
  这正是 Stage 1 用几小时而不是 2 天去检验的东西；
- 当前 run（100 epoch）结束后，用 10 分钟在最终模型上复测 oracle 探针，
  作为"能力缺口 vs 收敛补偿"的定论。
