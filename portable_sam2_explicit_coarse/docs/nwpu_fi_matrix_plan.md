# NWPU（VHR-10）fi 消融矩阵计划（v2，待审核，未授权启动）

日期：2026-09-06（v2 2026-09-07）。用户指令：主战场转向 NWPU，最终在
NWPU 上报告**完整消融矩阵**。本文档列出就位状态、协议决策点与开放
问题；**任何训练启动均需用户单独授权**。

## 0. 用户已裁决事项（2026-09-07）

- 报告口径：**沿用 val==test**（与 RSPrompter 基线惯例一致），不另设
  模型选择子集；矩阵按 best segm/mAP 报告。
- 目标：NWPU 完整四行消融矩阵为最终交付。

## 0.5 历史 NWPU 数据给出的协议结论（2026-09-07 复核）

| 运行 | 骨干 | 协议 | best | 位置 |
|---|---|---|---|---|
| vhr10_c5v2_400 | base_plus | 400ep cosine | 0.6476 | ep320 |
| vhr10_large400 | **hiera-large** | 400ep | 0.6171 | ep294 |
| vhr10_large600 | hiera-large | 600ep | 0.6533 | ep505 |
| **vhr10_c5v2_ft200** | base_plus | **400ep + 200ep@lr1e-4** | **0.6617** | ft 段 ep84（总 ~484） |

结论：(1) **不换 large**——同等 400ep 预算下 large 落后 0.031，多 50%
epoch 才追平，且仍输给 base_plus 的两段式；(2) **400ep 未收敛**——
低 LR 续训 200ep 增益 +0.0141，最优协议是 400+200 两段式（每 epoch
65 个优化步，400ep=26k 步，best 出现在 ~21k，ft 段在 ~31.5k 步创新高）。

## 1. 就位状态（已实现并 DRY_RUN 核验）

- `scripts/_run_vhr10.sh` 现透传 PE 学习率倍率（不再硬编码为 0）；裸
  fast400 默认仍冻结 PE。`vhr10_p2v2_dev.sh <a0|a1|a2|a3>` 固定 A 系列的
  100ep、四卡、fi、末轮保存和 D5-B 契约；它会清除 resume/init/path
  污染变量并锁住共同 P2/coarse 参数。DRY_RUN 已核验，未启动训练。
- A3 是 A0/PBM 的单阶段端到端 UDPR-K64 对照：仅增加 decoder-tail，P2 保持 off；
  不使用 A0 热启动，也不使用 tail-only 冻结。热启动 `udpr64` 只承担机制筛选，不进入
  A 系列同口径比较。A 系列 wrapper 显式封闭 dense/context/coarse schedule/ROI-SAM/
  point warmup/final-loss 与 Tail LR 环境面；污染环境回归已要求 A0/A3 config 恒等、
  A3 dry-run 为 `init=none, tail_only=0, lr_mult=1.0`。
- VHR-10 full-eval/manifest/bootstrap 已实际以历史 best checkpoint 在完整
  130 图上验收：10 类 category IDs=1..10、130 processed_image_ids、模型空间
  records 导出，且 `--resamples 0` mAP 与 inference 逐位一致。
- `scripts/ablations/vhr10_fi_overlay.sh`（fi 契约 = full_image +
  roi_balanced_dice，与 WHU 同一语义）。
- `scripts/ablations/vhr10_fi_matrix_rows.sh`：四行薄入口
  （point / point_box / point_box_mask / full），run tag
  `vhr10_fi_matrix_*` 与全部旧 VHR-10 结果隔离。
- DRY_RUN 契约行核验通过；**未启动任何训练**。

## 2. 携带的 WHU 经验（矩阵设计的输入）

- fi 契约在 WHU dev 上 +0.024~+0.037（6v4 臂）——跨数据集复现是本
  矩阵的第一目标。
- PE 适配（mask_downscaling 解冻 + LR 0.1）是 WHU 上唯一配对复现的
  正效应（+0.006~+0.008）；小门 α0.1 为最优 Mask 候选（D6）。
- dense/box 的 dev 级效应在 WHU 10%/15ep 分辨率之下（对照极差 0.009）。

## 3. 协议决策点（需审核定夺）

| # | 决策 | 选项与默认建议 |
|---|---|---|
| D1 | 训练规模 | **两层**：①dev 迭代层 = 100ep（6,500 optimizer steps；cosine 压缩、520/130 全量、fi 契约、seed44、单臂 ~4.5h），P2-v2 三臂 A0/A1/A2 与后续迭代在此层进行；②最终层 = **600ep 单段 cosine**（39,000 steps，论文协议）。100ep 是机制筛选，不可把负结果外推为 600ep 必然无效 |
| D1b | 骨干 | **base_plus（不换 large）**——数据支撑见 §0.5 |
| D2 | dense 旋钮 | P2-v2 A0 的冻结底座是 **D5-B**：detach=0/α0.5 + PE mask_downscaling 解冻、LR multiplier=0.1；本轮不可混入 D6 的 α0.1 |
| D3 | Mask 行配置 | A0 作为三臂共同 PBM 底座；A1/A2 除 P2 或 R1 loss 外逐字段相同 |
| D4 | 行数 | 四行（P/PB/PBM/Full）或先三行（无 P2）。**建议四行**：P2 在 WHU 无正证据，但 NWPU 10 类小目标的边界行为未测过，一行成本可接受 |
| D5 | 评估与噪声 | VHR-10 val=test 仅 130 图（10 类）——噪声比 WHU 62 图子集更严峻。**强制**：每行预测导出 per-image 记录 + 配对 bootstrap CI（工具已审计）；考虑 fixed split 的重复训练（至少关键配对） |
| D6 | 契约校验 | _run_vhr10.sh 无 WHU 式 validate_ablation_contract 前置。**建议**启动前补 vhr10 行的契约校验接入（小改动，下轮实施） |
| D7 | 报告口径 | **已裁决（§0）：沿用 val==test，best segm/mAP**；行间比较辅以 per-image 记录 + bootstrap CI（D5） |
| D8 | rot90×4 增强 | 0/90/180/270° 随机旋转（框、掩码、点同步变换；1024 方形画布无需补边），与现有 hflip/vflip+多尺度组合成 8 朝向——520 图的免费 4× 有效数据。需小改数据管线并验证 box/mask 变换。**注意**：启用则与 legacy 数字不可同表直比（legacy 对照行已有历史数据可并列表格但注明增强差异）；全行统一与否待审 |
| D9 | 提示几何 | coarse 64、2P2N 点预算、P2 搜索带 2/4 均为 WHU 建筑足迹调参；NWPU 10 类尺度谱宽（vehicle ~几十 px .. ground_track_field ~上千 px）。**建议**首行（point_box）跑完先出 per-class AP + 点质量遥测，再定是否按类自适应；不盲跑四行 |

## 4. 开放问题

- ~~130 图 val==test 的选型偏差~~（已裁决沿用，§0）；行间差距的
  bootstrap CI 与关键配对复跑仍保留（D5）。
- 10 类不均衡（bridge 94 .. airplane ~600）：per-class AP 附表必出
  （工具已有 VHR10_CATEGORIES 支持）。
- 与 RSPrompter 基线的同表对比数字来源（复跑 vs 引用）。

## 5. 依赖与预算粗估

- 前置：VHR-10 inference/manifest、p2_off 探针、paired bootstrap 的
  130 图/10 类 smoke 全部通过；不依赖 WHU 新训练。
- 150ep 方案：4 行 × (4 卡, ~1-1.5 天) 串行 ≈ 4-6 天；两行并行（2 卡
  ×2 组）≈ 2-3 天。400ep 方案相应 ×2.7。
- 投稿期时间约束下的建议路径：150ep 四行（并行 2 组）→ CI 分析 →
  若契约增益复现且 dense/PE 信号清晰，择优 1-2 行升 400ep 定稿。
