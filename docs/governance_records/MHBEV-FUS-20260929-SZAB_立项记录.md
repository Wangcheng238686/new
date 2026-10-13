> 2026-10-13 自 EXPERIMENT_GOVERNANCE.md「附录 V（迁出时第 1120–1233 行）」原文迁出；除本行外内容未改。

## 附录 V：立项记录 MHBEV-FUS-20260929-SZAB（小实例任务重定义 × P2 热拟合注入活机制 A/B）

**状态：预注册；训练类（A 路由·模块叠加型，训练级）。实施须独立实施审核 PASS 后启动。**
依据：H2_SCREEN_PASS（热拟合 P2 注入在冻结筛查下有边界增益）+ ORIG-AB（全量共适应，
`finetune_p2_inject_ab.py`，20260920-A/B）已建立训练级双臂框架；其未加权 4-epoch 尺度
分层基线（`ORIG-AB-sz`，仅加 AP_s/AP_m/AP_l 记录）显示普通注入对 `segm/AP_s` 无增益
（e4：B 0.1623 vs A 0.1670），且全部冻结探针（P5/P6/D2/R1/DEPTH-R1）一致指向 UAV
几何的边际信息集中在小建筑边界带。本实验把训练任务重定义为小实例加权目标，在同一
双臂框架下做注入机制的因果度量。

### V.1 谱系、数据与训练（两臂完全相同，唯一下变量 = 注入）

- 谱系与 ORIG-AB 逐键相同：`configs/ue_dual_stream_p2p3_sparse.py`；dual r1 e85 先载入
  后以单流 `ue_single_c5v2_control_tr128_va259_s44_r1/best_model_epoch86.pth` 覆盖；
  `drone_encoder./geometry_bev.` 冻结，其余（LoRA+neck+RPN/ROI heads）可训；E5 式
  最小二乘热拟合 conv（20 场景标定 + 80 场景拟合，FIT_SEED 20260925，1500 cell/场），
  γ init 0.25 仅 B 臂存在；train=官方 `ue_coco_with_uav` train 全量 712 场景（seed 44，
  subset 1.0），val=259 全量；6 epochs、AdamW lr 1e-4（inj 1e-3）、wd 0.01、accum 8、
  warmup 40、cosine→0.1、AMP、seed 20260928；空 GT batch 跳过逻辑不变。
- **脚本：**`scripts/ue/finetune_p2_inject_szab.py`（`finetune_p2_inject_ab.py` 的超集；
  未加权脚本保持冻结以便指纹对照）。唯一非申报语义差异：热拟合采集循环在 20 标定 +
  80 拟合场景收满后提前 `break`（原脚本继续空转前向剩余场景且不产出任何数）——采集集
  逐位一致（同样取前 20/80 个非空 GT 场景），且该阶段两个 encoder 处于 eval、无随机
  层、dataset 不消耗全局 RNG，故 RNG 流与拟合解不变；实施审核（C4）确认惰性。

### V.2 任务重定义（两臂共同，属于任务定义而非臂变量）

小实例加权损失：对 GT 实例 k，`a_k` = 1024² 输入画布上 GT mask 像素数（与 COCO AP_s
的面积定义同源），`w_k = 1/sqrt(max(a_k,1))`，无上下限截断。逐 batch 归一化：

1. **终末实例 mask 损失**（standard `mask_cross_entropy` 路径）：把"全像素普通均值"
   替换为逐正样本 ROI 均值的加权均值 `Σ w_i L_i / Σ w_i`（w_i = 该 ROI 所分配 GT 的
   w_k）。未加权时该式与原实现逐数值等价（每 ROI 像素数相同）。
2. **bbox head cls/box 损失**（V.7 修订版）：正样本 proposal 的 `label_weights` 与
   `bbox_weights` 乘以其分配 GT 的 `w_k` **按 batch 内正样本均值归一到 1**；负样本权重
   不变；`avg_factor` 口径保持 mmdet 默认（cls=非零权重计数、bbox=ROI 总数）。归一化
   保持正样本之间的 `1/√area` 相对形式，同时保持与未加权运行相同的正:负损失量级
   （原始 w 在本数据上均值 ~0.010，相对权重为 1 的负样本会把正类信号压低约百倍）。
3. **明确不加权**（保持原样）：RPN cls/box 损失（anchor 级，无实例语义）、coarse-mask
   辅助损失、P2 boundary refiner 损失、评估协议（与全部既往 259-val 完全同口径）。

### V.3 臂

| 臂 | 唯一变化 | 输出目录 |
|---|---|---|
| A | 加权任务，无注入（`--inject 0`，γ/injector 不存在，extract_feat 纯基座） | `MHBEV-FUS-20260929-SZAB-A` |
| B | 加权任务 + 热拟合 P2 注入（`--inject 1`） | `MHBEV-FUS-20260929-SZAB-B` |

### V.4 指标四要素与判定门

- **主指标：**last checkpoint（e6）259-val `segm AP_s`（COCO area<32²，标准推理）。
- **伴随指标：**逐 epoch `segm AP/AP50/AP75/AP_s/AP_m/AP_l`、boundary-F1、逐场
  small-instance mask-F1@0.5（小实例双侧同口径：GT 与预测均取 area<1024px²，预测另需
  score≥0.5；按 IoU≥0.5 一对一贪心匹配，P=匹配数/小预测数、R=匹配数/小 GT 数；
  双侧皆空记 1，单侧空记 0）；未加权基线
  `ORIG-AB-sz` e4 对照仅作 context，不构成门。
- **显著性：**逐场 small-mask-F1 的 `B−A` paired percentile bootstrap（10,000 draws，
  seed 20260936，报告 95% CI）；两臂 val 顺序由同 seed 保证一致，按场索引配对。
- **判定门 `SZAB_INJECT_PASS`（全部满足）：**
  (i) `AP_s(B,e6) > AP_s(A,e6)` 且 small-mask-F1 `B−A` CI 下界 > 0；
  (ii) 全局不退化守卫：`AP(B,e6)−AP(A,e6) ≥ −0.003` 且 `BF1(B,e6)−BF1(A,e6) ≥ −0.003`
  （H2 (ii) 同款）；
  (iii) B 臂注入通路活跃：γ(e6) 偏离 init ≥ 20% 且 engagement（训练期 `|delta|/|P2|`
  范数比逐 epoch 均值）> 0；
  (iv) 不变量：两臂 manifest 的 git commit、两个 ckpt SHA256、seed 三元组相同；除
  `inject` 标志外臂间无任何行为差异（代码审计确认单变量键集）。
  任一不满足 = `SZAB_INJECT_NO_SMALL_GAIN`。

### V.5 遥测三件套

1. **判读边界：**γ/engagement 与加权比遥测（逐 epoch `Σw`、加权/未加权 mask loss 比）
   只证明通路与重定义活跃，不构成收益证据；结论只由 V.4 主指标 + 门判定。
2. **标定校验：**warm-fit R² 归档；两臂初始模型 state（除 B 臂 injector/γ）hash 相等
   写入 manifest；B 臂 `inject["enabled"]=0` 分支与 A 臂代码路径逐行同一。
3. **机制遥测不替代结局验证：**所有 epoch 均以完整 259-val 标准推理评估（与 ORIG-AB
   完全同口径），不以训练侧字段代替。

### V.6 解释边界与端点

PASS 仅授权结论"在小实例重定义任务下，热拟合 P2 注入具有因果活机制收益（AP_s 口径）"，
并授权后续多 seed 复核或全预算训练提案；不构成全局 AP/BF1 增益主张。FAIL 关闭"当前
P2 热拟合注入形态在小实例任务上的活机制路线"，转数据设计（UAV_DepthBackfill 渲染
质量/实例分离）复盘。两臂各 6 epochs 约 3.5h，GPU 并行；工件全部落
`/data1/wangcheng/checkpoint/worktrees_dual-stream-dev_probe/`。

### V.7 修订与废弃记录

- **2026-09-30 V.2.2 修订（如上）。** 原文"乘以 `w_k`（分母不变）即对正样本的加权均值"
  在 cls 路径上不成立：分母（非零权重计数）同时计入权重为 1 的负样本。V1 运行
  （2026-09-30 18:08–19:59，GPU 0/1，seed 不变）证实：`sz_mean_pos_roi_w ≈ 0.010`，
  正类信号被压至负样本 ~1%，两臂 AP 于 epoch 1 从初始化 0.731 崩至 0.28-0.38，e5 仅
  爬回 0.60-0.64，AP_s(0.10-0.11) 低于未加权基线同期(0.16-0.17)——任务重定义整体劣化，
  与立项意图相反。两臂于 e5/e4 中止，工件归档为
  `MHBEV-FUS-20260929-SZAB-{A,B}-abandoned-v1weighting-0930195934`（作废工件，非门判定
  证据；臂内 B−A 方向仅供参考）。修订只改 bbox 路径一行：正样本权重除以 batch 内
  正样本权重均值（mask 路径 Σ 归一化下数学不变，遥测 sz_ratio 口径不变）。
  修订后须独立复核 PASS 方可重启。

### V.8 完成记录（2026-09-30）

- **判定：`SZAB_INJECT_NO_SMALL_GAIN`。** 两臂 6 epochs 完整跑完（GPU 2/3，V.7 修订版）。
  e6 主指标：AP_s A=0.1827 / B=0.1772（Δ=-0.0055）；small-mask-F1 配对 bootstrap
  （10k、seed 20260936）B−A=-0.0109，CI[-0.0407,+0.0187]，improved 19/259——门 (i) 双双
  FAIL。守卫 (ii) 全过（AP Δ+0.0025、BF1 Δ-0.0005）；(iii) 过（γ 0.2500→0.1836，偏离
  -26.6%；engagement 逐 epoch >0 但递减 0.0139→0.0085）；(iv) 全过（同 commit/ckpt/
  seed、init_state_hash_shared 一致）。
- **解读：**任务重定义本身有效（两臂 e6 AP_s 0.177-0.183、small-F1 0.51-0.52，均高于
  未加权基线参考 0.16-0.17）；但在此目标下热拟合 P2 注入对 AP_s 无增益且点估计为负、
  CI 含 0。与未加权 ORIG-AB（B≈A）、冻结探针链（T1 无语义占据增益；DEPTH-R1 仅为
  误差定位信息而非几何重建增益）方向一致：当前固定高度 RGB/几何 BEV 特征不携带
  卫星流无法自行提取的实例级 AP 相关边际信息——即使目标显式偏向小建筑。
- **处置：**按 V.6 关闭"当前热拟合 P2 注入形态 × 小实例重定义任务"的活机制路线；
  后续转数据设计（UAV 渲染质量/实例分离）复盘，不再做注入变体。工件与结论详见
  `docs/MHBEV-FUS-20260929-SZAB_conclusion.md` 与 SZAB-gate/summary.json。
