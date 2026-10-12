# NWPU VHR-10 失分解剖与抠分清单（v2）

日期：2026-09-20（v2 重构；v1 2026-09-19 的 SOTA 参照系考据与复核记录收缩进 §5 附录）。
性质：**纯分析报告**——只用手头已有数据与证据，不启动任何训练。
目标口径：**mask mAP 72**（用户指定：文献中确有模型能报 72 档，参照不在本机/未部署）。
当前位置：a3sgm last **68.2** / bestC（论文行）**68.3** / bestS **69.3** → **距 72 差 2.7~3.8 点**。
数据：NWPU VHR-10，RSPrompter 80/20 split，train 520 / val=**test** 130 图 743 实例 10 类，
COCO maxDets=100；a3sgm 形态 4×1×2（machine2，seed44，300ep）；无 TTA。
本文全部新分解基于 `inference_last_vhr10`（last 权重）的 dt/gt records 重算；
v1 已有结论（四权重指标、逐类 AP、IoU 曲线）经独立子代理复核（§5）。

---

## 0. 一页结论

**检测侧不丢分，丢分全部在掩膜边界，而且集中在小目标的掩膜边界。**

1. 检测侧：segm AP50 = 94.7（bestS 95.4）；GT 漏检（IoU≥0.5 无任何匹配）仅 **27/743 = 3.6%**；
   检测输出 771 vs GT 743，FP 净压力 +3.8%；bbox mAP 73.2 / AP75 84.4。
2. 掩膜侧三个定量事实（last 权重，greedy 匹配 @IoU≥0.5，716 个已匹配掩膜）：
   - **中位匹配 IoU 0.826；68% 卡在 [0.7,0.9) 带；21% 低于 0.75；仅 23.5% 达 0.9**——
     大量掩膜"离翻过下一档阈值只差一点"，这是 mAP 对边界改进高敏感的直接原因；
   - **掩膜质量随目标尺寸单调**（模型空间 sqrt(area)）：<32px 均值 0.74 → 32-64px 0.80 →
     64-96px 0.855 → ≥192px 0.97；**<64px 占已匹配实例 64%**（455/716）；
   - 全局 IoU 档曲线：0.945/0.940/0.931/0.901/0.843/**0.750**/0.610/0.471/0.315/0.108
     （0.50→0.95），严苛段崩塌与上一条同源。
3. 架构相关机制（假说+配置证据，非已证因果）：mask 分支 RoI 特征 =
   **RoIAlign output_size=14**（strides [4,8,16,32]，继承链 vhr10→whu1024→根配置
   `rsprompter_anchor_satS_v11_sam2_large_full.py:169`，无下游覆盖）——64px 目标落在
   14×14 网格上每格 ≈4.6px，边界细节低于特征分辨率；叠加 keep_ratio=False 各向异性
   拉伸（1024²），细长类（airplane/bridge）形变最重。
4. 逐类两桶失分：
   - **弱 5 类**（小/细长，高 IoU 全灭）：airplane 0.51 / bridge 0.37 / harbor 0.57 /
     vehicle 0.60 / ship 0.61；
   - **回归 2 类**（大块规则场地，输给自家基线 MaskDINO）：basketball 0.78 vs 0.86、
     baseball 0.83 vs 0.88——其中 basketball 的 bestS(E70)=0.874 不输基线，
     last 回落 −9.5 AP 属训练后期衰减同向（16 实例小样本，弱证据）。
5. 抠分四桶（§3）：**中位合计 +3~4 点 → 71~72 可及**；最大单项在小目标掩膜分辨率
   （TTA/高分辨率推理/按类几何），其次训练动力学（rot90 增强、两段式）。

---

## 1. 失分账本（last 权重，全部可由产物复算）

### 1.1 检测侧：几乎无失分

| 项 | 值 | 读法 |
|---|---:|---|
| segm AP50 | 0.9465 | 松匹配近饱和，超过所有已发表方法最优 AP50 |
| 漏检（GT 无 @0.5 匹配） | 27/743 = 3.6% | harbor 8、vehicle 8、ship 4、bridge 4、baseball 2、basketball 1；其余类 0 |
| 输出/GT 实例数 | 771/743 | FP 净压力 +28，NMS 后几乎无冗余 |
| bbox mAP / AP75 | 0.7317 / 0.8438 | 超文献最优 bbox（RSPrompter-anchor 70.3）2.9 点 |

漏检集中于密集小目标堆（harbor/vehicle），是 AP50 的次要损失源（≤+0.5 点量级），
不是主矛盾。

### 1.2 掩膜侧：匹配 IoU 直方图 + 尺寸单调性（本报告新增分解）

**逐类漏检与匹配 IoU 分布**（greedy by score，同类别，IoU≥0.5 记匹配）：

| 类 | nGT | 漏检 | <0.6 | 0.6-0.7 | 0.7-0.75 | 0.75-0.8 | 0.8-0.9 | ≥0.9 | 中位 IoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| airplane | 195 | 0 | 1 | 29 | 58 | 73 | 34 | **0** | 0.756 |
| vehicle | 154 | 8 | 2 | 6 | 10 | 27 | 95 | 6 | 0.826 |
| tennis_court | 78 | 0 | 0 | 3 | 3 | 8 | 30 | 34 | 0.895 |
| harbor | 77 | 8 | 1 | 3 | 6 | 15 | 36 | 8 | 0.824 |
| baseball | 63 | 2 | 0 | 0 | 0 | 2 | 10 | 49 | 0.934 |
| ship | 61 | 4 | 1 | 2 | 7 | 9 | 32 | 6 | 0.833 |
| bridge | 30 | 4 | 2 | 10 | 5 | 5 | 3 | 1 | **0.725** |
| storage_tank | 38 | 0 | 0 | 0 | 0 | 2 | 14 | 22 | 0.902 |
| gtf | 31 | 0 | 0 | 0 | 0 | 0 | 0 | 31 | 0.971 |
| basketball | 16 | 1 | 0 | 1 | 0 | 0 | 3 | 11 | 0.936 |

- **airplane 是最典型的"纯边界病"**：0 漏检、AP50 0.989，但 161/195（83%）匹配掩膜
  IoU<0.8、0 个≥0.9 → AP 只有 0.51。
- **bridge 中位 0.725**：一半掩膜在 AP75 阈值以下。
- **总量**：716 匹配中，[0.7,0.9) 带 487 个（68%）、<0.75 占 21%、≥0.9 仅 168（23.5%）。

**掩膜 IoU × 目标尺寸**（模型空间 sqrt(area)，px）：

| 尺寸档 | n | 均值 IoU | 中位 | ≥0.9 占比 | <0.75 占比 |
|---|---:|---:|---:|---:|---:|
| 0-24 | 3 | 0.675 | 0.711 | 0.000 | 1.000 |
| 24-32 | 22 | 0.756 | 0.767 | 0.000 | 0.364 |
| 32-48 | 315 | 0.797 | 0.799 | 0.102 | 0.279 |
| 48-64 | 115 | 0.799 | 0.793 | 0.035 | 0.191 |
| 64-96 | 123 | 0.855 | 0.884 | 0.382 | 0.122 |
| 96-128 | 75 | 0.865 | 0.910 | 0.520 | 0.107 |
| 128-192 | 31 | 0.863 | 0.887 | 0.452 | 0.194 |
| ≥192 | 32 | 0.969 | 0.972 | 1.000 | 0.000 |

**<64px 桶（455 个实例，64%）均值 IoU 0.80**：这正是 §1.2 直方图 [0.7,0.9) 带的主体。
掩膜质量不是均匀差，是被小目标拖垮的。

### 1.3 box↔mask 转化损失

bbox mAP 73.2 vs segm 68.2（**−5.0**）；bbox AP75 84.4 vs segm 74.9（**−9.5**）；
bbox AR100 77.2 vs segm AR100 71.7（**−5.5**）。框找得到、膜画不准的量化表述。

### 1.4 逐类 AP（v1 已复核，此处为完整性）

airplane 0.506 / ship 0.605 / tank 0.823 / baseball 0.829 / tennis 0.750 /
basketball 0.779 / gtf 0.991 / harbor 0.567 / bridge 0.365 / vehicle 0.598（均值 0.681）。
MaskDINO 对照：tennis −15.2、vehicle −14.3、harbor −7.5（我们赢）；basketball +8.5、
baseball +5.1（我们输，仅有的输给基线的类）。

---

## 2. 机制归因（为什么落后在这些地方）

1. **小目标 RoI 特征分辨率（首要嫌疑）**：mask 分支 RoIAlign output_size=14、
   featmap_strides [4,8,16,32]（SingleRoIExtractor 按框尺度选层，小框落到 stride 4/8 层）。
   64px 目标 → 14×14 网格 → 每格 ~4.6px；SAM2 decoder 的 256×256 输出是从这个
   条件网格上采样的，边界细节低于网格分辨率的部分只能靠先验脑补。
   §1.2 的尺寸单调性（<64px 0.80 vs ≥192px 0.97）与此一致。**注意：这是配置证据 +
   相关性，不是已证因果**；验证手段见 §3-A3。
2. **各向异性拉伸**：keep_ratio=False 拉到 1024²（协议选择，"非 bug"），75% 图像
   双边 <1024 即被放大且 x/y 比例不同——细长类（airplane 翼展、bridge）的形状先验
   与旋转/朝向多样性被压缩。协议层面杠杆，与 D8（rot90×4）同源。
3. **尾臂过拟合动力学**：a3sgm bestS@E70 后 last 衰减 −0.0117（A3 家族 −0.0233）；
   300ep 单余弦尾段对 segm 无正贡献；520 图小数据无旋转增强。basketball
   bestS 0.874→last 0.779 同向（小样本弱证据）。
4. **点位几何为 WHU 调参**（D9 遗留）：2P2N 点预算、coarse 64、P2 搜索带均为建筑
   足迹调参；NWPU 尺度谱宽（vehicle 几十 px → gtf 上千 px），弱 5 类全在谱系小端。
   K 点数预算轴已榨干（K64 定案，K32 掉带、K128 从头 0.6616<K64 0.6696）——
   瓶颈在"点位/几何"不在"点数"。
5. **已兑现与剩余的边界写入空间**：阈值锚定 margin 写 oracle 上界 +0.027
   （CI[+0.021,+0.032]，a3sgt 容器）；a3sgm 的 margin 尾已兑现大半
   （a3sgtm-dclip→a3sgm 平台 +0.032）；K128/256 外推仅 +0.003~0.004。
   **写入目标标定轴基本关账**——剩余 3-4 点不在尾轴上。

---

## 3. 抠分清单（按预期×成本排序；每项给证据、预期、验收门）

### A 桶：小目标掩膜质量（最大失分桶，<64px 占 64% 实例）

| # | 动作 | 直接证据 | 预期（点） | 成本 | 验收门 |
|---|---|---|---:|---|---|
| A1 | **TTA 推理（hflip + 多尺度）** | RS4D 官方在 NWPU 用 TTA；我们从未用；AP50 饱和说明检测稳定，TTA 主要收益在掩膜平均 | +0.5~1.5 | 低（推理侧，无训练；单卡数小时/权重） | 130 图配对，segm mAP Δ≥+0.005 且 bbox 不降 >0.005 |
| A2 | **高分辨率/分块推理**：小目标密集图 2× tile（或按图自适应），模型不变 | §1.2 尺寸单调 + <64px 64% 实例；RoI 14×14 对小目标信息不足的直接 workaround | +1~2 | 中（推理管线 + 评测仍在 1024 模型空间对齐） | 同上；另报 per-size 桶增量（<64px 桶应为主要收益） |
| A3 | **mask RoI 特征 14→28**（roi_layer output_size，或高分辨率 skip 增强） | 配置证据 §2.1 + 尺寸单调相关性 | 0~+1.5（不确定） | 高（架构改动需重训） | dev100 探针先行：平台 ≥+0.005 再升全量 |
| A4 | **按类/按尺度几何自适应**（D9 遗留）：弱类 2P2N→4P4N、框 jitter 按类 | K 轴饱和 + 弱类全在小端 + 点预算为 WHU 调参 | +1~2（弱类 AP +5~10 → 全表 +1~2） | 中（dev100 探针 ~4.5h/臂） | 弱 5 类 AP 均值 +0.05 且强类不降 >0.005 |

### B 桶：训练动力学（520 图小数据）

| # | 动作 | 直接证据 | 预期（点） | 成本 | 验收门 |
|---|---|---|---:|---|---|
| B1 | **rot90×4 增强**（D8 设计完成从未启用） | 尾臂 E70 后 −0.0117 衰减；细长类朝向敏感；520 图×4 免费数据 | +0.5~1.5 | 中（管线小改+验证；与 legacy 数字不同表，注明） | dev100 平台 ≥+0.005 |
| B2 | **两段式低 LR 续训**（last 续 100~200ep @1e-4） | 历史同构 +1.4（旧臂族 400+200：0.6476→0.6617）；尾臂动力学不同需探针 | +0.5~1.5 | 低（可从 last 续跑，不用从头） | 续训段平台 > last 平台 +0.005 |
| B3 | basketball/baseball 回归修复 | basketball bestS 0.874→last 0.779；baseball 稳定输 MaskDINO 5.1 | +0~1.4（公式上限 (0.085+0.051)/10=1.36，按折扣取执行预期） | 随 B1/B2 或自愈（basketball）；baseball 或需规则形状先验 | 逐类 AP 恢复至 ≥MaskDINO−0.01 |

### C 桶：口径内回收（不改模型能力）

| # | 动作 | 证据 | 预期 | 备注 |
|---|---|---|---:|---|
| C1 | 尾臂 mask_thr 迁移标定（非尾臂先例 +0.002~0.010） | nontail_calibrated_reeval_20260917 | +0~0.5 | 尾臂自锚定 0.4，b05 已证锚点不敏感——预期小，顺手做 |
| C2 | EMA 评测 | **实测 ckpt 无 ema_state 键**（last_model_epoch300.pth 顶层键无 EMA） | 需重训才可启用 | 降权；WHU 经验 EMA segm 微负 |
| C3 | lthpc 口径统一（论文数字） | 跨机偏移 +0.011~+0.021（A2/四臂实测外推） | +1~2（口径内） | 不改能力；论文表统一机器时自然发生 |

### D 桶：不投入（已证伪/关闭）

锚点 b05（0.5 锚输 0.4）；BM（纯框+sg dense 无增益）；R3 渲染器；dense 容量头；
box-dropout；UDPR K∈{16,32,128,256}（K64 定案）；迭代解码；64→128 coarse；
600ep 直跑；large@400ep（600ep/两段式下另议，见 §3 尾注）；授权 gate oracle（CI 跨零）。

### 合计与路径

中位估计：A1(+1) + A2/A4 取一(+1.5) + B1(+1) + B2(+1) ≈ **+3~4 点 → 71~72**。
上限场景（各桶上限）≈ +6~7；下限场景 ≈ +2（71 不到）。
**先零成本后训练**：等 WHU fast150 释放 GPU 期间先跑 A1（TTA 推理，单卡）；
GPU 释放后 dev100 探针批（B1/A4/A3 各一臂）；全量择优 + B2 续训收尾。
backbone large + 两段式 = 挂起的独立大项（历史 600ep 反超 +0.006），排在 A/B 之后。

---

## 4. 边界与不确定性（防过度解读）

1. 本报告的"漏检/直方图/尺寸"分解是 **last 权重、模型空间 1024²、greedy 匹配**的口径，
   与 COCO 官方匹配（全 IoU 档、按 score 贪心）在个别边际实例上有差异，不影响结论量级。
2. 尺寸用模型空间 RLE area；官方 metrics 的 APs=−1 是原图 area 口径的 bin 空问题
   （两口径均自洽，不互比）。
3. §2.1 RoIAlign-14 机制是配置证据 + 强相关，**未经干预实验证实**；A3 探针就是它的
   判定实验。
4. basketball 的 bestS→last 衰减基于 16 实例小样本，只作方向参考。
5. 所有"预期点"为证据内插/同构外推，不是预注册结果；立项时按 experiment-governance
   预注册四要素（指标/门/噪声标尺/判读规则）执行。

---

## 5. 附录

### 5.1 v1 参照系考据（一句话版）

可核实的最新文献锚（RS4D, TGRS 2026, arXiv 2606.25324, Table III，同 split 同协议）
已发表最优 mask mAP = RSPrompter(query) 67.5；自家同协议重跑最优 = MaskDINO 65.8。
用户口径的"72 档 SOTA"文献不在本机，未逐一核实——本文直接以 72 为目标值，不以此
下"领先/落后"结论。自家 RSPrompter-bf16 / RS4D-prompt 重训结果在 lthpc 未落盘。

### 5.2 v1 独立复核记录（2026-09-19）

独立子代理 14 项复核：12 PASS / 1 FAIL（算术标签，已改）/ 1 基本成立（披露建议 3 条，
均已落实）。数据底盘（四权重指标、逐类表、IoU 曲线、日志形态、尾臂轨迹、oracle/K、
文献表、论文主表口径）全部独立复算一致。v2 新增分解（§1.1 漏检计数、§1.2 直方图与
尺寸表、EMA 无键、RoIAlign=14 继承链）的复核见 §5.3。

### 5.3 v2 新增分解的复核记录（2026-09-20）

独立子代理 7 项复核 **7/7 PASS**（自建内联 python + pycocotools 重算，未用报告方脚本；
贪心匹配对"取最高 IoU"与"取首个过阈"两变体各跑一遍，计数一致——结论对贪心细节稳健）。
§1.1-1.2 全部数字（漏检 27/743 与分类、10 类×6 桶直方图、各中位、尺寸表 40 格、
68%/21%/23.5% 总量断言、<64px=455/716）与独立重算零偏差；EMA 无键、RoIAlign=14
继承链（含易混项排除：bbox roi 7 / canvas 128 / p2 roi 32 均属别的模块）、
roi_mask_size=28 仅训练目标尺寸（唯一使用点 sam2_mask_head_targets.py:223）、
测试粘贴走父类路径、因果边界与预期点表述——全部证实。修订建议 3 条（EMA 键清单
补 checkpoint_schema_version、B3 上限标签 1.36、本节回填）均已落实。

### 5.4 证据索引

| 项 | 路径 |
|---|---|
| a3sgm 四权重指标/记录 | `/data/wangcheng/checkpoint/portable_sam2_explicit_coarse/ablations/vhr10_p2v2_matrix300_a3sgm_pbm_sg_udprk64tm_tr1.0_va1.0/inference_{best,best_bbox,best_composite,last}_vhr10/` |
| a3sgm 训练日志（形态/轨迹） | `logs/ablations/vhr10_p2v2_matrix300_a3sgm_..._20260916_225843_pid1150355.log` |
| 漏检/直方图/尺寸分解 | 本文 §1.1-1.2；由上述 records 以 pycocotools RLE greedy 匹配重算（脚本 `/tmp/perclass_eval.py` 及内联代码，复核见 §5.3） |
| EMA 无键检查 | `last_model_epoch300.pth` 顶层键：checkpoint_schema_version/model/optimizer/scheduler/epoch/best_metrics/history/config_snapshot/architecture_contract/scaler（model 671 键中含 ema 的 0 个）；注：config_snapshot.training_args 中 ema_enabled=1 但未落盘 EMA 权重 |
| RoIAlign=14 配置链 | `configs/vhr10_baseplus_explicit_coarse.py` → `whu1024_baseplus_explicit_coarse.py` → `whu1024_baseplus_clean.py` → `rsprompter_anchor_satS_v11_sam2_large_full.py:169-174`（全配置唯一 mask_roi_extractor 定义，无下游覆盖） |
| 28×28 loss 契约 | `rsprompter/sam2_mask_head.py:84`（roi_mask_size=28 默认，训练目标尺寸）；测试粘贴走父类路径用 decoder 原生输出 |
| 全矩阵读数/判读 | `docs/evidence/box_prompt_dependence_vs_contribution.md`；`docs/evidence/p2v2_matrix300_four_arm_machine2_results.md` |
| 写入 oracle / K 敏感性 | `docs/evidence/k_sensitivity_results.md`；`diagnostics/a3sgt_tail_write_oracle_*_20260914/` |
| 基线重跑总表（含 MaskDINO 逐类） | `NWPU_VHR10_EXPERIMENT_TRACKING.md` |
| 历史两段式/large 证据 | `docs/nwpu_fi_matrix_plan.md` §0.5 |
| b05（锚点关闭） | `ablations/vhr10_p2v2_matrix300_a3sgm_b05_.../` + 日志尾（last 0.6797/0.7233；bestS 0.6964 仅尖峰口径） |
| GPU 现状 | 四卡 = WHU fast150（2026-09-19 04:59 起，whu1024_a3sgm_roilocal_full_fast）；b05 四权重推理链 `/tmp/b05_inference_chain.sh` 仍挂起等待 |
