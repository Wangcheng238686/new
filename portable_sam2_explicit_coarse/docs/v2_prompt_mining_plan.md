# V2 Prompt Mining 实施方案 v2.3（自包含终版）

> 交接文档：实施助手执行、规划方审查。v2.0（git 962f5e0）/v2.1（c9815aa）/v2.2
> （aad4b4a）仅存历史参考；**本文档自包含，实施只需本文 + 代码库**，唯一下列
> 例外：S3 前置门 B 的实验协议遵循
> `docs/prompt_consumption_and_p2_refinement_implementation_guide.md`（下称"指南"）。
> 分支 `machine2/fast-whu150`，主机 lthpc（4×RTX 3090 24GB，/data 大盘）。

## 0. 背景证据（已完成审计，勿重复验证）

模型：RSPrompter-anchor 系 + SAM2（冻结 Hiera+LoRA），两阶段检测 + prompt 挖掘
（coarse 画布 64×64 ROI-local + 2正2负形状点）→ SAM2 decoder 出掩码。
VHR-10（NWPU VHR-10，RSPrompter 520/130 划分，maxDet=100）现状：
best segm/mAP 0.6617（ft200）；RSPrompter-query 论文 0.675。

关键测量结论（决定本方案全部设计约束）：
1. 64×64 画布已达 0.954 IoU（VHR-10）——该分辨率饱和；
2. P2BR 画布精修 Δ≈0；P2 框精修/框抖动全程未激活（现配置即关闭态）；
3. decoder 只消费 sparse 点/框：去点 0.73→0.16；dense 基底与 s0/s1 置零无变化；
4. **负点病灶**：17% 的 N1 落在 GT 前景内（miner 从预测画布背景采样；airplane
   阴影被当背景 → 教 decoder 排除阴影）。airplane 逐类 AP 0.27-0.34
   （AP50 0.95 / AP75 0.01 断崖 = "飞机+阴影"复合标注）；
5. 8/24 研究（docs/p2_decoder_side_findings.md）：对已共适应 decoder 的 bolt-on
   **新增 token** 一律为负（oracle 点 -2.3pt）→ 消费端必须联合训练；
   （注意：移动既有 token 位置 ≠ 新增 token，未被该证据覆盖——S1.5 正是测这个）；
6. miner 在 no_grad 输出整数索引、coarse GT 在 mask forward 后构造、
   P2BR 硬编码 coarse_size=64、PromptEncoder 坐标全程浮点可微——四项代码事实
   决定下述工程约束。

## 1. 固定资产与操作基准

### 1.1 checkpoint 与评估
- 主基线（C0/各臂初始化与对照）：
  `/data/wangcheng/checkpoint/portable_sam2_explicit_coarse/ablations/vhr10_c5v2_ft200_tr1.0_va1.0/best_model.pth`
  （raw 权重，EMA_EVAL=0 协议下 best 即 raw；val segm/mAP 0.6617）
- 评估：`inference/infer_from_checkpoint.py --split validation`（=报告集），
  raw 与 EMA 双测；逐类/airplane 分 IoU 分析按
  `logs/test_eval/vhr10_c5v2/REPORT.md` §4 脚本逻辑（原尺度口径）。
- 数据：`/data/wangcheng/dataset/NWPU VHR-10 dataset/`，
  `coco_split/NWPU_instances_{train,val}.json` + `positive image set/`（路径含空格）。

### 1.2 唯一 launcher 与架构环境变量
入口模板：`scripts/ablations/vhr10_fast400.sh`（含完整协议）。架构导出段**必须原样
携带**（C4 事故教训：漏导出=静默降配，日志契约行必核对 architecture id/fingerprint）：
```
NECK_TYPE=pafpn  PROMPT_ROUTE=coarse  EXPLICIT_PROMPT_MODE=points_box_dense
P2_BOUNDARY_REFINER_ENABLED=1|0(见各步)  SAM_IMAGE_EMBED_STRIDE=16
SHAPE_DENSE_TRANSFORM=raw_logits  SHAPE_DENSE_DETACH=1
FINAL_MASK_COORDINATE_MODE=roi_local
P2_BOUNDARY_REFINER_PROJECTED_CHANNELS=64  P2_BOUNDARY_REFINER_MID_CHANNELS=64
P2_BOUNDARY_REFINER_LOSS_WEIGHT=0.05  SHAPE_PRIOR_LOSS_WEIGHT=0.10
```
训练协议（C0 与所有训练臂固定不变）：80ep、lr 1e-4、warmup 100 iter、seed 44、
AMP、4 卡 batch1×accum2（有效 8）、增强与 EMA 影子照旧、maxDet100、best-only。

### 1.3 统计与聚合规范
- 新 DEBUG 字段：每 50 optimizer step 打 rank-local `sum/count` 两个原始量；
  epoch 级 ratio 由 rank0 `allreduce(sum)/allreduce(count)` 汇总后打印；
- 现有 DEBUG-SP 是 detached 诊断（不可当训练监督）；新增可训练监督字段必须与
  诊断字段分名（后缀 `_loss` vs `_stat`）。

## 2. 实施序列

### S1 coarse GT 前移复用（前置工程，无训练）
把 jittered-ROI 对齐的 coarse target 构造从 mask forward 后（models.py
"ROI-local coarse-mask supervision" 段）前移到 _mask_forward 之前；同一份 target
复用于 coarse loss、后续点语义损失、DEBUG 统计。
**验收（零差异，全链路）**：raw coarse loss、refined coarse loss、P2BR loss、
全部 DEBUG 字段、最终 mask 输出——开关切换逐位一致。

### S1.5 推理期 oracle 负点门（一次推理，零训练，go/no-go）
**oracle 严格定义**：
1. 对象：仅处理与 GT 匹配的正样本 proposal（IoU≥0.5，逐 proposal 最近 GT 匹配；
   未匹配 proposal 的全部点原样保留）；
2. 动作：仅替换 N1；P1/P2/N2、标签、token 数全部不变；
3. N1 teacher：在匹配 GT 的**边界外 ε 环**（ε=8px，1024 坐标）内选"与原 N1 距离
   最小、且与 P1/P2/N2 满足现有间距约束"的合法点；环内无合法点则不替换并计数；
   禁止用远背景点替代；
4. 仅推理期替换，模型不动；基线 = ft200 best raw。
**输出 manifest（随结果归档）**：matched_roi_count、illegal_n1_count（原 N1 在
GT 前景内的 matched ROI 数）、actually_replaced_count、每图替换率分布、
被替换前后坐标对照表。
**判定**：逐图 paired bootstrap 95% CI（非点估计）：val segm/mAP CI 下界 > 0
或 airplane 逐类 CI 下界 > +3pt → 进 S2；CI 含 0 → 负点链条降级，主线转 S3。

### S2 消费端训练臂（PromptRobustifier）+ C0 对照
**训练点替换语义（严禁"GT 语义重标"——改 label 会更换 PromptEncoder 的
正/负 token embedding，破坏固定 2P2N 语言）**：
- clean 分支：miner 原点原样；
- corrective 分支：N1 替换到 GT 合法的边界外位置（teacher 规则同 S1.5），
  **label 保持 0**；环内无合法点 → 回退原点并计入 fallback_stat；
- 混合比例（主臂固定）：clean:corrective = 1:1 全程；退火（corrective→1.0）
  仅作为独立消融臂；
**推理口径不变**（miner 预测驱动，无 GT）。
**gap 定义与验收**：`gap(m) = metric(m, oracle-N1) − metric(m, miner-N1)`。
1. 新臂 vs C0，miner-N1 推理口径，逐图 paired bootstrap 95% CI 下界 > 0；
2. gap(new) ≤ gap(C0)；
3. oracle-N1 绝对指标单列（防"双指标同降导致 gap 收窄"的假阳性）。

### S3 连续有界偏移头（双重前置门 A+B 均通过才启动）

**前置门（v2.3 新增，缺一不可）**：
- **门 A（点位效用）** = S1.5 通过：移动既有 N1 对冻结 decoder 有稳定效用
  （paired bootstrap 95% CI）。**A 失败 → 整条 point-correction 分支停止**
  （不转 S3，S3 的前提正是点位修正有效）；
- **门 B（P2 增量信息）**：真实 P2 特征在相同 head、相同预算下，显著优于
  coarse-only 与最佳 sham-P2 对照——实验协议遵循指南 Phase 1（缓存
  coarse cue + base points + P2、同参数量轻量预测器、sham 对照；主指标为
  冻结 decoder 的配对效用，image bootstrap 95% CI 下界 > 0）。
  **A 通过但 B 失败 → 只允许非 P2 的 PromptRobustifier（coarse-only 偏移或
  纯 corrective 混训），不实现 P2 offset head**。

**互斥契约（强制）**：新变体默认
```
P2_BOUNDARY_REFINER_ENABLED=0   # 旧 P2BR 关闭（见下方 C0-off 定义）
P2_POINT_REFINER_ENABLED=1      # 新偏移头（新注册架构契约变体 id，如 c6_point_offset）
```
dual-P2（两者同开）仅作为后续显式消融。

**初始化与对照（v2.3 定义，保证单变量比较）**：ft200 属 P2BR-on 架构，关闭
P2BR 会改变 architecture id/fingerprint，训练器默认拒绝跨架构 `--init-from`：
- **C0-off 臂**：ft200 → P2BR-off，经 `--allow-cross-arch-init` 初始化，
  记录 missing/unexpected/excluded keys；先做**固定 checkpoint 的 P2BR on/off
  paired 输出检查**（顺带测得旧 P2BR 的真实推理贡献）；再按标准 80ep 预算训练；
- **S3 臂**：从同一 P2BR-off 初始化出发，**仅额外引入 zero-init point refiner**；
  S3 vs C0-off 即单变量比较，不混入"移除旧 P2BR"的效应。

**梯度契约（v2.3 精确化）**：对 offset head 而言，来自最终 mask 的唯一可微输入
路径为 coords+offset；P2/coarse cue 对偏移头输入 detach、不向上游回传。
**其余模块（MaskDecoder、shape prior 等）维持现有训练策略不变**，由同预算
C0-off 控制其影响；如需验证 refiner-only，显式冻结其余参数并新增 module-only
训练 scope 标志，作为独立臂。
**缓存契约（不得覆写现有整数缓存）**：
```
base_local_yx     # miner 整数索引，保留供现有 GT 诊断
base_coords       # image xy（float）
p2_refined_coords # 偏移后连续 image xy（float），仅供 PromptEncoder
```
refined 点的语义统计新增**连续采样式**字段（grid_sample 在 GT 图上取值），
不得把 float 点写回 `_last_shape_point_local_yx`。
**验收**：N1 误落率（可训练口径）17%→<2%；新臂 vs C0（各自关闭旧 P2BR 的口径）
配对 CI 为正。

### S4 128 residual coarse（最后，独立因素）
`coarse128 = bilinear(raw64) + zero-init residual`，residual 读取更高频 RoI/P2
特征（非仅改插值栅格）；旧 P2BR 关闭（coarse_size=64 耦合），与"P2BR-off+64"
的 C0 做正交消融；α 限幅（若做）为独立臂且先记录 ft200 实际 α。
正点入边界带：独立臂，不混入任何恒等主臂。

## 3. 全局停止门与失败行动
- 恒等起点：任何新模块加载后续训首 epoch 指标 ≈ 基线（ft200 先例 0.6399/0.6417）；
- val mAP 单边下行 >2pt 不回 → 冻结该新自由度、只收已验证收益；
- S1 零差异验收失败 → 不得进入任何训练步骤，先修前移实现；
- S1.5 CI 含 0 且 airplane 无增益 → 门 A 失败：**整条 point-correction 分支停止**
  （S2 与 S3 一并跳过），只剩 S1 的工程收益；后续方向重议（不自动转 S3）；
- 门 B 失败（真实 P2 无增量信息）→ 停 P2 offset head，只保留非 P2 的
  PromptRobustifier 路线（coarse-only 偏移或纯 corrective 混训）。

## 4. 审查节点（实施方暂停并交材料）
A. S1 零差异全套证据后；B. S1.5 manifest+CI 后（S2 go/no-go 裁定）；
C. S2 终值三项验收后；D. S3 变体契约行+缓存契约冒烟后；E. 任何停止门触发时。
材料：日志片段（契约行/DEBUG 新字段）、指标 json、oracle manifest、commit 列表。
