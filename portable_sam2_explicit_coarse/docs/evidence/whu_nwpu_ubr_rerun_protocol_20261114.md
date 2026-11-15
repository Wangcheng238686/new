# WHU/NWPU 单流重跑（UDPR→native UBR）协议预登记 2026-11-14

用户授权（2026-11-14）：论文单双流模型统一——旧 a3sgm 的 WHU/NWPU 论文行
（UDPR 尾）用新单流模型（native UBR，与双流开发对比的 S 代号基线模块级一致）
重训重测；训练预算/数据增强/选模/评测协议保持论文旧制。

## 立项路由（experiment-governance）

基座/训练动力学型：模块替换（UDPR→native UBR）+ 按论文协议全量重训，无冻结
基座可热启动 → 冻结诊断探针（指纹 parity / 零初始化恒等 / 4-rank smoke）→
直接全量（说明：热启动筛查无定义——尾模块替换改变梯度路径，且论文协议本身
是从通用 SAM2 初始化从头训）。

## 用户裁定（口径冻结）

- **架构一致性范围**：与 S 代号（compact S_t50 / S_D4_t50 / S_unified，三者
  权重结构逐键相同）在**模块设计层一致**——UDPR 删除 + native UBR
  （`NativeRoIRefiner`，6,625 参数、零证据 70→32、seed 2080 零初始化、FP32
  头）、无 UAV 参数/加载器、coarse64 shape_prior 保留在跑。
- **两项例外走论文旧口径**：① 分辨率族 = 1024（pad1024/embed64/mask_size
  1024²/shape_prior img_size 1024，不用 S 的 512 族）；② 坐标契约 = 论文各自
  原样（WHU `roi_local`，NWPU `full_image` + `roi_balanced_dice`）。
- 其余训练/评测协议全部保持论文旧制（见下）。

## 模型差分（唯一变量）

| 项 | 论文旧行 | 本次新行 |
|---|---|---|
| 尾模块 | UDPR `decoder_tail_refiner`（K64 点残差，38,217 参数，独立 margin loss） | 删除（`enabled=False`），换 native UBR 后置稠密残差（6,625 参数，零证据补零，主 mask loss 直接监督） |
| 尾数值精度 | 随全局（fp32/AMP） | UBR 头强制 FP32 island（AMP 下防下溢，随模块自带） |
| NWPU 辅助尾 clip | `--decoupled-aux-clip`（前缀 `decoder_tail_refiner.`，0.5） | 同语义，前缀等价为 `roi_head.mask_head.native_refiner.` |
| 其余全部键 | 论文 config 原样 | 逐键不变（指纹 parity 门验证） |

**不移植**（S 家族的 CitySample 合同，与本次无关）：512 分辨率族、
max_per_img 250 / 推理 nms_pre 2000 / AP250 口径、D4 增强、trusted score
重编码、两桶 clip、dual_target（后四项为双流/T-IT 专属或推理消费策略）。

**唯一新形态**：NWPU 的 full_image 版 UBR 安装分支——S 家族安装器只支持
roi_local（硬断言），坐标契约走论文（full_image）需放宽该断言并适配全图
lattice（1024→256）；act2 捕获自 `output_upscaling[-1]`、证据零填充、
refined logits 进原 roi_balanced_dice 监督、零初始化恒等不变。

## 冻结训练协议（克隆论文脚本，逐参不动，仅换 config/RUN_TAG/加旗标）

### WHU（EXP-0046，`whu1024_a3sgm_ubr_fast150_20261114`）
- 基线脚本：`scripts/ablations/whu1024_a3sgm_roilocal_full_fast.sh`；
  config：`configs/ablations/whu1024_ubr_fast150_20261114.py`（`_base_`
  a3sgm_whu1024.py，唯一 delta = UDPR 关停）+ `--native-ubr`。
- fast150：150ep、val 每 1ep、**fp32（AMP=0）**、有效 batch 8（本机
  4rank×1×2；历史 2×1×4 同有效 batch——模型无 BN，累积切分只改分组方式，
  数值合同等价）、seed 44、lr 5e-4 / wd 0.05 / warmup 100 iters、PE
  lr_mult 0.0（冻结）、全局梯度 clip、EMA 开但 tracking-only（ema-eval 0）、
  早停 E90 起 patience 20 / min_delta 5e-4 / window 5。
- A 系增强五元组：hflip 0.5 + vflip 0.5 + 多尺度 896–1152（value 模式）p=0.5。
- 选模：best=segm（严格大于，平局留早）；bestB(bbox)/bestC(composite)/last
  同存。四权重齐全后才进 test。

### NWPU（EXP-0047，`vhr10_a3sgm_ubr_matrix300_20261114`）
- 基线脚本：`scripts/ablations/vhr10_a3sgm_matrix300.sh`；config：
  `configs/ablations/vhr10_ubr_matrix300_20261114.py`（`_base_`
  a3sgm_vhr10.py，唯一 delta = UDPR 关停）经 `ABLATION_CONFIG` 挂入 +
  `--native-ubr`。
- matrix300：300ep、val 每 5ep 分片（VAL_SHARD_ACROSS_RANKS=1）、AMP、
  早停关、seed 44、PE lr_mult 0.1（downscaling 可训）、
  `--decoupled-aux-clip`（aux 前缀等价 native_refiner.）、同 A 系增强。
- 选模：best=composite（0.5*bbox+0.5*segm）；best/bestB/last 同存。

### 评测口径（两数据集同）
- WHU：best-segm 权重 → test 2220 图（`infer_whu_checkpoint.sh --split test`）；
  NWPU：best-composite 权重 → val==test 130 图（`infer_nwpu_checkpoint.sh`）。
- COCO 协议 maxDets [1,10,100]、`max_per_img` 100、推理 RPN `nms_pre` 1000、
  无 TTA、FP32 推理——与论文行同口径。

## 预注册四要素

1. **主指标**：WHU = test segm AP@[.50:.95]（best-segm 权重）；NWPU =
   val(=test) segm AP 与 bbox AP 双主指标（best-composite 权重）。
2. **伴随指标**（≥2 维度）：bbox AP、AP50/AP75、mask AP75（UBR 主打边界
   精修的敏感维度）、AR@100；训练遥测（loss/梯度范数/AMP 重试/显存峰值）。
3. **选择口径**：WHU best-segm、NWPU best-composite（均严格大于，平局留早）；
   不得根据 test 更换权重；test 仅终局一次。
4. **显著性方法**：新旧行同口径绝对差对照为主；若旧逐图 records
   （`logs/test_eval/`、旧 checkpoint 推理产物）可用，加逐图配对 bootstrap
   （10k 重采样，报 95% CI）；单 seed（44）局限明示，不声称跨 seed 置信。
5. **各级判定门**：
   - 探针门（拉全量前全过）：指纹 parity（当前代码 build 论文原 config ==
     config 头注指纹 `5d5f78ec…`/`ed7a422d…`）；模块一致性
     （NativeRoIRefiner 参数结构/键名/零初始化与双流侧 S 参考逐一相同，
     6,625 参数）；零初始化恒等（安装前后同 batch 前向逐位一致，两坐标
     契约分支各测）；4-rank smoke（真实 batch 2 update + 1 val、梯度有限
     且 native_refiner 收梯度、WHU fp32 / NWPU AMP、显存实测）；
     checkpoint→推理 rebuild→严格加载→2 图推理 round-trip；独立子代理
     实施+接线审计 VERDICT: PASS（落盘 docs/reviews/）。
   - 全量门：training_complete（epoch/updates 完整）+ 四权重齐全 + 论文
     口径 test 指标落盘。性能数字如实报告，**不设涨分门槛**；若新行回退，
     如实报告并归因，不做口径挑选。

## 旧行基线（对照）

- WHU Ours（UDPR）：test bbox **0.755** / segm **0.729**（2220 图，
  maxDets=100；论文行见 `docs/WHU1024_EXPERIMENT_TRACKING.md`）。
- NWPU Ours（UDPR，best_composite_model_epoch270）：数字 Phase 4 从旧评测
  工件/论文表回填本档。

## 登记与工件

- EXP-ID：**EXP-0046（WHU）/ EXP-0047（NWPU）**——全局下一个空位（双流
  登记册最大 EXP-0045；已查本地与 origin 均无 exp/EXP-0046、exp/EXP-0047）。
- 工件：本机 environment.local.sh 默认
  `/data1/wangcheng/checkpoint/portable_sam2_explicit_coarse/ablations/`
  `<RUN_TAG>_tr1.0_va1.0`（数据盘，余 254G 足够两个 run，不进 git）；结论 MD
  回填本档 + `docs/EXP_REGISTRY.md` 登记；分支纪律按
  `docs/machine_sync_protocol.md`（machine2/fast-whu150 共用分支，小步提交，
  push 前 fetch/rebase）。

## 风险预案

- NWPU full_image UBR 变体结构不可行（shim 语义与全图解码契约冲突）→ 停下
  向用户汇报方案选项，**不擅自**改回 roi_local（坐标契约已裁定走论文）。
- 指纹 parity 不符 → 先考古漂移来源（可能是共享代码静默演进），汇报后再继续。
- WHU fp32 1024 显存/时长未知 → smoke 实测后定与双流侧在跑臂的共租或排队。

## 中途工程观察与后续调整记录（2026-11-15，用户指示）

- **EXP-0046（WHU）保持 rank0-only 验证跑完**：实测验证 222–232 秒/epoch
  （627 图），占 epoch 墙钟 ~13.5%，全程 ~9.6h；为零风险与运行出处最简
  （无中断、无未 smoke 的组合路径）接受该开销，不中途切换。
- **分卡验证（VAL_SHARD_ACROSS_RANKS=1，分片全量精确重组）为工程等价
  优化，后续 WHU 家族 run 一律继承**：EXP-0047（NWPU）matrix300 合同
  自带；未来任何 fast 家族重跑/短调度家族 wrapper 必须显式带上。
  该项与判定口径无关（指标重组一致，训练器对截断分片有明确拒绝保护）。
- E17 参考轨迹：val segm 0.6836（E15 best）/ bbox 0.7558（E17 best），
  健康度正常；正式判读仍按预注册以 test 为准。

## 决策修订（2026-11-15，用户指示）：里程碑触发的一揽子加速方案

- **撤销**前节"分卡验证保持 rank0-only 跑完"的决策。修订为：**当 EXP-0046
  val segm/mAP 首次 ≥ 0.6948**（旧 UDPR 论文 run 的 ep98 val best，监视器
  已挂）时，在会话汇报里程碑，经用户确认具体参数后**一并实施**：
  1. 分卡验证（VAL_SHARD_ACROSS_RANKS=1，epoch 边界干净停 + RESUME_OWN
     续训）；
  2. 砍 epoch 预算（重发时 --epochs 改小；目标值届时按完整曲线的平台
     证据与用户共同拍定）。
- **可比性口径如实登记**：中途改 --epochs 后，余弦调度将按新总步数重锚，
  本 run 成为"里程碑截断"混合调度家族——与旧 150ep 家族不同表直比，论文
  正式对比行如需 150ep 家族口径，以本 run 截断前的轨迹 +（如必要）后续
  独立短调度家族 run 补足；截断合同本身作为加速产物单独登记。
- 触发时执行序：CPU resume smoke（resume×UBR×分卡组合预验）→ epoch 边界
  停 → 带参重发 → 首 epoch 指标连续性与分片重组正确性核验 → 回填本档。
- NWPU（EXP-0047）是否同样砍预算，届时按其自身发射与曲线单独决策。

## 里程碑执行记录（2026-11-15）

- **E22 达标**：val segm/mAP=0.7001 ≥ 0.6948（旧 UDPR 论文 run ep98 val
  best；曲线 E12-21 平台 0.66-0.685 后 E22 跳升）。best_bbox 0.7567(E21)。
- **用户拍定**：不 resume、不截断——EXP-0046 于 E22 停止（best E22 权重
  完整落盘，保存回执在案）；**从头重训干净的 fast60 家族（EXP-0048）**：
  60ep 余弦全程退火 + 分卡验证，其余合同与 EXP-0046 逐参一致；fast150 与
  fast60 为不同调度家族不混表；E22 截断点的 test 数字作为独立数据点登记。
- E22 best 权重 test 推理（2220 图，论文口径 maxDets 100）已启动，
  完成后回填本节并发射 EXP-0048。
