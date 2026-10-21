# val↔test 数据漂移与指标大幅下降：事件记录（2026-10-21）

**事件**：CitySample 定版数据集（`CitySample_SatUAV_TrainMinimal_v1_20261021`）
上的单流基线双臂（EXP-0024/0025）训练中期，val segm/mAP 达 **0.86**，
而 test 实测仅 **0.52–0.58**——表面落差 ~30 点。经审计分解：**无泄露**；
落差 = val/test 分布错配（主因）+ 评估上限伪影（~5.5 点，已修正）+
干净标签的机械涨幅。本文件为完整留档，含全部日志/权重/结果路径。

## 一、关键数字

### 训练日志中的 val（EXP-0025，trainer 全量 val 459 场/epoch）

来源：`/data2/wangcheng/checkpoint/citysample_v1/launcher_logs/EXP-0025-ddp4.log`

| epoch | val segm/mAP | val bbox/mAP |
| --- | --- | --- |
| 35 | 0.8289 | 0.8079 |
| 39 | 0.8349 | 0.8236 |
| 44 | 0.8456 | 0.8316 |
| 57 | 0.8602 | 0.8485 |

（权重内记录的 best_val：ep39 权重 0.8489、ep44 权重 0.8503、
ep57 权重 0.8610；日志行内 best 字段滞后一拍，以权重为准。）

### test 实测（459 场全量，`eval_citysample_test.py`）

| 权重 | 协议 | segm AP | bbox AP | 结果 JSON |
| --- | --- | --- | --- | --- |
| 0025 ep39 best | cap100+md100 | 0.5349 | 0.5719 | `exp0025_best_test.json` |
| 0024 ep43 best | cap100+md100 | 0.4881 | 0.5255* | `exp0024_best_test.json` |
| 0025 ep44 冻结 | cap100+md100 | 0.5233 | 0.5674 | `exp0025_cap100_test.json` |
| 0025 ep44 冻结 | **cap250+md250** | **0.5788** | **0.6325** | `exp0025_cap250_md250_test.json` |
| 0024 ep42 冻结 | cap250+md250 | 0.5272 | 0.5559 | `exp0024_cap250_md250_test.json` |
| 0025 ep57 冻结 | cap250+md250 | 0.5841 | — | boundary 工具内含 |

*0024 cap100 bbox 见 JSON；boundary AP（ep57）：14px 主口径 0.566、
7px 严格口径 0.469（`CITYSAMPLE-BOUNDARY-AP-20261021/exp0025_ep57_test.json`）。

## 二、归因（详见 `docs/CITYSAMPLE-SPLIT-AUDIT-20261021.md`，commit 03de1fb）

1. **val/test 分布错配（主因）**：val 32.6 实例/场、24% 小实例、面积中位
   2553px；test 98.0 实例/场、**72% 小实例**、面积中位 613px。
2. **评估上限伪影（−5.5 segm，已修正）**：41% test 场景 GT>100 实例
   （最大 238），模型 `max_per_img=100` 与 COCOeval 默认 `maxDets=100`
   双重掐住；同权重单变量消融 +5.5 segm/+6.5 bbox，增量集中小实例。
   规约（commit 187c29e，先于收线登记）：一切收口统一 cap250+md250。
3. **地理隔离干净（无泄露）**：val↔train 最近 450 m > 卫星幅面 360 m，
   test↔train 525 m；全部 3060 点坐标实测。
4. **资产级相似（次因，"相似"非"泄露"）**：47.5% val 实例在 train 有
   cos>0.97 近重复副本；val 内部点距中位 37.5 m（相邻场景 ~90% 重叠，
   459 场有效样本量远小于名义值）。
5. **机械涨幅**：review99_v3 清洗（多连通 20.6%→0%）使旧纪元口径的
   对比失真，部分"变高"是考卷变干净。

## 三、val 选权重不保证 test 更好（实证反例）

EXP-0025：ep39→ep44，val best 0.8489→0.8503（+0.14），同协议
（cap100）test 0.5349→0.5233（**−1.2**）——反转。规约：收线选择保持
预注册的 **last epoch 100** 口径（两臂同规则），best-val 仅作中期预览；
`best_model.pth` = 逐 epoch val segm/mAP argmax（训练器
`train/loops.py` `_save_verified_best_checkpoint`，"segm/mAP" 准则）。

## 四、路径清单

### 训练
- 日志：`/data2/wangcheng/checkpoint/citysample_v1/launcher_logs/EXP-0024-ddp4.log`、`EXP-0025-ddp4.log`
- 工作目录（含逐 epoch 检查点）：`/data2/wangcheng/checkpoint/citysample_v1/citysample_v1_512single_noaug_ddp4_s44_EXP-0024/`、`..._defaug_ddp4_s44_EXP-0025/`
- 启动器：`portable_sam2_explicit_coarse/scripts/ue/run_exp0024_ddp4.sh`、`run_exp0025_ddp4.sh`
- 预注册：`docs/EXP-0024-0025_preregistration.md`（含 2026-10-21 上限修正补充节）

### 推理用到的权重（全部为 EXP-0025/0024 的 best 系，非 last）
- 第一次 peek（移动目标）：当时 `best_model.pth`（0025 ep39 / 0024 ep43），文件已被后续 best 轮转覆盖
- 上限消融冻结副本：`..._EXP-0025/best_frozen_20261021b.pth`（ep44，val 0.8503）、`..._EXP-0024/best_frozen_20261021b.pth`（ep42，val 0.8444）
- 边界评估冻结副本：`..._EXP-0025/best_frozen_20261021c.pth`（ep57，val 0.8610）

### 推理结果（JSON + stdout 日志）
- 目录：`/data2/wangcheng/checkpoint/surfbev_b0/probes/CITYSAMPLE-TEST-PEEK-20261021/`
  - `exp0025_best_test.json` / `exp0024_best_test.json`（cap100 首peek）
  - `exp0025_cap100_test.json`、`exp0025_cap250_test.json`（md100 对照，证明 maxDets 伪影）
  - `exp0025_cap250_md250_test.json`、`exp0024_cap250_md250_test.json`
  - `*.stdout.log`（各次推理完整 stdout，含逐场进度与 COCOeval 打印）
- 边界 AP：`/data2/wangcheng/checkpoint/surfbev_b0/probes/CITYSAMPLE-BOUNDARY-AP-20261021/exp0025_ep57_test.json` + `exp0025_ep57_boundary.stdout.log`
- 工具：`portable_sam2_explicit_coarse/scripts/ue/eval_citysample_test.py`（`--max-per-img`，手工 COCO stats 绕开 pycocotools 非默认 maxDets 的 stats[0]=-1 坑）、`eval_citysample_boundary.py`

### 相关审计文档
- `docs/CITYSAMPLE-SPLIT-AUDIT-20261021.md`（隔离/相似/分布错配审计）
- `docs/DUAL-NEWERA-DIRECTION-20261021.md`（方向交付与旧结论可信度审计）

## 五、时间线（2026-10-21）

- 03:31 EXP-0024/0025 启动（commit 4a653d6）
- 上午 split 审计（03de1fb）→ PI 授权 test 首peek（cap100：0.535/0.488）
- 实例统计发现 41% test 场景>100 → cap/maxDets 消融 → +5.5 segm（187c29e 登记协议）
- val 选择反例实证（ep39 vs ep44）
- 边界 AP 首读（ep57：14px 0.566 / 7px 0.469，7b9b79f）
- 训练进行中（本文件落笔时 ~ep7x/100），收线按预注册 last 口径 + cap250/md250
