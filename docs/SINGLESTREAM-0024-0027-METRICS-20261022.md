# 新纪元单流四臂全指标台账（EXP-0024/0025/0026/0027，2026-10-22）

便于后续追踪的总账：每臂的训练/验证指标、测试推理指标（标准 segm/bbox
+ boundary 双口径全字段）、权重与文件指向。**协议演变警告**：跨臂对比
必须同口径（见 §5）；v1 与 v3 是不同划分，数值不互通。

## 1. 四臂总览

| | EXP-0024 | EXP-0025 | EXP-0026 | EXP-0027 |
| --- | --- | --- | --- | --- |
| 数据 | v1（459 场 val / 459 test） | v1 | v3（285 val / 413 test） | v3 |
| 增强 | 全关 | hflip 0.5 | 全关 | hflip 0.5 |
| 模型/协议 | `ue-whu-a3sgm-512-v1`，seed 44，DDP4，ep100 | 同左 | 同左（batch 2×累积1） | 同左 |
| workdir | `/data2/wangcheng/checkpoint/citysample_v1/citysample_v1_512single_noaug_ddp4_s44_EXP-0024/` | `.../citysample_v1/citysample_v1_512single_defaug_ddp4_s44_EXP-0025/` | `/data2/wangcheng/checkpoint/citysample_v3/citysample_v3_512single_noaug_ddp4_s44_EXP-0026/` | `.../citysample_v3/citysample_v3_512single_defaug_ddp4_s44_EXP-0027/` |
| 训练日志 | `.../citysample_v1/launcher_logs/EXP-0024-ddp4.log` | `EXP-0025-ddp4.log` | `.../citysample_v3/launcher_logs/EXP-0026-ddp4.log` | `EXP-0027-ddp4.log` |
| 启动器 | `portable_sam2_explicit_coarse/scripts/ue/run_exp0024_ddp4.sh` | `run_exp0025_ddp4.sh` | `run_exp0026_ddp4.sh`（见 0036d72） | `run_exp0027_ddp4.sh` |

## 2. 训练/验证指标

**训练器内部 val（逐 epoch COCO，maxDets=100；v1 val 上限不生效，v3 val
部分密集场 >100 故 v3 的内部读数偏低，见 §5）**

| | ep100 last: segm / bbox | 训练期 best: segm / bbox（epoch） |
| --- | --- | --- |
| EXP-0024 | 0.8374 / 0.8184 | 0.8517 / 0.8386 |
| EXP-0025 | 0.8615 / 0.8455 | **0.8690 / 0.8517** |
| EXP-0026 | 0.6988 / 0.7320 | 0.7051 / 0.7368 |
| EXP-0027 | 0.7096 / 0.7439 | 0.7112 / 0.7449 |

**同协议独立 val 读数（cap250+md250、通道序已修正、best 权重；仅 v3 对
做过——v3 val 密集场上限生效，此为可比口径）**：
`probes/CITYSAMPLE-TEST-PEEK-20261021/exp002{6,7}_v3best_VALcap250_val.json`

| | val segm AP（AP50/AP75, s/m/l） | val bbox AP |
| --- | --- | --- |
| EXP-0026 (ep74) | 0.7645（.9401/.8981, .7042/.8488/.9295） | 0.7996 |
| EXP-0027 (ep83) | 0.7732（.9487/.9084, .7119/.8546/.9264） | 0.8107 |

（v1 对无同协议 val 读数：v1 val ≤87 实例/场，md100=md250，训练器内部
best 即同口径。）

## 3. 测试推理指标（全部 cap250+maxDets=250、通道序已修正、best 权重）

### 标准指标（segm / bbox）
`probes/CITYSAMPLE-TEST-PEEK-20261021/exp002{4,5}_best_ep{68,72}_cap250_bgrfix_test.json`、`exp002{6,7}_v3best_cap250_bgrfix_test.json`

| | segm AP | AP50 | AP75 | AP_s | AP_m | AP_l | bbox AP | AP50 | AP75 | AP_s | AP_m | AP_l |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0025 v1 (ep71) | 0.7083 | 0.9138 | 0.8484 | 0.6710 | 0.8252 | 0.9206 | 0.7581 | 0.9268 | 0.8851 | 0.7317 | 0.8367 | 0.9299 |
| 0024 v1 (ep67) | 0.6596 | 0.8836 | 0.7747 | 0.5961 | 0.8476 | 0.9018 | 0.6969 | 0.9071 | 0.8015 | 0.6402 | 0.8519 | 0.9232 |
| 0027 v3 (ep83) | 0.8102 | 0.9580 | 0.9234 | 0.7170 | 0.9010 | 0.9426 | 0.8103 | 0.9583 | 0.9236 | 0.7411 | 0.8792 | 0.9463 |
| 0026 v3 (ep74) | 0.8023 | 0.9569 | 0.9199 | 0.7104 | 0.8921 | 0.9329 | 0.8002 | 0.9572 | 0.9203 | 0.7307 | 0.8691 | 0.9348 |

### Boundary AP（2%=14px 主口径 / 1%=7px 严格口径；bbox 无 boundary 口径）
`probes/CITYSAMPLE-BOUNDARY-AP-20261021/exp00{24,25,26,27}_*_test.json`（四文件均含 standard_segm 锚，与上表逐位一致）

| 14px | B-AP | B-AP50 | B-AP75 | B-AP_s | B-AP_m | B-AP_l |
| --- | --- | --- | --- | --- | --- | --- |
| 0025 v1 | 0.6985 | 0.9135 | 0.8468 | 0.6710 | 0.7977 | 0.7439 |
| 0024 v1 | 0.6471 | 0.8830 | 0.7722 | 0.5961 | 0.8126 | 0.7006 |
| 0027 v3 | 0.7743 | 0.9574 | 0.9203 | 0.7170 | 0.8394 | 0.7594 |
| 0026 v3 | 0.7627 | 0.9558 | 0.9067 | 0.7104 | 0.8258 | 0.7408 |

| 7px | B-AP | B-AP50 | B-AP75 | B-AP_s | B-AP_m | B-AP_l |
| --- | --- | --- | --- | --- | --- | --- |
| 0025 v1 | 0.6291 | 0.8977 | 0.8070 | 0.6381 | 0.6679 | 0.5316 |
| 0024 v1 | 0.5604 | 0.8547 | 0.6994 | 0.5546 | 0.6547 | 0.4564 |
| 0027 v3 | 0.6524 | 0.9548 | 0.8161 | 0.6659 | 0.6697 | 0.5401 |
| 0026 v3 | 0.6360 | 0.9425 | 0.7877 | 0.6595 | 0.6548 | 0.5006 |

stdout 全量日志：同目录 `exp00{24,25,26,27}_*_boundary.stdout.log`。

## 4. 权重指向

| 权重 | 路径（相对 workdir §1） | epoch | 被哪些读数使用 |
| --- | --- | --- | --- |
| 0025 best | `..._EXP-0025/best_model.pth`（= `best_model_epoch72.pth`） | 71 | §3 全部 0025 读数 |
| 0024 best | `..._EXP-0024/best_model.pth`（= `best_model_epoch68.pth`） | 67 | §3 全部 0024 读数 |
| 0026 best | `..._EXP-0026/best_model.pth`（= `best_model_epoch75.pth`） | 74 | §2 val 读数、§3 全部 0026 读数 |
| 0027 best | `..._EXP-0027/best_model.pth`（= `best_model_epoch84.pth`） | 83 | §2 val 读数、§3 全部 0027 读数 |
| 各臂 last | 各 workdir `last_model_epoch100.pth` / `last_checkpoint.pth` | 100 | 预注册收线口径；**尚无 test 读数** |
| 历史冻结副本（0024/25） | `best_frozen_20261021b.pth`（44/42）、`best_frozen_20261021c.pth`（0025 ep57） | — | 仅 2026-10-21 中期 peek/漂移记录使用，已被 best 读数取代 |

注意：**§2/§3 的全部现存 test 读数用的是 best-val 权重**；预注册
（`docs/EXP-0024-0025_preregistration.md`、`EXP-0026-0027_preregistration.md`）
的正式基线选择口径是 **last ep100**——last 权重的 test 收口是独立开封
事件，尚未发生。

## 5. 口径注意事项（跨表比较前必读）

1. **通道序修复（2026-10-22，commit 645aa40）**：此前所有独立 test 读数
   被低估 ~11 点（PIL RGB 入 bgr_to_rgb 预处理器）。本台账全部为修复后
   读数；修复前历史数字见 `docs/VALTEST-DRIFT-20261021.md`（含对照）。
2. **cap250+md250 规约（commit 187c29e）**：41% v1-test 场景 GT>100 实例；
   修复值 +5.5 segm。v3 的 val 也有密集场 >100，故 v3 内部 md100 读数
   （§2 上表）低于同协议 val 读数（§2 下表），两者不可混用。
3. **v1 与 v3 数值不互通**：v3 test 已重平衡（60 inst/场 vs v1 98）；
   v1 对按预注册降级为历史参照，新纪元基线 = v3 对。
4. **boundary 口径**：2%=14px 主 / 1%=7px 严格（对角线比例），边界带 =
   mask XOR 3×3×d 腐蚀、1px 零 padding、GT 面积保持原值；小物体边界带
   ≈ 整 mask（B-AP_s ≈ AP_s 为协议行为）。
5. **增强对照（test，best 口径）**：v1 hflip +4.9 segm/+6.1 bbox；
   v3 +0.8/+1.0；boundary 14px v1 +5.1 / v3 +1.2。

## 6. 工具与关联文档

- 评估工具：`scripts/ue/eval_citysample_test.py`（标准，`--max-per-img`、
  `--data-root`）、`scripts/ue/eval_citysample_boundary.py`（boundary 双口径）
- 预注册：`docs/EXP-0024-0025_preregistration.md`（含 cap 协议补充节）、
  `docs/EXP-0026-0027_preregistration.md`
- 漂移事件与修复：`docs/VALTEST-DRIFT-20261021.md`、commit `645aa40`、
  收线 `ba2506d`
- 本台账生成提交：2026-10-22（boundary 四臂读数 e2e5dd1 及本文件）
