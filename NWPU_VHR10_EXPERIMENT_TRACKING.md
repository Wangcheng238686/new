# NWPU VHR-10 对比实验追踪表

> **口径声明**：本表所有数字仅来自我们自己的重跑（日志/指标文件见各节指向），不引用 tex 或文献数值。
> 数据：`/data1/wangcheng/dataset/NWPU VHR-10 dataset`（路径含空格，shell 引用必须整体加引号）
> 标注：`coco_split/annotations.json`（650 图 / 3,921 实例 / 10 类 polygon，chaozhong2010/VHR-10_dataset_coco 人工补标，MIT）
> 划分（RSPrompter 官方 80/20）：train **520 全量**（`coco_split/NWPU_instances_train.json`，3,178 实例）/ val **130 兼任 test**（`coco_split/NWPU_instances_val.json`，743 实例，**无独立 test 集，val 即终评**）
> 图片目录：train/val/test 共用 `positive image set/`（无独立子目录）；`negative image set/`、`ground truth/`（bbox）实例分割不用
> 评测：COCO 标准协议 maxDets=100（每图实例 max 71，无需调大），IoU 0.50:0.95；无 TTA
> 类别（category_id 1–10）：airplane 757 / ship 302 / storage_tank 662 / baseball_diamond 391 / tennis_court 524 / basketball_court 159 / ground_track_field 163 / harbor 239 / bridge 124 / vehicle 604（全量）
> 超参原则：各方法按其原论文/官方仓配方，单卡适配仅限 batch×梯度累积与线性折算；**训练强度与 WHU-1024 版同构**（同一方法两数据集配方不变）
> 环境：cvt2（mmengine 系）、dt2（AdelaiDet）、mdino（MaskDINO）、mm2（RSIISN, mmdet2 栈）
> 日志根：`portable_sam2_explicit_coarse/logs/baselines/`；ckpt 根：`/data1/wangcheng/checkpoint/nwpu_vhr10_baselines/`
>
> **当前状态：前期准备已完成，全部等待 GPU 释放后 smoke → train → test。最后更新：2026-09-01**

## 总览

| 方法 | 状态 | 脚本 | 参数量（CPU 实测） | val(=test) bbox mAP | val(=test) segm mAP |
|---|---|---|---|---|---|
| **Ours (PromptMiner-SAM2)** | 📍 本机不跑（在别处训练/评测；本机 GPU3 上跑到 ep20 已停，遗留 ckpt 在 `checkpoint/portable_sam2_explicit_coarse/nwpu10/points_box_dense_p2br0/`，其中 ep17 best segm 0.5081 / ep20 best bbox 0.5997 可作参考，勿用于论文） | — | — | — | — |
| Mask R-CNN | ✅ test 已出 | `run_mmdet_nwpu.sh maskrcnn` | 44.0M | 0.501 | 0.491 |
| MS R-CNN | ✅ test 已出 | `run_mmdet_nwpu.sh msrcnn` | 60.3M | 0.515 | 0.527 |
| HTC (w/o sem) | ✅ test 已出 | `run_mmdet_nwpu.sh htc` | 77.2M | 0.589 | 0.513 |
| SCNet | ⏳ 未入队(WHU 曾三败) | `run_mmdet_nwpu.sh scnet` | 94.5M | — | — |
| Mask2Former | ✅ test 已出(GPU2) | `run_mmdet_nwpu.sh mask2former` | 44.0M | 0.603 | 0.615 |
| RTMDet-Ins-S | ✅ test 已出 | `run_mmdet_nwpu.sh rtmdet` | 10.2M | 0.596 | 0.463 |
| YOLO11s-seg | ✅ test 已出 | `run_yolo11_nwpu.sh` | 10.1M | 0.710 | 0.639 |
| CATNet | ✅ test 已出(GPU2) | `run_catnet_nwpu.sh` | 54.7M | 0.660 | 0.642 |
| CondInst | ✅ test 已出 | `run_adelai_nwpu.sh condinst` | — | 0.650 | 0.628 |
| SOLOv2 | ✅ test 已出 | `run_adelai_nwpu.sh solov2` | — | — (无框输出) | 0.518 |
| MaskDINO (R50) | ✅ test 已出 | `run_maskdino_nwpu.sh` | — | 0.657 | **0.658** |
| RSPrompter (anchor) | 🔄 bf16 重训中(09-06 起) | `run_rsprompter_nwpu.sh` | — | — | — |
| RS4D-prompt (anchor/点集) | 🔄 GPU2 在训(09-12 起, 作者默认协议 800ep/LSJ-1024) | `run_rs4dprompt_nwpu.sh` | — | — | — |
| RS4D-box | ✅ test 已出 | `run_rs4d_nwpu.sh` | — | 0.570 | 0.543 |
| RSIISN (Swin-T Cascade) | ✅ test 已出 | `run_rsiisn_nwpu.sh` | 95.9M | 0.581 | 0.629 |

> 执行队列：`scripts/baselines/queue_nwpu_gpu3_v2.sh`（GPU3 顺序队列；成功判定按产物校验而非脚本返回码——v1 队列曾因 run 脚本失败仍 rc=0 产生假完成标记，catnet OOM/condinst+solov2 np.bool 崩溃均由此漏检，已修正）。
> dt2 detectron2（/tmp/detectron2-0.6）np.bool→bool 已 patch（numpy 1.26 兼容），仅影响 NWPU 后续 condinst/solov2 训练，WHU 已完成不受影响。

## 已完成方法的完整终评指标（val=130 图 = test，COCO 协议 maxDets=100）

**榜单（截至 2026-09-06，12/14 有效完成；segm 降序）**

| 排名 | 方法 | bbox mAP | segm mAP |
|---|---|---|---|
| 1 | MaskDINO | 0.657 | **0.658** |
| 2 | CATNet | 0.660 | 0.642 |
| 3 | YOLO11s-seg | **0.710** | 0.639 |
| 4 | RSIISN | 0.581 | 0.629 |
| 5 | CondInst | 0.650 | 0.628 |
| 6 | Mask2Former | 0.603 | 0.615 |
| 7 | RS4D-box | 0.570 | 0.543 |
| 8 | MS R-CNN | 0.515 | 0.527 |
| 9 | SOLOv2 | — (无框输出) | 0.518 |
| 10 | HTC | 0.589 | 0.513 |
| 11 | Mask R-CNN | 0.501 | 0.491 |
| 12 | RTMDet-Ins-S | 0.596 | 0.463 |

待完成：RSPrompter（bf16 重训中，预计 09-07 出结果）、SCNet（未入队，WHU 三败前科）。
格局与 WHU 一致：迭代制/query 系（MaskDINO/M2F）与强预训练轻量模型（YOLO）在 segm 上占优；epoch 制两阶段系（Mask R-CNN/HTC）因 520 张小数据欠训练明显偏弱（1x=12ep 仅约 3300 迭代）。

**Mask R-CNN**（best ep12，test 日志 `maskrcnn_r50_fpn_1x_nwpu_bs8ai2_test_20260903_111624`）

| | mAP | AP50 | AP75 | APs | APm | APl |
|---|---|---|---|---|---|---|
| bbox | 0.501 | 0.821 | 0.577 | 0.532 | 0.523 | 0.374 |
| segm | 0.491 | 0.817 | 0.505 | 0.369 | 0.470 | 0.508 |

**MS R-CNN**（best ep11，test 日志 `msrcnn_..._test_20260903_114401`）

| | mAP | AP50 | AP75 | APs | APm | APl |
|---|---|---|---|---|---|---|
| bbox | 0.515 | 0.852 | 0.592 | 0.557 | 0.539 | 0.372 |
| segm | 0.527 | 0.826 | 0.568 | 0.455 | 0.524 | 0.517 |

**HTC w/o sem**（best ep12，test 日志 `htc_..._test_20260903_122435`）

| | mAP | AP50 | AP75 | APs | APm | APl |
|---|---|---|---|---|---|---|
| bbox | 0.589 | 0.869 | 0.677 | 0.538 | 0.593 | 0.518 |
| segm | 0.513 | 0.846 | 0.533 | 0.361 | 0.490 | 0.604 |

**RTMDet-Ins-S**（best ep292，test 日志 `rtmdet_..._test_20260903_153049`）

| | mAP | AP50 | AP75 | APs | APm | APl |
|---|---|---|---|---|---|---|
| bbox | 0.596 | 0.915 | 0.706 | 0.617 | 0.580 | 0.586 |
| segm | 0.463 | 0.792 | 0.472 | 0.317 | 0.445 | 0.624 |

**YOLO11s-seg**（test_out `test_metrics.json`，`yolo11s_seg_nwpu_default/yolo11s_seg_nwpu_default/test_out/`）

| | mAP | AP50 | AP75 | APs | APm | APl |
|---|---|---|---|---|---|---|
| bbox | 0.710 | 0.944 | 0.795 | 0.717 | 0.685 | 0.745 |
| segm | 0.639 | 0.939 | 0.673 | 0.528 | 0.606 | 0.746 |


**CATNet**（best ckpt，test 日志 `catnet_nwpu_r50_3x_bs4ai2_test_20260905_010648`，GPU2 独占重训）

| | mAP | AP50 | AP75 | APs | APm | APl |
|---|---|---|---|---|---|---|
| bbox | 0.660 | 0.927 | 0.777 | 0.684 | 0.643 | 0.646 |
| segm | 0.642 | 0.920 | 0.677 | 0.510 | 0.622 | 0.748 |

**Mask2Former**（test 日志 `mask2former_r50_nwpu_bs4ai4_300ep_test_20260905_103758`，GPU2）

| | mAP | AP50 | AP75 | APs | APm | APl |
|---|---|---|---|---|---|---|
| bbox | 0.603 | 0.779 | 0.668 | 0.632 | 0.590 | 0.604 |
| segm | 0.615 | 0.849 | 0.678 | 0.528 | 0.600 | 0.726 |

**MaskDINO**（best-by-val ckpt model_0075399，`test_out/test_metrics.json`；注意 run 脚本 test 需显式传 ckpt 且 PYTHONPATH=MaskDINO 仓库）

| | mAP | AP50 | AP75 | APs | APm | APl |
|---|---|---|---|---|---|---|
| bbox | 0.657 | 0.889 | 0.742 | 0.620 | 0.630 | 0.713 |
| segm | 0.658 | 0.896 | 0.715 | 0.519 | 0.633 | 0.772 |

**MaskDINO 逐类 segm AP**（harbor 0.492 / bridge 0.381 / vehicle 0.455 / airplane 0.498 为难类；ground_track_field 0.976 / baseball_diamond 0.880 / basketball_court 0.864 / storage_tank 0.825 为易类——与类别几何复杂度一致，小样本类波动大，论文引用 per-class 时注明 val 仅 30–195 实例/类）：
airplane 0.498, ship 0.614, storage_tank 0.825, baseball_diamond 0.880, tennis_court 0.598, basketball_court 0.864, ground_track_field 0.976, harbor 0.492, bridge 0.381, vehicle 0.455

**CondInst**（model_final，test 日志 `condinst_MS_R50_1x_nwpu_bs8lr05m_test_20260904_015904`）

| | mAP | AP50 | AP75 | APs | APm | APl |
|---|---|---|---|---|---|---|
| bbox | 0.650 | 0.901 | 0.796 | 0.680 | 0.634 | 0.608 |
| segm | 0.628 | 0.907 | 0.681 | 0.516 | 0.606 | 0.703 |

**RS4D-box**（best ep416，test 日志 `rs4d_bbox_nwpu_bs8_bf16_800ep_test_*`；segm APs 0.097 异常低——SAM 蒸馏原型对小目标掩码质量差，bbox APs 0.222 亦弱，但 APl 0.730 强，尺度两极分化明显）

| | mAP | AP50 | AP75 | APs | APm | APl |
|---|---|---|---|---|---|---|
| bbox | 0.570 | 0.825 | 0.654 | 0.222 | 0.514 | 0.727 |
| segm | 0.543 | 0.801 | 0.576 | 0.097 | 0.482 | 0.730 |

**RSIISN**（best_segm_mAP ep12，test 日志 `rsiisn_cascade_swinT_nwpu_bs2ai4_12ep_test_20260906_114532`；队列曾误报训练失败——其 best ckpt 命名为 best_segm_mAP_*.pth 不在队列 glob 内，实际训练+终评成功）

| | mAP | AP50 | AP75 | APs | APm | APl |
|---|---|---|---|---|---|---|
| bbox | 0.581 | 0.893 | 0.668 | 0.607 | 0.571 | 0.523 |
| segm | 0.629 | 0.897 | 0.671 | 0.489 | 0.611 | 0.689 |

**RSPrompter**：❌ 首轮失败已定位并修复重训。
- 诊断：ep3–12 val segm 实际在缓慢学习(0.000→0.032)，非多类适配 bug；ep33 `loss_mask` 在 **AMP fp16** 下溢出变 NaN 毒化权重（作者 models.py:669-675 有 "to avoid nan in fp16" 注释，此路径本就脆弱），此后 val 冻结在 segm 0.002/bbox 0.311。WHU 跑 fp16 属侥幸未触发。
- 修复：config `AmpOptimWrapper dtype float16→bfloat16`（4090 原生支持），毒化 ckpt 归档至 `rsprompter/nan_fp16_discarded_20260906/`，09-06 bf16 重训启动（GPU3，~20-24h）。

**SOLOv2**（model_final，test 日志 `solov2_R50_1x_nwpu_bs8lr05m_test_20260904_*`；无框输出，与 WHU 行为一致）

| | mAP | AP50 | AP75 | APs | APm | APl |
|---|---|---|---|---|---|---|
| segm | 0.518 | 0.763 | 0.551 | 0.376 | 0.488 | 0.587 |

## 前期准备完成清单（2026-08-31 ~ 09-01）

1. **数据核验**：官方原图 650+150 完整；COCO 掩码版 3,921 实例全部 polygon、iscrowd=0、零越界、md5 与数据集 README 一致；train/val 零重叠、并集=650。
2. **14 个基线配置 + 脚本**（见总览表）；全部通过：配置解析（对应环境）→ dataloader 实际加载（520/130 图零缺图）→ `bash -n` → **CPU 构建模型**（mmdet 系 6 个 + CATNet + RSIISN，参数量见总览表，num_classes=10 接线正确）。
3. **YOLO 标签转换**：`coco2yolo_seg_nwpu.py` 已执行（train 520/3178 行、val 130/743 行）；80 图 445 实例 IoU 抽样验证 mean=1.0000；无 stale labels.cache。
4. **SCNet 语义图**：`gen_nwpu_semantic.py` 已生成 520 张 → `derived_sem/train/`（前景=1 / 背景=255-ignore，与类别数无关）。
5. **各方法输入协议**：检测头系按官方多尺度（Resize(1333,800) 等）；Mask2Former/MaskDINO/RSPrompter/RS4D 为 LSJ-1024（官方 COCO 配方 / SAM 编码器原生 1024）——与 WHU 版同构，非 WHU 数据特性移植。
6. **detectron2 系注册**：AdelaiDet 新增 `adet/data/nwpu_builtin.py`（唯一共享源码改动：`builtin.py` 追加一行 import，不影响 WHU）；MaskDINO 新增 `train_nwpu.py`；id 映射 {1..10: 0..9} 已验证。
7. **Ours NWPU 多类适配**（2026-09-01 完成）：`configs/nwpu10_baseplus_{clean,explicit_coarse}.py`（bbox_head 10 类，SAM2 mask head 类无关零结构差异，全部环境开关保留）；训练入口 `run_nwpu10_explicit_coarse.sh` + wrapper `ablations/nwpu10_pafpn_coarse_points_box_dense.sh`（经 `DATASET_SINGLE_CLASS=0` + `DATASET_CATEGORY_MAPPING=1:0,...,10:9` 接入）；终评 `infer_nwpu_checkpoint.sh`（val 兼任 test）。trainer/eval/inference 的多类改动全部为 opt-in（默认单类路径逐字节保持旧行为），WHU 入口 DRY_RUN 回归通过、数据加载实测 520/3178+130/743、label 0–9 十类齐全。
   - ⚠️ 训练首跑后检查日志 label 分布（漏 export mapping 会 label=10 越界报错，wrapper 已统一导出，手工调 `_run_ablation.sh` 需自带）；本机实际加载 `environment.local.sh`（已加 `VHR10_DATA_ROOT`，换机器同步 `environment2.sh`）。
   - 注意 NWPU 图片长宽比不一，dataset 按 WHU 协议 keep_ratio=False 拉到 1024²（有长宽比畸变，协议选择非 bug）。

## 待办队列（当前状态 2026-09-05）
- ✅ 已完成 10/14（见上）；GPU2 队列（CATNet+Mask2Former）已全部跑完并空闲
- ✅ RSIISN 完成(09-06 11:46)；❌ RSPrompter 失败(loss_mask NaN)待 debug 重训；SCNet 未入队
- 未入队：SCNet（WHU 三败前科，如需单独启动）
- 备选补强（未执行）：1x 家族（Mask R-CNN/MS R-CNN/HTC）× 3 延长训练，排除小数据欠训练解释
## 附录：数据统计（本机实测）

- 图像尺寸：width 533–1728（中位 940），height 358–1028（中位 582）；644 种不同尺寸；75% 图像两边均 <1024
- 实例尺度：sqrt(area) 中位 42px、最小 12px——小目标占比高，APs 指标有参考价值
- 每图实例数：全量 max 71 / mean 6.0
