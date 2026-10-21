# CitySample Sat–UAV 对比实验追踪表（卫星侧基线）

> **口径声明**：本表所有数字仅来自我们自己的重跑（日志/指标文件见各节指向），不引用 tex 或文献数值。
> 数据：`/data1/wangcheng/dataset/CitySample_SatUAV_TrainMinimal_v1_20261021`（双流数据集，**基线只用卫星侧** `points/<id>/satellite/rgb.png`，UAV 五视角不参与）
> 划分（固定 70/15/15 空间隔离划分，**不得重划**）：train **2142** / val **459 选模**（save_best=coco/segm_mAP）/ test **459 仅终评**
> 评测：COCO 标准协议 IoU 0.50:0.95；无 TTA；单 ckpt
> 超参原则：沿用 WHU-1024 版配方（各方法按其原论文/官方仓配方，单卡适配仅限 batch×梯度累积与线性折算），只换数据与协议参数
> 环境：cvt2（mmengine 系）、dt2（detectron2/AdelaiDet 系）、mdino（MaskDINO）、mm2（RSIISN, mmdet2 栈）
> 日志根：`portable_sam2_explicit_coarse/logs/baselines_citysample/`；ckpt 根：`/data1/wangcheng/checkpoint/citysample_sat_baselines/`
>
> **状态：立项调查完成，协议 PENDING 项待拍板（见"协议决定"节），未启动任何实验。最后更新：2026-10-21**

---

## 数据集画像（2026-10-21 实测，来自 annotations/*.json）

| 项 | train | val | test |
|---|---|---|---|
| 图片数（512×512 卫星 RGB） | 2142 | 459 | 459 |
| 实例数（单类 building, id=1） | 126,214 | 14,980 | 44,988 |
| 实例/图中位数 | 50 | 28 | **89** |
| 实例/图 p95 / max | 131 / 208 | 68 / **87** | 199 / **238** |
| ≥100 实例的图数 | 304 | **0** | **192（42%）** |
| 空图数 | 0 | 0 | 0 |
| 目标尺寸 sqrt(area) 中位 | 31.4px | 50.5px | 24.8px |

标注事实：
- COCO 标准字段齐全（bbox/area/iscrowd=0），`segmentation` 为**未压缩列优先 RLE**（`{size:[512,512], counts:[...]}`），pycocotools 走 `annToRLE→decode` 原生可读；**独立 mask、保留既有重叠**（相邻建筑可交叠，不能由单值 ID 图重建）。
- `image_id` = 采样点编号；`file_name` 形如 `points/000163/satellite/rgb.png`（相对数据集根）。
- 数据集自检（VERIFICATION.json）：3060 点 mask 与官方逐像素一致、副本 SHA256 全验、reader 冒烟通过；**模型侧训练冒烟未做**（本次基线即首次模型接入）。
- 卫星相机为高空透视近似（非正射），对 2D 基线无影响，仅作记录。
- ⚠️ **val 与 test 分布漂移**：val 稀疏（中位 28、最大 87、目标偏大），test 密集（中位 89、最大 238、目标偏小）。选模峰不一定对应 test 峰，属数据集固定划分的客观属性，如实报告。

## 协议决定

### 已定（沿用先例）

| 项 | 决定 | 依据 |
|---|---|---|
| 选模 | val 459 上 segm mAP 峰值 ckpt（save_best=coco/segm_mAP），test 仅终评 | WHU/NWPU 先例 |
| 训练配方 | 与 WHU-1024 版同构（同 epoch/iter、同增强、同优化器），仅换数据/类别数/协议参数 | NWPU 先例（同方法跨数据集配方不变） |
| 无 TTA、单 ckpt | 是 | WHU/NWPU 先例 |
| SCNet | 跳过（WHU 三次复现失败史） | WHU tracking 记录 |
| val 选模 maxDets | **保持 100 即无损**（val 单图最大 87 实例 < 100） | 实测上表 |

### PENDING（待拍板，见 grill 清单）

| # | 项 | 我的建议 | 备选 |
|---|---|---|---|
| P1 | **test maxDets** | **300**（≥test 最大 238；与 MaskDINO 300 queries、YOLO 默认 300 对齐）；test 42% 图 ≥100 实例，保持 100 会把召回物理封顶 | 250 / 500 |
| P2 | **推理输出上限** | 全方法统一抬到 300（mmdet `max_per_img`、d2 `DETECTIONS_PER_IMAGE`、YOLO `max_det`、SOLOv2 NMS post），否则 maxDets 300 无意义 | — |
| P3 | **M2F num_queries** | 100→300 重训（否则输出硬顶 100 实例，test 上结构性吃亏）；偏离官方默认需在论文注明 | 保持 100 并如实报告容量上限 |
| P4 | **分辨率口径** | 维持"各方法按其原配方"：512 原图被各方法上采样（Resize(1333,800)→800²、LSJ/CATNet→1024²、YOLO→640），全上采样无下采样，对小目标有利 | 全体统一 512 原生（同分辨率公平但偏离先例） |
| P5 | **方法清单** | WHU 有效 12 个全上 + RS4D-box；RSPrompter(anchor) 可选补（WHU fp16 NaN / bf16 被 kill，两边均无终版数） | 13 个（含 RSPrompter）或 12 个 |
| P6 | GPU 与排队 | 按用户指定卡数跑串行队列（恢复当年 queue 的产物校验机制，避免假完成标记） | — |

## 方法清单与状态

| 方法 | 状态 | WHU 版 config 起点（移植派生源） | 移植改动点 |
|---|---|---|---|
| Mask R-CNN | ⏸ 待命令 | `mmdetection/configs/whu1024/mask-rcnn_r50_fpn_1x_whu1024.py` | 数据路径、max_per_img→300 |
| MS R-CNN | ⏸ 待命令 | `mmdetection/configs/whu1024/ms-rcnn_r50-caffe_fpn_1x_whu1024.py` | 同上 |
| HTC (w/o sem) | ⏸ 待命令 | `mmdetection/configs/whu1024/htc-without-semantic_r50_fpn_1x_whu1024.py` | 同上 |
| Mask2Former | ⏸ 待命令 | `mmdetection/configs/whu1024/mask2former_r50_whu1024.py` | 数据、num_queries(P3)、**补 save_best**（WHU 两次踩坑：checkpoint hook 疏漏无 best，只能日志解析） |
| RTMDet-Ins-S | ⏸ 待命令 | `mmdetection/configs/whu1024/rtmdet-ins_s_300e_whu1024.py` | 数据、确认 num_queries=300、max_per_img |
| CATNet | ⏸ 待命令 | `CATNet/configs/whu/cat_mask_rcnn_r50_3x_whu1024.py` | 数据、max_per_img |
| CondInst | ⏸ 待命令 | `AdelaiDet/configs/WHU/condinst_MS_R_50_1x_whu1024.yaml` | 数据、DETECTIONS_PER_IMAGE→300 |
| SOLOv2 | ⏸ 待命令 | `AdelaiDet/configs/WHU/solov2_R50_1x_whu1024.yaml` | 数据、NMS post→300（无框输出为方法属性） |
| MaskDINO | ⏸ 待命令 | `MaskDINO/configs/whu/maskdino_R50_whu1024.yaml` + `train_whu.py` | 数据（queries 默认 300 已合规） |
| RS4D-box | ⏸ 待命令 | `RS4D/configs/rsprompter/`（WHU 用作者 config 原样） | 数据 |
| RSIISN | ⏸ 待命令 | `RSIISN/configs/whu1024/rsiisn_cascade_swinT_whu1024.py` | 数据、max_per_img |
| YOLO11s-seg | ⏸ 待命令 | 脚本制（`coco2yolo_seg.py` 转换 + ultralytics 默认配方） | 数据转换（512 图/子目录路径）、max_det→300 |
| RSPrompter (anchor) | ❓PENDING P5 | `RS4D/configs/rsprompter/rsprompter_anchor-whu.py`（bf16 修复版口径） | 数据；上次 300ep 在 E27/E229 被 kill，本次需完整预算 |
| SCNet | ❌ 跳过 | — | WHU 三次复现失败 |

## 启动脚本与评测设施（已从 git 历史恢复）

2026-10-21 从快照 `73c8eab` 恢复 5cf4d58（R12 清理）删除的 47 个基线脚本至 `portable_sam2_explicit_coarse/scripts/baselines/`（磁盘现存图件脚本未覆盖）：

- 分家族 run 脚本：`run_mmdet_whu1024.sh` / `run_catnet_whu1024.sh` / `run_adelai_whu1024.sh` / `run_maskdino_whu1024.sh` / `run_rs4d_whu1024.sh` / `run_rsiisn_whu1024.sh` / `run_yolo11_whu1024.sh` / `run_rsprompter_whu1024.sh`
- 统一推理入口：`infer_whu1024.sh`（自动选 val best ckpt，指标落 `test_out/test_metrics.json`）
- 评测器：`whu_test_mmdet.py`（mmdet 系）、`whu_test_adet.py`（AdelaiDet 系）、`eval_maskdino_whu.py`、`eval_yolo_whu.py`、`eval_rsiisn_pkl.py`
- 转换器：`coco2yolo_seg.py`
- 队列模板：`queue_nwpu_gpu3_v2.sh`（含产物校验而非返回码判定——v1 假完成教训）

CitySample 侧新脚本将按 `_citysample` 后缀另建（WHU 版保持原样不动）；评测器需增加 `maxDets` 参数（现硬编码 100，`eval_yolo_whu.py:51` 的 `max_det=100` 等共 4 处）。

## 风险与注意

1. **maxDets 是本数据集第一协议风险**：test 42% 图 ≥100 实例、最大 238。不抬 maxDets 与推理上限，所有方法的 AP 都被结构性压低（召回封顶）。
2. **M2F/查询类容量**：100 queries 物理上限 100 实例/图（P3 必须拍板）。
3. **重叠实例**：标注保留人工重叠，训练目标生成按各框架默认重叠消解（后写优先级不同框架略异），评测无影响。
4. **训练时长预估**：train 2142 图 ≈ WHU 2943 的 73%，各方法墙钟约按 WHU 经验 ×0.73 折算。
5. **日志/ckpt 新根**：不复用 `logs/baselines/`（WHU/NWPU 史迹保持只读），新开 `logs/baselines_citysample/` 与 `/data1/wangcheng/checkpoint/citysample_sat_baselines/`。
6. 数据集 README 要求：`test 只用于最终评估`；划分不得重划；本归档不授予公开分发许可（论文使用需注意数据授权表述）。
