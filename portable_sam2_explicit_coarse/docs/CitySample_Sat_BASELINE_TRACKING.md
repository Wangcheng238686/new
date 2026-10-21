# CitySample Sat–UAV 对比实验追踪表（卫星侧基线）

> **口径声明**：本表所有数字仅来自我们自己的重跑（日志/指标文件见各节指向），不引用 tex 或文献数值。
> 数据：`/data1/wangcheng/dataset/CitySample_SatUAV_TrainMinimal_v1_20261021`（双流数据集，**基线只用卫星侧** `points/<id>/satellite/rgb.png`，UAV 五视角不参与）
> 划分（固定 70/15/15 空间隔离划分，**不得重划**）：train **2142** / val **459 选模**（save_best=coco/segm_mAP）/ test **459 仅终评**
> 评测：COCO 标准协议 IoU 0.50:0.95；无 TTA；单 ckpt
> 超参原则：沿用 WHU-1024 版配方（各方法按其原论文/官方仓配方，单卡适配仅限 batch×梯度累积与线性折算），只换数据与协议参数
> 环境：cvt2（mmengine 系）、dt2（detectron2/AdelaiDet 系）、mdino（MaskDINO）、mm2（RSIISN, mmdet2 栈）
> 日志根：`portable_sam2_explicit_coarse/logs/baselines_citysample/`；ckpt 根：`/data1/wangcheng/checkpoint/citysample_sat_baselines/`
>
> **状态：协议已拍板（2026-10-21，见下）；首发 RSPrompter(anchor) 与 YOLO11s-seg 无增强版已立项。最后更新：2026-10-21**

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

### 已拍板（2026-10-21 用户决定）

| 项 | 决定 | 依据 |
|---|---|---|
| 选模 | val 459 上 segm mAP 峰值 ckpt（save_best=coco/segm_mAP），test 仅终评 | WHU/NWPU 先例 |
| 训练配方 | 与 WHU-1024 版同构（同 epoch/iter、同优化器），仅换数据/管线/协议参数 | NWPU 先例 |
| 无 TTA、单 ckpt | 是 | WHU/NWPU 先例 |
| SCNet | 跳过（WHU 三次复现失败史） | WHU tracking 记录 |
| val 选模 maxDets | **保持 100 即无损**（val 单图最大 87 实例 < 100） | 实测上表 |
| **P1/P2 test maxDets** | **250**（test 单图实例最大 238 < 250；推理输出上限全方法同步抬 250） | 用户拍板 2026-10-21 |
| **P3 M2F num_queries** | **250**（届时重训） | 用户拍板 2026-10-21 |
| **P4 分辨率** | **512 原生，不允许上采样**（各方法 resize 一律 identity 512；"Ours 可上采样但暂未上采样"——**后续可能调整**，届时基线同步修订重跑） | 用户拍板 2026-10-21 |
| **P5 方法清单** | 按本表执行（后续可能变更） | 用户拍板 2026-10-21 |
| **P6 GPU** | **四卡预算、可共租**（只要不 OOM，与用户其他任务共存） | 用户拍板 2026-10-21 |
| **无增强对比层** | **首发实验（RSPrompter/YOLO11s-seg）为无数据增强版**：与 Ours 单流卫星无增强版公平对比。无增强=去翻转/尺度抖动/裁剪/mosaic/HSV 等全部训练期几何与色彩增强；COCO 预训练初始化保留（初始化≠增强，WHU 协议一致） | 用户指令 2026-10-21 |

## 方法清单与状态

| 方法 | 状态 | WHU 版 config 起点（移植派生源） | 移植改动点 |
|---|---|---|---|
| **RSPrompter (anchor) — 无增强版** | 🔄 **训练中（E4 起，16:23 resume）**：300ep bf16 bs1×ai2 显存 15.5G，0.67s/iter，eta ~5.3 天。**E3 事故已修**：val 崩于 SAM patch_embed 的 1024 硬校验——根因是 base config 的 BatchFixedSizePad(1024) 仅训练期生效（训练时 512 图零填充进 1024 画布，故 3ep 正常），我覆盖 test 管线时误删了作者的 Pad 步骤；修复=test 管线补 `Pad(1024, 零填充)`（与训练一致，补边非上采样，P4 不破，训练路径零改动 E3 ckpt 直接续）。20 图探针全链路通过（E3 segm mAP 0.009，对照 NWPU 早期轨迹 ep3-12 从 0 爬起，正常）。首个全量 val 在 E6（val_interval=3，与 WHU 版一致）。

**E11 中断（21:35）＝用户手动终止**：此前"挂起死锁"推测作废——用户确认是手动杀的（21:05 起停止推进为其暂停过程），我误判为异常挂起并自动续训，经用户指出后已再次停止（22:00 前）。**当前状态：⏸ 暂停待命**，ckpt 无损（best_coco_segm_mAP_epoch_9.pth=segm 0.5080 + epoch_10.pth），恢复命令 `bash scripts/baselines/run_rsprompter_citysample.sh resume <gpu>` 从 E11 接续，等用户指令 | `RSPrompter-release/.../configs/rsprompter/rsprompter_anchor-whu-1024.py` | 新 config `rsprompter_anchor-citysample-512-noaug.py`：无增强管线（去 RandomFlip+LSJ，identity Resize 512 守卫）、max_per_img=250、bf16、300ep bs1×ai2、AdamW 2e-4+clip0.1 不变；启动 `run_rsprompter_citysample.sh`（smoke/train/resume/test，test 走 `citysample_sat_test.py` maxDets=250） |
| **YOLO11s-seg — 无增强版** | ✅ **完成**（100ep 跑满 exit=0，best.pt 终评 test 459 已收：**bbox mAP 0.6865** / **segm mAP 0.5035**（maxDets=250）；val(mAP50-95 box 0.746/mask 0.675)→test 落差反映 val稀疏大目标 vs test密集小目标的分布漂移；segm AP75 0.522 vs AP50 0.859=原型掩码边界精度短板；指标文件 `yolo11seg/.../test_out/test_metrics.json` + pred_coco.json 已落盘） | 脚本制（`coco2yolo_seg.py` 转换 + ultralytics 默认配方） | `coco2yolo_seg_citysample.py`（唯一 stem=采样点号，全图同名 rgb.png；未压缩 RLE 先 frPyObjects；转换零损耗 126214/14980/44988 与 GT 一致）+ `run_yolo11_citysample.sh`（imgsz=512、mosaic/HSV/flip 等全关、100ep bs16 seed0 auto 优化器）+ `eval_yolo_citysample.py`（max_det=250、maxDets=[1,10,250]） |
| Mask R-CNN | ⏸ 待命令（后续方法默认沿用无增强+512 原生口径，除非另行指令） | `mmdetection/configs/whu1024/mask-rcnn_r50_fpn_1x_whu1024.py` | 数据路径、max_per_img→250、无增强管线 |
| MS R-CNN | ⏸ 待命令 | `mmdetection/configs/whu1024/ms-rcnn_r50-caffe_fpn_1x_whu1024.py` | 同上 |
| HTC (w/o sem) | ⏸ 待命令 | `mmdetection/configs/whu1024/htc-without-semantic_r50_fpn_1x_whu1024.py` | 同上 |
| Mask2Former | ⏸ 待命令 | `mmdetection/configs/whu1024/mask2former_r50_whu1024.py` | 数据、num_queries(P3)、**补 save_best**（WHU 两次踩坑：checkpoint hook 疏漏无 best，只能日志解析） |
| RTMDet-Ins-S | ⏸ 待命令 | `mmdetection/configs/whu1024/rtmdet-ins_s_300e_whu1024.py` | 数据、确认 num_queries≥250（默认 300 合规）、max_per_img |
| CATNet | ⏸ 待命令 | `CATNet/configs/whu/cat_mask_rcnn_r50_3x_whu1024.py` | 数据、max_per_img |
| CondInst | ⏸ 待命令 | `AdelaiDet/configs/WHU/condinst_MS_R_50_1x_whu1024.yaml` | 数据、DETECTIONS_PER_IMAGE→250 |
| SOLOv2 | ⏸ 待命令 | `AdelaiDet/configs/WHU/solov2_R50_1x_whu1024.yaml` | 数据、NMS post→250（无框输出为方法属性） |
| MaskDINO | ⏸ 待命令 | `MaskDINO/configs/whu/maskdino_R50_whu1024.yaml` + `train_whu.py` | 数据（queries 默认 300 ≥250 合规） |
| RS4D-box | ⏸ 待命令 | `RS4D/configs/rsprompter/`（WHU 用作者 config 原样） | 数据 |
| RSIISN | ⏸ 待命令 | `RSIISN/configs/whu1024/rsiisn_cascade_swinT_whu1024.py` | 数据、max_per_img |
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
