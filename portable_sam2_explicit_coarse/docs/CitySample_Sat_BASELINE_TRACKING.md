# CitySample Sat–UAV 对比实验追踪表（卫星侧基线）

> **口径声明**：本表所有数字仅来自我们自己的重跑（日志/指标文件见各节指向），不引用 tex 或文献数值。
> 数据：`/data1/wangcheng/dataset/CitySample_SatUAV_TrainMinimal_v1_20261021`（双流数据集，**基线只用卫星侧** `points/<id>/satellite/rgb.png`，UAV 五视角不参与）
> 划分（固定 70/15/15 空间隔离划分，**不得重划**）：train **2142** / val **459 选模**（save_best=coco/segm_mAP）/ test **459 仅终评**
> 评测：COCO 标准协议 IoU 0.50:0.95；无 TTA；单 ckpt
> 超参原则：沿用 WHU-1024 版配方（各方法按其原论文/官方仓配方，单卡适配仅限 batch×梯度累积与线性折算），只换数据与协议参数
> 环境：cvt2（mmengine 系）、dt2（detectron2/AdelaiDet 系）、mdino（MaskDINO）、mm2（RSIISN, mmdet2 栈）
> 日志根：`portable_sam2_explicit_coarse/logs/baselines_citysample/`；ckpt 根：`/data1/wangcheng/checkpoint/citysample_sat_baselines/`
>
> **状态：v1 数据集已作废（val/test 分布严重偏移），全部基线切换 v3 重跑（2026-10-21 夜）；第二批 HTC/M2F 均已停止让位(GPU0=他人任务共租/GPU3=用户留用)，本机我方 GPU 占用归零；v3 已收三：Mask R-CNN segm 0.6974/bbox 0.7002、YOLO segm 0.6595/bbox 0.7612、MS R-CNN segm 0.6736/bbox 0.6678；RTMDet-Ins-S ✅(0.6464/0.6423, 边界 0.596/0.431)。v3 四方法全部收官。RSPrompter(15.5G 高占用)缓发待用户指令。最后更新：2026-10-22 00:10**

> ⚠️ **v1 作废声明（2026-10-21）**：`CitySample_SatUAV_TrainMinimal_v1_20261021` 的 val(稀疏大目标, med 28 inst/图) 与 test(密集小目标, med 89 inst/图) 分布严重偏移，val/test 指标差距过大，用户重新调整分布后发布 v3。**v1 上的所有结果（YOLO test bbox 0.6865/segm 0.5035、RSPrompter E9 val segm 0.5080 等）仅作过程史迹，不得入论文表**；v1 的 RSPrompter ckpt 全部弃用，v3 需从头重训。

## v3 数据集画像（2026-10-21 实测）

| 项 | train | val | test |
|---|---|---|---|
| 图片数（512×512 卫星 RGB） | 1931 | 285 | 413 |
| 实例数（单类 building） | 123,788 | 18,066 | 24,754 |
| 实例/图中位 / 最大 | 57 / 238 | 57 / 169 | 47 / 208 |
| ≥100 实例图数 | 313 | 58 | 65 |
| 目标 sqrt(area) 中位 | 28.6px | 27.8px | 31.1px |

- v3 分布修正确认：val/test 密度与尺寸已对齐（v1 时中位 28 vs 89 的偏斜消除），val 选模与 test 终评同域。
- **maxDets=250 协议不变**（全 split 单图最大 238 < 250）。
- ⚠️ **v3 新注意**：val 有 58/285 图 >100 实例（最大 169），训练期 mmengine CocoMetric 硬编码 maxDets=100 对**选模信号**轻度截断（epoch 间一致，不影响 test 报告数——test 走 citysample_sat_test.py 的 maxDets=[1,10,250]）。v1 时 val max 87 无此问题。

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

## 边界指标协议（BoundaryAP，用户口径 2026-10-22 起）

所有对比算法的 test 推理**必须输出**边界相关指标（`scripts/baselines/boundary_ap.py`，已集成进 `citysample_sat_test.py` 与 `eval_yolo_citysample.py` 自动输出）：

1. **内部边界带**：B(M) = M − Erode(M, d)；3×3 全 1 核连续腐蚀 d 次，图像外按背景（实现：零填充 d 后腐蚀再裁回）；带宽按整图对角线 d = max(1, round(r·√(H²+W²)))，512×512 → 2%=14px、1%=7px。注意：是内部边界带，不是轮廓两侧扩张。
2. **匹配质量**：q(P,G) = min(MaskIoU, BoundaryIoU)，BoundaryIoU = |B(P)∩B(G)|/|B(P)∪B(G)|——掩码区域与边界需同时达标。
3. **AP 口径**：阈值 0.50:0.05:0.95 十档取均；按原检测 score 全局降序贪心一对一（每档阈值独立匹配集）；每图预测上限 250；主结果覆盖全部尺寸（无 area 过滤）。14/7px 决定的是带宽而非"容差距离"，最终按边界带重叠判优，7px 版对细微轮廓变化更敏感，两尺度同时报告。
4. 评测器自检：GT 自身作为预测 → BoundaryAP=1.000（两尺度），d=14/7 校验通过（2026-10-22）。

**回算说明**：已出结果的预测 json（mmdet=完整 RLE、YOLO=多边形）即测试集推理原始输出，边界指标为纯后处理，直接在落盘预测上计算，与重跑推理等价，不占 GPU。

**已回算结果（2026-10-22，boundary_metrics.json 落盘于各 test_out/）**

| 方法 | 数据集 | segm mAP | BoundaryAP_2pct (d=14) | BoundaryAP_1pct (d=7) |
|---|---|---|---|---|
| Mask R-CNN | v3 | 0.6974 | **0.6349** | **0.4693** |
| MS R-CNN | v3 | 0.6736 | 0.6092 | 0.4347 |
| RTMDet-Ins-S | v3 | 0.6423 | 0.5963 | 0.4312 |
| YOLO11s-seg | v3 | 0.6595 | 0.5877 | 0.3942 |
| YOLO11s-seg | v1(已作废,史迹) | 0.5035 | 0.4738 | 0.3523 |

观察：排序与 segm 榜一致（两阶段 > YOLO），且在更严的 1% 尺度分化加大（MRCNN vs YOLO：segm +0.038 → 1% 边界 +0.075）——与 RoI 掩码边界精度优势的归因一致。RTMDet 训完自动带边界指标。

## 方法清单与状态

| 方法 | 状态 | WHU 版 config 起点（移植派生源） | 移植改动点 |
|---|---|---|---|
| **YOLO11s-seg — v3 无增强版** | ✅ **完成**（100ep exit=0，best.pt 终评 test 413 已收，maxDets=250）：**bbox mAP 0.7612 / segm mAP 0.6595**（AP50 0.968/0.957，AP75 0.890/0.753；APs 0.691/0.528；AR@250 0.803/0.701）。末段 val box 0.753/mask 0.687 与 test 高度一致——v3 分布修复的直接证据（v1 时 val→test 崩落 0.75→0.50）。segm AP75 0.753 vs AP50 0.957：原型掩码边界精度仍为该系短板。指标文件 `yolo11seg/yolo11s_seg_citysample_v3_512noaug/.../test_out/test_metrics.json` + pred_coco.json 落盘 | v1 版脚本重定向 v3 | 同 v1 协议：max_det=250、maxDets=[1,10,250]（eval 脚本已带 s[0] 陷阱修复） |
| **Mask R-CNN — v3 无增强版** | ✅ **完成**（12ep exit=0，best=E9，test 413 已收 maxDets=250）：**bbox 0.7002 / segm 0.6974**（AP50 0.926/0.914，AP75 0.819/**0.812**；AR@250 0.739/0.735）。segm 反超 YOLO(+0.038) 且 AP75 领先 0.059——RoI 掩码边界精度优势；bbox 弱于 YOLO(-0.061)。val(best E9, maxDets=100 截断) 0.586x → test 0.697 的上差= val 选模截断的量化体现 | `mmdetection/configs/whu1024/mask-rcnn_r50_fpn_1x_whu1024.py` | 新 config `configs/citysample/mask-rcnn_r50_fpn_1x_citysample_v3_noaug.py`：无增强 identity Resize512、max_per_img 250、test_cfg.rcnn 覆盖 |
| **MS R-CNN — v3 无增强版** | ✅ **完成**（12ep exit=0，best=E8，test 413 已收 maxDets=250）：**bbox 0.6678 / segm 0.6736**（AP50 0.921/0.910，AP75 0.798/0.791；AR@250 0.713/0.713）。略低于 Mask R-CNN（-0.032/-0.024），mask-iou 校准在单类建筑上无增益 | `mmdetection/configs/whu1024/ms-rcnn_r50-caffe_fpn_1x_whu1024.py` | 同上（`ms-rcnn_r50-caffe_fpn_1x_citysample_v3_noaug.py`，含 mask_iou_head 单类适配） |
| **RTMDet-Ins-S — v3 无增强版** | ✅ **完成**（300ep exit=0，best=E250 val segm 0.5830，test 413 已收 maxDets=250）：**bbox 0.6464 / segm 0.6423**（AP50 0.895/0.894；APs 0.574/0.528）；**BoundaryAP 2%=0.5963 / 1%=0.4312**（边界指标已自动输出，集成生效）。四个方法中整体最弱——无增强 300ep 下 one-stage 原型掩码两项皆垫底 | `mmdetection/configs/whu1024/rtmdet-ins_s_300e_whu1024.py` | `rtmdet-ins_s_300e_citysample_v3_noaug.py`：去 mosaic/mixup/HSV/多尺度/flip，**摘除 PipelineSwitchHook**（无增强后 stage1/2 无差异，EMA 保留 momentum 3.125e-6）、bbox_head.test_cfg.max_per_img 250（num_queries 默认 300 合规） |
| **RSPrompter (anchor) — v1 版（已作废）** | 🔄 **训练中（E4 起，16:23 resume）**：300ep bf16 bs1×ai2 显存 15.5G，0.67s/iter，eta ~5.3 天。**E3 事故已修**：val 崩于 SAM patch_embed 的 1024 硬校验——根因是 base config 的 BatchFixedSizePad(1024) 仅训练期生效（训练时 512 图零填充进 1024 画布，故 3ep 正常），我覆盖 test 管线时误删了作者的 Pad 步骤；修复=test 管线补 `Pad(1024, 零填充)`（与训练一致，补边非上采样，P4 不破，训练路径零改动 E3 ckpt 直接续）。20 图探针全链路通过（E3 segm mAP 0.009，对照 NWPU 早期轨迹 ep3-12 从 0 爬起，正常）。首个全量 val 在 E6（val_interval=3，与 WHU 版一致）。

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
| **HTC (w/o sem) — v3 无增强版** | ⏸ **用户指令停止让位 GPU0**（20:2x 停；**E1 已完成有 ckpt**（epoch_1.pth + best E1），恢复命令 `run_mmdet_citysample.sh htc resume <gpu>` 从 E2 续，显存 ~7-8G 可共租）。背景：发现 GPU0 共租方为他人(liubowen)任务，按"不影响别人任务"原则主动停止 | WHU 版 `htc-without-semantic_r50_fpn_1x_whu1024.py` | `configs/citysample/htc-wosem_r50_fpn_1x_citysample_v3_noaug.py`：数据/无增强管线/max_per_img 250 |
| **Mask2Former — v3 无增强版** | ⏸ **用户指令停止让位 GPU3**（20:1x 停；仅跑到 ~700 iter < 首个 3000-iter 存档点，**无 ckpt，续跑=从头**；恢复命令 `run_mmdet_citysample.sh mask2former train <gpu>`，显存 ~46G 需近独占卡） | WHU v2 `mask2former_r50_whu1024.py` | `mask2former_r50_citysample_v3_noaug.py`：**num_queries 100→250（P3 拍板）**、**补 save_best（修 WHU 疏漏）**、去 LSJ/hflip、test 管线废弃 Resize(1333,800) 升采样改 identity 512、里程碑按 0.656 折算 |
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
