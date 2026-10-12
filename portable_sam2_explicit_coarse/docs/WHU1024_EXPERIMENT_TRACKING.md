# WHU-1024 对比实验追踪表

> **口径声明**：本表所有数字仅来自我们自己的重跑（日志/指标文件见各节指向），不引用 tex 或文献数值。
> 数据：`/data1/wangcheng/dataset/WHU`（1024×1024 COCO 实例版，单类 building）
> 划分：train **2943 全量**（=官方 2793 标注图 + 150 无标注背景图，一律不滤空图）/ val **627 选模**（save_best=coco/segm_mAP）/ test **2220 仅终评**
> 评测：COCO 标准协议 maxDets=100，IoU 0.50:0.95；无 TTA；test 推理统一 `infer_whu1024.sh`（CocoMetric/COCOEvaluator）
> 超参原则：各方法按其原论文/官方仓配方，单卡适配仅限 batch×梯度累积与线性折算
> 环境：cvt2（mmengine 系）、dt2（detectron2 系，`/data2/wangcheng/envs/dt2`）
> 日志根：`portable_sam2_explicit_coarse/logs/baselines/`；ckpt 根：`/data1/wangcheng/checkpoint/whu1024_baselines/`
>
> **最后更新：2026-08-28 23:00**

## 总览

| 方法 | 状态 | 训练数据增强（各按其论文/官方配方） | val best (segm/bbox) | test bbox mAP | test segm mAP |
|---|---|---|---|---|---|
| **Ours (PromptMiner-SAM2)** | ✅ paper版已出 | 水平翻转 p=0.5，仅此一项（其余增强全关） | (best ep98) | **0.755** | **0.729** |
| CATNet | ✅ | 三向翻转（水平/垂直/对角线）p=0.75 + Resize(1024,1024)（方形原生图上为 no-op，等价其论文"(1400,800)"设定） | 0.7470 / 0.7800 (ep31) | **0.755** | **0.727** |
| CondInst | ✅ | 多尺度短边 640–800（RandomChoice）+ 水平翻转 p=0.5（detectron2 官方 1x 配方） | 0.7054 / 0.7690 (iter50k) | 0.742 | 0.697 |
| HTC (w/o sem) | ✅ | Resize(1333,800)（1024 方图→800²，论文短边800配方）+ 水平翻转 p=0.5 | 0.7120 / 0.7400 (ep10) | 0.727 | 0.692 |
| Mask R-CNN | ✅ | 同 HTC：Resize(1333,800) + 水平翻转 p=0.5 | 0.7050 / 0.7410 (ep12) | 0.709 | 0.682 |
| MS R-CNN | ✅ | 同 HTC：Resize(1333,800) + 水平翻转 p=0.5 | 0.7000 / 0.7350 (ep10) | 0.688 | 0.677 |
| SCNet | ❌ 放弃(三次失败) | — | — | — | 无法在本环境稳定复现, 表格引用文献值并注明协议差异 |
| Mask2Former (300ep v2) | ✅ | LSJ-1024：尺度抖动 0.1–2.0× + 随机裁剪 1024 + 水平翻转 p=0.5（官方配方） | 0.6810 / 0.6460 (iter220700) | 0.614 | 0.645 |
| SOLOv2 (1x) | ✅ | 多尺度短边 640–800 + 水平翻转 p=0.5（与 CondInst 同款官方配方） | (见训练日志) | — (无框输出,与文献行"bbox --"一致) | **0.668** |
| RS4D-box | ✅ | LSJ-1024：尺度抖动 0.1–2.0× + 随机裁剪 1024 + 水平翻转 p=0.5（作者 config 原样） | 0.6720 (ep648) | 0.645 | 0.641 |
| RSPrompter (anchor) | 🔄 v2 重训(bf16) | LSJ-1024 + 水平翻转 p=0.5（作者 anchor 配方，AdamW 2e-4 + clip0.1；fp16→bf16 数值修复） | — | — | — |
| RTMDet-Ins-S | ✅ | Mosaic + MixUp + YOLOXHSV + 多尺度Resize(0.5–2.0)+Crop640 + 翻转0.5（280ep 切 stage2 去 mosaic/mixup） | 0.6220 (ep130) | 0.642 | 0.598 |
| MaskDINO (R50) | ✅ | LSJ-1024：尺度抖动 0.1–2.0× + 随机裁剪 1024 + 翻转 p=0.5（官方配方） | 0.7541 / 0.7329 (iter441450) | **0.733** | **0.733** |
| YOLO11s-seg | ✅ | ultralytics 默认配方：Mosaic1.0 + HSV + 多尺度0.5–1.5 + 翻转0.5（末期关 mosaic） | (best 由 ultralytics fitness 选) | **0.767** | 0.668 |
| RSIISN (Swin-T Cascade) | ✅ | 多尺度 [(1200,800)…(800,400)] + 水平翻转 p=0.5（官方配方） | 0.6900 (ep10) | 0.741 | 0.721 |

> 增强口径说明：对比表采用"各方法按其原论文推荐增强 + 同数据同评测"的口径（即 CATNet/RS4D 等保留各自更强增强、Ours 保留轻增强设计），增强差异是方法配方的一部分，不额外统一。

**test 指标梯度（bbox mAP）**：YOLO11s 0.767 > CATNet 0.755 > CondInst 0.742 > RSIISN 0.741 > MaskDINO 0.733 > HTC 0.727 > Mask R-CNN 0.709 > MS R-CNN 0.688 > RS4D 0.645 > RTMDet-Ins 0.642 > Mask2Former-v2 0.614（11/11 有效基线齐；segm 榜：MaskDINO 0.733 > CATNet 0.727 > RSIISN 0.721 > CondInst 0.697 > HTC 0.692 > MaskRCNN 0.682 > MSRCNN 0.677 > YOLO11s=SOLOv2 0.668 > M2F 0.645 > RS4D 0.641 > RTMDet 0.598）

---

## ✅ CATNet (CAT Mask R-CNN, R50)

| 项 | 指向 |
|---|---|
| 配置 | `CATNet/configs/whu/cat_mask_rcnn_r50_3x_whu1024.py`（36ep SGD 0.01@有效16，原生1024，三向翻转0.75） |
| 训练日志 | `catnet_whu1024_r50_3x_bs4ai2_train_20260826_010653_pid1097113.log`（+`_resume_20260826_163019_pid19503.log` 服务器重启续训） |
| checkpoint | `catnet/cat_mask_rcnn_r50_3x_whu1024_bs4ai2/`（best=best_coco_segm_mAP_epoch_31.pth） |
| test 推理日志 | `catnet_whu1024_infer_20260826_221246_pid148831.log` |
| test 指标文件 | `catnet/.../test_out/test_metrics.json` |
| val 轨迹 | ep1 0.583 → ep14 0.737 → ep22 0.741 → **ep31 0.7470** → ep34 降LR后未超越 |

**test 全量 COCO 指标（2220 张，best ep31）：**

| | mAP | AP50 | AP75 | APs | APm | APl |
|---|---|---|---|---|---|---|
| bbox | 0.755 | 0.940 | 0.872 | 0.830 | 0.622 | 0.205 |
| segm | 0.727 | 0.941 | 0.865 | 0.778 | 0.611 | 0.304 |

## ✅ HTC (R50-FPN 1x, without-semantic)

| 项 | 指向 |
|---|---|
| 配置 | `mmdetection/configs/whu1024/htc-without-semantic_r50_fpn_1x_whu1024.py`（12ep SGD 0.02@有效16，Resize(1333,800)） |
| 训练日志 | `htc_wosem_r50_fpn_1x_whu1024_bs4ai4_train_20260827_102619_pid338088.log` |
| checkpoint | `htc/htc-without-semantic_r50_fpn_1x_whu1024_bs4ai4/`（best ep10） |
| test 推理日志 | `htc_whu1024_infer_20260827_203316_pid507808.log` |
| test 指标文件 | `htc/.../test_out/test_metrics.json` |

| | mAP | AP50 | AP75 | APs | APm | APl |
|---|---|---|---|---|---|---|
| bbox | 0.727 | 0.906 | 0.825 | 0.759 | 0.373 | 0.036 |
| segm | 0.692 | 0.907 | 0.816 | 0.714 | 0.382 | 0.071 |

## ✅ CondInst (MS-R50 1x)

| 项 | 指向 |
|---|---|
| 配置 | `AdelaiDet/configs/WHU/condinst_MS_R_50_1x_whu1024.yaml`（90k iter，bs8 lr0.005 线性折算，多尺度640-800） |
| 训练日志 | `condinst_MS_R50_1x_whu1024_bs8lr05m_train_20260826_220744_pid146891.log`（+`_resume_20260827_003621_pid184368.log` d2竞态崩溃后续训） |
| checkpoint | `condinst/condinst_MS_R_50_1x_whu1024_bs8lr05m/`（val best=iter 49999） |
| test 推理日志 | `condinst_whu1024_infer_20260827_195645_pid507807.log` |
| test 指标文件 | `condinst/.../test_out/test_metrics.json`（d2 百分制已对齐为小数） |
| 备注 | loss 0.877→0.756；总训练 18.5h；APs 0.785 极强 / APm 0.164 弱（FCOS 系小物体特长） |

| | mAP | AP50 | AP75 | APs | APm | APl |
|---|---|---|---|---|---|---|
| bbox | 0.742 | 0.900 | 0.824 | 0.785 | 0.164 | 0.016 |
| segm | 0.697 | 0.902 | 0.809 | 0.721 | 0.210 | 0.035 |

## ✅ Mask R-CNN (R50-FPN 1x)

| 项 | 指向 |
|---|---|
| 配置 | `mmdetection/configs/whu1024/mask-rcnn_r50_fpn_1x_whu1024.py`（12ep SGD 0.02@有效16） |
| 训练日志 | `maskrcnn_r50_fpn_1x_whu1024_bs8ai2_train_20260827_103745_pid342433.log` |
| checkpoint | `maskrcnn/mask-rcnn_r50_fpn_1x_whu1024_bs8ai2/`（best ep12） |
| test 推理日志 | `maskrcnn_whu1024_infer_20260827_195645_pid507810.log` |
| test 指标文件 | `maskrcnn/.../test_out/test_metrics.json` |

| | mAP | AP50 | AP75 | APs | APm | APl |
|---|---|---|---|---|---|---|
| bbox | 0.709 | 0.899 | 0.809 | 0.742 | 0.304 | 0.029 |
| segm | 0.682 | 0.900 | 0.800 | 0.705 | 0.315 | 0.063 |

## ✅ MS R-CNN (R50-caffe-FPN 1x)

| 项 | 指向 |
|---|---|
| 配置 | `mmdetection/configs/whu1024/ms-rcnn_r50-caffe_fpn_1x_whu1024.py`（bs4×累积4=有效16，lr 不变——GPU3 显存适配） |
| 训练日志 | `msrcnn_r50_caffe_fpn_1x_whu1024_bs4ai4_train_20260827_105848_pid351378.log`（首版bs8因显存改bs4重启） |
| checkpoint | `msrcnn/ms-rcnn_r50-caffe_fpn_1x_whu1024_bs4ai4/`（best ep10） |
| test 推理日志 | `msrcnn_whu1024_infer_20260827_201458_pid516504.log` |
| test 指标文件 | `msrcnn/.../test_out/test_metrics.json` |

| | mAP | AP50 | AP75 | APs | APm | APl |
|---|---|---|---|---|---|---|
| bbox | 0.688 | 0.894 | 0.800 | 0.726 | 0.311 | 0.018 |
| segm | 0.677 | 0.896 | 0.795 | 0.700 | 0.346 | 0.030 |

---

## ✅ Mask2Former (R50, mmdet 移植版) — v2: 300ep 重训完成

> **v1→v2 变更依据**：v1 用 epoch 等价换算（50ep=14.7万样本，仅 COCO 原配方 2.5%），val 至终点仍陡涨（50k→70k: 0.536→0.615）为欠训练实锤；且与 CondInst 的迭代字面照抄（490ep 量级）口径不一致。v2 改 300 WHU epochs（88万样本，bs4×累积4，里程碑 196354/212416），目标进入其他基线样本量级。
> v1 旧结果已备份：`mask2former/mask2former_r50_whu1024_bs2ai8_50ep_undertrained_bak/`（test bbox 0.536 / segm 0.582 仅作参考不入表）
> v2 启动：GPU2，SCNet v3 完成后由 `watch_scnet_then_m2f.sh` 自动接力，预计 9/2 出结果

| 项 | 指向 |
|---|---|
| 配置 | `mmdetection/configs/whu1024/mask2former_r50_whu1024.py`（AdamW 1e-4@有效16，50ep=73575 batch-iter，LSJ-1024） |
| 训练日志 | `mask2former_r50_whu1024_bs2ai8_train_20260827_001751_pid179432.log`（exit=0 完整跑完） |
| checkpoint | `mask2former/mask2former_r50_whu1024_bs2ai8/`（**无 save_best**——调度改写时疏漏；val best 由日志解析=iter 73000，推理显式指定 `iter_73000.pth`） |
| test 推理日志 | `mask2former_whu1024_infer_20260828_*.log`（进行中） |
| val best | iter 73000：segm 0.6220 / bbox 0.5750（**显著低于其他基线**，50ep 等价调度在 WHU 偏短，如实报告） |
| test 推理日志 (v1) | `mask2former_whu1024_infer_20260828_221553_*.log`（v1 结果见上，仅参考不入表） |

**v2 (300ep) 训练/推理：**

| 项 | 指向 |
|---|---|
| 训练日志 | `mask2former_r50_whu1024_bs4ai4_300ep_train_*.log`（exit=0，220725 iter 跑满） |
| checkpoint | `mask2former/mask2former_r50_whu1024_bs4ai4_300ep/`（best val 由日志解析=iter 220700，用 final `iter_220725.pth` 推理） |
| test 推理日志 | `mask2former_whu1024_infer_20260902_111152_pid2288880.log` |
| test 指标文件 | `mask2former/mask2former_r50_whu1024_bs4ai4_300ep/test_out/test_metrics.json` |
| val 轨迹 | 41 次验证；中途 ~0.65 平台 → 末期爬升至 0.6810（cosine 末段+多下降点生效） |

**test 全量 COCO 指标（2220 张，iter_220725 ≈ best val 时刻）：**

| | mAP | AP50 | AP75 | APs | APm | APl |
|---|---|---|---|---|---|---|
| bbox | 0.614 | 0.827 | 0.707 | 0.652 | 0.511 | 0.307 |
| segm | 0.645 | 0.841 | 0.756 | 0.664 | 0.568 | 0.397 |

> v1→v2 提升：bbox 0.536→0.614、segm 0.582→0.645（欠训练修复确认）。test 仍低于 val 约 3.6 个点（test 集更难），且低于 M2F 家族在 COCO 上的相对位次——如需在论文中讨论可注明其 query 式设计在 WHU 密集小建筑上不占优。

## ❌ SCNet (R50-FPN 1x) — 三次尝试后放弃

| 版本 | 修复动作 | 结局 |
|---|---|---|
| v1 | 原始 | 梯度爆炸 NaN（补 detectron2 风格裁剪 35） |
| v2 | 语义图加载修复 + 2 类语义 | 语义头自激（loss_semantic_seg→3740）全网爆（补 BN） |
| v3 | 语义头加 BN | 训练平稳走完 12ep 但 bbox 仅 0.37、mask 卡死 0.169 且逐 ep 变差 |

结论：mmdet 移植版 SCNet 的语义融合与"从实例掩码派生的合成语义"（信息冗余、无增量）深度不兼容；原版依赖 COCO-Stuff 真实 stuff 标注。**表格 SCNet 行引用文献值并注明协议差异。** 训练日志：`scnet_r50_fpn_1x_whu1024_bs8ai2_train_*.log`（三份）

⚠️ SCNet 踩坑全记录（三次发散的完整根因链，写论文/rebuttal 可用）：
1. mmdet 移植版缺 detectron2 默认梯度裁剪 → 已补 max_norm=35
2. mmengine 数据序列化路径丢 `seg_map_path` → `serialize_data=False` 绕过
3. **`num_classes=1` 的语义头在数学上不可用 CE 监督**（softmax 恒 1 → loss 恒 0）→ 语义图改 2 类（building=1/bg=0）、`num_classes=2`
4. **无归一化的残差语义头自激发散**（v2 在 ep1-3 loss_semantic_seg 冲到 3740 拖垮全网）→ 语义头加 `norm_cfg=BN`（对齐原版 detectron2 带归一化的实现），v3 训练稳定（loss 3.5、语义 0.25）

## ✅ SOLOv2 (R50 1x)

| 项 | 指向 |
|---|---|
| 配置 | `AdelaiDet/configs/WHU/solov2_R50_1x_whu1024.yaml`（1x=90k iter；eval 仅 segm——SOLOv2 无框输出，与文献行"bbox --"口径一致） |
| 训练日志 | `solov2_R50_1x_whu1024_bs8lr05m_train_20260826_234047_pid169450.log`（+`_resume_20260828_*.log` d2竞态后续训，8/29 14:13 exit=0 完整收官） |
| checkpoint | `solov2/solov2_R50_1x_whu1024_bs8lr05m/model_final.pth` |
| test 推理日志 | `solov2_whu1024_infer_20260829_141615_pid1056085.log` |
| test 指标文件 | `solov2/.../test_out/test_metrics.json` |

**test 全量 COCO 指标（仅 segm）：**

| | mAP | AP50 | AP75 | APs | APm | APl |
|---|---|---|---|---|---|---|
| segm | 0.668 | 0.864 | 0.769 | 0.696 | 0.410 | 0.054 |

## 🔄 RS4D-box

| 项 | 指向 |
|---|---|
| 配置 | `RS4D/configs/rs4d/rs4d_bbox-whu-1024.py`（AdamW 9.2224e-5@bs8 bf16，800ep cosine，LSJ-1024，蒸馏权重 20250227-021854_500k） |
| 训练日志 | `rs4d_bbox_whu1024_bs8_bf16_800ep_train_20260826_002914_pid1084425.log`（+`_resume_20260826_163019_pid19502.log` 重启续训） |
| checkpoint | `rs4d/rs4d_bbox_whu1024_bs8_bf16_800ep/` |
| 训练完成 | 800ep 跑满（9/3 07:03），best ep648 segm 0.672 |
| test 指标文件 | `rs4d/rs4d_bbox_whu1024_bs8_bf16_800ep/test_out/test_metrics.json` |

**test 全量 COCO 指标（2220 张，best ep648）：**

| | mAP | AP50 | AP75 | APs | APm | APl |
|---|---|---|---|---|---|---|
| bbox | 0.645 | 0.876 | 0.747 | 0.228 | 0.692 | 0.785 |
| segm | 0.641 | 0.878 | 0.752 | 0.199 | 0.692 | 0.801 |

> ⚠️ 其 fork 版 CocoMetric 的 s/m/l 分档结果反常（s 低 m/l 高，与其他全体方法相反），all/mAP/AP50/AP75 档可信，s/m/l 引用需注意。

---

## 🔄 RSPrompter (anchor) — 第 10 个基线

| 项 | 指向 |
|---|---|
| 配置 | `RSPrompter-release/configs/rsprompter/rsprompter_anchor-whu-1024.py`（AdamW 2e-4 wd0.05 + clip0.1 + AMP fp16，300ep cosine，LSJ-1024，bs1×累积2） |
| 启动脚本/日志 | `run_rsprompter_whu1024.sh`；`rsprompter_anchor_whu1024_bs1ai2_300ep_train_20260829_212613_pid1181770.log` |
| checkpoint | `rsprompter/rsprompter_anchor_whu1024_bs1ai2_300ep/` |
| ❌ 结局 | 300ep 跑满但**权重全线 NaN**（epoch_297–300 全部 472 tensors NaN/Inf，更早 ckpt 被 max_keep 清理）→ 推理 0 实例、val 全程无指标即为表象。根因：AMP fp16 数值爆炸（官方推荐 DeepSpeed，fp16 备选路径对 SAM 全参微调不稳）。已拍板 bf16 重训（20260906 21:00 起，GPU3 与 NWPU 任务共存 37GB，0.75s/iter，独占后 ~2.7 天/共存 ~4 天）；NaN 旧目录归档 `rsprompter_anchor_whu1024_bs1ai2_300ep_nan_bak_20260906/` |

⚠️ 部署记录（query 版弃用经过）：
1. query 版（Mask2Former 式 100-query 解码器）**bs2 与 bs1 均在 48G 卡 OOM**（全量微调 SAM ViT-B + 高分辨率逐 query dynamic conv 激活爆炸；作者原配置靠多卡 DeepSpeed ZeRO-2 才能跑）；已给 SAM encoder/decoder 包装类打梯度检查点补丁仍不够 → 按用户决定改用 **anchor 版**（Mask R-CNN 式，bs1 仅 22.7GB ✓）
2. 本发布版 `mmdet/rsprompter/__init__.py` 为全延迟导入不触发注册 → custom_imports 需指到 `.models`/`.datasets` 子模块
3. DeepSpeed 未安装 → 走官方注释的 AMP 备选路径（保留 gradient_clipping=0.1）；SAM ViT-B HF 权重经 hf-mirror 下载至 `work_dirs/sam_cache/`

---

## ✅ RTMDet-Ins-S — 第 11 个基线（cvt2/mmdet 官方实现）

| 项 | 指向 |
|---|---|
| 配置 | `mmdetection/configs/whu1024/rtmdet-ins_s_300e_whu1024.py`（CSPNeXt-S，300ep AdamW 0.004@有效256=官方8×32，640 尺寸，EMA+两阶段增强切换） |
| 启动脚本/日志 | `run_mmdet_whu1024.sh rtmdet`；`rtmdet_ins_s_whu1024_bs4ai64_300ep_train_20260829_230440_pid1220775.log` |
| checkpoint | `rtmdet/rtmdet_ins_s_whu1024_bs4ai64_300ep/` |
| 单卡换算 | bs4×accum64（实测 bs32=46.6G/bs16=25.4G/bs8=24.2G(含碎片) 挤占 RSPrompter 改小微批）；EMA momentum 2e-4/步→3.125e-6/micro-iter 等价换算；SyncBN 单卡自动转 BN |
| test 推理日志 | `rtmdet_ins_s_whu1024_bs4ai64_300ep_test_20260901_153916_*.log`（best=ep130） |
| test 指标文件 | `rtmdet/rtmdet_ins_s_whu1024_bs4ai64_300ep/test_out/test_metrics.json` |
| val 轨迹 | ep70 0.620 → ep130 0.6220(峰) → 末期 0.614 |

**test 全量 COCO 指标（2220 张，best ep130）：**

| | mAP | AP50 | AP75 | APs | APm | APl |
|---|---|---|---|---|---|---|
| bbox | 0.642 | 0.856 | 0.753 | 0.694 | 0.129 | 0.017 |
| segm | 0.598 | 0.858 | 0.712 | 0.633 | 0.141 | 0.109 |

---

## 🔄 MaskDINO (R50) — 第 12 个基线（IDEA-Research 官方 detectron2 栈）

| 项 | 指向 |
|---|---|
| 配置 | `MaskDINO/configs/whu/maskdino_R50_whu1024.yaml`（AdamW 1e-4 bs16 LSJ-1024 AMP grad-clip0.01，300 queries DN=seg） |
| 训练入口 | `MaskDINO/train_whu.py`（WHU 三 split 注册 + 自研 AccumAMPTrainer 梯度累积） |
| 启动脚本/日志 | `run_maskdino_whu1024.sh`；`maskdino_R50_whu1024_bs2ai8_300ep_train_20260829_225817_pid1218044.log` |
| checkpoint | `maskdino/maskdino_R50_whu1024_bs2ai8_300ep/` |
| 单卡换算 | bs2×accum8=官方总16（bs4 OOM>46G：DN+300query+9层decoder 远重于 M2F；注意 detectron2 IMS_PER_BATCH 单卡即每步样本数）；300 WHU ep=441450 micro-iters，里程碑按 COCO 原比例 88.886%/96.296% |
| 环境 | 新建 `mdino`（torch1.13.1+cu117 + detectron2 v0.6 + MSDeformAttn 编译 + pillow9.5 + numpy1.23.5）；dt2 的 detectron2@9eb4831 太老（缺 file_io/AMPTrainer）不可复用 |
| 预训练 | `pretrained/R-50-torchvision.pkl`（注意：本地原 R-50.pkl 是 MSRA-caffe 版，MaskDINO 需 torchvision 版，已另下） |
| 训练/推理日志 | `maskdino_R50_whu1024_bs2ai8_300ep_train_20260829_225817_*.log`；test `..._test_20260906_111051_*.log`（model_final=best val 时刻） |
| test 指标文件 | `maskdino/maskdino_R50_whu1024_bs2ai8_300ep/test_out/test_metrics.json` |
| val 轨迹 | ep40 0.659 → ep62 0.707 → ep175 0.723 → **最终 0.7541**(iter441450) |

**test 全量 COCO 指标（2220 张，model_final）：**

| | mAP | AP50 | AP75 | APs | APm | APl |
|---|---|---|---|---|---|---|
| bbox | **0.733** | 0.908 | 0.827 | 0.776 | 0.381 | 0.082 |
| segm | **0.733** | 0.903 | 0.825 | 0.771 | 0.362 | 0.064 |

---

## 🔄 YOLO11s-seg — 第 13 个基线（ultralytics 8.4，AGPL-3.0 已知悉）

| 项 | 指向 |
|---|---|
| 数据 | `dataset/WHU/yolo_seg/`（COCO→YOLO seg 多边形转换，339 空图保留为负样本；`coco2yolo_seg.py`） |
| 启动脚本/日志 | `run_yolo11_whu1024.sh <mode> <gpu>`（train=ultralytics 全默认配方：100ep bs16 imgsz640 auto 优化器，官方 COCO 预训练 yolo11s-seg.pt 起训）；`yolo11s_seg_whu1024_default_train_20260829_224153_pid1208212.log` |
| checkpoint | `yolo11seg/yolo11s_seg_whu1024_default/yolo11s_seg_whu1024_default/`（best.pt/last.pt + results.csv） |
| test 评估 | `eval_yolo_whu.py`：predict→COCO json→pycocotools maxDets=[100]，12 指标 + 可选 mask-only 可视化（conf0.001/iou0.7/max_det100） |
| 训练/推理日志 | `yolo11s_seg_whu1024_default_train_20260830_135951_pid1396098.log`（v3）；`yolo11_whu1024_infer_20260830_151316_pid1404913.log` |
| test 指标文件 | `yolo11seg/yolo11s_seg_whu1024_default/yolo11s_seg_whu1024_default/test_out/test_metrics.json` |
| ⚠️ 重训记录 | v1 坏标签(x/y拼接bug)、v2 labels.cache 吃旧缓存，均归档 `badlabel_discarded_*`；v3 修复后终版 |

**test 全量 COCO 指标（2220 张，best.pt v3）：**

| | mAP | AP50 | AP75 | APs | APm | APl |
|---|---|---|---|---|---|---|
| bbox | **0.767** | 0.915 | 0.854 | 0.793 | 0.516 | 0.024 |
| segm | 0.668 | 0.914 | 0.791 | 0.690 | 0.491 | 0.026 |

---

## ✅ RSIISN (Swin-T Cascade) — 第 14 个基线（ESWA'23 官方实现）

| 项 | 指向 |
|---|---|
| 仓库 | `RSIISN/`（Sherlock1018/RSIISN；Cascade Mask R-CNN + Swin-T + CSA/MSA/SRL 模块） |
| 配置 | `RSIISN/configs/whu1024/rsiisn_cascade_swinT_whu1024.py`（AdamW 1.25e-3 12ep 多尺度，CIoU+soft-nms+DiceLoss，官方补丁文件替换 mmdet2.23 对应源码） |
| 启动脚本 | `run_rsiisn_whu1024.sh <mode> <gpu>` |
| checkpoint | `rsiisn/rsiisn_cascade_swinT_whu1024_bs4ai2_12ep/` |
| 单卡换算 | bs4×accum2=官方总批8（mmcv GradientAccumulativeOptimizerHook） |
| 协议偏离声明 | 官方 test 为多尺度+翻转 TTA；本协议全基线不开 TTA → 单尺度 (1200,800) flip=False；rcnn max_per_img 1000→100 |
| 环境 | 新建 `mm2`（torch1.13.1+cu117 + mmcv-full1.7.2 + mmdet2.23.0 + 官方 5 个补丁文件） |
| 训练/推理日志 | `rsiisn_cascade_swinT_whu1024_bs2ai4_12ep_train_20260830_153624_*.log`（+resume）；best ep10 重推 `result_best_ep10.pkl`（samples_per_gpu=1） |
| test 指标文件 | `rsiisn/rsiisn_cascade_swinT_whu1024_bs2ai4_12ep/test_out/test_metrics.json` |
| ⚠️ 官方补丁坑 | (a) 补丁版 coco.py 硬编码作者路径 /home/zwc；(b) 模型 batch>1 推理输出空（tools/test.py 继承 samples_per_gpu=2）→ spg=1 重推；(c) results2json 的 image_id 与 GT 不符 → eval_rsiisn_pkl.py 从 pkl 按 data_infos 真实 id 评估 |

**test 全量 COCO 指标（2220 张，best ep10）：**

| | mAP | AP50 | AP75 | APs | APm | APl |
|---|---|---|---|---|---|---|
| bbox | 0.741 | 0.900 | 0.844 | 0.773 | 0.365 | 0.078 |
| segm | 0.721 | 0.901 | 0.839 | 0.748 | 0.393 | 0.088 |

---

## Ours (PromptMiner-SAM2) — 完整重跑

| 项 | 内容 |
|---|---|
| 权重 | `paper_promptminer_rd_p2_whu_full_tr1.0_va1.0/best_model_epoch98.pth`（paper 正式版，全量 train 100ep） |
| test 推理 | worktree@7e9423f `test_only_from_ckpt.sh`（ckpt 契约校验需匹配代码版）；日志 `viz_ours_paper_e98_tr1.0_va1.0_20260830_204512_*.log` |
| val best | ep98（segm 0.6948 / bbox 0.7354 于 p2pc10_calibrated；paper 版同类） |

**test 全量 COCO 指标（2220 张，paper e98，maxDets=100）：**

| | mAP | AP50 | AP75 | APs | APm | APl |
|---|---|---|---|---|---|---|
| bbox | 0.755 | 0.839(mAP75列见日志) | 0.839 | — | — | — |
| segm | **0.729** | 0.901 | 0.840 | 0.435 | 0.779 | — |

> segm 0.729 为当前全部方法第一（CATNet 0.727）；bbox 0.755 与 CATNet 持平、低于 YOLO11s 0.767。
> （bbox AP50/APs 等完整 12 项见 test_metrics.json）
| 启动脚本 | `.worktrees/dual-stream-dev/.../scripts/ue/run_ue_single_stream.sh`、`run_ue_dual_stream.sh` |
| 训练日志 | `.worktrees/p2-v2-e2e/.../logs/ablations/p2pc10_baseline_rd_p2_from_sam2_tr0.10_va1.0_20260826_161049_pid9231.log`；`.worktrees/dual-stream-dev/.../work_dirs/ue/{ue_single_c5v2_control,ue_dual_c5v2_p2p3_sparse}_tr128_va259_s44_r1/` |
| val best (bbox/segm) | —（待训完） |
| test 全量指标 | —（用 `infer_whu1024.sh` 同协议评测后填入） |

---

## 附录 A：WHU-1024 关键统计（本机实测）

- 划分：train 2943（143,252 实例）/ val 627（15,926）/ test 2220（70,063）；全 1024×1024 单类
- train 2943 = 2793 官方标注图 + **150 张 `11xxx` 系列无标注背景卫星图**（零实例，全部参与训练）
- maxDets=100 天花板：test 中 209 张图（9.4%）GT>100 实例、占全部实例 33.5%，单图最多 140；全局理论 recall 上限 ≈96.3%（对全体方法同等生效）

## 附录 B：推理/可视化统一入口

- 全量 test 推理：`bash portable_sam2_explicit_coarse/scripts/baselines/infer_whu1024.sh <model> <gpu> [vis]`（自动选 val best ckpt；指标落 `work_dir/test_out/test_metrics.json`；vis 生成仅 mask 叠加、每实例独立颜色、无置信度的 PNG）
- 新增四基线入口：RTMDet-Ins → `run_mmdet_whu1024.sh rtmdet test <gpu>`（whu_test_mmdet.py）；MaskDINO → `run_maskdino_whu1024.sh test <gpu> <ckpt>`（eval_maskdino_whu.py，官方 DatasetMapper test 预处理）；YOLO11 → `run_yolo11_whu1024.sh test <gpu>`（eval_yolo_whu.py）；RSIISN → `run_rsiisn_whu1024.sh test <gpu>`（mmdet2 tools/test + 日志解析 12 指标）
- 本项目自身可视化：`scripts/visualize_instances.py pred --mask-only`（口径与基线一致）
- 所有训练/推理日志：`portable_sam2_explicit_coarse/logs/baselines/{runtag}_{mode}_{时间戳}_pid{pid}.log`（含每分钟 GPU 监控 .gpu.log）

## 附录 C：已修复并留档的部署问题

1. evaluator `ann_file` 显式本地化（否则继承 data/coco 路径，CATNet 首跑因此在验证时崩溃）
2. torch≥2.6 `weights_only` → 所有脚本加 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`
3. SCNet 三层根因（梯度裁剪缺失/序列化丢 seg 路径/C=1 CE 恒零），详见 SCNet 节
4. AdelaiDet 空图兼容（339 张无标注图：CondInst mask 分支/SOLOv2 GT 构造/空 batch loss 三处保护）
5. MaskDINO 与 dt2 的 detectron2@9eb4831（2020）不兼容（缺 `utils.file_io`/AMPTrainer）→ 新建 mdino 环境装 v0.6；MSDeformAttn 需在对应环境重编译（`python setup.py build install`）
6. mmengine/urllib 直连 openmmlab 下载预训练报 SSL 证书错 → curl 预下载到 `~/.cache/torch/hub/checkpoints/` 后以 `init_cfg.checkpoint` 本地路径注入（RTMDet cspnext-s）
7. detectron2 无原生梯度累积 → 自研 `AccumAMPTrainer`（bs4×accum4；裁剪经 build_optimizer 包装在 unscale 后生效，与官方一致）
8. ultralytics 装入 cvt2：`--no-deps` 安装 + 单独补 filelock≥3.16（AsyncFileLock），mm 栈版本零改动，已验证共存
9. mmengine EMAHook 不感知梯度累积（每 micro-iter 更新）→ RTMDet EMA momentum 按微步等价换算 2e-4/步→3.125e-6/micro-iter(bs4ai64)
10. **YOLO 标签 x/y 拼接 bug**（20260830 发现并修复）：COCO→YOLO 转换器把多边形写成 `[全部x, 全部y]` 而非 `x1,y1,x2,y2` 交错 → 标签乱线化（抽样仅 5.6% 与 GT 吻合）。症状：ultralytics 内部 val 对(同样乱的)标签 box mAP 0.487 假高、mask≈0.002 假低、对真 GT 全零；曾先后误诊为 image_id 映射/category_id/rect 差异。修复后重转标签抽查 903/903 全吻合，旧模型作废重训（`badlabel_discarded_20260830/`）。教训：格式转换必须做"转换→反转换→与源对齐"量化抽查
11. **YOLO 第二层坑：labels.cache**（20260830）：修复标签重转 txt 后未删 `yolo_seg/labels/*.cache`，重训仍吃坏缓存（val 指标与坏模型逐位一致暴露）。删 cache 后 v3 重训 loss 结构立刻正常（seg 5.9→1.2 量级）。教训：改标签必删 ultralytics cache
12. mm2（mmdet2.23 栈）依赖对齐：pip mmdet 2.23.0 要求 mmcv-full≤1.5.0（官方日志实为 1.4.8，源码编译）；RSIISN 补丁需 timm==0.6.13；mmcv1 系配 yapf<0.41（0.43 移除 verify 参数）
13. dt2 环境：THC 头移除补丁、numpy/Pillow/rapidfuzz/OpenCV 版本适配
14. 强杀 mmengine 训练后 dataloader worker 会成孤儿进程持有 CUDA 上下文——清理：`ps -ef | grep <config名> | grep -v grep | awk '{print $2}' | xargs kill -9`
15. detectron2 `MapDataset` 多 worker 竞态（IndexError 越界）为瞬态故障，resume 即可（CondInst/SOLOv2 各中招一次）
16. Mask2Former `IterBasedTrainLoop` 按 batch 计数（非优化器步数），调度换算勿踩；其 checkpoint hook 需显式 save_best
17. mdino 环境隐患：detectron2 pip 依赖装新了 Pillow(11)/numpy(2) → 需降 pillow 9.5 + numpy 1.23.5；后台链路 conda activate 偶发不生效 → 一律绝对路径 python
18. RSIISN 官方补丁两处坑：(a) 补丁版 coco.py:339 硬编码作者路径 /home/zwc → 验证时 PermissionError，改为使用传入 jsonfile_prefix；(b) 其模型代码在 **batch>1 推理时输出空结果**(tools/test.py 继承 samples_per_gpu=2) → test 需 samples_per_gpu=1 重推；另补丁版 results2json 产出的 image_id 与 GT 不符(loadRes IndexError 被吞成 "empty") → 用 eval_rsiisn_pkl.py 从 pkl 直接按 data_infos 真实 id 转 COCO 评估

## 附录 D：参数量对比（20260830 实测，各方法实际训练 checkpoint 统计）

口径：推理时总参数（含 BN running stats 等 buffer，不含优化器/EMA 副本），单位 M。
Ours 为 SAM2 Hiera-B+ 全量微调架构（LoRA 注入），总参数即推理参数；可训练增量另注。

| 方法 | 参数量 (M) | test bbox/segm mAP |
|---|---|---|
| **Ours (PromptMiner-SAM2)** | **113.06** | 待重跑完回填 |
| RSPrompter anchor (SAM ViT-B) | 117.05 | ❌ 训练 NaN 报废 |
| RSIISN (Swin-T Cascade) | 95.89 | **0.741 / 0.721** |
| HTC (R50) | 77.21 | 0.727 / 0.692 |
| MS R-CNN (R50-caffe) | 60.28 | 0.688 / 0.677 |
| CATNet (R50) | 54.72 | 0.755 / 0.727 |
| SOLOv2 (R50) | 46.28 | — / 0.668 |
| MaskDINO (R50) | 46.27 | **0.733 / 0.733** |
| Mask2Former (R50) | 44.06 | **0.614 / 0.645** (v2 300ep) |
| Mask R-CNN (R50) | 44.02 | 0.709 / 0.682 |
| CondInst (MS-R50) | 34.04 | 0.742 / 0.697 |
| RS4D (RSMamba-base) | 33.23 | **0.645 / 0.641** |
| RTMDet-Ins-S (CSPNeXt-S) | 11.66 | **0.642 / 0.598** |
| YOLO11s-seg | 10.08 | **0.767 / 0.668** |

Ours 分模块：SAM2 Hiera-B+ backbone 70.60M (62.4%) + SAM2 其余(prompt enc/mask dec 等) 6.95M + 新增 miner/head 35.51M；LoRA 增量参数仅 1.06M。
SCNet(R50, 弃用) 91.77M 仅供参考不进对比表。

## 附录 E：边界掩码指标（Boundary AP，20260903 实测）

口径：每实例 mask 的边界带（**COCO 2019 挑战官方 w = 2·√A/20，面积自适应**，最小 1px）作为分割结果走标准 COCO 匹配（IoU 阈值 0.50:0.95，maxDets=100，与主表同协议）；评估器 `scripts/baselines/boundary_eval.py`（bbox 窗口化 2.6ms/实例，多进程）。fix2（固定 2px）经验证无区分度弃用；fix4 为严苛参考口径（另存各模型 `test_out/boundary_fix4.json`）。
预测来源：各模型 best ckpt 在 test 2220 上重推落盘 `test_out/pred.segm.json`（GPU1 串行 20260902-03）或复用既有 pred_coco.json（RSIISN/YOLO）。

| 方法 | Boundary AP | Boundary AP50 | Boundary AP75 | APs | APm | APl |
|---|---|---|---|---|---|---|
| CATNet (R50) | **0.223** | 0.621 | 0.090 | 0.283 | 0.142 | 0.019 |
| SOLOv2 (R50) | 0.202 | 0.565 | 0.077 | 0.247 | 0.090 | 0.002 |
| RSIISN (Swin-T) | 0.200 | 0.598 | 0.050 | 0.256 | 0.021 | 0.006 |
| CondInst (MS-R50) | 0.179 | 0.552 | 0.045 | 0.241 | 0.014 | 0.002 |
| HTC (R50) | 0.144 | 0.514 | 0.009 | 0.198 | 0.009 | 0.000 |
| Mask2Former v2 | 0.141 | 0.467 | 0.024 | 0.192 | 0.130 | 0.129 |
| Mask R-CNN (R50) | 0.139 | 0.495 | 0.009 | 0.191 | 0.006 | 0.000 |
| MS R-CNN (R50) | 0.136 | 0.486 | 0.010 | 0.185 | 0.014 | 0.000 |
| YOLO11s-seg | 0.112 | 0.433 | 0.009 | 0.152 | 0.065 | 0.002 |
| RTMDet-Ins-S | 0.068 | 0.328 | 0.000 | 0.106 | 0.002 | 0.000 |

**关键观察**（对论文有用的点）：
1. Boundary 排名与 segm mAP 排名显著不同：YOLO11s（bbox 第一）边界垫底区（640 低分辨率 mask 粗糙）；SOLOv2 直接法分格预测的 mask 边界反而优于多数两阶段法
2. Boundary AP75 全体极低（≤0.09）——高 IoU 下的边界对齐是全体方法的共同难点（为边界增强类方法留出论文叙事空间）
3. CATNet 三向翻转增强 + 原生 1024 分辨率 → 边界也第一，与 segm 一致
4. MaskDINO Boundary AP **0.234**/0.627（第一，与其 segm 0.733 第一致）；RS4D 0.120/0.434（其 format_only json 的 category_id 为 0 起始索引，评估器已归一化修复——曾因此全零）
5. RSPrompter 因训练 NaN 报废无边界数据
