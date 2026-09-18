# WHU-1024 剩余对比实验部署计划 v1（待审批）

> 生成日期：2026-08-26 ｜ 状态：**计划，未实施**
> 已在跑：RS4D-box（GPU 0，~9月2日完）、CATNet（GPU 1，~8月26日下午完）
> 本计划覆盖剩余 6 个方法：Mask R-CNN / MS R-CNN / HTC / SCNet / CondInst / SOLOv2 / Mask2Former（7 个方法，Mask2Former 二选一方案）

## 0. 统一协议（与已跑两个实验一致）

- 数据：`/data1/wangcheng/dataset/WHU`，COCO json，train **2943 全量不滤空图**，val **627 选模**（save_best=coco/segm_mAP），test **2220 仅终评**，单类 `building`，`maxDets=100`，无 TTA
- 超参原则：各方法**按其原论文/官方仓推荐配方**（优化器/调度/增强/输入处理一律照抄，见各组明细）
- checkpoint：`/data1/wangcheng/checkpoint/whu1024_baselines/<model>/`；日志：`logs/baselines/{runtag}_{mode}_{ts}_pid{pid}.log`（沿用现有脚本模板）
- 卡：仅 GPU 0/1（48G）。排序原则：GPU 1 在 CATNet 完成后先承接 detectron2 组（周期长），GPU 0 在 RS4D 完成后承接其余

## 1. 环境方案（经实测核查）

| 环境 | 覆盖方法 | 依据 |
|---|---|---|
| **cvt2（复用，零新装）** | Mask R-CNN、MS R-CNN、HTC、SCNet、Mask2Former(mmdet 移植版) | 已验证 cvt2 内置 mmdet 3.3.0 的 5 个官方配置全部解析通过（mask-rcnn/ms-rcnn/htc/scnet/mask2former_r50_8xb2-lsj-50e） |
| **新建 `/data2/wangcheng/envs/dt2`** | CondInst、SOLOv2（AdelaiDet）、Mask2Former(官方优先方案) | AdelaiDet 基于 Detectron2（README 锁 commit 9eb4831），与 cvt2 的 mmcv 2.x 完全不兼容；Mask2Former 官方仓同为 detectron2 系，可共用此环境 |

**dt2 环境搭建步骤**（一次性的，约 1-2 小时）：
```
conda create -p /data2/wangcheng/envs/dt2 python=3.9 -y
# torch cu117（4090 经 PTX JIT 可用；AdelaiDet 官方验证过的最高 torch 版本线）
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
conda install -p /data2/wangcheng/envs/dt2 -c nvidia cuda-toolkit=11.7   # 匹配的 nvcc，编译 detectron2/MSDeformAttn
# detectron2 @ 9eb4831（AdelaiDet 推荐锁定 commit）：经 ghfast.top 镜像 clone 后
FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="8.9" CUDA_HOME=$CONDA_PREFIX pip install -e <detectron2 目录>
# AdelaiDet 自身（含 CUDA 扩展）：
cd /home/wangcheng/project/AdelaiDet && FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="8.9" pip install -e . --no-build-isolation
# Mask2Former 官方（编译 MSDeformAttn 核）+ panopticapi：
cd /home/wangcheng/project/Mask2Former && FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="8.9" python setup.py build install
pip install git+https://ghfast.top/https://github.com/cocodataset/panopticapi.git（或镜像 clone 后本地装）
```
风险点：① 4090+cu117 首次启动各 kernel 有 JIT 编译延迟（缓存后消失）；② 若 AdelaiDet 在 torch 1.13 报小错，社区常见补丁即可修（备选退 torch 1.10 但 4090 不支持，故不退）；③ detectron2 编译若 sm_89 不识别则加 `TORCH_CUDA_ARCH_LIST="8.0+PTX"`。

## 2. Group A：mmdetection 家族（cvt2，GPU 0/1 空闲后跑，每个都很短）

复用 cvt2 已装 mmdet 3.3 的 `.mim` 入口（与 CATNet 跑法相同：PYTHONPATH 不需要，直接 python .mim/tools/train.py）。为每个方法写一个 WHU config（模板 = CATNet 那份的 dataloader/协议段 + 各自官方 coco 基配置的 model 段）。

| 方法 | 基配置（mmdet 3.3 内置） | 超参（官方 1x 配方原样） | 输入 | 单卡适配 | 预计时长 |
|---|---|---|---|---|---|
| Mask R-CNN | mask-rcnn_r50_fpn_1x | SGD lr0.02@bs16、wd1e-4、12ep、milestones[8,11]、hflip0.5 | Resize(1333,800)→1024图缩为800²（论文 shorter-side-800 配方忠实执行；torchvision R50 已在本地缓存） | bs8×accum2=有效16，lr 不动 | ~1.5h |
| MS R-CNN | ms-rcnn_r50-caffe_fpn_1x | 同上 1x + MaskIoU 头 | 同上；**caffe 版 R50 权重需从 download.openmmlab.com 拉**（国内可达，待验证） | 同上 | ~2h |
| HTC | htc_r50_fpn_1x | 1x、lr0.02@bs16、hflip0.5（无语义分支版本） | 同上 | bs4×accum4（HTC 显存更高） | ~3-4h |
| SCNet | scnet_r50_fpn_1x | 1x、lr0.02@bs16 | 同上 | bs8×accum2 | ~3-4h |

注意：这四个的文献值（表内行）出自 RS4D 论文的 LSJ-1024@512 协议，我们按各论文原始 800 短边配方在 1024 数据上重跑——与 CATNet/RS4D 行同属"各自配方+同数据同评测"口径，论文里注明即可。

## 3. Group B：Mask2Former（推荐方案 B1，备选 B2）

**B1（推荐）：mmdetection 移植版 @ cvt2**，配置 `mask2former_r50_8xb2-lsj-50e_coco`：
- 超参：AdamW lr1e-4、wd0.05、backbone×0.1、bs16、**50ep、LSJ 1024 crop**（与 RS4D 同款输入管线，对 WHU 1024 是原生分辨率）
- queries=100、ce 2.0 / mask 5.0 / dice 5.0、点采样损失照抄
- 单卡：bs2×accum8=有效16（1024² transformer 显存高，4090-48G 装得下 bs2；若余量大再 bs4×accum4）
- 预计：50ep×184 步×8accum×~1.2s ≈ **20-24h**
- 优点：零新环境、已验证解析通过；缺点：非官方仓（但 mmdet 移植与官方配方对齐）

**B2（备选）：官方仓 @ dt2**：maskformer2_R50_bs16_50ep.yaml，配方同上，另需编译 MSDeformAttn；若 B1 训练曲线异常再切。

## 4. Group C：AdelaiDet @ dt2（GPU 1，CATNet 完成后立即承接）

需先写 WHU 的 detectron2 数据集注册脚本（COCO json 三 split，注册 `whu_1024_train/val/test`，PIL 可读 TIF 无需转格式）。

| 方法 | 配方（官方仓 yaml 原样） | 超参 | 时长估算 |
|---|---|---|---|
| CondInst | MS_R_50_1x | SGD lr0.01@bs16、90k iter、steps(60k,80k)、warmup1k、多尺度短边640-800、MAX_PROPOSALS 500 | 90k 步：bs8×accum2@~800px ≈ **2-3 天** |
| SOLOv2 | R50（仓库只发 3x=270k，**建议改用论文同样报告的 1x**：steps(60k,80k)/90k，与 CondInst 调度对齐） | SGD lr0.01@bs16、warmup1k、多尺度640-800、bitmask | 1x ≈ **2-3 天**（若坚持 3x≈8 天，默认不建议） |

detectron2 迭代数照抄的口径说明：90k iter 在 WHU 上≈490 个 epoch，RS4D 一系文献复现也是这么跑的（他们 800ep 更长），保持照抄可比性最好。

## 5. 执行时序（默认排队方案）

| 时间 | GPU 0 | GPU 1 |
|---|---|---|
| 现在~8/26 下午 | RS4D 训练 | CATNet 训练 |
| 8/26 下午起（CATNet 完） | RS4D 训练 | **dt2 环境搭建 + CondInst 训练（2-3天）** |
| CondInst 完（~8/29） | RS4D 训练 | SOLOv2 1x 训练（2-3天） |
| ~9/2（RS4D 完） | Mask2Former B1（20-24h）→ Mask R-CNN → MS R-CNN → HTC → SCNet（合计~12h） | SOLOv2 继续/测试 |
| 收尾 | 各 test 模式跑 test.json 终评 | 同左 |

全部完成预计 **9月3-4日**。所有实验共用现有 runtag 脚本模板（smoke/train/resume/test 四模式 + GPU 监控）。

## 6. 待你拍板的三个点

1. **SOLOv2 调度**：1x（推荐，2-3天）还是仓库默认 3x（8天）？
2. **Mask2Former 方案**：B1 mmdet 移植版 @ cvt2（推荐）还是 B2 官方仓 @ dt2？
3. **Group A 输入口径**：按各论文 (1333,800)→800²（推荐，忠实配方）还是统一 LSJ-1024（RS4D 一系文献口径）？

批准后我将按顺序实施：建 dt2 环境 → 写 7 份 WHU 配置 + 注册脚本 + runtag 脚本 → 各自 smoke → 按时序挂正式训练。
