# 独立训练的 1／3／5 UAV 视角敏感性接线

日期：2026-10-28。状态：接线与 CPU 验证完成；未发射新训练，真实四卡 SUTC 门仍待执行。

## 配置与模型

`ue_ubr_native_sutc_runtime_20261028.build_runtime` 新参数：

```python
# 在 scripts/probes 可导入的环境中调用；这是模型工厂，不是训练命令。
rt = build_runtime('C', view_count=3,
    subset_cache_root='/data/wangcheng/cache/ue_v1_view_subsets_20261028')
# 或显式指定 uav_view_ids=[0, 144, 288]，两种参数互斥。
```

| 预设数量 | 固定 yaw ID（度） |
|---|---|
| 1 | 0 |
| 3 | 0、144、288 |
| 5／默认 | 0、72、144、216、288 |

支持指定其他非空、无重复合法 ID 子集；内部按采集 yaw 的标准顺序排序。不是根据 test 表现选择组合。预设组合不能代表所有拍摄方向的均值或变异。

S 保持纯卫星，不构造 UAV loader/分支，审计有效视角数为 0。U/C 的上游与 T/C 的尾部使用同一组实际所选视角，不允许一路三视角、另一路暗中保留五视角。

网络仍为五个 canonical 源槽位，源 token ID 编码仍是 `slot*16384+token`。**只读取、编码所选 RGB**，未选槽位直接放零，不进编码器；候选引用未选槽位立即报错。保持共享参数、LoRA、八候选上限、每视角两候选与源 stride4 不变。保留五槽不是对未选视角进行隐式编码，也没有新增参数。

`geometry.shape=(1,5,512,512)` 的 5 指 height/dx/dy/viewcount/support 五种几何通道，**不是五个视角**。这些通道对每个子集重新生成；viewcount 保持原始 `/5` 固定缩放，分子仅含实际选中且位于最近表面1m内的相机。一个观察者始终编码为 .2，不随子集数改单位。

新增 `SubsetNativeEvidenceG` 只替换 G 的源读取，保留原几何 CNN、写回 hook、交互、实例目标与补框。T 的 `TailP0Evidence` 使用同一源读取函数，不引入上游路径。默认五视角 U/C 继续走原 G 源码；T 的五视角读取保持原归一化/采样顺序。

## 深度与缓存

新增 `data/ue_view_subset_inputs.py` 及 `scripts/probes/ue_build_view_subset_cache_20261028.py`。

1. 只读取所选 yaw 的原始 EXR camera-Z（R 通道 cm 转 FP64 m）与相机参数。
2. 未选深度槽在投影之前置无效；从原始所选点重新去重、排序、每视角截断及全局 top8。
3. 重新生成 FP64 源/卫星锚点、最近表面高度、dx/dy、实际覆盖、视角计数。上游7维候选几何再从这些锚点计算，包含重算的最近锚点深度差。
4. 不过滤旧 top8 充当重建，不复用五视角高度/支持缓存。不能靠过滤找回已经被旧 top8 挤掉的候选。
5. 缓存按精确子集目录隔离，并记录 data root、split SHA、代码 SHA、所选原始深度与相机 SHA、全部图片 ID 及每个 NPZ SHA。已有不一致产物拒绝覆盖；恢复时从原始输入复算并逐数组核对。
6. 正式 runtime 必须有该子集完整 train/val/test 缓存；缺失立即失败，不自动回退。`--ids` 生成的工程子集不能用于正式 runtime。

全量缓存准备（CPU，不启动训练）：

```bash
/data/wangcheng/envs/cvt2/bin/python \
  /home/wangcheng2021/project/new-uav-dual/portable_sam2_explicit_coarse/scripts/probes/ue_build_view_subset_cache_20261028.py \
  --cache-root /data/wangcheng/cache/ue_v1_view_subsets_20261028 \
  --view-count 3
```

1／5 视角分别替换最后的数量。也可用 `--view-ids 0 144 288`。单个 builder 串行构建并持有子集锁，不抢占 GPU；全量缓存本轮未构建。

## 证据与限制

38 项 CPU 测试通过：包含开关/消费回归，五视角旧数学等价，删除深度的不变性，top8 重建能恢复遗漏候选，未选 RGB 不被编码且无梯度，G 单视角与旧读取相同有效锚点时等价，数量配置路由及缺缓存提前拒绝。

真实 train 图 001234 的三个子集缓存与加载已完成：

| 数量 | 保留候选 | 支持卫星像素／262144 |
|---|---:|---:|
| 1 | 21425 | 145316 |
| 3 | 58316 | 224727 |
| 5 | 91150 | 241881 |

五视角新缓存的 index、FP64 anchor、geometry5 与现有旧缓存逐值一致；实际 RGB 文件读取数分别为 1／3／5，工程局部缓存被正式 loader 正确拒绝。这是一张训练图的接线证据，不能外推覆盖单调性或 AP。

收据：`/data/wangcheng/checkpoint/ue_native_sutc_engineering_20261028/view_subset_CPU_gate.json`，缓存在同目录下 `view_subset_cache/`。门脚本 `scripts/probes/ue_view_subset_cpu_gate_20261028.py` 记录代码和旧缓存 SHA。未执行真实模型四卡前反传、正式250实例资源门、完整缓存构建或任何敏感性训练，不能据此宣称训练/AP已验证。

后续各数量从同一 generic 初始化独立训练，预算、四卡/有效batch、增强、选权、全val/test协议相同；输出目录和续训合同必须包含所选 ID 与缓存 manifest SHA，禁止不同数量复用 optimizer/已训练检查点。当前旧 ABC wrapper 不接受该配置，不得直接改参数冒充新实验。
