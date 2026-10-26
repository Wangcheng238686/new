# 正确 RGB-D E67：测试集固定检查点 A–E 干预

## 固定对象

用户在四臂train/test补充评价后明确要求固定正确RGB-D权重进行五组测试。
使用先前固化的E67（SHA `7e125debe1e76edd76fc15a2d17a770571f499b66c94f87211089634cbdc3557`），
本次再独立复制权重；不是当前滚动best，不重新训练，不按test选权。
全459test、FP32、关闭增强、eval/inference_mode、maxDets250。

|组|输入或路由改动|工程核验|
|---|---|---|
|A|正确输入|复用同权重完整测试记录，并单图复核预测逐值一致|
|B|源RGB替换为编码器自身的归一化均值|归一化RGB逐值为零；geometry/index/anchor仍为原对象|
|C|供体整组五视角RGB|固定test顺序循环移位229张，459供体映射预先保存；原目标geometry/index/anchor不变|
|D|正确RGB＋参考平面|使用独立sampling-origin水平面重建的全部fields/support/index/anchor；文件审计禁止实测cache读取|
|E|关闭UAV路径|enabled/internal均false，prepare_uav为空操作；不读UAV RGB、标定或任何UAV几何缓存，context/outputs始终为空，追加UAV候选数为0|

C是保留目标几何的外观错配诊断，输入在物理上不一致，不能称真实场景反事实。
D保留同一标定和RGB、联合替换所有实测深度派生输入；不能单独归因高度或某一字段。
E保留本联合检查点已共同训练的卫星网络和原有检测/掩码逻辑，包括原正式推理的score映射，
不是另训纯卫星检查点；关闭后的下降可能反映共同适应，不能独立证明全部UAV净收益。
原路径original_path仍含内部交互时的影响，不能把只去追加候选的original_path当作E。
合同核验仍可读取缓存manifest/BUILD_COMPLETE身份元数据，单独记录，不属于推理消费的几何NPZ。
输入审计同时包装OpenCV imread（原生I/O不触发Python open审计）；D禁止实测NPZ/EXR，E禁止UAV图像、标定和几何NPZ。

## 实现、调度与产物

新增 `scripts/analysis/ue_joint_counterfactual_20261026.py`，进程内包装已提交的补充评价Runtime，
复用其推理、COCO和只读状态核验，不修改原28个训练数值源或既有评价脚本。
运行根 `/data/wangcheng/checkpoint/ue_joint_counterfactual_E67_20261026`。

- `manifest.json`：同一权重、A记录SHA、固定供体映射、源码SHA和预定bootstrap设置。
- `groups/{A..E}`：每组合同；A引用同权重基准产物，B–E完整推理的records/metrics/complete；每组单图工程gate另存。
- 完成收据绑定实际输入文件SHA、2295个供体RGB（C）、原几何对象保留计数（B/C）、无UAV推理检查计数（E）。
- `owner`：GPU2共租，按free6144MiB准入（4096工作预算＋2048余量；原正式250实例门peak3230MiB＋context在预算内），B/C/D/E逐项串行，不抢占或自动重试失败任务。
- A–E全部就绪后CPU分析合并Mask/bbox/Boundary AP2%/1%，原路径Mask/bbox及平均预测/追加候选数。
- A−B/C/D/E的合并Mask AP采用500次同图bootstrap，seed20261021；每组前三次累积对原冻结COCO参考核对。区间不覆盖训练种子方差或多次探索。

所有组都保存相同GT/图序，并核对权重文件、参数/持久buffer摘要不变、梯度为空和零优化器更新。
单图gate只核接口与不变量，不报告为完整test收益；完整科学结论等待五组产物及配对分析。

```bash
python scripts/analysis/ue_joint_counterfactual_20261026.py prepare --root /data/wangcheng/checkpoint/ue_joint_counterfactual_E67_20261026
python scripts/analysis/ue_joint_counterfactual_20261026.py owner --root /data/wangcheng/checkpoint/ue_joint_counterfactual_E67_20261026
```

五组 `eval --group A --limit 1` 等单图gate需先完成，owner才允许启动。用cvt2环境，正式owner为子进程显式选择物理GPU2。
