# 检测阶段漏失定位探针

## 目标与固定合同

用户授权补充检测链路诊断。固定EXP-0017 E73纯卫星检查点、v1开发test459图、FP32/eval、无增强、无训练或参数更新。已有UAV探针的框/分数保持一致，本探针首先定位卫星检测链路的损失位置，不直接检验UAV检测增益。

独立入口：`scripts/probes/ue_detection_attrition_probe_20261023.py`。
产物：`/data/wangcheng/checkpoint/ue_detection_attrition_20261023/`。

## 实际阶段

1. 全部未回归RPN锚框。
2. 全部RPN回归框，尚未每层top-k。
3. 实际每层前1000个候选。
4. RPN尺寸过滤（宽/高严格大于0）。
5. 按特征层分别进行RPN NMS，IoU阈值0.7，尚未全图截断。
6. RPN全图最多1000个候选。
7. 实际RPN候选经过RoI框回归，尚未筛前景分数。
8. 前景分数严格大于0.05；不另加前景argmax判定。
9. RoI NMS，IoU阈值0.5，尚未截断。
10. 实际最终最多250个框。

RPN/ROI的实际返回值与独立回放逐值核对；各图最终框/类别/原scores与已有459图纯卫星基座记录逐值一致。工程门另外在实例数最多的test图回放原完整mask推理，核对最终掩码也完全一致。探针不修改阈值/NMS/预算。

## 指标与归因边界

- bbox IoU阈值提前固定为0.5、0.75。每阶段保存候选数、每GT最大IoU以及一对一最大匹配容量。
- 任意候选覆盖的分母为全部GT；相邻建筑可能共享同一个好框，因此额外以稀疏二部图最大基数匹配统计容量。只用容量总数，不将不唯一匹配身份用于尺寸归因，也不称COCO按分数匹配召回/AP。
- 分组按真实mask面积small<1024、medium<9216、large；密度固定<=50、51–100、>100。mask面积分组不同于旧canonical evaluator的bbox面积。
- 逐相邻阶段同时统计失去覆盖和恢复覆盖；框回归可能改变覆盖，不将全链路强制解释为单调漏斗。
- 对最终无覆盖GT，记录最后一次有覆盖后在哪一步丢失；从未有覆盖者单列。这是路径描述，不是“只改该步骤就能恢复最终AP”的因果证明。
- 最终框没有任何GT达到指定IoU、以及最大匹配容量剩余，是背景/重复/定位误差代理，不等同COCO FP类别。不把bbox高IoU失败直接称背景误检。
- 对每GT额外记录RoI回归后达到IoU阈值的候选最高前景分数，帮助区分框存在但分数不足与框定位不足。

## 工程验收与运行

1. 两个相同GT、单个候选的合成例：任意覆盖2、一对一容量1；空候选容量0。
2. 实际RPN与RoI回放恒等；基座GT/框/类别/scores身份核对。
3. 最大实例图原完整mask推理逐值回放，测allocated/reserved峰值。
4. 每阶段子集链的任意覆盖和最大基数容量不得增加；两段回归单列恢复。
5. 最终漏失分类计数守恒；完整模型参数/buffer SHA与41份既有冻结source及实际MMDetection/MMCV方法源码SHA核定。
6. 持久owner先工程门，再重新读取各卡空闲显存，按实测峰值+2048MiB共租准入；保留其它训练。独立输出progress/state，不新增训练逐步遥测。

正式命令：

```bash
/data/wangcheng/envs/cvt2/bin/python scripts/probes/ue_detection_attrition_probe_20261023.py
```

`gate/`只代表工程门；根目录完整459图`complete.json`及`RESULTS.md`才代表本轮完成。开发test已反复诊断，不作为独立最终泛化认证。依据结果再讨论UAV进入RPN前FPN，还是进入候选分类/定位阶段，不能提前宣布某入口有效。

## 启动验收

最大实例图（dataset index223，238个GT）的工程门已通过：RPN/ROI回放、原完整mask推理、基座检测输出、冻结SHA与源码均一致。峰值reserved2406MiB、allocated1883.7MiB。耐久owner随后重新准入GPU3，共租执行459图；运行进度见progress.json。

新增独立CPU验收入口`scripts/analysis/ue_detection_attrition_verify_20261023.py`已启动，等待根目录完整产物；将复核459图/44988GT/40932最终DT身份、GT真实mask面积、来源SHA和计数守恒，并独立用pycocotools float64重算最终任意覆盖，公开与主诊断FP32在阈值附近的差异。`verification.json`才代表这项独立验收结束，进程启动不等于验收完成。
