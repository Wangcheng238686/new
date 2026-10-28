# Native适配器卫星query×几何四格诊断

目的：固定geometry训练E10权重，区分新增卫星局部适配、几何字段及公共偏置的即时作用。不新增训练。
保留原卫星/SAM2/UDPR全部路径；“query关闭”只将新增适配器的query1×1投影输出32通道置零（含其bias），不是移除原卫星模型输入。
几何关闭仅置零四字段；mixer卷积bias、alpha和UAV支持全保留。
四格：query on/geo on（已有correct）、query on/geo off（已有zero）、query off/geo on（本轮correct）、query off/geo off（本轮zero，bias-only）。
无RMS匹配，本轮判断各路径的原始固定权重函数作用；结合已有RMS匹配结果解释幅度限制。
原始计划和源码SHA在`/data/wangcheng/checkpoint/ue_native_query_factorial_20261022/plan.json`，结果前固定。
FP32，完整test459，同一检测框/分数/GT逐图核对，卫星state SHA前后一致。
两条预定Mask AP比较：geometry-only−bias-only、satellite-query-only−bias-only，各500次同图配对bootstrap。
query关闭属于分布外干预，不能代替独立训练；bias-only仍带UAV支持，不是纯卫星容量训练控制。
运行产物与最终统计不得混为一谈。原空间五项诊断仍独立运行完整Boundary AP及验收。

## 最终核定

本问题已完成固定模型干预及交叉验收，最新结论见[最终可信结论](UE_NATIVE_CREDIBLE_CONCLUSION_20261022.md)。早期“几何被利用”不得再解释成已证实的正确几何推理净增益；新增query适配作用已由四格核定。
