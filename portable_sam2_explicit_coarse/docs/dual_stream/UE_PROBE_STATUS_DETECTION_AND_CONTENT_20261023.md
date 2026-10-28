# 检测与内容探针阶段更新（2026-10-23）

## 1. 检测漏失：459图完整完成且独立验收通过

`ue_detection_attrition_20261023/verification.json`通过：44988 GT、40932最终DT，全部框/分数/类别及GT mask面积与基座一致；主FP32与独立pycocotools float64最终覆盖判定在.5/.75均0差异。

全部RPN回归框可覆盖44969/44988个GT（IoU≥.5），实际每层top-k后43049；最终任意框覆盖41456，一对一最大匹配容量40376。后两者不能混作同一召回定义。

最终3532个无任何框达到IoU .5的GT，按最后一次失去覆盖位置分：RPN top-k1838、RPN NMS128、RPN1000上限1、RoI回归395、分数过滤313、RoI NMS841、从未覆盖16；最终250上限0。RPN top-k及RoI NMS合计约76%。其中top-k1838有1833个mask-small，RoI NMS841有832个mask-small。

这支持优先检查预筛选排序和相邻实例保留，而非继续提高maxDets。覆盖存在不等于可靠候选；分类/定位会恢复或丢失覆盖，上述路径统计不保证修改单一步骤就提高AP。尚未新增UAV检测训练或报告检测AP收益。

## 2. 跨划分方向诊断已完成

A/D各train/val/test固定4图的结果在`ue_rgb_content_generalization_audit_20261023/RESULTS.md`。D的4个test图正确RGB相对constant/donor均使mask loss降低，平均−.004177/−.001627；A在test反而仅1/4有利，与train/val倾向不同。这支持路线之间有泛化差别，不能笼统称所有RGB只在训练有用。每划分仅4图、GT正proposal诊断，不替代完整AP。

已有全459净纠错仍显示D在大量小实例上改善微弱、正负抵消；两项证据应共同解释。

## 3. 新完成训练的实际任务指标

### N2-shared

正确RGB E10 Mask AP64.5809，纯卫星64.5510，匹配常量训练64.5527。

- 对卫星Mask +0.0299pp，名义95%区间[−0.0666,+0.1373]pp。
- 对匹配常量训练Mask +0.0282pp，区间[−0.0087,+0.1576]pp。
- 对匹配常量训练Boundary AP2% +0.1221pp，区间[+0.0253,+0.1755]pp。
- 对匹配常量训练Boundary AP1% +0.0716pp，区间[+0.0473,+0.2287]pp。

存在RGB训练内容的边界正信号，整体Mask尚未通过。固定CP正确对常量Mask依赖区间为正，但donor点值64.5896高于correct64.5809，不能称正确配对输入全面最佳。仍为单seed、开发test的多路线名义区间。

### S1-R

正确RGB E10 Mask64.5547，常量训练64.5536；内容差+0.0011pp，区间跨零。对卫星仅+0.0037pp；未形成有意义整体Mask增益。

### N2原版、D辅助

N2训练及完整预测已完成；一个独立GPU梯度审计失败阻断后续CPU AP统计。独立CPU恢复器已启动四项原定比较，不变更原估计器。
D辅助zero已完成E10；正确/常量仍在E10。三项zero离线梯度审计曾CUDA OOM，不能把失败说成正常完成。

## 4. 资源修复与保留证据

S²DB共租owner此前将FP32离线梯度审计误归3500MiB推理预算，多个审计并发造成OOM。只停止调度owner，训练子进程保留；旧state/log/amendment留档，失败任务保留previous_attempts后重排。新增offline_gradient预算20000MiB及既有2048MiB余量；原模型/训练/评价数值source及检查点不改。新owner实际重跑后zero E1已完成，E10继续。

`ue_post_analysis_recover_20261023.py`只恢复N2已具备全部依赖的四项CPU比较，两个线程上限；原GPU拓扑诊断失败仍单列，未伪装为整体完成。其恢复状态保留在原研究state及独立日志。

修复只涉及资源调度，不能从OOM推导模型方法失败。
