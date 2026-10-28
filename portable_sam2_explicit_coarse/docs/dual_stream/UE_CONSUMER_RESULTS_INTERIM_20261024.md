# 已完成S1-R/N2-shared结果与J1状态

读取完整FP32 test预测与500次图像配对bootstrap。单seed、开发test，区间未做多重探索校正。单位pp=AP绝对值乘100。

| 组 | 正确训练Mask AP | 常量训练Mask AP | 正确−基座pp | 正确−常量pp及95%CI |
|---|---:|---:|---:|---|
| S1-R | .645546998 | .645536443 | +.003717 | +.001056 [-.003444,+.030141] |
| N2-shared | .645808654 | .645526765 | +.029882 | +.028189 [-.008727,+.157617] |

基座.645509830。两组未确认整体Mask AP净增益或训练RGB内容增益。S1-R的训练内容差在两个Boundary AP也跨0；与基座的BoundaryAP1%微升主要由常量臂也获得，不能归为正确RGB收益。

N2-shared有局部边界内容信号：正确−常量BoundaryAP2% +.122059pp CI[+.025265,+.175543]；BoundaryAP1% +.071617pp CI[+.047307,+.228741]。正确CP同权重correct−constant的Mask +.104636pp CI[+.005413,+.184981]，但donor点值.645896082略高于correct.645808654，因此这不证明正确配对RGB特异性。共享×内容交互在Mask及两个Boundary AP的区间均跨0，未证明共享接线优于旧N2。

S1-R点级统计：正确训练正确输入净改正29690点，常量训练24299点，正确CP donor28785点；这些是重复RoI/选点计数，不是AP。局部点改正明显但全图MaskAP几乎不变，不能把点级诊断包装为实例AP收益。

J1仍训练，读取时正确E6/update1570、常量E7/update1605（总2670），尚无E10完整评估；不得用训练loss比较增益。

旧N2训练及完整预测已完成，离线FP32梯度审计E1因共租显存不足而失败，原owner已退出，后续CPU比较被阻塞。独立repair_runtime.py在原产物根启动owner3782604，保留原失败日志/退出码，单GPU0任务且free≥20000MiB再准入，CPU分析单任务并行。E1/E10顺序补诊断，后续补齐分析，不改模型、权重、训练或数值source。S2DB其它owner另有zero-aux梯度审计OOM，不将它误称为训练失败。

产物分别位于ue_udpr_point_roi_20261023、ue_post_activation_shared_20261023、ue_joint_encoding_20261023；旧N2恢复证据位于ue_post_activation_20261023/repair_20261024。
