# 同CVT投影尺度检查：实施与六图验收

2026-10-15，EXP-0006/0007，冻结E72 strict EMA。**状态：实施静态审核与6图提取smoke独立验收PASS；完整281图和小头拟合尚未运行，暂无尺度收益结论。** 原双流生产模型/配置/遥测均未改。

## 检查设计

同一learned CVT256、相同project_source/pixel/mass/quality，分别生产投影stride16(32²)与stride4(128²)。粗32以bilinear align_corners=False重建128；四路粗/细原位及错位均在128坐标统一乘I=(fineD>0)&roll(fineD>0,32,32)，再同stride4 RoIAlign16²与同256→32 Gaussian矩阵seed1813。公共quality/coverage取fine同I。错位两轴128卫星像素，循环roll仅人为对照，非真实标定。

共同I控制输出曝光位置，但粗池化及bilinear仍可能包含mask外上下文；不是内容来源完全相同或原native32入口逐值等价。公共sat32/extra32/prob/target/boundary/boxes/身份从上轮严格SHA的281缓存逐值复制，geometry前5保留、后2替换，不把上轮结果作本轮基线。

SG/S64/C/CX/F/FX六cell、Linear参照与20513参数Nonlinear主，共12小头；固定128train/153val、seed1831、imageuniform再ROIuniform、batch64、AdamW1e−3/1e−4、last1000steps、train-only normalization，不使用旧读出权重。训练未clip残差的MSE，末次clip(p+f)评16²occupancy Brier；边界/二值IoU/支持/尺寸/建筑等权/clamp为伴随。

四主比较C对S64/CX、F对S64/FX，均>=.001且四比较Bonferroni98.75%图像配对CI下界>0，每stage两门均过才支持。Fine−coarse方向为coarseBrier−fineBrier，95%区间只描述；只有F两门过且direct区间下界>0才可称该匹配读出条件细尺度更好，不证明真实AP或不可逆信息损失。所有门/seed/预算沿上轮，不按旧成绩改。

## 六图实际验收

固定3train(7/15/25)+3val(287/288/295)，共240ROI。以下经独立CPU与主代理检查：

- 全部source/protocol/旧cache/新cache/矩阵SHA及实际GT身份匹配。
- 旧9项复用字段、geometry前5逐值一致；两尺度Dmask CPU核、同I/采样/缺失域与范围合法。
- correct hookon/off全FPN/SAM/PE/highfeatures逐值exact；CVT/输入/payload不变、参数版本不变、hook恢复。
- 共同支持零ROI对应四个UAV视觉块全零，全部ROI保留。
- 提取timer内36.1476秒、allocated峰2.1296GiB，6cache共39,449,438字节；manifest complete=False是smoke按设计标记，不是遗漏正式完成。

原native32 ROI与固定重建128 ROI relative L2为.0162–.0366（约1.62%–3.66%），仅界定路线差，不据此调插值或选图。不能把此误差当分割退化。

## 资源事件与当前停止点

GPU上四个UE渲染进程各约11–12GiB，smoke使用GPU0 allocator20%约4.9GiB硬cap，不修改UE。初始化期间本探针出现Dl/page wait、日志暂空、GPU仅context约254MiB；主代理核整机内存约109/125GiB、swap约35GiB、IO full压力约72–85%。随后原进程解阻并自然完成，TERM仅针对自身PID的尝试返回No such process（已退出），session19830 exit0。没有成功发送终止信号、没有重启、没有清理/终止UE。timer外初始化等待约150–160秒为估计，wrapper无阶段时间戳不能报精确值。

独立审核和runner均建议在当前实测资源压力下不启动完整281提取或读出拟合。虽然显存峰可容纳，不能据此保证host/磁盘/900秒wall预算；这是真实资源准入未通过，不是工程效果FAIL。没有任何scale Brier/IoU/AP结果，不编造门判定。

## 下一步恢复

环境压力回落后，先核源SHA/资源条件，不必重跑已通过且source不变的smoke；按审核准入进行train-only32step计时，再单次full281冻结提取/全部cache验收，再固定1000step读出与唯一末次val，最后独立结果审查。共享渲染时保留allocatorcap并将初始化纳入外部wall预算。若正式四门不过，按协议关账，不通过加seed/宽度/步数寻找阳性。

[预注册](probes/results_20261015/PROJECTION_SCALE_PREREGISTRATION.md)、[smoke manifest](probes/results_20261015/projection_scale_smoke_manifest.json)、[独立审核全文](../reviews/UE_PROJECTION_SCALE_20261015.md)、[恢复步骤](probes/results_20261015/PROJECTION_SCALE_RESUME.md)。协议原文含草案标题，保留执行前SHA；实际状态以manifest与本报告为准。

精确脚本在scripts/analysis/ue_projection_scale_{extract,fit,summary}.py，沿固定/tmp/helper依赖，非通用入口。缓存与完整资源/进程/log在`/tmp/ue_projection_scale_probe/`；大缓存不进git。sourcegit=5767ee2（上轮独立关账），如smoke记录不同以运行manifest为准。
