# 同CVT池化与固定重建尺度探针：独立审核

审核者frozen_probe_audit，不参与extractor/fit/summary开发；只写/tmp。

## 定版静态实施审核

VERDICT：PASS，按root明确委托准唯一3train+3validation特征smoke GPU0，900秒/20GiB/5GiB。当前不准full281或fit；后续分级准入。

同一E72 strict EMA最后CVT256，source/pixel/massquality均相同，调用生产投影32²与128²。coarse先bilinear升128 align_corners=False再shift；fine原生128。四图按同fine D支持与roll32交集I掩码，公共fine q/c同I，同stride4 RoIAlign16、同seed1813压缩；粗细输出位置支持匹配，但粗格/context和重建混合不相同，不能称纯native尺度因果。

核对checkpoint SHA/architecture与上一轮精确一致，三个固定矩阵重生SHA等旧；源old cache SHA和GT box/slot/image及smoke实际globalid/mask对账。公共sat/extra/prob/target/band/身份逐字节复用，geometry前5旧变量、后2新fine q/c，必须重fit全部SG/S64基线，不拿上一轮分数相减。Dfine mask及smoke CPU两尺度bincount mask对生产q>0精确核对。actualROI flags、allshape/finite/range、读取CVT/input/参数hash与correct hookon/off完整features checks均已落实。

smoke保存native32ROI与固定重建128ROI差，仅描述，不按其选择插值或新增阶段。roll32finecell固定物理128像素两轴，wrap是人为OOD。无GT选source、channel或支持，不后筛ROI。

## fit及summary静态审核

已逐diff对比上一轮已审实现：仅目录及U/UX→F/FX字段替换，C coarse缓存命名沿cvt32但manifest明确已重建新表示。固定12头、同架构配对初始化、相同sample seed1831/1000steps/64batch/AdamW/last不变；train-only norm、身份-only前检、全部12fit结束后才valtargets评估保持。

主nonlinear四contrast仍C/F各对S64与本stage错位，幅度.001及98.75%图像sum/count CI下界>0；F−C95%仅描述，fine_conditionally_better要求F两支持门及预定描述区间正，不绕容量/空间对照。linear伴随不能择优；没有新超参或效应门。新6cache验收及同字节正式协议copy后才可准唯一32step train-only timing，fullfit/root另准。

## 实际运行环境阻断与可恢复状态

最终状态：**静态实施审核PASS；smoke尚未完成，结果审核PENDING。环境阻断，不是工程效果FAIL，也不是整个任务完成。**

root已批准唯一6图特征smoke，采用GPU0 wrapper allocator上限20%约4.9GiB，与UE驻留共享；无fit或full提取准入。runner报告PID1157297初始化约1分31秒仍处Dl、wchan=wait_on_page_bit_common，GPU仅context约254MiB，无manifest/log或可解释cache。root同时核到内存约109/125GiB、swap34GiB、I/O full avg10约85%，有明确内存/磁盘压力证据；因此指定超过180秒无输出只停止自身smoke PID，不碰UE、不重启或转CPU拟合、不追加任何GPU尝试。

本审核不把该初始化停滞当作模型/候选/效应门负结果。上轮stage最终PASS已完成关账，不能与本轮未验收混同。当前保留协议、实现、源SHA与原运行日志，归档口径为“已实现、静态审通过、验证未完成”。若最终停进程状态/退出码由runner随后反馈，应补记录而非推定已经停掉。

恢复后待办依次为：在root重新准入资源条件下完成原唯一6图smoke；独立验源SHA/旧REUSE逐值/两尺度Dmask及共同I/无扰动/预算；再固定正式协议同字节并准train-only32step timing；root另批full281提取、验全部cache后再批固定12头fit；最后完整结果独立审核。期间不调整主指标/四门/矩阵/seed/超参，不根据旧开发结果选择新设置。

本轮未产生任何可报告的Brier/IoU/AP或尺度效果，也未授权生产训练/模型改动。

## 运行竞态更正与实际六图验收

上述“无产物/阻断”是初始化等待时的阶段性状态，现已被实际exit0及工件覆盖。原唯一smoke在elapsed约3分05秒检查时已输出train样本，runner按root停令TERM自身PID1157297返回No such process，原因是原进程已经自然完成；session19830 exit0。没有新run、没有杀其他进程、UE未动。约150秒初始化I/O等待与manifest内提取36.1476秒应分开报告，不能把前者藏入“36秒完成全部启动”。

VERDICT：六图smoke实际结果PASS。完整提取与fit仍未准入，效果结果PENDING。

审核者亲自CPU逐六cache核SHA/source/protocol、240RoI（train195/val45）、所有形状/finite/range和实际globalid隔离。旧sat/extra/prob/target/boundary/身份逐数组精确复用，geometry前5逐值相同、后2为新fine q/c；common coverage等于公共geometry最后字段且≤original coverage。四C/F原/shift共同支持零目标全部为0，train/val visible建筑无交叉；checkpoint/architecture/三个投影矩阵身份等旧。

运行记录correct hookon/off全部features精确、源CVT/input/payload/参数version不变、两尺度CPU Dmask核验、hooks恢复均通过。实际RoI flags aligned=True、sampling_ratio=0、spatial_scale=.25。6图共同支持零数量0/0/5/1/1/2，主分母仍保留；native32ROI与重建128ROI的relativeL2约.0162–.0366仅路由描述，不能据此调插值或挑样本。

manifest内部36.1476秒、peak allocated2.1296GiB、cache37.6219MiB，满足smoke预算。不能将含额外on/off前向的六图时间机械线性外推full；也不能因模型峰显存可容纳就忽略host内存/I/O。

资源建议：**保留已通过smoke，等待资源压力下降，再由root重新批准完整281图；当前不建议共享启动full或fit。** root最新观测I/O full avg10约72.6%、host内存111418/128550MiB、swap35727MiB，且本次已发生约150秒初始化page wait。full预期约10–15分钟、缓存约2.6–2.7GiB，但页回收/磁盘争用会影响UE与900秒操作预算，smoke模型2.13GiB不能证明host有余量。无需新CPU拟合/重试GPU来证明环境问题。该建议是基于实际资源与预算的准入判断，不是尺度效果FAIL。

后续待办仍为root另批full、独立cache验收、同协议train-only32step timing及再批12头fit/最终结果审。此次审计未新增GPU或任何fit。
