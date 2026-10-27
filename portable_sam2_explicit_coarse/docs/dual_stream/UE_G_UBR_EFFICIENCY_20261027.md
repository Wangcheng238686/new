# G＋UBR实现效率优化与三臂重新发射

2026-10-27。用户要求先明确效率方案、派子代理实施和主代理审核，完成后停止旧两组UBR并重新拉三组。只做实现优化，不以缩网格、删候选、改loss或裁窗替代。

## 固定最终实现

- Context保留完整当前预测池、source调用边界和native32解码批。只贴概率，不将32通道特征无用地贴到512²后丢弃。
- 单图邻居binary sum仅算一次，按稳定candidateID减自身；对<=250个bool预测，FP32整数求和/减法精确等价。没有增加GT邻居或新的NMS。
- 训练native解码仍N1原序，UBR **训练microbatch1**；省掉UBR内部嵌套checkpoint，保留完整decoder→paste→UBR→loss checkpoint。
- 每次active计算图闭包缓存首次train-mode forward的detached SDT/band，重算复用。只算一次精确scipy EDT，不偷用eval context的mask、不跨sample/update缓存。
- 推理native按原source/chunk32一次解码，缓存128² native logits及32ch特征（250实例FP32特征约500MiB）；UBR **推理microbatch4**，避免第二次SAM2解码。
- 保持512²目标网格、band8像素、每图64匹配预测上限、原native mask和coarse/检测/实例损失、UBR .1及其整图非空band实例分母。模型参数对象/形状和初始化不变，微批不改变GN逐实例定义；浮点舍入以真实门审查，不宣称整个训练轨迹bitwise不变。

## 实测依据与边界

实际profile修正了静态推断：邻居求和、featurepaste不是首要耗时。旧FP32的64active诊断前反传约15.48秒；同步分段SDT forward/backward各约1.93/1.96秒、native decoder约2.09/1.60秒；neighbors合计约.155秒、featurepaste约.188秒。分段为inclusive重叠时间，不能直接相加当百分比。

探索门attempt3：AMP仅UBR目标前反传 old约17.35秒，fast train1约11.38秒（约1.52×）、train4约11.34秒（约1.53×），但AMP tail reserved为3468与7000MiB，fullobjective train4峰值10122MiB。故最终保train1/infer4，micro4训练没有收益却阻碍三组共租。这里仅UBR目标诊断包含真实UAV/G/SAT特征和梯度，省略原task losses，不能冒充完整global8训练速度。推理250旧/新非零输出exact binary，概率max约1.2e-7，探索提速1.25–1.29×。

attempt1受共租诊断cap7040限制，在FP32微批4反传触发本进程allocator OOM，旧任务未受影响。attempt2的AMP tail-only checkpoint受gate测试顺序/weightcache影响失败；gate已更正完整目标为真实主loss→实例→UBR顺序，tail-only诊断禁AMP weightcache，正式完整目标仍cache=True。均保留原失败产物，不把诊断故障称模型精度退化。

最终数值源变更后另做attempt4，不将探索门通过标记冒充最终源码验收。最终门同时比较FP32/AMP loss、全部梯度、ID/assignment/context/band；250自然池zero/nonzero推理；两次四rank完整目标AMP更新含空rank，检查参数副本SHA、正式训练及FP32推理峰值。它是单批工程证据，不是AP或多训练种子重复性证明。

## 源文件

|新增文件|职责|
|---|---|
|rsprompter/uav/ubr_detector_fast.py|独立fast controller，复用相同refiner参数/capture hook|
|tests/test_ubr_detector_fast.py|非零loss/梯度、分母、checkpoint SDT缓存、eval批次、empty行为|
|scripts/probes/ue_ubr_fast_gate_20261027.py|四卡真实工程等价/分段profile/全目标资源门，保存实际全repo依赖源|
|scripts/probes/ue_ubr_fast_runtime_20261027.py|原通用初始化后替换controller，参数对象不变|
|scripts/probes/ue_ubr_fast_run_20261027.py|进程内复用原训练engine、独立配置/准入/源合同|
|configs/ablations/ue_g_ubr_fast_protocol_20261027.json|新三臂固定10轮与train1/infer4实现合同|
|scripts/probes/ue_ubr_fast_owner_20261027.py|新三组四卡共租/依赖/test调度，真实full-repo闭包与峰值预算|
|scripts/analysis/ue_ubr_fast_restart_audit_20261027.py|归档旧状态、严格核PID/starttick/命令后停旧owner及两组，保留全部产物|

旧在训源均未热修改；新runner通过进程局部绑定复用oldengine，不改变别的进程全局变量。正式source需全部Gitblob一致后准入。

## 旧实验归档与恢复

旧根`/data/wangcheng/checkpoint/ue_g_ubr_short_20261027`完整保留。G_no_tail已真实完成E10/2670，其完成CP SHA与全部94实际数值源均匹配历史Gitblob。旧owner误报failed来自漏记SAM2 prompt_encoder/transformer延迟依赖，不是源漂移。

实现、41项CPU组合门及独立审查完成后，按用户指示安全停止旧owner和两组UBR，8rank确认退出，归档`retirement_for_fast_restart_20261027`。为释放FP32梯度检查显存，这一步发生在最终attempt4之前；新三臂发射仍必须等待最终四卡门通过。

新根`/data/wangcheng/checkpoint/ue_g_ubr_fast_20261027`，三臂G_no_tail/G_UBR_geometry/G_UBR_zero_geometry全部从相同通用common_init重跑，不接旧半程权重；相同seed44、样本顺序、10轮2670更新、每实验4卡batch1×accum2=global8、原100轮LR日程前10轮、增强/EMA/FP32评价不变。每epoch保存last，E1/E10完整459val，三臂全E10后固定E10完整test，不因test改预算/选权。

新owner按正式AMP全目标+FP32推理实测最大reserved峰值＋context512预约，每卡另留2048余量，最多三组且按四卡共同余量准入；FP32反传诊断峰值单列，不误当正式训练峰值。已完成参考臂也重新训练用于新实现同合同对照。没有新增AP收益结论。

## 最终验收（attempt4，主代理复核）

有效门`/data/wangcheng/checkpoint/ue_ubr_fast_gate_20261027/attempt4/complete.json` passed=true。四rank finaltrain1/infer4：FP32 loss与梯度在锁定容差内，AMP全部383个梯度张量逐值相同，loss误差<=3.73e-9；ID/assignment/context/band exact。250自然pool非零推理binary/ID/框/分数 exact，概率maxabs1.19e-7；zero概率exact。pool真实含UAV来源45/48/54/63个，补上之前foundation池满SAT时未覆盖的来源分支。

无插桩仅UBR目标前反传提速1.45–1.46×；FP32完整250推理提速1.24–1.27×，不能称整轮训练等比加速。正式完整目标AMP每rank1图无累积的两次工程更新含rank3空例，参数副本SHA一致；正式执行峰值6550MiB（约6.40GiB），FP32梯度诊断峰值4350MiB单列。新owner每组预约7062MiB、全卡再留2048，允许三四卡组共租。

CPU41项组合测试主代理重跑通过，旧93份冻结数值源全部未改变。新final源全repo实际import闭包及快照在attempt4目录，正式发射前要求提交Git、通过fast运行器恢复门；后续运行状态见新root owner_state/rank_launches/progress。

## 实际发射与来源核定

数值提交`246e4aa32db80572bf387545c5bb17bcdb7f2cdf`。fast运行器独立四卡恢复门全部passed（model、显式synthetic AdamW moment、scaler、5类RNG next-five exact），产物`/data/wangcheng/checkpoint/ue_ubr_fast_resume_gate_20261027/resume_gate/G_UBR_geometry/complete.json`。

新owner1485153已实际启动三组四卡，共12rank：geometry launcher1485368/ranks1485379/81/82/83，no_tail launcher1485369/ranks1485395/96/98/99，zero_geometry launcher1485370/ranks1485386/87/88/89。独立发射审查核PID/startticks/rank/world及初始化/合同均一致；三臂97份实际数值源为owner100来源的精确子集，SAM2延迟依赖完整，全部与上述Gitblob一致。owner状态running3/failures空，各卡总占用约19–20GiB，尚无报错。

初始发射不等于训练完成或AP收益。后续最新状态看`/data/wangcheng/checkpoint/ue_g_ubr_fast_20261027/owner_state.json`及各臂progress/epoch/complete；日志25更新采样，最后训练记录250后可能是本轮最后17更新或完整val阶段，不能仅凭该记录判定卡死。
