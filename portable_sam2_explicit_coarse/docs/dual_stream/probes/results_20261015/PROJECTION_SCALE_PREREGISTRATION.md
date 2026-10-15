# 同CVT投影尺度读出诊断：统计预注册草案（禁止未审执行）

固定E72 strict EMA correct双流表示；同128train/153val、12621/4496GT RoI、同global_id隔离。沿上轮16²occupancy任务、6cell、12读出、seed1831/同初始化同采样表、不因上一轮成绩更换超参。旧fixed Gaussian256→32 seed1813不改；公共卫星512→32 seeds1811/1812不改。32通道压缩阴性不能否定完整256信息。

唯一轴为相同learned CVT最后BEV256经过production stride16与stride4投影。细/粗source/pixel/massquality完全相同，区别target归桶/池化尺度。root已定共同坐标方案：C32 bilinear(align_corners=False)固定重建到128，F128原生，四图在128层aligned/roll(32,32)之后都乘I128=(Dfine>0)&roll(Dfine>0,32,32)，同stride4 RoIAlign16；shared finequality/coverage同I，6cell同geometry。这个对照是“池化加固定重建的可读性”，不是纯native32消费因果；粗细信息本身也有支持混合，不能将输出高分直接归因真实细边界。

公共geometry的q/c已变，SG/S64必须按本轮重新拟合，旧成绩不可直接相减。旧sat32/sat_extra32/prob/target/boundary/boxes/globalid与新提取逐值对账复用，不重算标签或改变卫星PE/读出sampler。原/commoncoverage新共同支持字段只描述；全体主包括I无支持RoI，不按支持后筛。

## 六cell和读出（冻结沿上轮）

SG=sat32+geometry7，第二视觉block零且明确有效变量较少；S64=sat32+extraSat32+geometry7；C/CX=sat32+coarse/shift32+geometry7；F/FX=sat32+fine/shift32+geometry7。input统一71。C/F同Gaussian与相同重建后采样坐标，SC是同有效通道容量对照但卫星PE和多层路径语义不同，不能当严格语义等价。

Linear=Conv1×1(71→1)；Nonlinear主=Conv3×3(71→32,pad1)+GELU+Conv1×1(32→1)。AdamW lr1e−3 wd1e−4，last1000steps，batch64，imageuniform再ROIuniform有放回，seed1831，6cell同arch init逐值相同、同采样序列，不共享拟合weights。train channel population norm std floor1e−6，normalization只能训练。目标residual GT16−p_sat16，训练未clip MSE；末次一次val输出clip(p_sat+f,0,1)。固定输出unclip Brier/clamp比例与trainloss100step趋势，不按val选epoch/seed/架构/width，不续训。

prefit仅允许val身份字段读入核image/slot/globalid/boxes/SHA与隔离，valtarget/features只能所有12headlast训练结束后一次载入/评估。不正式更新网络/无AP。提取资源预算≤15min峰显存20GiB/cache5GiB；train-only32steps timing审后一次确认，12headfit≤15minGPU1，不看val调预算。

## 唯一主任务和判定（root已定）

主每ROI16²softoccupancy Brier，全4496RoI等权。GT64 area16，边界固定7²pad0 dilate XOR erode再area16仅伴随；binary16 IoU、size/support/建筑等权、unclip/clamp同时报告，不换终点。

固定保留四主支持contrast：CvsS64/CX、FvsS64/FX，controlBrier−stageBrier正为好，各须>=.001且Bonferroni四比较98.75%CI下界>0（1000 imagepaired sum/count，seed1830，quantiles .00625/.99375）。stage两个门同时通过才称预算匹配的可读互补支持。F−C固定定义coarseBrier−fineBrier（正为fine更好），固定95%imagepaired区间，仅定位描述。只有fine两支持门均过且FvsC描述差正向才允许“本条件更细投影可读性更好”，不能单凭fine>C绕过容量或空间对照门；绝不据此称生产分割/AP必提高。不增加第五门，不把该描述差当生产pool瓶颈因果。

Nonlinear是唯一主，Linear同优化预算参照不替代门。四门失败则停止此设置，不换超参、随机矩阵或验证子组重试。.001沿上轮工程最小支持线，不代表16²对真实512边界一定有意义。图像bootstrap不覆盖空间/重复建筑/选checkpoint/单seed；开发集已反复检视，非独立test。

## 失败/验收

所有缓存源/矩阵/协议SHA、GT身份、公共sat/标签逐值对账、fine/coarse geometry分母、commonmask以及shift相同、不改模型参数/hook/输入/sourcefeatures。真实投影生产对账由独立审查，读出统计由主代理独立重算。只/tmp工件，不改工作树。先协议与代码静态审，再提取smoke、trainonlytiming、完整提取、正式fit各自准入，失败保留证据不借失败改效应门。

## 工程定版：同CVT池化+固定重建读出

extract_projection_scale.py仅抽E72 strictEMA correct UAV冻结特征，不创建optimizer/不反传、不AP；未准入前不执行。same281 imageIDs继承/tmp/ue_stage_shape_probe/features/run_manifest.json（128hashtrain/153val），旧每图cache SHA/GT框/slot/image_id必须精确匹配。sat32/sat_extra32/prob/target/boundary_weight/boxes/global_id/image_id/gt_slot逐数组复制，禁止用旧native32 C或旧geometry当本轮输入；旧公共前5几何变量保留，新q/c重新计算。因此SG/S64与两stage都须在新共同geometry上重fit，旧成绩不能作为新对照。

每图同次网络forward只读hook捕获CVT最后BEV256×128²。原project_source/project_pixel/mass/quality不变，调用真实production BEVToSatellite得到32²及128²。C128=bilinear(C32,size128,align_corners=False)，F128=production fine128。V=(Dfine>0)，I=V & roll(V,32,32)，Dfine为massquality按fine目标桶累加分母，必须assert Dfine>0与生产qfine>0逐值相同。四图C/F和各自roll(32,32)在共同128²坐标上都乘I，公共qfine/cfine也乘I；同RoIAlign(spatial_scale.25,output16²,aligned/sampling_ratio继承生产flags)。ROI全保留，缺失/共同支持0伴随，不后筛主门。

错位在粗升采样之后进行循环roll32finecell=128卫星像素两轴，边界wrap明确人为OOD，不是camera错标定；不偷换成native32 roll8再升采样。两尺度均保留全部原投影entry，I仅在读出前作用；粗池化向量可含mask外子格上下文，bilinear可混邻粗格，不能称内容来源支持完全相同。共同I保证输出采样曝光位置相同，不能证明细尺度只增加形状而不改变context。此轮比较生产池化尺度+固定重建路径，不是native32消费者一变量因果。

smoke固定原3train+3val：correct hookon/off所有FPN/SAM/PE/highfeatures SHA逐值exact、CVT读取前后SHA/inputRGB/satellite/payload不变、参数_version和全model hookcount恢复；独立Dfine mask核对、共同I四路同采样，旧cache复用数组精确匹配。另保存native32RoI16对重建128RoI16的RMS/MAE/relativeL2，只描述路由差不调插值、不增加第三stage、不按偏差选图。所有source/helper/旧cache/模型/checkpoint/manifest/git/protocol/矩阵SHA保存；上一extractor间接helper链需同时固定。

schema：cvt32/cvt_shift32/fine32/fine_shift32及sat32/sat_extra32均[N,256,32]，两stage与shift使用同256→32矩阵seed1813；卫星矩阵seed1811/1812不变，SHA沿旧核。geometry[N,256,7]前5来自旧，后2=fineq/finecoverage*I经共同RoI，original_coverage=未掩finecoverage RoI、common_coverage=同I finecoverage RoI均[N,256]只描述不加输入。prob/target/boundary_weight16²与GT64变换沿旧字节复用，公平容量与统计超参见上段。两stage多次随机压缩或更高维本轮不实施。

资源预估：上轮281一次冻结提取614秒（含候选uniform/256cov），新轮去掉二者但新增fine128投影和重建RoI，因此预计同量级，必须6图smoke实测后核full，不保证旧速度。操作门full900秒、allocated20GiB、cache5GiB，超限停审incomplete、不按收益停止。缓存/权重仅/tmp。六图smoke审查后full281与fit另准入；此协议本身不授权自动运行GPU。
