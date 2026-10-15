# 等权候选替代表示 vs learned CVT：统计协议草案 v2（未执行）

2026-10-15。原尺度轴草案已撤回保存RETRACTED_SCALE_AXIS_DRAFT.md，不本轮混合。当前只比较同一五视角DepthCVT.source之后的tokens：按原candidate有效anchor/token等权得每query256维替代表示(U)，与训练所得两层CVT最后query256维(C)。原candidate可能几何多anchor同token，应按生产candidate票而非擅自token去重；作者须记录选择规则。U不是所有原始tokens的完整保留，C包括learned query/位置/残差/FFN，差不等于不可逆信息损失。

## 同一读出坐标与支持

U/C均用相同hascandidate过滤、原project_source/pixel/massquality、production stride16池化，再同GT框RoIAlign得到16²，256→32同一个Gaussian矩阵seed(沿1813)、SHA一致；同卫星32与额外卫星32矩阵1811/1812。卫星沿生产多层RoI14再bilinear16，U/C直接stride16 RoI16；U/C两stage sampler相同但不称卫星同sampler，卫星额外PE亦不同，SC容量对照和共享xy无法完全去除该混杂。共同same支持/coverage/quality、同局部/全局位置输入，所有cell统一。RoI16²不同旧14→8路径，故数字不直接与旧Brier相减。不能给一个表示更大空间支持/通道预算/PE后宣称CVT损失。边界仍是16²占据代理，非512细轮廓。

训练图固定复用旧hash选128、验证153，manifest/IDs/building隔离与历史相同；train12621 RoI/val4496。所有标准化、拟合、优化/超参只用train；val一次最终评估，不早停不调参不换seed。当前验证集反复探索且已选E72，结果只开发诊断。

## 六个固定cell

- SG：sat32+公共几何。
- S64：sat32+extraSat32+公共几何，容量对照。
- C：sat32+CVT32+公共几何。
- CX：sat32+错位CVT32+公共几何。
- U：sat32+等权候选32+公共几何。
- UX：sat32+错位等权候选32+公共几何。

C/U/S64/CX/UX视觉64维与公共输入预算完全一致；S较少有效变量仅基线，不以零列伪称相等容量。shift固定卫星32²map roll(8,8)，即两个轴各128px且循环wrap；几何不变造成OOD，非真实标定。四UAVmap都乘共同intersection I=(Doriginal>0)&roll(Doriginal>0,8,8)，aligned/shift一律同I再同ROI；公共quality/coverage也乘I，六cell同geometry。原coverage/commoncoverage额外缓存只描述，不增加input，全体主包含I无支持ROI，不筛选。不能依据val选shift幅度。

## 目标与读出能力

原卫星coarse64²概率与GT64²二值target，以area-pool到16²。拟合残差r=GT16−p_sat16，部署预测clip(p_sat16+f(inputs),0,1)。原coarse保持相同冻结dual-trained satellite pre-UAV路径，不是E66单流。

主唯一终点：每ROI16² soft occupancy Brier，再ROI等权全val均值。伴随：GT64固定3像素半径7²方形边界带pool到16²的带占比加权Brier、二值16²IoU、small/medium/large、支持为零与非零、建筑内平均再建筑等权；不得将boundary/子组后选替换主终点。未覆盖目标保留全体主结果，另共同支持组描述，不后选过门。

两读出固定并完整报告：Linear=Conv1×1(in71→1)，Nonlinear=Conv3×3(in71→32,pad1)+GELU+Conv1×1(32→1)，所有cell输入统一71维（sat32+second32+公共7维），S的second32为0但明确有效维较少。主readout固定Nonlinear，Linear伴随，不据val择优。标准化只train每cell各变量均值/std，零块保持0，不为某stage额外PE；常数/截距通过conv bias。

AdamW固定lr1e−3、wd1e−4、1000steps、batch64 RoI、单seed(1831)，配对cell及架构初始化和训练样本序列一致；架构不同无法逐参数相同，但同架构六cell权重初始化逐值相同。每step先均匀选train图，再图内均匀选RoI，可有放回；同一batch所有cell完全同身份。模型残差输出的训练损失为未clip residual的MSE（f−(target−p_sat))²，validation唯一固定输出clip(p_sat+f,0,1)计算主Brier，未clip Brier和clamp比例固定伴随。避免训练clip造成零梯度；所有cell一致。无earlystop，无schedule/warmup/val tuning，最后step唯一评估，不留best。训练loss每100steps只机制记录，不据此追加steps；NaN/实现失败停审。

小头可在GPU1训练，正式网络冻结不反传，cached features detach；不称生产模型训练或热启动。1000×64=64000 ROI抽样约train12621的5.1倍，但image均匀→ROI均匀导致概率不相等；1638万cell彼此相关，不能以此证明数据够或无过拟合。trainloss下降及final train/val差完整报告，若未收敛只说明固定优化预算，不能事后续训。

train-only timing smoke固定32steps，只核实际steps/显存/数据/梯度，禁止评val或按loss收益选width/seed。预算按root提取≤15min/VRAM<20GB/cache<5GB，12小头拟合预算≤15min GPU1（先train-only计时核定），超预算先主代理审定缩steps或width并修改协议一次，不接触val。

## 预注册主比较和门（提案，待root审）

主readout固定局部非线性，线性只伴随，不能择优。为避免六cell×多比较随意选阳性，建议两个stage支持判据各须同时超过容量与本stage错位：
C相对S64及CX；U相对S64及UX，共四预定contrast，controlBrier−stageBrier正为好。四compare Bonferroni各98.75%区间quantiles(.00625,.99375)，1000 imagepaired sum/count bootstrap seed1830。工程幅度各>=.001且下界>0；此.001是16²新任务预设工程最小支持线，不与旧8²改善直接横比；任一stage两门都过才称该stage有预算匹配的可读互补支持。

C与U直接差U_Brier−C_Brier同时给固定1000 image-paired 95%CI(quantiles .025/.975，同seed1830抽样)，仅描述，不能把缺乏同预算卫星控制的差异视为信息保留；该直接差不能绕过四门当成功判据。两stage均不过则停止方向，不扩大hidden/通道/seed/多val调参；若只有U通过C未过，可称等权候选在本读出条件优于CVT的可访问互补线索，不称原信息被抹去；若C通过U未过则相反。都通过则候选/CVT都有可读信号，不按最大值选架构。所有伴随/六cell/linear均完整报告。

两个stage不共享拟合weights，同架构相同初始化与batch序列；各cell独立训练。

原256维通道的train std/有效秩只描述坍塌，不能恢复32压缩丢失语义。source/depth/camera对齐和不同tokens语义不由读出得分证明。GT框排除检测，主Brier不是最终SAMIoU/AP，也不能外推完整生产集/独立test。

## 预算定版前必须补齐

非线性train-only GPU计时（不看val），六cell最大内存、缓存每ROI16²×32维估计17117×256×32×4约561MB/featureblock，4个U/C/shiftblock约2.24GB+sat/extra；数据可逐图流式不全内存。逐step同batch可串行12head或按预算分头训练，初始化/采样表预保存SHA；固定1000steps不因loss或val追加。特征提取成本旧281图约358秒，新U聚合预计需smoke实测；无正式GPU准入前不得执行。

独立实施+smoke/结果审查、全部SHA、feature参数version/cache身份、source/channel/PE/support匹配、不修改模型和生产遥测；仅/tmp工件，审后主代理归档。

## 工程定版（仅冻结特征提取，执行前）

实际extractor：/tmp/ue_stage_shape_probe/extract_stage_pair.py；helper：stage_pair_extraction_draft.py。E72EMA strict原architecture snapshot，correctUAV，原manifest与128hashtrain/153val固定IDs复用旧complementarity。仅/tmp写缓存，正式网络无optimizer/backward。原cache8²/32维不足重建raw/U/C/SAMprepost，本轮必须一次新冻结提取；不加入SAMprepost或投影尺度轴。

U取真实DepthCVT.source之后tokens（detailed stride8+semantic上采样，经Conv/GN/SiLU）；按原candidate_index有效anchor slot取anchor_token对应256维value等权平均，重复token不擅自去重，不按GT/视角质量选择。不加入位置/视角/query/bias/FFN，因而是替代聚合设计，不是原始多视角全信息或可逆原表示。无候选0；emptyanchors且全invalid时严格返回0。C为原DepthCVT两层最终128²×256输出，原无候选mask保持。

两臂使用完全相同production BEVToSatellite32²、project_source/pixel/massquality、target归桶/quality/coverage。共同mask I=(D>0)&roll(D>0,8,8)（D为massquality分母；生产q>0与D>0精确等价，不能用rawcoverage替代）。四图U/C/Ushift/Cshift先投影32²，shift循环roll(8,8)，然后都乘相同I；q/c公共几何也乘I。ROI全部保留，不按支持后筛主分母。原coverage与commoncoverage单独缓存仅描述，不增读出输入。人为循环位移仍是OOD，不是物理UAV/camera错配；支持交集会牺牲可用数据，不能按结果改变mask。

U/C sampler相同：RoIAlign输出16²，spatial_scale1/16，aligned/sampling_ratio沿原生产maskRoI flags。公共卫星仍为preUAV neck+原extraPE+原多层RoI14²，bilinear到16²；不是与U/C完全相同sampler或PE语义，六cell共享此卫星底座及SC容量控制，不将其差直接当跨阶段信息损失。两臂同256→32 Gaussian seed1813，公共sat512→32 seeds1811/1812、矩阵SHA保留。无新增PE。GT64/粗概率64 area16；GT64 boundary按padding外0的dilation7²与erosion7²不等带，再area16为边界权重，阈值/带宽不根据结果调。

schema：每图sat32/sat_extra32/cvt32/cvt_shift32/uniform32/uniform_shift32均float32[N,256,32]，geometry[N,256,7]=[p16,localy,localx,logw,logh,q,c]；prob/target/boundary_weight/original_coverage/common_coverage均[N,256]；boxes/global_id/image_id/gt_slot身份。manifest同旧complete/weights/strict_load/parameter_versions_unchanged/selected_image_ids/records（split/image_id/n_rois/path/SHA），附helper/source/git/architecture/checkpoint/矩阵SHA和峰显存/耗时。

仅train两臂matchedI之后、共同RoI16²域原256通道流式sum/cross covariance，CPUdouble 2线程，一口径，不另增supportpositive协方差；输出std/有效秩/谱不作为PCA或调seed，不保存原full256大缓存。

准入：先3train+3val（旧选择列表前3）smoke。固定16hascandidate query linspace取样，用CPUfloat64独立token枚举等权均值核对；firstlayer prehook仅1次并清理，全module hook计数恢复。correct hook-on/off全FPN/SAM/PE/high返回值SHA逐值exact；原卫星coarse直接计算与zeroUAV完整head捕获coarse逐值exact。参数_version、模型输入/几何/RGB/卫星、被读取原neck/CVT/U张量SHA前后不变；两臂q/c逐值相同；GTlocalglobal身份/顺序核对。独立审查+root准入后才运行smoke，smoke结果审核后正式281另准入。

预算：旧281冻结提取357.76秒，本轮预估≤2倍但必须smoke实测，不保证旧速度；full15分钟、GPU peakallocated<20GiB、cache<5GiB操作门，超限停审并标incomplete，不按收益停止。拟合12小头的train-only计时和完整fit由另一代理单独协议/审查，不和正式模型训练混称。提取脚本未执行时不得伪称资源已验。
