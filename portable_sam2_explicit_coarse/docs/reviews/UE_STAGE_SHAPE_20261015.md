# 分阶段形状探针独立审核

审核者frozen_probe_audit，只审协议/实施/工件，不参与extract或fit开发；不改生产或工作树，不自动执行GPU。

## 首版静态发现

VERDICT：PENDING，暂不准运行。已读工程可比性、统计协议及extract/helper；root允许审核者在修正静态PASS后准3train+3validation唯一特征smoke，禁止fit/full281。

候选等权stage U在实际DepthCVT.source之后的tokens按原candidate有效slot等权，非GT选view/token；与C同256维、hascandidate、production stride16投影和seed1813压缩。U已经聚合，不是无损原始token；C含learned query/位置/残差/FFN，不能将差异归因单attention。原卫星公共路径是multilevelROI14再bilinear16，U/C直接stride16ROI16；公共六cell卫星一致，不能宣称所有表示同采样或以绝对方差推断信息损失。

执行前待闭合：统一PREREG；correct观察hooks开关完整features逐值hash；固定candidate样本独立均值核对；first-layer prehook调用/清理；空anchor全无候选安全返回0；GT64边界band area16缓存以支持预注册伴随终点。

shift对照roll8cells移动内容及零支持域，但公共目标geometry不变；需root冻结如何使U/C/UX/CX匹配支持，或明确OOD支持不匹配局限。审核建议原D>0与roll(D>0)交集作为四臂共同mask、全ROI主分母不后筛，待root决策，不自行改定义。

fit只可在train归一化/优化、paired init与batch，1000last固定，无val调参。当前fit尚未出现，禁止准入fit/full281；特征smoke不得输出任务得分供选择协议。

## 修订版特征提取静态准入

VERDICT：PASS，按root委托准入唯一3train+3validation特征提取smoke，GPU0；不准fit、不准full281。

统一PREREG已落盘，固定SG/S64/U/UX/C/CX、配对小头计划、四contrast及预算；拟合实现仍另审。当前extract补齐correct on/off全部原features hash、全module hook/prehook counts及candidate调用1、16固定无GT选query的CPU double候选均值对账、原zero-UAV coarse逐值回放、empty anchor安全零返回、GT64形态学band area16缓存。source/payload/RGB/satellite hash与参数version检查保留。

root已冻结共同支持I=(D>0)&roll(D>0,8,8)。四个U/C原/shiftmap均在roll后乘同I，公共quality/coverage亦乘I；原coverage和commoncoverage只描述、不增小头输入。实际用production q>0取得有效D支持，两者在当前有限正分母下相同；所有ROI主分母保留无支持实例，无GT选择支持。train原256协方差在同I后RoI域累积，不据val拟合压缩。

已准确披露公共卫星ROI14→16与两stage直接ROI16差异及roll循环wrap的OOD。小头主Nonlinear、Linear伴随，不以本smoke结果改变特征/seed/超参。提取计时/显存及缓存预算仍须由smoke实际验收决定full281能否准入，当前PASS不提前批准。

## 六图特征smoke结果审核

VERDICT：PASS。审核者按root委托准唯一train-only32step timing GPU1，禁止validation targets评估及fullfit；fullextract281建议root可在原900秒/20GiB/5GB硬预算内准唯一一次，实际启动仍由root决定。

首版因PE=None被hash辅助直接调用tensor_sha而AttributeError，未产生cache；原脚本/log保留。修复通用feature_hash跳过NonePE/空high，未改变模型/数学定义，独立静态复审通过后重跑。另新增实际split globalid身份集合和互斥assert，不用于选择样本。

独立CPU核验修复版manifest6图、240ROI（train7/15/25共195，val287/288/295共45），complete=false与smoke范围一致，strict EMA无missing/unexpected、无optimizer/model训练、参数version不变。当前script/helper/stagehelper/protocol/生产source SHA全部一致。每图on/off全部原features精确hash、candidate16样本均值、coarse原zero-UAV接线精确回放、hook清理、input/payload不变及U/C projection quality/coverage一致的运行assert与记录均通过。

六份cache逐数组形状/finite/ranges、geometry prob等于公共prob、common coverage等于geometry最后通道且≤original coverage、GT slot/image身份正确；所有共同支持0目标四stage视觉特征均0，没有stage独占支持。train158/val19 visible globalids独立从cache重建，与manifest一致且无交集。原256协方差仅train n=195×256=49920，两stage相同；不在val学压缩或挑参数。

实耗21.384秒、peak allocated2.757GiB、缓存38.774MiB。按ROI比例全缓存约2.70GiB可满足5GB；smoke含额外correct与zero-UAV replay而正式仅一次前向，不能直接按每图smoke成本声称完整提取低于900秒，完整运行须保留硬停预算。smoke无Brier或val选模。

## 小头fit与统计静态审核

当前fit/summary静态PASS：6cell×linear/nonlinear共12小头，同架构初始化逐值相同、同1000×64身份抽样表；train-only每通道标准化、零块保持0、独立权重AdamW固定1000last。训练未clip residualMSE，final预测clip、未clip与clamp比例伴随。主nonlinear四固定contrast用98.75%图像sum/count配对CI和.001幅度，linear及C−U95%仅伴随，不绕门。

修订后训练前仅读取cache指定身份字段，核SHA/slots/boxes/真实与声明globalid互斥；validation targets/features未加载用于归一化或选择。正式fit才要求128train/153val与12621/4496RoI；timing只用已验smoke train195，固定32steps，不评val。norm JSON与sampletable、权重SHA均记录。正式PREREG由DRAFT同字节copy并核对提取协议hash，禁止偷偷修改协议后继续旧cache。

所有fullfit/validation结果均未准入；等待32step timing实际预算和梯度有限性验收。

## train-only32step timing验收

VERDICT：PASS，支持预算可行性；不单独准fullfit，待281cache验收及root最终审批。

独立CPU验收12头权重SHA/last32、有限梯度与参数记录、固定optimizer、无validation targets读取/评估、正式与DRAFT协议同字节及feature协议hash一致。由3train195RoI原缓存重算全部normalization均值/std和49920cell分母，吻合JSON；按seed1831独立重生1000×64配对image/slot采样表，逐值精确一致。norm/sample/fit脚本与feature manifest SHA均正确。

32step fitloop3.2619秒，线性1000step外推101.9345秒，peak allocated约31.68MiB。只说明该固定小头预算在smoke输入下可行，未含281缓存加载/final train+val评估，不是正式性能benchmark。没有以trainloss选width/seed/超参或追加steps。

## 完整提取后执行状态（主代理记录）

独立审核代理在完整提取结束前因服务额度中断。主代理执行其已编写且未修改的independent_cache_check.py，281份SHA/schema/finite/支持/身份合同全部PASS：12621train/4496val，2239/237建筑不交，2,879,598,544字节，614.031秒，峰显存2,316,940,288字节。此为独立编写检查器由主代理执行，不称审核代理亲自验收正式结果。

在既有协议/fit实施与train-only timing独立PASS、完整缓存检查PASS基础上，主代理准唯一1000step×12小头正式读出拟合。冻结源模型不加载/更新，无超参变动；完整结果独立审核尚PENDING。


## 正式结果状态：PENDING独立复核

主代理完成唯一正式12头1000step读出，最后统一train/val评估，CPU汇总与另脚本Brier/边界/建筑均值及四主bootstrap门重算PASS。小头fitloop28.49秒，加载/拟合/末次评估合计92.75秒；正式源模型未加载或更新。最终独立审核代理因服务额度中断，尚未亲自审核完整cache或用固定weights重放正式预测；以上主代理复算不能替代该独立结果节点。四工程门全FAIL，两stage支持门FAIL。不得以当前未完成审核结果启动新训练。
