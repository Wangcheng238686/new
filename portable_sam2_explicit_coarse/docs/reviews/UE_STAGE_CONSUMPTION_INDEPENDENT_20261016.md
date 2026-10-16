# 独立机制复核：信息是否被消费、为何无明显净收益

审查对象 /home/wangcheng2021/project/new-uav-dual/portable_sam2_explicit_coarse，2026-10-16。只读源码/报告/轻量JSON，未运行模型、梯度、拟合或训练。核对了生产dual/CVT/projection/fusion/dense源码，以及complementarity/stage提取、拟合与汇总定义和对应JSON。未逐个复算全部大缓存，不把已有独立审核等同本次亲自重放。

## VERDICT

“UAV完全没被读取/梯度没到UAV”不成立。当前确实消费配对RGB相关信号，主要沿SAM stride16图像embedding；但它形成了强协同依赖，尚未形成相对独立训练卫星单流的可靠净优势。现有证据不足以把问题唯一归到数据冗余、CVT丢信息、梯度隔离或SAM压制。最有依据的结构解释是：UAV主要进入较粗SAM视觉入口，而多数mask RoI与高分辨率视觉入口没有直接UAV补充；粗形状有可被SAM消费的条件空间，但冻结UAV表征目前仅读出微弱形状增量。后半句受测量压缩、任务和域限制，不能用作UAV原始信息贫乏的证明。

## 按证据强弱排序

1. **已证实现象：训练形成替代性依赖，净互补尚未成立。** dual AP .584958，single .583000，差+.196pp，图像bootstrap[-.201,+.494]pp；dual zero掉7.530pp，供体RGB掉3.432/3.539pp。zero直接跳过编码/BEV/融合（dual_detector.py:75-77），同时移除query/geometry相关表征；RGB供体固定非RGB几何，但制造RGB-D不一致。这两者只证明已训模型依赖，不能作为新增信息AP贡献。独立单流和双流特征可发生重新分工，正确UAV可能主要补回双流卫星分支训练时让渡的能力；这是最简相容解释，尚非已完成匹配训练控制的因果结论。

2. **已证结构缺口：局部形状入口覆盖与注入尺度不匹配。** dual_detector.py:103-116在post-neck仅改P3/P4与SAM embedding；configs/ablations/ue_rgbd_dual.py:14指定levels=(1,2)，:148原样返回high。GT框3657/4496=81.34%读取P2，small/medium均在此；故FPN供体变化下这些RoI根本不接收直接UAV扰动（UE_FUSION_ROUTE:83-87），不能以FPN小敏感性判“FPN忽略RGB”。P2也不是SAM high-res图像通路。整个模型仍沿SAM消费UAV，不能写成81%目标没使用UAV。预测框实际比例未由GT比例证明。

3. **有依据的风险：空间具体性弱或形状增量未保留/未被现有读出访问。** 旧SU对SG/SX的Brier改善.000359613/.000186775，对S64 .000341644；stage非线性C对S64/CX改善.000182947/.000598295，U对S64为-.000509908；均未过预设.001幅度门，但正差区间已经支持微弱条件可读信号，FAIL不能写成零互补。UAV RoI内方差比例val .41685，sat .69557是空间特异性线索；不是信息量计。production stride8视觉token→单4m BEV向量→mass*quality桶平均（cvt.py:51-53、168-182；projection.py:20-27、44-55）有明确混合可能，同token多anchor不增加视觉分辨率。候选等权stage也失败，且C优于U，故不能继续断言learned attention就是主要损失点；共同source/aggregation/projection可能已限制两者。

4. **已证训练路径限制，但未证根因：提示生成器不直接接受最终mask反馈。** dense_prompt_utils.py:31-32按配置detach；mask_head.py:451-459明示coarse消费者只有detach dense和no_grad点；coarse仍由GT BCE+Dice辅助loss学习。PE冻结参数不阻断对输入梯度，CopySlices可微，因此能改dense反馈，但当前不是全UAV缺梯度：SAM图像直路接mask/tail，FPN接检测/coarse，共享CVT/UAV汇合。粗loss=.1不足以断言梯度弱；门控接近全开、残差范数大也不是因果信息占比。需要分loss同batch梯度才可判监督方向/冲突，现有总norm不能判。解除detach无法给P2凭空补UAV，也不能修复support0。

5. **已证局部限制：197/4496的32²框中心support为0，dense提示完全消去，其中small占197/881=22.36%。** factorial此组换GTdense输出不变，构成具体算子限制；不是几何coverage0。小目标UAV投影support0另为148，旧全体184；新stage91是共同intersection及采样改变，不能合并。不能把4.38%局部问题扩写为整体净收益主因；小目标支持非零组仍无明显读出优势。

6. **净差归因受评测边界限制。** 单seed、同153图用于checkpoint选择及反复机制开发；4496出现仅237建筑，图像bootstrap未覆盖空间相关/种子/选模。GT框IoU、GT辅助选择预测框、occupancy Brier均不是部署AP。全量GT框双流比单流低.002304，也不代表全AP更差；检测排序及框条件有补偿，小AP差可能抵消多种因素。现有候选/NMS案例不能据此立首要根因，且无需重复已有全量审计。

## 各探针到底证明什么

- whole-zero/large BEV shift：整个辅助路径与对应关系依赖；不隔离RGB/深度/query，也不测新增可用信息。
- donor RGB：在几何固定时RGB相关条件依赖；物理失配与统计变化使其不能分离新语义、外观先验、训练适应。
- CC/CD/DC/DD：冻结激活替换效应主要沿SAM；不是从头单入口性能消融，不证明FPN无需或融合已最优。
- SAM s0/.5/1：固定权重统一减弱不能改善，.5掉2.075pp；可排除“现在直接减半就好”。不能排除强融合导致训练中的卫星能力让渡、局部门控错误或不同起点的弱融合训练。
- P2冻结跨尺度迁移：变化确实到RoI/coarse/dense/decoder，粗质量本身几乎没改善（-.00000324）；不能指控SAM压制已形成的有用信号，也不能否定专训P2。不是原路径有效信息验证。
- GT提示factorial：dense-only+.015619 IoU，point-only+.004540；SAM在GT框、特定幅度/提示分布下能消费更准形状。它给的是条件存在性，不是可达到的全模型上界，更不是UAV可预测性。GT±4相对预测±4+.016068，排除“这一次二值化幅度就解释全部”但不是纯置信因果分解。
- frozen ridge：train-only fit及建筑隔离支持微弱可泛化条件形状读出；来源是双流训练过的pre-fusion卫星neck，不是独立单流。32维随机压缩、8²occupancy、逐格加性线性限制大量任务规则。岭谱保留89.2%-99.9%、有效自由度68.985/71、非零方差排除数值抹除/全坍塌，不证明语义保留。
- stage C/U：同source后256→32、同stride16投影、同16²任务、匹配小头容量，C优于U而两者未过幅度门，足以否定“本轮简单等权明显救回形状”；不构成五阶段信息守恒审计。U自身已聚合，C含query/PE/FFN/residual，C-U不能命名attention纯效应。旧8²与新16²Brier不能直接相减。

## 不能循环论证的三条链

“GT形状有效→UAV应当生成→解除detach有收益”中两次箭头都未被实验支持。
“供体掉分→RGB是互补→SAM饱和所以增益消失”先混依赖和净互补，后把范数当成错误诊断。
“读出FAIL→BEV丢形状→细网格一定恢复”把受压缩读出阴性当原始内容阴性，并把结构可能性当性能因果。

## 最小下一步

本次不建议再做zero/donor/强度/GToracle/P2跨尺度迁移/改读出seed或步数。这些已回答各自的问题。

若问题是**信息在哪里丢了**，唯一优先冻结候选是同一CVT输出、相同32维与同16²小头、同共同支持和ROI/训练样本，比较真实stride16与较细投影；只改空间汇聚，附同预算卫星与错位对照。需事先定义幅度门、量纲、正控制、no val tuning。先证明这个测量确能利用任务相关空间细节，避免负结果再次不可解释。阳性只定位投影可读性，仍不等于最终AP；阴性只关闭该空间汇聚假设，不能要求继续扩宽扫参。如果目标转为**是否原始RGB有独立训练价值**，最终需要匹配架构/几何/预算的训练对应控制和独立空间test；已有冻结探针不可能代替，但本次不启动。

若希望只判断**修改dense反馈是否工程可行**，另做单batch无step的分lossautograd，对coarse/mask/tail分开核定reachable、方向cosine、norm ratio、support0阴性及真实训练RoI峰值；此项与空间可读性是不同问题，不应同轮改结构并宣称解决净收益。
