# v1冻结基座UAV适配器开发协议与修订记录（2026-10-21）

当前状态（2026-10-22）：原八臂已完成训练；补齐early/contextual两条常量RGB训练对照后共十臂。
用户已将主要有效性判断改为固定E10的test评价，控制训练与test矩阵仍运行中。
test进度见[测试报告](UE_V1_TEST_PROBE_PROGRESS_20261022.md)，val诊断见[阶段报告](UE_V1_PARALLEL_PROBE_INTERIM_20261022.md)。下文保留初始设计、事前登记和
后续操作修订的时间顺序，早期“未启动/尚无结果”等语句仅描述当时状态。本记录不改变EXP-0017或正式数据划分。

## 已核定基座

- EXP-0017 run_state.stage=complete；封存清单SEALED_SATELLITE.json。
- 权重：/data/wangcheng/checkpoint/ue_minimal_single_20261021_EXP-0017/checkpoints/best_model_epoch73.pth
- 封存SHA256：dd9106f4afe8ba72936316d7efa9621ca0c05d24836fa28c72ea322bed84ace1（读取封存记录，本轮未再次全文件计算）。
- 按validation segm AP250选择，weights=model，非EMA。训练日志best_segm约.8546；最终test segm/mAP=.6455098296975914，bbox/mAP=.6854287640342888。本机约20.9pp Mask AP差距，不混同4090用户报告约30pp。
- 数据：/data/wangcheng/dataset/CitySample_SatUAV_TrainMinimal_v1_20261021。

## 研究问题与数据边界

在相同v1训练/验证划分下，冻结该卫星基座，训练新增UAV分支，检验是否能提高已有实例的掩码质量。这可以验证消费接口，不构成新划分的正式结果，也不解决当前跨区域泛化差距。

训练仅用v1 train，开发仅用v1 val；本轮不重复用已评估test调适配器。后续若调整划分，必须检查基座已见图像/区域是否进入新val/test，不能直接沿用权重并宣称新held-out性能。正式新划分必要时从通用初始化重训单流，再迁移结构设计。

## 最小实现

1. 载入完整封存卫星权重；原SAM2、检测neck/RPN/box、CSPM、mask decoder、UDPR均冻结并保持eval语义，防止运行统计/随机层漂移。
2. UAV编码器从通用预训练初始化，允许LoRA；只训练LoRA、一个源尺度投影、一层局部聚合及残差适配器。不继承旧失败双流权重，不恢复基座优化器、调度器、epoch或EMA状态。
3. 深度与相机恢复可见表面并投到卫星实际透视坐标，保留源视角，不压成旧独立4mBEV。
4. 每个预测/训练RoI用自身bbox定义局部查询网格，读取其卫星区域query与对应UAV候选。以原RoI顺序形成feat_s0空间大小的残差，在sam2_decoder.conv_s0之后、SAM2最后一级上采样加feat_s0前注入；基座全图特征重复路径不改。该RoI残差与基座全图重复特征如何共同影响输出仍需实际验证，不声称入口已经有效。
5. 无候选处增量严格为0；零贡献初始化能打开梯度。冻结模块仍保留对新增特征的输入梯度，不能用no_grad包住残差之后的整个解码链。
6. 沿用原卫星mask监督和相关既有掩码损失；检测只读、不训练。不要新增UAV标签、几何实例ID、边界loss、第二个mask头或同时融合FPN。
7. UAV参数使用独立存储，避免共享可训练LoRA意外改变卫星骨干。训练时同一次读取的RGB/深度/相机必须对应；卫星光度增强不应改变几何。

## 验证顺序

- 必要工程检查：关闭分支复现基座；不同RoI中心/尺度的投影坐标检查；无支持处恒等；残差打开后梯度确实到达源LoRA；冻结参数和运行状态不漂移。卫星纯特征提取可no_grad；含新增残差的解码不可no_grad。
- 第一轮建议固定10epoch作为开发预算，关闭EMA、无早停，last作为预先约定的筛查结果；该预算是建议值而非已启动配置。学习率在接线与梯度核定后固定，不直接照搬单流全组5e-4。
- 同一冻结基座比较：新增分支关闭；真实UAV适配器。若真实UAV出现改善，增加同结构同预算的“常量RGB外观+正确深度/相机”训练对照，区分外观增量；后者仍有几何和适配容量，不能称全无信息。
- 同检查点zero/donor RGB只判依赖性，不代表相对基座净收益；donor RGB与正确深度不一致要披露。关闭残差与喂零RGB是不同操作。
- 正式训练阶段各方法统一按val Mask AP250选checkpoint；报告同checkpoint Mask AP/AP75、Boundary AP（2%，1%伴随）及bbox/召回。冻结检测预测框和原检测分数以明确比较对象；若现有mask质量评分会反馈最终实例分数，应明确固定/重算策略并同口径重评基座，不默认分数一定不变。
- 不要求冻结检测入口的bbox AP提升；以Mask AP净提升判断实例分割收益，若仅Boundary AP改善则收窄结论。小幅提升需后续种子/配对不确定性验证。

本协议优先回答“UAV信息能否在固定基座上被有效消费”，不把低成本开发筛查代替最终联合训练与新划分泛化验证。

## 2026-10-21 实施前预注册补充（正式筛查尚未启动时）

- 路由：A类模块叠加，直接消费探针同时作为冻结基座热启动筛查；已有历史几何支持/干预作为背景，新增真实样本smoke已验证恒等与梯度，不把它当收益证据。
- 实现：source stride4的SAM2 FPN0，源深度stride2逐点camera-Z投影；卫星128²候选表，每格最多8个源token，每视角最多2个，按已观测卫星深度选近表面且优先覆盖不同视角。此表是候选索引，不是压成一个BEV特征。无目标深度，不声称精确遮挡认证。
- 查询：每个RoI的128²局部网格映射到卫星512坐标，读取该4px格内候选；32通道单层余弦注意力（温度4），6项轻量元信息。没有额外全局搜索或几何监督。RoI分块4、非重入checkpoint仅降低激活内存，不改变输出公式。
- 训练：v1全部2142train，10epoch，seed44，单卡batch1、累积8（每轮2136图、267更新，丢弃随机顺序尾部6图）；AdamW，LoRA与适配器统一LR1e-4、wd.01，50update线性warmup，余弦降到.1倍；AMP scale512、global clip.5；原卫星光度增强沿用，源RGB无空间增强。EMA关闭，无早停，主比较固定epoch10/last，不用best挑尖峰。
- 可训练范围：UAV LoRA r16/alpha32/dropout.05，不开放UAV neck；新增源/query投影、key/value及geometry投影、输出投影、单个零初始化残差系数。选择UAV neck冻结是本探针限定，不能声称等同旧双流的LoRA+neck。
- 主终点：459val Mask AP@[.50:.95] maxDets250的last−关闭分支差值；伴随bbox/AP75及Boundary AP2%/1%。bbox及scores需逐图不变；若不变门失败先归因/修复，不用结果宣称UAV收益。
- 判定：点估计>0为方向性信号；预先要求至少+.002绝对AP（+.2pp）且image-paired bootstrap 95%CI下界>0才称“此固定检查点和v1验证集上的明确增益”。500次paired image重采样，seed20261021。此CI不覆盖训练seed或空间相关性，不能称最终可复现泛化结论。
- 机制诊断：训练last完整val执行正确/关闭/常量外观/循环donor RGB。donor保持深度相机，形成外观几何不一致，仅测依赖。若真实UAV出现净改善，追加同预算常量RGB+正确几何的独立训练对照，不把推理时置常量替代训练对照。
- 遥测：每次optimizer update记录末次microbatch的loss、alpha、候选格支持率、原始/应用残差RMS、基座RMS、RoI数；标注latest_forward，不能解释成全epoch均值或像素边界覆盖率。梯度范数是累积8张后、裁剪前的全可训练参数范数。遥测只诊断，不参与loss、候选选择、选权或判定。
- 工件：/data/wangcheng/checkpoint/ue_v1_adapter_probe_20261021/；入口bash scripts/ue_rgbd.sh adapter prepare|smoke|train|eval。checkpoint内嵌基座/数据/脚本身份、全部超参；仅新增分支权重，不进入原inference自动识别，使用adapter eval严格重建。

### 起步阶段口径修正记录

初版459val AMP推理基座Mask AP=.8532678452，比训练日志E73的FP32验证约低.0013。
因此正式对照统一FP32推理（训练仍AMP），不能用两种精度的数直接计增益。
早期训练在epoch1未完成时中止，未评估其训练后收益；记录保留preflight_train_aborted，
从同seed同基座重新开始10epoch。checkpoint使用启动时冻结的源码/contract副本，防后续分析脚本
变更污染运行身份。正式训练目录correct；中止产物不进入last比较。

### 已完成校验与运行身份

- FP32完整459val关闭分支：Mask AP=.8545524587821675，bbox AP=.8330386567044331，
  与E73训练日志四位小数一致。后续净差使用这份同入口缓存，不使用AMP诊断基线。
- smoke：真实train图2712，零初始化mask/box/score逐值一致；非零alpha但全无候选时mask逐值一致；
  三步反传后LoRA梯度非零，卫星state SHA不变；峰值自有分配2799.7MiB。
- bootstrap校准：前8张val固定重复采样[0,0,1,2,3,4,6,7]，复用匹配与重建COCO的
  Mask/Boundary2%/Boundary1% AP误差均为0；原顺序重累积也复现。校准不作为模型结果。
- 正式训练PID及完成链由工件run_state.json登记；`scripts/probes/ue_v1_adapter_finish.py`
  仅在启动时确认live的训练owner退出且last存在后执行完整干预和CPU分析，失败记录终态。
  真实UAV相对基座点估计>0时自动补同预算常量外观训练对照。
- 本轮尚未给出训练后结果。需检查最终源码合同、10epoch/2670更新、冻结状态及完整输出后落判。

### 全缓存来源审计

2601个train/val缓存逐项核对了其camera及5份深度的SHA256、形状、索引范围、token去重、
每格每视角<=2候选、几何元信息有限/范围和源视角投影回环。全部通过，最大代数回环误差
8.5742e-10px。证据：工件geometry_cache_audit.json及同目录audit_geometry_cache.py。
此回环仅检验矩阵实现内部一致，不是外部像素配准精度或可见性证明。

## 用户授权后的并行方向扩展（各新路线训练前登记）

用户明确允许并行尝试不同方向，以尽快找到可行入口。原highres/fine训练保持不变；新增：

| 臂 | GPU | 消费入口 | 源特征 | 外观 |
|---|---:|---|---|---|
| correct | 0 | conv_s0后、最后一级上采样前 | 原stride4 FPN0 | 正确RGB |
| geometry | 1 | 同correct | 同correct | 固定归一化均值RGB，正确深度/相机 |
| early | 2 | per-RoI image embedding、SAM2 two-way transformer前 | 同correct | 正确RGB |
| contextual | 3 | 同correct | stride4细节 + stride16语义投影到同一源网格 | 正确RGB |

- geometry不是新的外观路线，而是并行归因对照；它仍包含几何、卫星query及适配容量。
- early通过256通道残差进入32² embedding，使UAV可影响mask token交互；RoI-local坐标合同
  与原高分辨率入口一致。输出投影通道32→256是入口要求的容量差异，不伪称参数量完全相同。
- contextual针对已核实的SAM2 FPN0没有高层top-down语义、其梯度仅到早期块的问题；加入
  一个256→32投影并上采样相加，仍只有一套候选/聚合/消费入口。它是源表示方向，非同时改入口。
- 所有新臂同seed44、v1 train/val、10epoch last、累积8、相同LR/损失/评价；独立输出和smoke。
- 原分支对关闭基座的比较不变；统一报告全部3条真实UAV路线，不只报最高者。
  500次95%图像配对区间是逐比较的名义区间，未校正多路线筛查，也不覆盖训练随机性/空间相关性。
  发现方向后仍需新seed/预先固定评价验证，不把挑出的赢家当最终泛化证据。
- early/contextual若相对基座出现正向信号，追加各自同配置常量外观训练对照；correct的对照
  已由geometry并行完成，不重复训练。完成顺序由各臂实际进程决定，先完成者先评价/分析。
- 新模块local_adapter_variants.py和multi_finish监督器属于独立probe路由；原correct正在运行的
  源码已冻结归档，未热修改其模型。停止旧等待监督器后由新监督器接管，只替换后处理调度。

### 固定中途观察节点（所有臂E1完成前登记）

为在并行路线完成前获得可比较信号，固定在epoch1、epoch3各做一次完整459val FP32诊断，
均比较同一关闭基座并核对GT/框/分数一致。由interim监督器读取已写完checkpoint后的progress文件
触发，独立保存产物。不做中途选权、早停或修改预算，不用这些数字替代epoch10主结果，最终
统一披露这些观察点。该设置服务用户要求的同步出结果，不改变正式主终点。

### 并行实施校验

新增三臂的真实样本smoke均通过零初始化逐值复现、无候选恒等、卫星state不变和LoRA非零梯度。
峰值自身分配：geometry2799.7MiB、early3206.8MiB、contextual2821.6MiB。
最后smoke步非零LoRA张量数：原fine为6、early为7、contextual为61；这是梯度路径证据，
不是表达能力或收益排序。固定snapshot和原始smoke.json在各自工件目录中。

### 2026-10-22 候选保留诊断（不修改在跑架构）

对原raw审计固定16train+16val，按原独立实例内侧1px边界union复核：val原始全分辨率点的
1px邻域支持88.12%；当前4px候选格中“有候选”的边界比例88.38%，但该格保留的8个候选
真实投影点在边界1px邻域的比例仅63.95%；允许跨格搜索已保留点可达74.97%，本格2px为80.65%。
train对应raw1px95.07%、格内有候选94.83%、本格候选1px75.93%、跨格候选1px86.10%。
投影点按最近像素取整，邻域为Chebyshev距离；候选UV由float16元信息恢复，有量化误差。
源特征感受野大于点，所以以上不是信息覆盖上界、分割精度或实际配准准确率。
它提示后续若需要，可优先研究更细候选索引/查询邻域和候选保留方式，而非再增加全局融合入口。
脚本及逐图证据位于工件audit_candidate_boundary_support.py、candidate_boundary_support.json。
初次诊断脚本未转换uncompressed RLE而报错，修正为frPyObjects后完成，未影响训练/评价读取器。

### 固定E1首条观察（2026-10-22，非主终点）

correct/highres/fine的epoch1完整459val FP32：Mask AP=.8524340273540962，
相对同入口关闭基座.8545524587821675为-.0021184314280712213（-.211843pp）。
bbox AP=.8330386567044331，与基座相同；interim监督器逐图核对GT、框、分数、标签一致。
此结果不支持“第1轮已有净增益”，不外推为10轮终点或UAV假设失败；维持原固定预算。
原始产物correct/interim_epoch1/correct/，进度interim_state.json。

## 2026-10-22 原生几何方向与同容量内容对照（训练前登记）

由已完成的候选保留诊断触发，新增两条小模型探针，原三条RGB-D路线及geometry对照均继续：

- native_geometry：全部512²源深度点按相机合同恢复三维，投到卫星512²原生栅格，保留离卫星
  最近的UAV已观测表面，输入为相对采样原点高度/100、相邻有效像素高度差x/y（/10、clip±5）、
  与最近表面相差<=1m的观测视角数/5。无完整卫星深度，仍不声称完整可见性认证。
- native_zero：相同原生support掩码、相同卫星query和完全相同可训练模块，4项几何内容置零。
  它仍保留观测支持信息，是条件于相同support的内容/容量对照，不是完全无UAV信息。
- 模型：卫星高分辨率特征投影32通道；每RoI在局部128²网格读取卫星query和原生几何，
  一个36→32的3×3卷积、GELU和32→32的1×1卷积生成feat_s0残差。没有新增mask head，
  冻结原SAM2/CSPM/UDPR，沿用原mask+UDPR损失。不编码RGB、不训练UAV LoRA；它服务几何
  信息可消费性的正交诊断，不能据此代替或宣称完整RGB-D分支已成功。
- 成对初始化相同（适配器初始化seed45，随后训练随机性seed44）；与其他臂相同10epoch last、
  有效batch8、LR1e-4、warmup50、cosine到.1、wd.01、AMP训练/FP32验证。
- 主比较native_geometry相对基座及native_zero的Mask AP250，伴随Boundary2%/1%、AP75与bbox；
  500次image-paired bootstrap95%、seed20261021。清晰信号仍需净提升>=.002且名义CI下界>0；
  对“几何内容有增量”的主张还需geometry−zero为正且相应CI下界>0。
- 两臂使用独立目录和GPU1/3的剩余显存，与已有任务并行；GPU资源检查与smoke通过后启动。
  缓存/data/wangcheng/cache/ue_v1_native_geometry，仅train/val。
- 入口bash scripts/ue_rgbd.sh nativegeo prepare|smoke|train|eval；模型与原有probe隔离。
  工件目录native_geometry、native_zero；各自保存native源文件/runner快照与SHA。

### 原生几何对照的归因边界

native两臂均将RoI对齐后的卫星query直接送入卷积，因此零几何内容臂也可能通过新增容量或
更直接的卫星局部特征消费改善单流。不能把native_geometry对原基座的提升全部归于UAV；
几何内容贡献必须看geometry−zero的配对结果。与原RGB-D注意力臂相比还同时改变了表示、
算子和query消费方式，不能把跨家族差值解释为单独的“分辨率因果消融”。

原生缓存全2601点审计通过：来源camera/5份深度SHA与已核验候选缓存一致，形状、无效位置、
高度梯度clip与视角计数均符合合同；平均图像像素支持84.94%，不是GT前景覆盖率。
同固定32图的原生边界支持train直接89.45%/1px95.10%，val直接79.99%/1px88.22%；
采用连续坐标floor栅格，与原raw诊断nearest-round略不同，不能把约0.1–0.5pp差异当配准改进。
原生路线保留了更多近边界观测，但仍未证明分割收益。证据native_cache_audit.json、
native_boundary_support.json及各自CPU脚本。

### 并行资源协调与native中途节点（native E1完成前登记）

原六条训练继续并行。考虑GPU1/3承载额外native臂后仅余约4.6–5.6GiB，所有新增中途验证
统一在GPU2串行，通过interim_gpu2.lock协调；已完成E1结果不重跑。只重启已排空的interim
监督器，未中断训练。native两臂也固定E1/E3全459val诊断，所有选择与正式终点仍为E10 last。
队列等待不改变checkpoint身份或评价口径。native最终评价在两条native训练都结束后于GPU1顺序执行。

## 2026-10-22 04:23中断与恢复修订（操作性修订，非事前声明）

07:47复查发现六条探针、监督器和原执行session均已消失；各训练日志在04:23:33–47停止，
无Python traceback，也未找到该时段新的内核OOM记录；具体终止信号/原因未获证，不伪造退出码。
机器未重启，另一个v3单流任务仍存活。不能把state中的training当live证据。

- 最后日志update：correct1645、geometry1576、early1620、contextual1275、native_geometry2509、native_zero2587。
- 为保持RGB-D方向与成对对照可比，四臂统一从**共同完整E4/1068更新**恢复；native两臂从
  **共同完整E9/2403更新**恢复。选择依据是共同完整进度，不看验证指标；不将中断轮次当终点。
- 原日志、原frozen contract和源码快照，以及被回退的更晚完整checkpoint保存在各臂
  interrupted_20261022_0423/；最终连续history只包含被最终权重继承的更新，原计算开销不隐瞒。
  parent checkpoint SHA/完整epoch/优化器/scaler已实际torch.load核验，见recovery_parent_checkpoints.json。
- 旧checkpoint未保存RNG，故**不是bit-exact续跑**。恢复RGB-D E5..10、native E10采用
  epoch_seed=1000000+44*1000+epoch，重设Python/NumPy/Torch/CUDA及DataLoader generator；
  内容对照同族同恢复点同随机流。epochs仍10、全局更新2670，LR/warmup/cosine/AMP状态继承原预算。
  如early/contextual触发新常量RGB训练对照，其E1..4沿原seed44轨迹，E5..10也采用同逐轮seed规则。
- 此后每完整epoch原子写checkpoint，保存Python/NumPy/Torch/CUDA/loader RNG，记录完整parent合同链，
  并逐轮核对卫星state SHA。最终history需核对1..2670连续、每epoch267次更新。
- 新恢复owner及训练/评价子进程使用独立session，owner已确认PPID=1；不再依赖长时间执行工具session。
  入口scripts/probes/ue_v1_probe_recover.py；统一状态recovery_state.json，同时维护run_state/native_state。
- 跨家族恢复阈值不同（4与9），且此前架构不同；不把家族差值当单因素因果消融。所有结果只作
  本v1开发筛查，并披露中断与随机流修订；不冒充完整未中断的seed44复现实验。

### 已完成E3观察（恢复前轨迹，仍在最终权重共同前缀中）

| 路线 | E3 Mask AP250 | 对基座差值pp |
|---|---:|---:|
| correct | .8535504620 | −.100200 |
| geometry常量RGB | .8536112314 | −.094123 |
| early | .8543067351 | −.024572 |
| contextual | .8538687232 | −.068374 |
| native_geometry | .8544438583 | −.010860 |
| native_zero | .8538368366 | −.071562 |

E3全部仍未超过基座；原生几何相对零内容点估计约+.0607pp，但不等于对基座净收益，
没有E3 bootstrap，不据此改预算或选权。最终仍待E10完整评价/控制和区间。

## 特征尺度稳定化对照（2026-10-22，训练前登记）

原highres/contextual轨迹中source_delta_rms由约.15增长到100–200，alpha缩到约1e-4，
实际注入RMS仍约.03–.08；early、native未出现同等幅度漂移。这是训练特征幅度/参数化风险，
不是已证的相机cm/m错误，也不能直接判成AP失败根因。为检验该机制，补充最后一对路线：

- unit：原highres/fine源尺度与入口不变，对32通道源投影逐token做无affine LayerNorm，再对
  输出残差向量做无affine LayerNorm；alpha保留为唯一自由幅度。原query/候选/几何属性/损失不改。
- unit_geometry：相同归一化、相同初始化及训练预算，RGB置均值，保留正确几何。
- 两项归一化作为一组“特征尺度稳定化”干预，不宣称分别识别每一项的因果贡献。无额外可训练参数。
- 完整10epoch last、有效batch8、LR1e-4等同原RGB-D臂；同seed44初始，E1..4按原随机流，
  E5..10用epoch_seed=1000000+44*1000+epoch，与恢复后的RGB-D预算/随机规则一致。
- 主比较unit对基座与unit_geometry；伴随所有原AP/Boundary2%/1%和500次名义配对CI。
  正向阈值与前述相同；只报基座增益不能证明真实RGB外观贡献。
- 输出LayerNorm在AMP下可能提升dtype，因此残差在加入receiver前显式转回query/receiver dtype，
  保持零系数时soft logits逐值恒等。强化smoke新增zero_init_logits_exact；初版仅硬mask恒等
  记录保留pre_dtype_calibration，不进入正式训练。
- 完成检查需验证source_delta_rms<=1.01（有效支持外为0，浮点容差），以及LoRA梯度非零、
  全无候选恒等、冻结基座SHA不变。该范数界只是实现标定，不是收益判定。
- GPU0/2各一条，独立目录unit、unit_geometry，监督器unit_state.json。

本轮训练矩阵至此固定为八臂，除已有协议要求的early/contextual正向后匹配控制，不再根据
中途AP继续增加模型路线。最终结论必须覆盖全部八臂、已预定控制及固定观察点。

## 用户修改主要评价目标：固定终点测试集评价（2026-10-22，读取新增探针test结果前登记）

用户明确要求：不以v1高指标验证集上的增益作为目标，将有效性判断放到测试集。
此指令替代前述本轮“仅train/val、不读test”的范围限制。旧记录保留作为历史，不删除旧结果。

- 训练仍只读原2142 train；原459 val及E1/E3、E10结果保留为诊断。不恢复基座优化器、不改结构、
  LR、训练轮数或已选定的E10 last；不依据test挑epoch，也不只测试val正向路线。
- 测试矩阵在新增test结果产生前固定：原八臂，加early/control和contextual/control两条同结构
  常量RGB训练对照，共十臂。early/control已在运行；contextual此前因val非正未触发，现无条件补齐，
  不等test方向决定是否训练，避免缺少同族归因。固定10轮、seed44、E5..10显式逐轮重设规则相同。
- 主终点改为原459 test上的Mask AP@[.50:.95]、maxDets250，比较固定E10 last对同入口FP32
  关闭分支基座；伴随Boundary AP2%/1%、AP75、分尺度AP、bbox/召回。沿用+.002绝对AP且
  500次image-paired bootstrap名义95%下界>0的“明确净增益”参考门槛；仍完整报告较小正向信号。
- 所有十臂均做正确输入完整test；四条真实RGB臂correct/early/contextual/unit还做相同检查点的
  常量RGB与循环donor RGB干预，共18次适配器推理，加一次基座。donor仍只置换RGB，保留本图几何。
- 十臂各自对基座，以及五个匹配内容比较（correct−geometry、early−early/control、
  contextual−contextual/control、unit−unit_geometry、native_geometry−native_zero）均计算三个
  指标及500次同图配对区间；bbox/分数/类别/GT须逐图一致。推理干预点估计不代替训练归因。
- 新CLI `--eval-split val|test`只允许eval动作；train/smoke/prepare禁止设test。默认val保持旧调用。
  test输出独立放置test_evaluation及test/，记录split、权重SHA、精度、实际459图、原检测scores和cap250，
  不覆盖val缓存。统一分析器新增split来源标记，不改AP/Boundary/重采样数学定义。
- 已运行训练的进程与源码快照不热更新；新增eval功能不改训练数学，所有源码版本仍按实际快照核验。
  修改前分析器归档analysis_before_test_scope.source.py，既有数学校准继续适用。
- 所有结果出来后统一判断，不依据test新增结构/延长预算或反复挑赢家。若再用这些结果指导后续开发，
  新数据划分的正式评价仍需保留未用于开发的test；本轮不声称解决v1分布问题或完成新划分泛化验证。
