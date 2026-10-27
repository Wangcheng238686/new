# G＋UBR分批实施计划：少改动、可解释、逐批审核

2026-10-27。用户批准细化并由子代理实施，主代理负责审核与汇报。全量100epoch预算不在本次发射范围。历史训练与冻结数值源保持不变，新文件独立，禁止未声明改动旧模型/权重语义。

## 1. 一条论文主线，四个工程批次

任务主线为“几何增强实例发现，共享空间修正器消费几何改善掩码”。不声称无损压缩、完整三维重建或所有模态必要。P0/P1/P2分开是为定位原因，不是增加三个研究模块。

| 批次 | 只解决什么 | 保持什么 | 交付/判定 |
|---|---|---|---|
| P0 当前实施 | 恒等起步空间UBR、预测邻居工具 | 不接生产、不改变候选/模型 | CPU行为门＋四rank组件梯度/资源门，不是AP证据 |
| P1 接线前先坐标核定，再换尾部 | 新opt-in UBR mask/ROI head，UDPR不实例化 | 当前G/旧交互/中心头与loss/候选评分不变 | 概率paste与几何坐标门、真实模型四卡门、替换短比较 |
| P2 共享候选确认 | 补充候选加入共享RoI训练与判别 | 固定P1尾部与监督权重 | 分来源曝光/净补漏/新增误检，不与P1混为同一改动 |
| P3 有证据才做 | 先8→10候选，再实例条件RGB-D读取 | 不同时加宽骨干/删旧CNN/改loss | 各自单改动与匹配内容对照 |

## 2. 固定首版简单选择

- UBR40ch，hidden48/depth7单conv残差块、GN8+SiLU；带宽8原生目标像素。输入prob/SDT/neighbor detach，decoder feature可微。末层零初始化，无额外alpha门。
- 输出改为无界residual：z_final=z+B*r。它保留带内原z的直接梯度、初始输入网格精确恒等，不再使用b+r绝对替换。它仍是空间边界带卷积修正，不回到64点UDPR。此处是用户认可讨论后的合同修订，不称远端原加法探针逐位复现。
- 部署比较符固定prob>=0.4（已读本地mmdet FCNMaskHead），SDT与neighbor二值化使用同一比较符。empty/full mask SDT以无边界语义产生空band；不自动全图修改。首版不加入连通域重写。
- loss维持检测四项各1/final1/coarse .1/UBR .1/实例1；实例内部尺寸 .1；UBR focal gamma2、无额外alpha，Dice仅分母eps1e-6。各目标的职责和分母不互相平均，不动态调lambda。
- 同一整图/窗口的几何/特征坐标显式输入，core不自动resize。GN裁窗不等价整图：窗口为新模型明确合同，不能伪称数值等价；未核定窗口规则前资源先按整网格测。
- 不加来源embedding、动态采样配额、learnable温度、额外可见性loss、模态dropout或PCGrad。

## 3. P0实际文件

- `rsprompter/uav/unified_boundary_refiner.py`：SDT/band、40ch空间UBR、unweighted带内focal+Dice，图等权归一化、checkpoint可选。
- `rsprompter/uav/ubr_prediction_context.py`：纯预测候选确定性NMS/lineage与self排除、同步neighbor sum；禁止GT输入。候选metadata可先选择再解码，避免为被NMS过滤的框生成掩码。
- `tests/test_unified_boundary_refiner.py`、`tests/test_ubr_prediction_context.py`：非恒等输入/梯度/空分支/归一化/候选顺序等有意义行为检查。
- `scripts/probes/ue_ubr_component_gate_20261027.py`：四卡UBR组件AMP更新、空rank、参数一致性、250实例顺序推理峰值/时间；synthetic输入，不是完整SAM/G资源或AP验收。结果单独存/data checkpoint新目录。

## 4. P1必须先定：RoI语义、概率paste与有限背景

当前实际配置`final_mask_coordinate_mode=roi_local`，旧训练用GT RoI裁剪目标，旧推理先sigmoid再按bbox paste概率。SAM2捕获embedding的来源不意味着可直接当卫星512²逐像素输入。

P1第一步仅新增RoI→卫星适配器与landmark/checkerboard/paste等价门：明确把raw预测与32ch特征采用同一已声明映射。概率paste不同于logit paste；零residual要求native/core logit逐值恒等，最终旧路径二值/分数完全一致，概率误差受预定epsilon约束，不承诺0/1反logit仍逐位恒等。bbox外用有限背景logit的epsilon必须记录，否则-inf加residual不能向框外增长。也不能为了修此点悄悄切full_image目标。

新增opt-in maskhead/ROIhead+decoder capture shim，constructor强制UDPR disabled；旧文件保持冻结，旧模型重建隔离。新decoder沿既有hook捕获32ch而不借用UDPR tail_outputs，逐调用清空，防止分块读旧张量。

## 5. P1训练邻居合同

上下文来自当前模型无梯度预测，复用提取特征，候选集合/阈值/NMS/上限与当前P1推理族一致；不能将GT正RoI直接当邻居集合。mask均为精修前预测，同步构建，不依赖实例处理顺序。排除self依靠稳定预测candidate ID和NMS归属，不靠GT assigned instance ID。

P1仍沿旧正RoI mask训练，对补充框的训练覆盖缺口公开保留，不宣称端到端统一完成；P2才修复。参考neighbor工具的标准NMS不能直接替换P1旧评分/保护prefix规则：集成时必须使用P1当前已选集合或明确等价核验，不能把helper误用变成额外候选策略改动。P2再使用统一NMS路径。

## 6. P2共享候选：不添加复杂配额

RPN<=1000、UAV<=250，框detach后合并，进入原assigner/RandomSampler。总RoI256，pos_fraction=.25，GT proposal沿旧add_gt=True；首版来源盲采样，不新增每来源quota。仅离线报告两来源正/负曝光，确认UAV长期缺乏训练曝光后才考虑固定配额。

保留中心/offset/logwh直接监督。补充正框接受共享分类/回归/mask/UBR，负框接受共享分类；几何头的框经离散解码不做STE。推理所有候选走共享RoI后统一NMS/top250，不增加预测头。取消原.25人工评分和protected-prefix属于P2显式合同变化，需要旧策略对照。

## 7. 最小评价与预算纪律

P0只工程；P1先过坐标/初始化/空rank/真实反传资源/续训RNG门，再比较S与旧单流、D0与旧G尾部替换。尾部效应使用同上游D0/Dc/Dr，Dc只常量化UBR端口，上游RGB/深度不变。匹配constant是内容结论的必要对照，不因正确臂先涨就省略。

短预算只诊断优化与净纠错，不用10epoch阴性否定方向；正式预算、val选权、非劣界与CI规则在正式发射前锁定，不按test排名定参数。本次不启动100epoch、不承诺短门通过代表AP收益。训练实验按用户要求每实验四卡，空闲共租/不足排队，不抢占其它任务。

## 8. 审核与责任

实现代理分别负责core、context；第三代理独立核工程接线。主代理核代码、重跑组合CPU门、核四卡进程与实际产物、按任务source SHA/Git提交归档。只暂存本轮文件及本轮指南/字段新增段落，不吞入旧dirty修改。新字段仅工程产物，不扩逐步训练遥测、不参与loss/门控/选权。
