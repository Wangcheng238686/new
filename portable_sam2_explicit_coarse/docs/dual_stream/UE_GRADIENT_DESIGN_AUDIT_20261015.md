# 双流提示路径：梯度与工程设计核定

日期：2026-10-15。审核者：gradient_design_audit。只读源码／既有工件核定，无GPU运行、无训练、无backward、无optimizer创建、无生产文件改动。

## 身份、证据及口径

仓库`/home/wangcheng2021/project/new-uav-dual`，HEAD `bd8e6ad7f755903c131babd0b856358cf895f483`，核读时工作树干净。下列源码路径以该仓库的`portable_sam2_explicit_coarse/`为根；SAM原生源码另行给出。E72 strict EMA原探针manifest：`docs/dual_stream/probes/results_20261015/p2_prompt_trace_run_manifest.json`，model fingerprint `fcbb8af545a25d5998469cb6a0fe6ed14fdde81ddb7ed9c2fc2b5064de362fb4`，manifest SHA `495841b2174242c87661e6b40dba2ddcca13d64b13798116efc8c933c679ca59`。

已阅读`docs/EXPERIMENT_GOVERNANCE.md`及`docs/dual_stream/UE_GT_PROMPT_FACTORIAL_20261015.md`。现有GT提示结果只证明在153图/4496次出现的GT共同框条件下，冻结SAM可以利用这组提示；不证明训练解除detach有效，也不证明UAV可实现该形状改善。

VERDICT：源码接线核定完成；存在明确的dense提示梯度隔离，未发现解除detach后另一个必然阻断coarse-logit梯度的算子。不能据此宣布解除detach是优先修复或能够带来性能收益。无需为重复验证静态可达性额外消耗GPU。

## 1. 实际现状：训练路径与参数

### 1.1 提示生成与loss

- `rsprompter/cspm/shape_prior_injector.py:99`建立SmallMaskDecoder；`:106`的context函数不使用SAM embedding；`:119`抛弃context/box/图像索引，`:123`只用RoI特征生成64²粗logits。该分支真实为roi_only。
- `rsprompter/cspm/mask_head.py:473`至`:497`将同一粗logits交给dense画布；`:546`交给点挖掘。
- `rsprompter/cspm/shape_point_miner.py:223`显式no_grad，`:224`对mask_logits.detach后sigmoid，再阈值、形态学距离及选点。点路径不会向粗头传梯度；离散选点不是仅移除no_grad即可合理端到端优化。
- `rsprompter/cspm/dense_prompt_utils.py:31`至`:32`在配置为真时detach粗logits，当前manifest dense_detach_input=true。
- `rsprompter/detection/anchor_roi_head.py:197`计算最终mask loss，`:206`附加tail loss，`:210`附加coarse loss；`:313`至`:317`把coarse_loss原张量挂到loss_shape_prior，并未detach loss本身。
- `rsprompter/cspm/mask_head_targets.py:276`至`:297`调用真实CoarseMaskLoss；`:197`至`:274`按正proposal裁GT并nearest到coarse大小。`coarse_mask_loss.py:158`至`:177`为sigmoid Dice+BCE；权重由`:106`与forward结尾生效。当前继承`configs/a3sgm_whu1024.py:175`至`:182`：weight=.1、fixed、BCE=1、Dice=1、boundary/distance=0。
- `rsprompter/cspm/mask_head_targets.py:354`至`:367`的标准最终mask loss独立加入loss_mask；`train/loops.py:114`至`:139`合并全部真实loss，`:204`反传。

因此，当前粗头的学习仅直接来自coarse辅助监督；这不等于整个UAV分支只靠coarse loss学习。**.1是loss系数，不能凭系数断言其梯度弱**：梯度还取决于目标归一化、链式Jacobian、不同模块尺度与裁剪。

### 1.2 PE冻结、形状gate与直接视觉路径

- `rsprompter/cspm/mask_head_init.py:135`至`:174`冻结稀疏PE和当前WHU继承配置的mask_downscaling；`configs/a3sgm_whu1024.py:144`至`:146`明确train_mask_downscaling=False。原训练日志无prompt_encoder optimizer组，与配置一致。
- `rsprompter/cspm/mask_head_helpers.py:91`至`:100`对shape_dense_alpha_raw取sigmoid；E72实际值0.3281454742；这不是UAV融合alpha或gate。
- `rsprompter/cspm/mask_head.py:573`直接调用PE，**没有no_grad**。`:614`至`:620`先按box support限制PE残差，再由shape alpha乘入。PE冻结参数并不等于冻结输入梯度。
- `rsprompter/detection/dual_detector.py:103`至`:117`分别融合post-neck FPN和SAM image embedding，`:148`返回high不变；`:150`至`:173`将x交给RPN/bbox/RoI、将embedding和high交给mask分支。
- `rsprompter/cspm/mask_head.py:378`至`:410`直接将SAM embedding和high-res送decoder。`rsprompter/decoder/sam2_decoder.py:188`至`:206`用conv_s0/s1处理high-res，调用真实SAM decoder，全程无detach/no_grad。
- `rsprompter/decoder/decoder_tail_refiner.py:85`至`:96`的tail值路径是coupled的；只有选位置逻辑detach，mask loss及tail margin可以回到decoder、SAM融合和共享BEV，而不是只有tail自己的参数。

### 1.3 实际optimizer证据

原始日志：`/tmp/ue_uav_experiments/20261014_ad_seed42/dual/train_EXP-0006_pid745647.log:93`至`:101`。

| 组 | 可训练参数量 | 初始记录LR | 结构上接收的主要损失 |
|---|---:|---:|---|
| 卫星backbone LoRA+可训练neck | 1,492,528 | 2e-5 | 检测、coarse、直接SAM mask/tail路径 |
| shape_prior粗头 | 960,385 | 2e-4 | 当前仅coarse辅助loss |
| SAM mask_decoder组 | 4,082,497 | 1e-5 | 最终mask、tail margin（具体子头可unused） |
| decoder_tail | 38,217 | 2e-4 | 最终mask、tail margin |
| other | 37,130,794 | 2e-4 | neck/检测头等各自路径；含shape_dense_alpha_raw |
| UAV backbone LoRA+可训练neck | 1,492,528 | 2e-5 | FPN检测/coarse路径与SAM直接mask/tail路径汇合 |
| CVT | 5,518,672 | 2e-4 | 同上共享BEV路径 |
| fusion | 494,854 | 2e-4 | P3/P4接受检测及选中层coarse；SAM接受mask/tail |

分组源码`train/optimizer.py:53`至`:103`、`:182`至`:207`。iou_head另有132,612参数，optimizer有组不证明其有监督/非零梯度；分割score仍detector，不能将组存在误作贡献证据。

原日志`:138`至`:141`已有UAV、CVT、FPN融合、SAM融合非零有限梯度遥测。但这是各loss合计的训练梯度，不能据此分解每个loss贡献，也不能比较不同模块norm断言“有效信息大小”。

裁剪源码`train/optimizer.py:231`至`:256`将tail/UAV/CVT/FPN/SAM融合分成aux groups，其余为base；`train/loops.py:230`至`:236`根据resolved开关每组max_norm=.5或全局.5。**粗头和shape_dense_alpha_raw当前都在base裁剪集合**。解除detach会改变base集合内的梯度组成，甚至影响其它base参数裁剪比例；不要未经测量直接加粗loss权重，也不在同一实验顺便改变clip分区。

实际原训练日志`:111`明确`Clip decomposition ON: base 663 tensors at max_norm 0.5 + aux groups [6, 418, 59, 16, 8]`，所以当前不是全局共享一个.5预算；base内部仍共享裁剪预算。

## 2. 仅改dense.detach_input=False，是否真的可微？

### 算子链核定

1. `dense_prompt_utils.py:33`温度除法可微，当前temperature=1。
2. `:40`至`:41`clamp[-8,+8]在区间内可微，区间外梯度为0；不是普遍梯度截断，亦不能忽略饱和比例。
3. `:93`新建常量canvas，`:116`至`:122`双线性插值并切片赋值。该赋值对源logits形成CopySlices计算图，不是把logits转numpy或detach。box floor/ceil/.item决定离散位置，**不对box坐标提供梯度**，但不会切断源logits梯度。
4. 原生`/home/wangcheng2021/project/new-uav-dual/sam2/sam2/modeling/sam/prompt_encoder.py:57`至`:65`是conv/LayerNorm/GELU的mask_downscaling；`:134`至`:137`及`:195`至`:196`直接调用，没有no_grad。权重requires_grad=False时，仍可向输入canvas传播梯度。
5. `mask_head.py:614`至`:620`固定support乘法、sigmoid alpha乘法仍可微。support0 ROI的所有dense残差梯度严格为0；解除detach不能修复它。非零support也不保证每个coarse像素收到非零梯度。
6. decoder与coupled tail无额外全路detach。若损失在当前ROI有有效Jacobian，可回到coarse logits，再到RoI/neck以及当前实际使用的FPN融合层。

**结论是静态可达，不是已经测得当前样本梯度大小、方向一致性或资源峰值。** 本次不额外跑GPU。需要正式进入训练方向前，再单batch分loss autograd诊断，保持无optimizer.step；只报告reachable/finite/nonzero及冲突，不报告性能改善。

### 配置和合同边界

模块构造器接受detach=False（`mask_head_init.py:265`至`:275`没有硬拒绝）；但不能将此等价为“改旧配置然后无痕strict复现”。

- `rsprompter/architecture_contract.py:44`至`:51`对全部规范化model config求SHA；detach改变必然改变model_fingerprint。
- UE `rsprompter/uav/contract.py:55`至`:65`用同一fingerprint；architecture_id字符串可能仍相同，**字符串相同不代表合同相同**。
- `inference/ue_rebuild.py:24`至`:29`校验旧snapshot合同；不能编辑旧snapshot或绕strict检查，把新训练路径说成旧架构。
- 研究性梯度诊断可以先按旧合同strict重建，再运行时只改head.dense_prompt_cfg的非持久属性，记录before/after及counterfactual身份，参数/输出前向应完全相同；不能保存成冒充正式新合同的checkpoint。
- 正式新训练应新配置、新fingerprint、新run/EXP登记并从通用预训练独立开始，不能把旧UE单流权重热启动当最终双流结果。解除detach属于训练动力学型，冻结诊断并不替代性能实验；遵守governance B路由，而不是把冻结oracle PASS当自动训练授权。

## 3. P2与SAM high-res并非同一入口

`anchor_detector.py:125`至`:164`先从backbone outputs选SAM embedding、高分辨率特征，再单独经neck产生x；`dual_detector.py:103`融合x时已经post-neck，不会回流改写SAM high-res。

`anchor_roi_head.py:130`将FPN送mask RoI extractor；mask head legacy p2_feature/renderer_features参数在`mask_head.py:339`明确不消费。P2仅经RoI→粗头→提示接触SAM，不存在秘密direct P2图像通路。

当前`configs/ablations/ue_rgbd_dual.py`的levels=(1,2)只构造P3/P4融合，P2未融合。153图GT RoI中P2有3657次，其粗头输入没有当前FPN UAV残差；SAM embedding融合也不会经roi_only粗头补入UAV。**解除detach只解决“loss能否教粗头生成有用dense”问题，不会凭空给读取P2的粗头提供UAV细节。** UAV backbone仍可通过SAM直路学习。

这里“没有直接UAV输入”只指当前forward中mask RoI的显式特征通路：共享卫星backbone参数在双流训练期间已受SAM融合后分割梯度影响，因此不能把P2表征称为完全未受UAV训练影响，也不能将其当作独立训练单流特征。

若UAV互补探针支持针对粗误差修正，最小P2候选是独立P2残差适配器，只改mask RoI使用的P2，保留原RPN/bbox输入及P3/P4/SAM入口，配合原coarse GT监督。这样可先归因粗提示而不同时改检测。该建议属于待验证设计，不表示应立即训练。不要继续把P3融合器跨尺度复用的零收益当作训练过的P2否定结果，也不建议一次扩所有FPN层。

## 4. 条件式决策表（须与另两并行任务合并）

| dense形状/幅度拆分 | UAV互补证据 | 优先工程方向 | 当前不允许推出 |
|---|---|---|---|
| 预测形状±4已带来主要改善，GT等幅度额外改善小 | 任意 | 先核校准/量化及预测框下部署效果；无需自动改梯度 | 二值化必然普适或UAV无用 |
| 等幅度GT形状仍明显更好 | UAV特征对粗误差有独立增量 | 优先局部P2/RoI适配或coarse残差修正；保留原aux监督 | 新增模块一定改善AP |
| 等幅度GT形状仍明显更好 | UAV无可靠增量或未测 | 暂缓UAV粗头模块训练，核对特征质量、配准/可见性与诊断容量 | GT上界证明UAV可学 |
| 有dense空间且最终mask梯度与coarse目标需协同 | UAV证据支持或先做通用结构对照 | 进入单batch分loss梯度诊断；通过后才立新detach训练臂 | 梯度可达即一定收益 |
| 小框support0占比问题 | 任意 | 单独做support规则冻结对照，保留点分支 | 解除detach或增加alpha可解决support0 |

辅助监督权重.1先保持。不建议只用GT框训练局部修正头：GT框冻结探针是条件诊断，真实训练/部署读取proposal boxes，GT-only学习会改变框分布。若以后做局部预算训练，应沿原positive proposal分配/采样和GT crop target、匹配同预算卫星-only对照，记录重复建筑/空间划分，不能用验证集训练或选择通道。当前未实施此训练。

## 5. 最小实施方案与验收守卫（未来候选，非本次改动）

### A. dense梯度反馈候选

- 新配置唯一model键差：`roi_head.mask_head.dense_prompt_cfg.detach_input` True→False；保持PE冻结、shape alpha初始化、coarse loss=.1、clamp/temp/support、点挖掘、融合levels、LR、clip、采样和阈值不变。
- 先做无更新梯度核定：旧合同strict重建，固定样本、框/assignment、视觉输入，两个cell只变detach；前向coarse/点/canvas/PE/dense/final/logits/loss逐值相同，参数hash及version不变。
- 分开最终mask、coarse辅助、tail margin求梯度，禁止只看总loss；检查coarse输出、粗头、RoI、FPN融合、SAM融合、CVT/UAV LoRA的reachable/finite/nonzero；PE参数始终无grad；support0和clamp饱和列为预定阴性/限制。
- 对同一参数组报告mask-vs-coarse梯度cosine及norm ratio，仅是局部冲突证据。不同模块norm不可直接排序“信息强弱”；若新增遥测须先定义字段/DDP归约/标定，不能临时往正式训练日志塞未定字段。
- 资源：PE在detach=True且参数冻结时无需保留输入梯度图；False会额外保存dense卷积输入/中间激活供反传。训练`anchor_roi_head.py:189`直接整批positive RoI，不使用预测路径`:413`的32 chunk。不能沿用冻结推理chunk32显存当训练预算；需测实际positive RoI数与AMP下peak。没有本次实测资源数，不能承诺显存稳定。
- 验收通过只准“可达/数值/资源工程核定”；后续性能以独立正式臂完整bbox+segm AP、固定协议为准。前向完全一致意味着现有冻结IoU不可能区分detach开关优劣。

### B. P2/RoI局部适配候选

- 新模块独立命名、参数显式进入optimizer及clip组；不可误将P2模块留在未审查other组。零残差初始化，先验证disabled/zero alpha时全输出逐值复现。
- 仅mask用P2拷贝分叉，RPN/bbox用原x；shared BEV仍可接受其它loss，不声称整个UAV分支训练隔离。
- 训练前完成新旧snapshot/fingerprint键差审核、原臂指纹回归、strict重建、梯度可达/coverage0阴性、gate初值与资源检查。保持P3/P4/SAM原入口便于单变量归因；不同时解除detach、修改support、增加aux weight。

## 6. 总结

可以同步完成形状/置信拆分、UAV互补诊断和工程核定；第三项的性能设计决策必须等前两项证据。当前可确认“提示dense终局梯度被主动隔离”以及“P2粗头当前无直接UAV输入”；不能确认这是净收益不足主因。保留SAM视觉直路、点分支及coarse监督，下一步选择应由形状/幅度及UAV可预测性结果决定。
