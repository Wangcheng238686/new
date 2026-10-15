# GT形状提示条件oracle冻结探针

2026-10-15；EXP-0006/0007；仅当前冻结开发子集153验证图、4496次出现、237栋建筑，非独立test。用户两个描述是同一核心对照，本轮唯一两组，不重复GPU计算。

## 固定干预

baseline：E72 strict EMA、正确UAV、原模型所有特征和GT共同框，逐项复现p2_prompt_trace/full/baseline GT IoU。
joint_gt_prompt：实际head.get_coarse_targets得到GT框内64²二值target（floor/ceil crop+nearest），以固定有限logits前景+4/背景−4替换shape_injector.mask_decoder输出，原后续转换同时生成点与dense提示。oracle hook先于只读observer注册；替换后的粗logits也是输出coarse_outputs，故oracle coarse IoU非空target时为1（空union按原quality口径为0），仅输入构造校验，不叫学习质量提升。固定幅度不搜索、不依结果更改。

只改变这一个中间输出；原GT框、RoI、P3/P4、SAM image embedding/位置编码/high_res、点挖掘规则、dense detach/support/gate、mask阈值.4、decoder/tail/权重不变。不修复small目标支持，不训练、不建optimizer、不改生产源码、不算RPN/AP。oracle是条件诊断，GT形状与原预测分布不一致；联合点+dense无法分别归因。GT64局部形状不是完整512分辨率真值，bbox量化、重采样、相邻实例背景和负点可用性均有限。无改善不能否定一切更好提示；改善也不能证明UAV可生成此提示。

## 准入和不变量

静态独立审查后仅3图smoke（287、288、295）；baseline observer开关最终mask逐像素相同；同一注入hook返回原output.clone()的sham，各阶段和最终mask精确相同。正确GT原输出注入实际进入point和canvas，原粗logits仍与baseline精确一致，RoI不变；每chunk按生产32处理，target offset/GT boxes顺序/数量精确。正式153只baseline+joint_gt_prompt，无额外sham以减少算力；全153须另获主代理准入。

记录模型参数_version、特征hash、原RGB/深度/几何hash不变，所有hook移除；所有张量有限；shape64²、canvas128²、dense32²；support=0时dense必须仍为no-mask基底。manifest记录checkpoint流式SHA、代码/依赖SHA、GT manifest SHA、git状态、配置/alpha/阈值、无optimizer、耗时。baseline终点逐GT精确对账前轮，source原粗logits分位数/mean/std用于判断OOD，不反过来选择幅度。

## 逐GT输出与主比较

baseline.json/joint_gt_prompt.json沿前轮trace字段：image_id/gt_slot/global_id/bbox_area/roi_level、gt_iou、coarse_iou/BCE、predicted_coarse_iou（仅原预测）、coarse_target_fg_pixels、original_coarse_logit_quantiles(min/p05/p50/p95/max)/mean/std、oracle_input_exact、point valid/correct/labels/coords/local_yx、dense_support_cells/no_mask_delta_norm、sparse token长度、阶段roi/coarse/canvas/pe_dense/sparse/dense/sam_raw/tail_final delta RMS/MAE/relative L2与变化计数、point变化、paste XOR和实际阈值翻转。oracle_input_exact在baseline为false（不适用），oracle必须全true。

主终点全4496个GT框最终paste IoU的joint_gt_prompt−baseline，不仅P2。1000次配对image sum/count bootstrap，seed1760；探索性headroom门均值>=.005且95%CI下界>0。该门不是统计泛化/净增益证明。报告全部分组：small(<32²)/medium(<96²)/large、P2/nonP2、dense support0/非0、237建筑内部均值再等权；不按结果后选分组。点准确率仅有效槽，GT64采样；support不是UAV几何coverage。保持两组框和权重固定，最终IoU改善才说明当前decoder可在本条件下从更准确形状提示获益；连续阶段变化不等于有益变化。空/全前景局部GT的点slot失配单列解释。
