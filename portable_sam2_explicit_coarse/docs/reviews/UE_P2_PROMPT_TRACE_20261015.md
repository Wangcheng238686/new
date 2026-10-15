# P2读取路径只读追踪独立审核

日期2026-10-15，审核者frozen_probe_audit，未参与实现，只读CPU核验。

## 实施审核全文

P2读取路径追踪首次定版实施审核：VERDICT：PASS，可进入原预注册3图smoke；153正式须待hook-on/off逐像素验收，不授权训练或新方案。
只读静态核对：Observer各pre/forward hooks均返回None，仅detach/CPU副本，不改模型输入/输出；按一次head→coarse→miner→PE→decoder→head完成序列记录，每chunk结束current清空；merged严格按chunk顺序cat，boxes顺序及chunk数/长度跨臂检查充分。captured ROI==coarse输入、coarse==output.raw_logits、miner输入==coarse的assert与当前实际调用链一致。decoder pre拿的是实际送入的sparse/dense，decoder post在tail前CPU复制raw，head post拿tail后，避免tail状态污染raw。
P2四组构造与上轮相同，donor BEV在目标几何下提取、finally恢复RGB；不再RPN/AP。每GT终点IoU精确对账原四臂，非P2 ROI/coarse/paste逐值不变；连续其他stage若因整chunk全invalid token2↔6而变化，预注册已单列批处理结构效应。sparse补齐只改CPU编码副本、不改实际序列，保留真实token count；点变化仅有效并集、移动距离按双方有效，正确。finalpaste XOR与raw/tail低res阈值翻转分别记录，不混淆。
CPU读E72 snapshot确认roi_only、dense detach_input=True、points_box_dense、restrict_dense_prompt_to_box=True、PE全freeze、coarse auxiliary固定weight .1、coarse阈值.5；源码miner内部torch.no_grad+mask_logits.detach，dense变换先detach_input。故final loss不沿coarse生成提示反传属于静态接线事实；本轮不反传，不应写成梯度大小/训练瓶颈已证。SAM embedding直连decoder仍存在。
GT质量target调用实际get_coarse_targets，输出[N,64,64]与coarse匹配，floor/ceil+nearest裁剪口径正确，仅诊断、不入模型。point local_yx用于采样同target，invalid不计错误。support32²取实际box-support函数；zero_support时dense==no_mask_base有强assert。阈值由test_cfg .4实时计算logit−.405465，不误用0；coarse .5/0logit独立。
提醒解释：raw/tail flip是低分辨率logit网格用同阈值的辅助诊断；实际部署还含paste插值，应以paste_xor确认最终变化。relative L2或dense_final_to_pe_delta_ratio不能跨stage解释信息保真/有用性；若都变但质量不升，不能说decoder压制了有用信息。未发现阻断项，待smoke。

## 默认chunk属性修复

首个smoke在manifest前因直接访问未显式设置的mask_roi_chunk_size而退出，没有可解释结果。改为与生产predict_mask一致的int(getattr(roi_head,'mask_roi_chunk_size',32) or 0)，保留0整批语义；未改模型。原smoke.log保留，smoke_fixed.log用于修复版。

## 汇总静态审核全文

Trace汇总静态审核 VERDICT：PASS，待smoke实际端点及形状验收。生产chunk fallback修复已确认：int(getattr(...,32) or 0)，0时整批，符合predict_mask；首个AttributeError未产生结果，保留原log即可。
汇总的stage strict_changed与RMS>1e−6计数为逐GT到达数；不能跨stage直接比较mean_relative_l2作信息保真。point_slots正确按每槽有效数作分母，point_correct已被valid mask约束；invalid槽不计错。point_changed_roi_count/slot total的分母分别GT和四槽有效并集，报告百分比时应保留此区别，不默认除4496×4。
P2/nonP2、尺寸、dense-support0/非0组按baseline身份固定且所有臂身份精确一致，237建筑等权正确。seed1750共1000次同图重采样，coarseIoU/finalIoU各臂−baseline使用sum/count，匹配预注册。没有新增益门，correct−donor只是描述。
joint_counts是观测事件重叠而非干预因果证据：PE变化而finaldense不变可能来自box支持/valid替换；dense变化而paste相同含阈值/paste因素；sparse_dense_both_same_paste_changed尤其须结合真实token count（CPU补0的2↔6序列不等价）解释。当前limits已明确。
小建议（非阻断）：记录中已有point_move_distance_sum/point_both_valid_count，若正文报告点平均移动，须按双方有效总数除，不除所有槽/GT；当前汇总未输出该平均，不能用point_changed数替代。静态门/质量方向解释无越界。

## 修复版smoke准入审核全文

VERDICT：P2读取追踪修复版三图smoke工件验收 PASS，准入原预注册唯一153图只读追踪；不追加GPU对照/AP/训练。
独立CPU验收：manifest3图287/288/295、45GT、complete=false，strict EMA无missing/unexpected、无训练optimizer、参数version未变；当前script/helper/prereg/donor-map SHA均精确一致。3条不变量全True，包括每臂hook-on/off逐像素mask相同、端点IoU相同、非P2 ROI/coarse/paste相同、hooks移除、几何RGB恢复；BEV hook均2。
4组各45条身份/顺序相同，GT层级26P2/19P3；全部180个endpoint IoU逐项精确等上轮各自模式。baseline所有阶段delta=0、paste XOR=0；其他模式19个P3 ROI/coarse strict_changed均false且paste XOR=0。stage RMS/MAE/relativeL2、粗IoU/BCE、点coords/local_yx/valid/correct、dense支持和token数均数值/分母合法；各组2个support0实例dense_no_mask_delta_norm精确0，invalid点未算正确。生产运行形状assert（coarse1×64²/canvas1×128²/dense256×32²）全部通过才产出DONE。
实际字段：chunk_size32、三图chunk=[13]/[14]/[18]，GT boxes顺序对账通过；本smoke未遇>32的多chunk图，正式运行既有merge顺序/数量/boxes assert将继续约束，未扩大测试。dense_shape_alpha=.32814547419548035、detach_input=True、coarseaux=.1；mask_thr=.4/logit−.4054651081081643，符合真实部署。paste改变数q19/donor19/one21仅smoke描述，不作质量结论。
首次缺属性异常在manifest前终止无解释结果；使用生产默认getattr32的修复不改变模型/探针数学定义。未发现阻断项。

## 正式153图结果独立审核全文

P2→提示→最终mask读取追踪独立结果审核全文

VERDICT：PASS。未发现阻断项。本轮只完成既有冻结P2迁移四组的只读路径定位，不新增益结论，不授权新探针、训练或默认模型变更。

审核者frozen_probe_audit，未参与被审脚本设计开发。只读CPU核验/tmp/ue_p2_prompt_trace.py、/tmp/ue_p2_prompt_trace_summary.py及p2_prompt_trace/{full,summary.json,PREREGISTRATION.md}。未运行GPU、反传、训练或重复AP。

1. 来源及身份：正式manifest complete=true、153图/4496出现，strict EMA无missing/unexpected，checkpoint、manifest SHA及image顺序与历史相同，无训练optimizer、参数version不变。当前probe/helper/donor_maps/预注册SHA独立重算均匹配manifest。四臂各4496条记录，(image_id,gt_slot,global_id,bbox_area,roi_level)逐项一致且与上轮gt_identities一致，建筑237栋、GT官方P2/P3计数3657/839。

2. 端点及不变量：四臂共17984个GT框最终IoU逐值精确匹配上轮各自模式。153条invariance全部要求项True，BEV hook每图2次；非P2 ROI/coarse/paste不变、RGB几何恢复、features hash及observer移除均通过。baseline所有stage delta/点改变/paste XOR均0。正式observer_smoke_pixel_exact=False是预注册节省成本而未重跑off；先前修复版3图四臂on/off逐像素相同已独立验收PASS，不能把正式false误报为观察器扰动失败。

3. chunk/形状：独立以每图GT数和生产chunk32重算每图chunk_sizes，153图全部一致，其中42图实际覆盖多chunk，记录拼接身份/boxes顺序及总数一致。当前实现运行时对ROI/coarse输入一致、miner/coarse一致、coarse输出一致及coarse[1,64,64]/canvas[1,128,128]/dense[256,32,32]形状有assert，完整产出证明通过。没有为验收额外扩大GPU测试。

4. 汇总数值：独立从17984条逐GT记录重新聚合四臂×all/P2/nonP2/small/medium/large/dense_support0/nonzero的全部质量均值、stage mean RMS/relativeL2、strict与RMS>1e−6到达计数、点有效/正确/准确率分母、点改变、coarse/raw/tail/paste变化、joint_counts与dense/PE差范数比，均与summary一致(<1e−12)。点correct未包含invalid槽；准确率按每槽有效数量作分母。阶段的数值改变与质量提高必须分开解释。

5. 主要到达结果：预定correct_q在3657个P2 GT中，ROI/coarse/canvas RMS>1e−6均3473；PE dense3444、实际decoder sparse1282、实际decoder dense3204、raw及tail logits3397；最终paste改变2092。非P2的ROI/coarse/paste均不变。P2变更确实可沿coarse→提示进入decoder并改变最终mask，不能称完全未被读到或被单点彻底阻断。各阶段覆盖集合并非严格嵌套，sparse/dense分支、阈值与RMS口径不同，不可简单用相邻计数相除称信息存活率。

6. 质量及区间：按seed1750独立重生1000×153配对图像采样，以图像sum/count重算三变体各coarse/final的六组差值CI、237建筑等权及全部分组，均与summary<1e−12。correct_q−baseline coarseIoU=−.000003238725159，95%[−.000046264669620,.000035852425263]；finalIoU=−.000031878300372，95%[−.000134290851408,.000081723782434]。两区间均跨0。建筑等权分别+.000019791727112/+.000012637926761。这与上轮终点基本持平相符，未显示粗质量先明显改善而后被decoder消除，不能说“有用信号被decoder压制”。较大one也只是压力对照，不后选。

7. dense支持缺口：197/4496为support32²=0，全部属于small（197/881=22.36095%）；四臂这些实例dense_no_mask_delta_norm均精确0，实际decoder dense为原no-mask基底。仅说明这一盒内32²网格支持定义下dense分支未注入；sparse点/box仍可能存在，不能说这些目标完全没有提示或完全绕过UAV。支持基于GT框，预测框部署比例可能不同。跨stage dense/PE范数比同时含box支持、valid替换、alpha及有限精度，不能独立等同门控值或信息保真率。

8. 真实配置/静态路径：mask_thr=.4、对应logit−.4054651081081643，coarse阈值.5；shape_dense_alpha=.32814547419548035、dense_detach_input=True、coarse auxiliary weight=.1、roi_only、points_box_dense、chunk32与此前checkpoint/源码核定一致。FPN P2在当前roi_only接线中经ROI→coarse→sparse/dense提示进入SAM decoder；它不是额外SAM高分辨率视觉输入，SAM image embedding/high-res仍为原值。源码dense detach及point no_grad使final-mask loss不沿这两个coarse提示生成路径回传；这是继承接线的静态事实，本轮inference_mode未测梯度大小，不证明训练动力瓶颈或bug。SAM图像embedding直连segm路径仍存在。

9. 解释边界：raw/tail的logit翻转是在低分辨率网格用相同阈值的辅助量，实际部署仍含paste插值，以paste XOR判断最终变化。coarseIoU/BCE和点准确率只针对实际get_coarse_targets生成的ROI-local64² GT诊断，不是输入模型的oracle，也不是AP。读出观测不分别隔离sparse/dense/tail因果；2↔6真实token序列需按记录结构区分，补0副本不能当行为等价。本次四臂没有token-count change，但该限制仍成立。

10. 范围/复现限制：固定E72与当前153开发验证图、4496重复出现/237建筑，非完整生产集或独立test。图像bootstrap仅探索性，不覆盖空间相关、选模及训练seed方差。本次从已保存逐GT stage统计重聚合，没有重新生成原始中间张量/像素mask；端点以历史IoU逐值对账，像素无扰动证据来自已验收smoke与正式运行assert。未重算RPN/bbox/AP，上轮固定框条件AP结论沿用，不能将路径到达替代结局收益。

最终可支持的表述：P2迁移扰动已到达coarse、提示和SAM decoder，部分最终mask也改变；现有冻结P3权重跨尺度迁移并未产生稳定有益质量变化。不能据此宣布decoder压制有用信息、P2无用、应解除detach或应立即训练新结构。
