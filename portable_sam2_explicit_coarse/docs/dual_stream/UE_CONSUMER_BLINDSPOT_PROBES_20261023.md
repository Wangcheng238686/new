# 子代理盲区复核：S1-R与N2-shared

## 独立复核与优先级

用户授权利用卡0/1剩余显存并派子代理复核。`probe_blindspot_review`与`consumer_topology_review`独立审查，均指出N2 capture顺序问题；后者进一步从实际训练合同、tail target及paste发现S1错误坐标合同。主代理逐处核实后，优先两个确定接线问题；仅屏蔽tail辅助梯度/降低tail权重方向后置。L1四图梯度较强或冲突不足以证明所有训练被tail损害。

## 纠错：旧S1无效并停止

此前依据“完整image/highres embedding重复”推断UDPR输出格为全图，是错误的。实际EXP-0017合同 `final_mask_coordinate_mode=roi_local`；`anchor_roi_head._apply_tail_loss`用每proposal的GT crop监督native选点；`mask_head._predict_by_feat_single`按bbox paste。**监督与部署决定输出格语义，不能只看输入是否裁剪。**

旧S1直接把native选点映射到512全图，读取了错误UAV位置；点级GT诊断同样错误。旧CPU gate只证明错误假设自洽，没有核对真实crop/paste。旧S1训练与诊断不得作为正确对齐UAV证据的检验。

已停止仅本任务自有的旧S1 owner3378228、两个train process group3378246/3378251及summary3379439。其baseline已退出；旧权重/日志/source保留，`state.json`标记retired_coordinate_error，另存state_before_retirement.json、state_retired.json和retirement.json。没有终止其它owner或修改其数值source。终止前校验实际cmdline；初次处理已退出baseline的空cmdline时审慎停止脚本，再排除zombie后完成，未误杀其它任务。

本结论不把旧L/N2统一判成错坐标：最后GELU的mask输出有roi_local监督；前层重复global feature与后层ROI语义的转换更复杂。G0空间诊断本来使用bbox映射，不因旧S1错误而整体无效。

## S1-R：RoI坐标修正的UDPR点证据

新模型 `udpr_point_roi_probe.py`，独立runner `ue_udpr_point_roi_probe.py`。点位置为：

`uv = bbox_xy1 + ((col+.5)/W, (row+.5)/H) * (bbox_xy2-bbox_xy1)`。

显式要求模型roi_local；越界点无支持。UAV候选读取和点级GT诊断共同采用该变换。原mask logits、tokens、选点、UDPR MLP/幅度/损失/冻结范围均不改。参数名/初始化和旧S1一致，直接复制相同common_init.pth；不继续错误版本的已训练权重。

CPU门覆盖128/256格、完整框与平移/缩放框，以及与mask目标所用aligned RoIAlign对线性x/y坐标图的独立对照；最大误差容许6e-5像素（仅数值合同，不代表真实几何误差）。并验证只写选点、零W回放、非零W/bias无支持回放及source/query梯度。完整GPU门与单张val点诊断门通过。

采样边界：上述RoIAlign对照针对final mask的连续RoI约定；UDPR原有tail target另经get_coarse_targets的floor/ceil整数crop再nearest resize，并非同一采样器。本轮保持该既有监督不变，点级GT诊断使用浮点bbox映射后的nearest图像标签，不冒充与tail训练target逐值一致。该离散差异与旧S1错误的整幅坐标语义是两个问题。

产物 `/data/wangcheng/checkpoint/ue_udpr_point_roi_20261023`，三步训练reserved3044MiB，250RoI FP32 reserved3044MiB；准入空闲6877MiB。

## N2-shared：共享增强后的高分辨率特征

新模型 `post_activation_shared_probe.py`继承原N2，注入后将相同enhanced tensor显式同步wrapper._captured_upscaled_embedding。用weakref指向wrapper，不把冻结模型注册为adapter子模块。不改原N2 source或运行中权重。

独立门在同权重/非零alpha/同val图下切换sync：原SAM raw logits精确一致；同步版UDPR输入与实际增强输出是同一tensor；与旧N2的UDPR feature确有变化；卫星指纹不变。不是靠最终mask涨跌倒推接线正确。

产物 `/data/wangcheng/checkpoint/ue_post_activation_shared_20261023`，训练/250RoI FP32 reserved3442MiB，准入空闲7375MiB。初始权重与旧N2完全一致。

## 固定训练与调度

两组各正确/常量RGB一臂，共4训练，从各组同一随机初始化独立训练；v1/EXP-0017、10epoch/2670更新、每卡batch1累积8、seed45/逐epoch1044000+epoch、AdamW LR1e-4/wd.01/warmup50/cosine、clip.5、AMP512、无EMA、E10 last保持既有协议。常量臂保留深度/姿态/几何候选，不是完全移除UAV。

统一owner `ue_consumer_blindspot_schedule.py`，根 `/data/wangcheng/checkpoint/ue_consumer_blindspots_20261023`，owner3385135。仅GPU0/1，共租原任务；每卡本组最多2个GPU任务，每次准入要求3GiB+1.25×实测reserved空闲，并保存快照。两组峰值之和约6.4GiB，相对起始卡0/1空闲18.1/18.2GiB有余量；不因“空闲”跳过测量。

首次分配：卡1 roi_correct3385173/shared_correct3385183；卡0 roi_constant3385178/shared_constant3385188。以state.json为实时依据。未抢占原D_zero、L系列或旧N2。

启动后四条训练均已完成至少6次真实参数更新，loss/grad有限；实测整卡0/1使用约12.9/12.8GiB，分别剩约11GiB，未见OOM。不是仅Popen成功的启动报告。

各组完整test基座disabled、正确CP correct/constant/donor、常量CP constant；各臂val；每组5项预定500次配对bootstrap比较（净增益/训练内容差/固定CP依赖），覆盖MaskAP250及BoundaryAP2%/1%，不按test选权重。

独立CPU `ue_shared_capture_interaction.py`（3385136）等待旧N2与共享版四臂完整，核验10epoch样本顺序/初始化/检测输出，再计算共享×内容四臂配对交互。S1-R点级汇总等待器3385137读取正确坐标版本，旧S1错误诊断不混入。

当前只有门和启动证据，尚无新组完整AP；旧S1错误不等于所有历史内容归因结论失效，也不证明修正后一定涨点。现有test为开发集、单seed图像bootstrap不覆盖跨训练seed/地理相关。
