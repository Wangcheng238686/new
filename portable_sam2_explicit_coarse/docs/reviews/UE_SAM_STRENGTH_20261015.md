# SAM残差强度冻结探针独立审核

日期2026-10-15。审核者frozen_probe_audit，独立只读，未参与脚本编写。

## 静态实施审计全文

SAM强度脚本首次定版静态审计：未见阻断问题；可运行且仅运行预注册3图smoke，153正式需待smoke验收。
审计点：ConfidenceFusion返回satellite+applied，无inplace；hook不返回值、不修改原输出，捕获inputs[0]/output[0]恰对应E0/E1，finally移除且len==1；s0直接E0、s1直接E1、s05=E0+.5*(E1-E0)，符合有效浮点残差定义。FPN完整提取/RPN/bbox各一次；predict_bbox调用与官方roi_head.predict的extraPE、rescale、test_cfg一致。每scale deepcopy(det)，mask推理不会污染下个scale；对det原值和每scale完整导出框分数均有精确检查。mask_head/_last字段仅当次输出/遥测，未见读取前次结果的缓存路径。GT框固定hash、s1全RLE/IoU回放、RGB和几何前后hash、feature hash、参数version均有检查。s1的GT IoU参照full_fixed_box而非correct_replay，可由已通过上一轮精确对角关联成立。
限制：目前仅静态审核，hook_removed字段为finally.remove语义支持而非检查hook dict，足够但待smoke运行完成；新预注册AP bootstrap实现未随此脚本提供，离线汇总首次定版仍需审计其缓存点估计与重复图像重新赋ID验证。不得把该阶段PASS写为结果审核PASS。

## Smoke和汇总准入审计全文

VERDICT：SAM强度3图smoke验收 PASS；允许进入原预注册唯一一份153图冻结探针，不授权扩大扫描/训练。
独立CPU逐对象核验：287/288/295共45GT，manifest complete=false符合smoke、strict EMA无missing/unexpected、parameter版本未变，无optimizer；脚本/helper/预注册SHA与manifest一致。3条invariance全True且hook_calls均1。3强度各90完整预测，image/category/bbox/score均精确复现历史原模型；s1全部预测（含RLE）与历史精确相同，45个GT IoU和共同框SHA精确相同；三强度固定框SHA一致、bbox全部指标相同。IoU s0=.698714612665、s05=.767823556760、s1=.793012887743，仅smoke，不作结论。
汇总静态审核PASS：COCOeval缓存仅单类/all面积/maxDet100，因此evalImgs每图一个；aggregate用每个重采样位置独立slot重设params.imgIds及_paramsEval，重复图像不会被去重。已对照当前安装COCO accumulate源码确认只用这些位置索引，而非evalImg内原image_id重新去重。full-point和预设重复序列[0,0,2,5,9,9,11,15]的重新赋fresh-ID普通COCO校验均设assert，失败即停；500次seed1720同抽样跨强度、1000次seed1721 IoU sum/count、建筑等权、固定.5主对照及三种预注册门均实现正确。正式运行后须确认这些assert实际通过并独立审核结果，目前只是静态PASS。无GPU/文件改动。

## 正式结果审核全文

VERDICT：PASS（预注册153图冻结SAM残差强度探针结果审核；不授权训练、扩大扫描或修改默认模型）。
审核者frozen_probe_audit，未参与被审脚本设计与开发。本轮只读CPU复核正式产物、源码、预注册和既有历史端点；未修改文件。
1. 正式manifest complete=true、153图/4496出现，图像顺序/manifest SHA/checkpoint与历史correct_replay相同；strict EMA load无missing/unexpected、无optimizer/训练、参数version不变。脚本/helper/预注册SHA均独立重算一致。153条invariance图像顺序正确，布尔项全部True、hook_calls均1。
2. 三强度各6514完整预测，image/category/bbox/score逐条与历史原模型精确相同；全部bbox指标精确不变。s1完整dt导出（含mask RLE）、全部4496个GT框IoU及框hash与历史精确相同，s1全部AP字典亦相同。三强度GT共同框hash完全一致。
3. 从保存的逐GT IoU独立重算means：s0=.7962265355492919，s05=.8303887268095695，s1=.8417685247679710。1000次图像sum/count配对bootstrap seed1721独立复现：s0−s1=−.0455419892186791，95%[−.0516850713691542,−.0399973834331214]；s05−s1=−.0113797979584014，95%[−.0133627847049248,−.0095111356761960]。237建筑等权差分别−.0373876361714976/−.00907261372760775，summary一致<1e−14。
4. 尺寸出现数881/2510/1105；s0−s1 small/medium/large=−.007036527702/−.039322259151/−.090369893352；s05−s1=.000773324932/−.009030920133/−.026404761404。半强度small组微正必须保留，不能称所有目标/尺寸均变差。
5. 官方AP点估计保存值s0=.5082877866175155，s05=.5642117676092067，s1=.5849579676428078，bbox均.6184538725985026。读取全部500份配对AP bootstrap样本独立重算差值CI：s0−s1=−.0766701810252923，95%[−.0823364943635523,−.0668029520069161]；s05−s1=−.0207462000336012，95%[−.0217534017326503,−.0161894157225867]，与summary一致<1e−14。summary记录seed1720/500次、full_point_exact与duplicate_image_direct_exact=True；脚本这两处为实际执行的assert，summary完整产出证明其通过。未另行重跑完整COCO；AP核验范围是已审实现、精确s1历史端点、保存点值及bootstrap样本再汇总。
6. 预注册门独立落判：support_overstrong=false；half_near_tie=false；original_better_than_both_reductions=true；bbox_exact=true。符合“原强度优于这两种减弱干预，未支持过强假设”。.5为主对照，无后验挑scale。
7. learned sam_alpha=.21367643773555756，平均||E1−E0||/||E0||=.8548182954196057，只作机制描述，不能单凭数值称过强。结果只针对E72当前权重、该153图冻结开发集、正确RGB-D配对、FPN保持原值时的s0/.5/1推理干预。s0仍有UAV→FPN，不能称单流；不能证明原强度全局最优、别的强度必差、训练修改无益或UAV净收益。bootstrap只覆盖图像重采样，不覆盖空间相关/选模/seed。原始GT框mask未保存，因此IoU本次从逐GT工件重汇总，未从像素重生成。
未发现阻断项；可证据先行关账“简单统一减弱SAM残差改善”的当前冻结假设，不自动训练。

### 尺寸AP边界补充全文

补充已实际读取核验的尺寸AP边界，请并入结果审核全文第4条：半强度small GT共同框IoU均差+.000773324932只是固定GT框条件的微小均值变化；完整segm AP_s从s1 .46596179345767935下降到s05 .46152572670721165（差−.0044360667504677），s0 .4481805846472968。不能宣称小目标增益。medium/large完整AP亦下降；关SAM的大目标GTIoU差−.0903698933517619较大，但不能脱离框口径外推每个大目标或训练收益。其余数值、AP样本CI、预注册门全部已独立通过，上一条全文已给。
