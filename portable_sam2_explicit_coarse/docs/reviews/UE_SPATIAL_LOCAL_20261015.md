# 空间对齐与局部注入质量独立审核

日期2026-10-15。审核者frozen_probe_audit，独立只读，未参与脚本实现。

## 首次实施审核全文

空间/局部探针静态实施审计已完成：VERDICT：PASS，可进入预注册3图smoke；153正式须等smoke验收。无GPU/文件改动。全文要点：
1. translate的归一化网格使用采样(x−dx,y−dy)，align_corners=False、zeros/no wrap，正dx/dy确为右/下；32²下.25cell=512²下4px。独立AST提取实际函数CPU运行全部脉冲、边缘、线性ramp合成检查通过。
2. hook只读返回None，E0/E1及原confidence/coverage捕获正确，finally移除且调用数1。identity直接E1精确回放；其他变体仅E0+变换有效差分残差。残差原无支持处为0有assert；平移后原coverage>0 mask再裁切；symmetric=四个经相同裁切的shift算术平均。该操作同时包含bilinear平滑及支持边缘残差损失，不能称纯几何重投影；预注册已经披露，方向vs symmetric对照合适。
3. FPN/RPN/bbox一次，mask每臂deepcopy det，bbox/score逐值历史对账；features参数hash/version及RGB/非RGB payload前后检查合理，未见跨cell缓存读取污染。identity全RLE/GT IoU历史端点、bbox AP不变量有assert。
4. local regions使用7×7方形核、constant0边界，即半径3膨胀/腐蚀，band=dilate\erode；band IoU限制union/intersection于同band，inner FN除以侵蚀面积、空inner null，far FP除以GT面积。独立CPU 12×12方形sanity得inner36/outer324/band288，完美预测各指标1/0/0。它是自定义GT框局部统计，非标准论文BoundaryIoU；边缘band会被图像裁切，clipped分组需保留。
5. fine support/quality依据project_pixel行优先= y*512+x，mass加权quality与生产投影相同；生产coarse桶=(pixel//512//sy)*w+(pixel%512//sx)，nearest上采样只用于归属，未替换生产几何。coarse confidence是几何quality，不是语义置信。fine_quality_supported是支持像素质量的等像素均值，非实例内全点mass再次加权，报告应照此描述。
6. 相机world/local使用正确e/t与origin，roundtrip仅内部代数一致性；不能证明真实标定。小边界提醒：代码每相机试采256格点后过滤无效深度，因此实际是每图5cam各最多256有效点；若某相机全部无效会跳过，目前只报告总count。smoke验收应检查count>0且如实称有效采样，不要硬写每cam恰好256已通过。若要严格宣称五相机均验证，建议记每相机valid count或assert每相机至少1，不改变探针结果。
当前未发现阻断问题。汇总及同时区间/分组实现待另审。

## 采样记录修正

正式运行前已增加per_view_valid_samples和每yaw有效数>0断言，预注册明确最多256有效点；3图smoke重新执行，原初版日志保留smoke_initial.log。

## 汇总静态审核全文

汇总脚本静态审核 PASS：8个预注册contrast正确=四方向各减identity/symmetric；seed1730同一1000×153重采样，图像内sum/count保持GT等权，max over8 |boot−point|逐bootstrap后q95作所有contrast统一半径，符合预注册。direction门同时检查AP≥.005、vs identity IoU≥.002/同时下界>0、vs symmetric IoU≥.001/同时下界>0；smooth门符合预注册。固定6cell均输出，无按最高值重选。237建筑等权、支持0/.25/.75分箱、size1024/9216、clipped及size内支持分箱正确；半强度join依历史同image/GT顺序且s1框hash一致。Spearman仅输出rho/n无p值，局部关系为观察性，不能推广为GT门控收益。建议归档说明分组/相关按4496重复出现计算而非独立建筑，support与size/coverage也相关，rho不构成因果或独立显著性。新版per-view有效计数及每yaw>0断言已确认，修正了空相机漏检。smoke新run当前log到2图，待DONE我做工件验收。

## 新版smoke准入审核全文

VERDICT：新版空间/局部探针三图smoke工件验收 PASS；准入原预注册唯一一份153图冻结运行，不授权扩大平移幅度/方向扫描或训练。
必要CPU验收已完成：manifest为287/288/295共45GT、3图complete=false、strict EMA无missing/unexpected、无训练optimizer、参数version未变；当前probe/helper/预注册SHA均与本次manifest一致，synthetic_checks全True。3条不变量全True、hook_calls均1。每图5个yaw各256有效采样（15view均非空），计数总和吻合，三类相机最大误差均<1e−6。六cell各90完整预测，全部image/category/bbox/score精确复现历史原模型；identity完整dt（含mask RLE）及45GT IoU/框SHA精确回放。六cell框SHA和所有bbox指标完全一致。六cell局部bandIoU、内部FN、远外FP数值有限且合法；45条geometry的支持/quality/coverage/conf等有限、范围合法。
smoke均值identity .7930128877433；right .7928660356799；left .7935223413669；down .7925900022922；up .7927800916683；symmetric .7933700554033。仅作执行验收，不从3图选方向或作机制结论。无GPU、无文件修改。

## 正式结果独立审核全文

VERDICT：PASS（预注册153图空间/局部冻结探针结果审核；所有方向与平滑改善门均未通过，不授权扩大扫描、训练或修改默认模型）。
审核者frozen_probe_audit，未参与被审脚本设计开发。只读CPU检查正式工件、当前probe/helper/summary及预注册；无GPU、训练、文件改动或重复完整官方AP。
1. 正式manifest complete=true、153图4496次出现，图像顺序/manifest SHA/checkpoint与历史correct_replay相同；strict EMA无missing/unexpected，无训练optimizer，参数version未变。当前probe/helper/预注册SHA独立重算与manifest一致。153条invariance全部布尔True、hook_calls均1、顺序正确；synthetic_checks全True。
2. 六cell的全部完整预测image/category/bbox/score与历史原模型逐条精确一致，bbox全部指标不变；identity完整dt含mask RLE与历史精确相同，4496个GT共同框IoU/hash和全部保存AP字典亦精确回放。其他cell框hash相同。AP其余cell本次核对保存点值和报告，未另跑官方COCO。
3. GT IoU均值独立重算identity/right/left/down/up/symmetric=.841768524767971/.841252655594338/.841987472660026/.841366154654553/.841850209152768/.842161858225894。按seed1730重生1000×153图像采样，以sum/count重算全部8个contrast，最大绝对bootstrap偏离q95共同半径=.00032408214256612634，与summary精确一致(<1e−13)。8条件均值、共同同时区间、正负>.05计数和237建筑等权结果均独立复现。smooth−identity=.000393333457922487，单独探索区间[.000214542458365234,.000558775005317635]，建筑等权=.000187804353504727。
4. 六cell全部local_metrics聚合均独立复现，identity自定义3px方形形态学band IoU=.647614194109471（n4496）；内部FN=.0319416415789615（n4296，200个空内部排除）；远外FP/GT面积=.09268489899632705（n4496）。不能把band IoU与global IoU .841769直接作高低质量比较，也不能称标准BoundaryIoU。局部指标从已存每GT统计再聚合，原GT框预测mask未保存，本次未从像素重生成。
5. 与full_stage的(image_id,gt_slot)/global_id/bbox_area逐项核对，237栋身份正确；强度探针s05与s1框hash均一致、s1 IoU等于当前identity，再按同一图/GT顺序独立join半强度差。per_gt_joined全部4496行及所有支持/尺寸/截断/尺寸内支持分组的n、均值、各shift差均复现<1e−13；Spearman rho与有效n也全部重算一致。
6. 低/中/高fine支持组n=852/1292/2352，baseline IoU=.780489/.811843/.880406；半强度均差−.009677174390/−.019841162395/−.007348570250。基线差并不说明低支持处是有害污染，不能据此关闭；每个尺寸内band均值随支持分箱提高只是观察关系，且面积、截断、遮挡和重复出现可共同影响。全局band均值并非按支持单调（低组.648839，中组.582749，高组.682803），因此不要笼统声称全局支持越低边界一定越差。局部分组没有独立干预/AP收益验证。
7. 153×5个view计数合法且全部为256有效点，总195840。最大same-camera UV误差5.3433e−11px，camera-Z误差7.3896e−13m，sat world/local UV误差5.6843e−13px；与summary一致。这只验证使用同一相机矩阵的代数自洽，不能证明渲染真实标定、像素中心或跨视角语义正确。
8. 预注册落判独立执行：四方向directional_support均false，symmetric_smoothing_support=false，bbox_exact=true。所有方向的完整segm AP均低于identity，亦无方向GT IoU同时超过identity和symmetric的工程门。smooth AP−identity=.0001807795009721，即+.01807795个百分点，远小于+.005门；不能写成已证实净增益。正确结论为“本轮四方向±4px有效残差干预未检出预注册支持”；不能说全空间对齐正确或所有尺度偏移均无效。
9. 限制：固定E72权重与153图冻结开发子集；不是完整生产集/独立test。平移是含gate图案的learned有效残差，经bilinear及原支持mask裁切，混合平滑与边界损失；不是重新投影或相机标定干预。bootstrap覆盖图像重采样，未覆盖空间相关、模型选择或训练seed；4496出现并非4496独立建筑。GT局部统计不可直接变成部署门控；不据本轮自动训练、加P2、改默认或扩扫描。
未发现阻断问题，可按失败门证据关账本轮统一±4px方向偏移/对称平滑改善假设。
