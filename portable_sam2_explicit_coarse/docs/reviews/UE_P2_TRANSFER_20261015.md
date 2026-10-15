# P2冻结跨尺度迁移探针独立审核

日期2026-10-15。审核者frozen_probe_audit，未参与脚本设计开发，只读CPU审核。

## 首次实施审核全文

P2迁移探针首次定版静态审核：VERDICT：PASS，可进入预注册3图smoke，正式153须待smoke验收。不授权完整检测路径或训练。
已核验：correct/donor均用目标几何，hook只读捕获BEV和P3输入输出；finally恢复原RGB并移除两个hook，RGB/非RGB hash均检查。native通过原生产BEVToSatellite投128²，再用原P3 fuser(P2,U,Q,C)，无随机新参数；correct/donor Q/C逐值相同。R2/R2D无支持处严格0；upsample使用原correct P3实际差分，bilinear align_corners=False并mask至P2支持。q/.25和one/1是预定有效残差倍率，baseline原P2精确回放。
所有packs除x[0]外P3/P4等FPN、SAM、PE/high均同对象；生产extraPE参数无关、仅依网格而非P2值，不会把变更传播给其他层。RPN/bbox只用原x一次且历史框score精确检查，所有臂deepcopy det只推mask，因此AP应称固定框score条件segm AP，bbox自然相同。GT非P2按官方map识别且逐像素mask相同有assert。特征hash/参数version/端点RLE/GTIoU检查合理，未发现跨臂缓存污染。
解释边界符合预注册：跨尺度gate/GroupNorm/特征分布偏移，负结果不否定新训P2价值；donor RGB-D失配非净内容贡献；native vs upsample不是严格纯分辨率消融。supported_rms_ratio为支持内L2范数比，因delta无支持处0、分子分母同支持元素数，等于支持内RMS比。
需修正文案小矛盾：预注册写“hook调用各一次”，实际correct和donor两次完整提取，bev_hook_calls/p3_hook_calls应各2（每次提取各1）；hooks在后续native fuser调用前已移除，逻辑正确。建议把预注册明确为“两次提取各一次，总计各2”，与源码一致后再记录正式hash。

## 文案修正

正式运行前已修正hook计数文字；smoke运行前原协议字节保存在PREREGISTRATION_SMOKE_ORIGINAL.md用于旧manifest hash核验。计算源码无变化，独立审核接受不为文字修正重跑GPU；正式run记录修正后的hash。

## 三图smoke验收全文

VERDICT：P2迁移三图smoke验收 PASS，准入原预注册153图固定框mask探针；不授权完整RPN/bbox改动、扩倍率或训练。
只读CPU工件验收：3图287/288/295、45GT，complete=false正确，strict EMA无missing/unexpected、无训练optimizer、参数version未变。probe/helper/donor-map SHA一致；smoke prereg SHA精确匹配保留的PREREGISTRATION_SMOKE_ORIGINAL.md，当前只修正hook文案且正式将用新hash的处理合理，无须重复GPU。
3条不变量全True，BEV/P3 hook各实测2次并移除、RGB/几何恢复。P2/P3形状正确；残差及支持内比均有限非负，baseline为0，one=4×q验证通过，q残差全图范数比约1.44%–2.37%，非异常爆炸（仅数值健全非收益证据）。
5cell均90完整预测，全部框score与历史逐条精确相同，bbox全部指标不变；baseline完整dt/RLE、45GT IoU/hash精确回放。官方公式独立重算GT层级26P2/19P3，所有cell层级一致，19个非P2 IoU逐项精确相同；源码逐像素非P2相同assert及工件nonp2_gt_masks_exact=True亦通过。bandIoU/IoU数值有限合法，身份文件45条层级一致。
smoke IoU baseline/native_q/donor_q/upsample_q/native_one=.793012887743/.793143117291/.793295299231/.793154104543/.793473515915；仅执行验收，不从三图小变化选臂。AP均仅固定原检测框score的条件segm AP，不称新完整检测推理收益。

## 汇总静态审核全文

P2汇总静态审核 VERDICT：PASS（结果仍待正式工件验收）。500次条件AP bootstrap seed1741同采样跨5臂，单类/all面积/maxDet100下缓存每图1slot，fresh-ID重复图像对账及full-point精确对账均有实际assert；当前COCO accumulate实现兼容该独立slot聚合。1000次IoU seed1740保持图像sum/count实例等权；237建筑等权、尺寸/官方RoI分组、band差及非P2 IoU/band精确相同检查正确。主q固定，one只压力对照，无后验选主臂；主门额外要求条件AP CI下界>0，两个附加门仅要求IoU CI下界>0及各自AP/IoU阈值，吻合预注册。protocol和effect字段均明确conditional_ap，不能写成完整新detector AP。run_full_modified_detection_next=main仅应解释后续单独探针准入条件，不等于授权自动训练/改默认。未发现阻断问题。

## 正式结果独立审核全文

VERDICT：PASS（预注册153图冻结P2跨尺度迁移结果审核；主门与两个附加门均未通过，不追加完整修改检测探针、不训练）。
审核者frozen_probe_audit，未参与被审脚本设计开发；只读CPU核验，无GPU、文件改动或重复全部官方COCO。
1. manifest complete=true、153图4496次出现，原image顺序/manifest SHA/checkpoint一致；strict EMA无missing/unexpected，无训练optimizer、参数version不变。当前probe/helper/donor-map/当前修正文案后的PREREGISTRATION SHA均与正式manifest一致。153条invariance布尔项全True、BEV/P3 hook各2次，恢复RGB/几何、移除hooks、仅P2改变均通过。
2. 五cell各6514完整预测，所有image/category/bbox/score逐项与历史原模型相同，全部bbox指标不变；baseline完整dt含RLE、GT框IoU/hash、全部保存AP字典精确历史回放。独立按官方公式从GT框重算3657个P2/839个P3，与身份/每臂层级一致；839个非P2 IoU及bandIoU逐项完全不变，运行源码逐像素mask相同assert和nonp2_gt_masks_exact亦通过。
3. 独立重算五臂GT IoU：baseline .841768524767971，nativeq .841736646467599，donorq .841807435379449，upsampleq .841747665319352，nativeone .841842899286706。seed1740重生1000次图像采样sum/count，全部六条件差/CI、237建筑等权、尺寸/RoI组、band差和>.05正负计数与summary误差<1e−13。
4. 预定主q−baseline GTIoU=−.0000318783003717，95%[−.000127392736835,.0000776684306540]；建筑等权+.0000126379267611，band差−.000125151421126。small/medium/large差−.000113921556/−.000051923160/+.000079065324；非P2差0。q−donor=−.0000707889118498，95%[−.000185787427315,.000065479373849]；q−upsample=−.0000110188517528，95%[−.0000767264083994,.0000568417900076]。不能据小数点方向宣称净内容或原生细网格优势。
5. 条件segm AP点baseline/nativeq/donorq/upsampleq/nativeone=.584957967642808/.585078983662457/.585038097966155/.584911964097701/.585178752680417。读取500份保存配对bootstrap样本独立重算全部六差及CI，与summary<1e−13。主q−baseline条件AP=+.000121016019649，即+.0121016个百分点，95%[−.000705421599637,.001052221092403]；one−baseline=+.000220785037609，95%[−.000873139067439,.001229482727079]。q−donor=+.0000408856963020，95%[−.000706887537853,.000793724256397]；q−upsample=+.000167019564756，95%[−.000354516900898,.000852152500261]。这些区间全部跨0，且工程阈值未达。
6. summary记录500次AP seed1741、1000次IoU seed1740及full_point_exact/duplicate_image_direct_exact=True；已审脚本在这些字段前实际执行官方点估计和fresh-ID重复图像普通COCO对账assert，完整summary产出证明通过。本次未另跑完整COCO；AP独立核验范围为保存点值、baseline精确历史端点、保存bootstrap样本再汇总及已审实现。
7. 所有机制字段独立重聚合一致。P3 alpha=.102067865431309；q residual norm比均值.0225477887658414、支持内RMS比.0256526754146405；one分别.0901911550633658/.102610701658562。值均有限非负，one=4×q，不能用范数健全解释收益。GT身份global_id/bbox_area与full_stage逐项匹配，237建筑等权身份正确。
8. 独立执行预注册门：main_transfer_support=false、paired_rgb_support=false、native_grid_support=false、run_full_modified_detection_next=false。正确结论为“本轮固定P3权重跨尺度迁移到P2、预指定.25残差未显示预注册改善支持”；不后验选择one，不追加完整检测，不训练。
9. 边界：AP仅固定原检测框/score条件下的分割AP，bbox自然不变，不能叫完整新detector AP或检测收益。新残差借用P3 gate/GroupNorm/语义分布，负结果不否定新训P2融合器价值，也不说明P2冗余。donor沿用RGB-D失配/train供体局限，native−upsample不是严格纯分辨率消融。仅153冻结开发图/4496重复出现/237建筑，非test；bootstrap不覆盖空间相关/选模/训练seed。bandIoU为自定义3px形态学局部指标，非标准BoundaryIoU。原GT预测mask未保存，本次IoU/band从逐GT工件重新汇总，并非重新生成像素mask。
未发现阻断项；可按预注册证据关账本轮迁移假设，不据近零条件AP变化宣称收益，不据负结果否定P2训练价值。
