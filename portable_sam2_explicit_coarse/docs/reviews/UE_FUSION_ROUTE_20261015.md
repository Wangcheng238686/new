# FPN/SAM入口冻结探针独立审核

日期2026-10-15。审核者implementation_integration_review，只读，未参与脚本编写。

## 实施审核全文

VERDICT：修复版实施审核 PASS。可进入预注册的两套供体、各153图冻结诊断；不授权自动训练。

原 smoke 的明确阻碍已关闭：inference tensor 无版本计数，现改为特征前后 SHA256 比较，保留参数版本检查，不改变计算路径。

已核验：

- 四 cell 正确拼接：CC/CD 共用正确 FPN，DC/DD 共用供体 FPN；第二字母只选择 SAM embedding。
- PE、高分辨率特征和未融合 FPN 层逐值一致。两次特征提取后的遥测状态不参与预测；head 显式使用传入特征，未发现跨 cell 特征缓存污染。
- 修复后3图、45 GT smoke 完成，CC/DD 正式预测及 GT 共同框 IoU 精确回放，特征 hash 与参数版本不变。
- 独立检查 smoke：CC/CD 的框与分数完全一致，DC/DD 同样一致，符合 SAM embedding 替换边界；四 cell 共同框 hash 一致。
- 汇总脚本的 bootstrap 使用图像内差值之和／重采样 GT 总数，保持实例均值口径；同一重采样跨 cell、跨供体复用。条件效应、交互和预注册门实现一致。

解释必须保持条件性：固定 GT 框下的 FPN 路径包含 ROI 特征和提示生成；完整 AP 下还包含 RPN、回归与 NMS。不能把任一入口效应称为该模块的独立净信息贡献。CD/DC 是激活失配诊断，不能据此宣布最佳训练结构。

当前 smoke 为 `complete=false`，正式结论仍须等待全部153图的不变量、对角回放和结果审核。未修改文件。

## 结果复核状态

正式运行和主代理汇总已完成。独立结果审核回复尚未取得，不标记结果审核PASS；该记录待补齐。

## 补齐的独立结果审核全文

VERDICT：PASS（冻结开发子集融合入口结果审核；不授权训练或外推论文净收益）。
审核者frozen_probe_audit，未参与脚本实现；只读CPU检查，无GPU/训练/文件改动。独立阅读EXPERIMENT_GOVERNANCE、预注册、probe/helper/summary和结论MD。
1. donor_a/b的manifest均complete=true、153图4496出现、strict EMA load无missing/unexpected、参数version不变；153×2条invariance_checks全部布尔True，图像顺序一致。当前脚本、helper、预注册、供体映射SHA逐项匹配manifest。
2. 从8份固定框逐GT数组独立重算4496均值、1000次图像sum/count配对bootstrap seed1710、237global_id内平均再等权，10个条件效应/CI/建筑均值与summary相差<1e-14。A CC/CD/DC/DD=.841768524768/.823876547419/.841776881549/.823888327570；B=.841768524768/.823266335185/.841684855828/.823203917321。SAM四差=.017891977349/.017888553979/.018502189583/.018480938507，FPN最大绝对差=.000083668940，交互=.000003423370/.000021251076。主要沿SAM门通过，其余门false，正确执行预注册。
3. 两供体CC和DD的全部完整预测导出（包括mask RLE）与rgb_content_audit历史端点逐对象精确相同，GT框hash及IoU数组同样精确。保存的对角AP字典全部精确相同。所有CC/CD与DC/DD的image/category/bbox/score列表精确一致。为控制成本未重新运行官方COCO；交叉cell AP只核对保存结果与报告，非独立重新估计。
4. CPU加载E72 checkpoint配置：mask/bbox均SingleRoIExtractor、strides=[4,8,16,32]，未覆盖finest_scale，已查当前环境官方默认56。直接执行官方源码map_roi_levels函数于保存的全部4496 GT框，计数独立复现[3657,839,0,0]；fusion levels=(1,2)即P3/P4。融合发生在neck之后，无后续跨层回流P2。mask用独立mask_roi_extractor；renderer_features/p2_feature/prompt_rois虽传入，但mask_head.forward声明legacy且实际未消费，无启用旁路推翻该解释。
5. 结论MD数值与边界合适：只能称既定冻结RGB干预的主要敏感路径；不证明模块bug、FPN无用、UAV净收益或最佳训练结构。GT分配不等于预测框分配。原始GT框预测mask未保存，因此本次重算是从逐GT IoU工件重汇总与历史数组对账，未从像素重新生成GT框IoU。
未发现阻断问题。当前MD“独立结果审核待回复”应由主代理据此更新。
