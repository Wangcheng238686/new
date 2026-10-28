# Native真实来源核查（P2，2026-10-22）

## 范围与结论

按已审核计划第4节实施CPU核查；没有训练、GPU推理，也没有修改冻结模型源码或共享缓存。

**八图的实际native最近表面来源已重建，并与八个历史cache的support和四通道float16 fields逐值相同，所有差异计数为0。** 八图cameras.json和全部40份深度文件哈希也与历史cache report相同。这里只证明来源重建忠实于冻结算法、现有缓存没有由此次重建引入差异，不证明物理标定或遮挡正确。

**固定24点原始影像审看：24点均保留“无法判断”，没有作外观一致认证，也未确认明确系统性冲突。** 原因是固定点多处于重复屋顶设备、窗格、暗色道路、水面或影像边缘，缺少能够独立标记的唯一共同结构。没有将投影落入GT或代数回环当作视觉一致性。不能把“没有确认冲突”解释成“全部对应正确”。

此次结果未产生要求暂停后续诊断的已确认系统性错误，但也没有消除真实同表面对应的不确定性。后续P1仍需自己的授权与工程门，本任务没有运行P1。

## 固定抽样与代表性

图ID先排序，以单个`Generator(PCG64(20261022))`先从train、再从val无放回各选4张；抽样名单在重建前写入manifest。顺序未改变：

- train：1238、1842、1828、2512。
- val：472、716、692、1488。

|split / ID|实例数|真实mask面积 small / medium / large|整图native支持|GT union内支持|
|---|---:|---|---:|---:|
|train 1238|152|121 / 29 / 2|87.30%|92.17%|
|train 1842|45|1 / 40 / 4|82.96%|91.17%|
|train 1828|48|22 / 26 / 0|93.42%|97.66%|
|train 2512|51|22 / 28 / 1|90.88%|91.42%|
|val 472|18|3 / 10 / 5|41.13%|42.85%|
|val 716|25|6 / 18 / 1|89.64%|87.99%|
|val 692|18|2 / 11 / 5|44.81%|48.40%|
|val 1488|51|20 / 30 / 1|87.56%|92.77%|

共408实例。真实mask面积阈值为32²和96²，与canonical bbox面积分组区分。八图含1张>100实例图、2张GT支持<50%图，但不是测试集代表性样本，也没有按结果补抽图。所有图的三个分组均非空，没有缺组。

每图内部、边界、背景各1点：逐实例mask减2次3×3腐蚀的并集定义内边界（图外为0）；内部=union减边界，背景=非union。每组与native support相交，按行优先像素列表由`PCG64(SeedSequence([20261022,image_id,group]))`选择，group=内部0、边界1、背景2。24点未移动。

## 重建与逐值核对

脚本：`scripts/analysis/ue_native_source_provenance_20261022.py`。独立CPU产物目录：

`/data/wangcheng/checkpoint/ue_native_source_audit_20261022/`

按冻结`native_geometry_probe.py`原顺序复算：五视角yaw 0/72/144/216/288，每视角源像素全512²、行优先；OpenCV深度EXR第2通道即R，camera-Z厘米除100为米。world/camera平移厘米除100为米，轴为x右y下z前。

所有有效源射线经同一真实卫星透视投影；连续UV在[0,512)内后以floor落目标像素。`np.lexsort((sat_camera_Z,target_pixel))`取每像素最近表面，排序与并列值来源顺序保持原样。

- selected射线决定height：世界Z减样本中心Z，先写float32高度缓冲，再除100。
- dx/dy分别来自当前像素与右/下相邻selected表面的float32高度差；两像素均有support才有效，除10并clip至±5。它们不一定是同一个物体上的表面差分，边缘可混合表面。
- viewcount单独从所有投影射线统计：相对该像素最近sat-Z不超过1m、每视角最多计一次，除5。不是仅检查height射线，也不是源camera-Z差。
- 无支持字段归零，最终与历史cache相同float16存储后逐值比较。

全部8个历史cache已存在，因此没有生成替代reference，没有写共享cache。`*_cache_check.json`均为`cached-reference`，support mismatch=0、fields mismatch=0、max abs difference=0。相机、深度来源hash逐份匹配缓存report。

源camera UV→world→原源camera UV的全有效射线最大绝对回环误差在`8.53e-13`至`2.50e-11`像素之间。这只是同一矩阵链的内部一致性，不能认证真实RGB/depth对应。目标floor量化不算投影错误，连续坐标和落点坐标均已保留。

## 全量来源与24点产物

- `manifest.json`：先封存的图名单、算法、种子、源码/标注SHA与单位。
- `audit.json`：八图summary、24点、cache逐值核对、原始文件hash。
- `sources/{ID:06d}.npz`：**所有通过投影筛选的源射线**的view、source UV/camera-Z、world XYZ、sat连续UV/Z、height、target pixel；原始拼接selection index、selected列表、512² selected_index_map、1m near布尔标记、support与fields。没有用RGB候选8点代替native。
- `{ID:06d}_points.json`：每点height selected射线；dx/dy邻点完整来源射线；所有1m近表面来源射线按视角分组；原始5×5源深度局部数值。四字段的来源分开披露。
- `contacts/{ID:06d}.png`：每图3行，依次固定内部/边界/背景点；各行含卫星RGB+GT轮廓、卫星局部、源RGB、源局部、深度图、深度局部。红十字为原固定点，不是独立人工识别点。
- `visual_review.json`：逐点视觉结论与原因；补充生成JSON中的pending状态。未给出任何独立识别坐标或距离，因为没有作外观一致判定。

接触图已全部通过view_image逐张审看：001238、001842、001828、002512、000472、000716、000692、001488。
另对000716固定背景点查看无标记的同点64²原RGB裁图（`000716_bg_raw_zoom.png`）：该点位于水面，邻近屋顶位于裁图下方；避免把红标记附近的屋顶误认为采样点内容。这没有补点或改变点位置。

## 对下一步的约束

原始RGB能提示屋顶、立面、道路等内容，但未提供外部surface/object hit身份。例如val 472/692固定点源图可见高楼立面；native表面证据不能统称屋顶，单靠GT区域不能把它升级成屋顶对应真值。

来源核对排除了“本次追溯拿错了native候选”“重建字段与旧缓存不同”这两类实现问题；没有排除卫星视角遮挡、跨像素表面混合、渲染RGB/depth细微错位、单位以外的系统标定误差，也没有检验消费入口混合坐标语义。

需要物理量化同表面误差时，应另取得renderer的object/surface hit身份或独立对应标注。不得用24个无法判断的点推成对应正确率、整库校准通过或UAV增益已证实。

## 主代理独立复核

主代理另写`scripts/analysis/ue_native_source_root_verify_20261022.py`，不调用子代理的重建函数，
直接在原厘米坐标中作齐次变换，从五视角全部原始深度重新找固定目标像素的最近表面及1m视角数。
独立核对8图名单、原文件SHA、历史cache support/fields、24点及dx/dy邻点源像素、深度、世界坐标和差分。
全部通过，记录于`root_verification.json`。主代理也查看了全部8张接触图，保留视觉对应无法判断的边界。
该独立数值复核仍共享原始相机/渲染深度，不是外部标定真值。
