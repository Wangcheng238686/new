# 紧凑双流联合模型：工程实施状态

## 当前交付

**最新前台核对（2026-10-25 11:20）**：纯SAT约E31，三个UAV臂约E21–E22，实际四训练PID均活；E1/E3/E10的24个固定train函数梯度诊断已全部完成，控制进程正常退出。最新共同完成E20 Mask为RGB-D83.5102、SAT82.4716、常量RGB83.1073、参考平面82.1281。尚未完成100轮及正式test/19项收尾；以下各早期快照保留历史意义，不表示当前轮次。


**2026-10-25早期观察快照**：四臂E1训练、459val和实际CP冻结/AdamW/样本顺序审计均完成，四训练PID仍存活并已进入E2。E1 Mask AP为SAT67.6711、RGB-D66.4703、常量RGB68.4940、参考平面65.4744；完整RGB-D的早期Mask净收益未成立，bbox有较高点值。不以E1排名改预算、入口、权重或loss，也不把val早期差值当最终test/稳定模态贡献。详细表及限制见文末。


**启动时状态（2026-10-24，取代更早“尚未启动”快照）**：四臂100epoch联合训练已启动，owner PID269383；satellite/rgbd/constant_rgb/reference_plane子PID依次269429/269430/269431/269432，GPU依次2/3/0/1。本次前台已核对真实/proc状态和至少125/75/75/75次更新。PID及步数仅为观察快照，后续须核对当前句柄与检查点，不据本段认定仍在运行或已完成。新训练未加载E73任务权重；还没有新联合AP结论。

启动前补齐AR@250字段命名、同批AMP溢出重算、非空optimizer恢复和科学启动复核。全部工程收据在`/data/wangcheng/checkpoint/ue_compact_joint_20261024`；早期未训练版本归档`pre_AR250_label_fix/`与`pre_amp_retry/`，共同模型初始化未改变。参考平面3060缓存已完整重哈希，旧14臂119任务研究已全部完成并独立前台复核；它不代表本轮联合目标完成。


这是一条独立可选的新联合路径，不修改当前14臂冻结探针及原单流入口。目标仍包括完整联合训练、各输入匹配消融及任务收益验证；本文件不把工程门当成目标完成。

|文件|实际行为|
|---|---|
|`rsprompter/uav/compact_joint.py`|`CompactInteraction`沿用已检验的单入口交互/实例头，只将UAV源换成P0紧凑编码器；`CompactJoint`保留原卫星可训练范围，原任务和直接实例loss共同训练，支持常规forward/loss。构造函数不加载任何任务权重。|
|`data/ue_joint_inputs.py`|显式选择measured/reference_plane及正确/常量RGB；绑定同图camera SHA。实测路径核对native/anchor来源相同并包含五depth SHA；reference路径只读独立参考缓存且缺失时失败，不回退实测数据。返回几何、RGB、index、FP64 anchor及来源。|
|`scripts/analysis/ue_compact_joint_foundation_gate_20261024.py`|实际torch.load白名单只允许通用SAM2；导出共同新鲜卫星初始化，最大GT train图3次真实AMP更新、组梯度、冻结参数及显存门。不是正式训练运行器。|

## 初始化与可训练范围

从静态`ue_minimal_single.py`继承封版SAM/CSPM/UDPR/coarse64/max250结构，仅通用SAM2权重初始化。不加载EXP-0017 E73或其它遥感任务检查点。卫星LoRA、原FPN/检测头、既有可训练解码器/CSPM/UDPR遵循单流配置；PE等原冻结项继续冻结。UAV只保留P0有效LoRA及新增几何/交互/实例头，不训练原始UAV backbone权重。

首轮门实际记录5次torch.load，全部路径为`/data/wangcheng/pretrained-models/sam2/sam2_hiera_base_plus.pt`。共同卫星新鲜状态SHA为`4b7cd9ad9c291a26edf955bedbb8413c61df175ce39d1b5cb37c27c9be739c3e`，添加UAV模块前后逐值一致，与E73 SHA不同。导出的`common_satellite_init.pth`是共同基础初始化，不是已训练模型。

## 已核验与限制

初次真实门发现父级nn.Module.cuda递归迁移未执行MMEngine BaseModel的设备元数据更新，导致预处理器仍把输入放CPU。已在外层cuda/to/cpu显式调用子级设备接口修复。随后最大208GT训练图2464的3次AMP更新通过，scale512无跳步，冻结参数逐值不变；satellite LoRA、decoder、UAV LoRA、geometry、writeback各组梯度有限且末步非零，峰值reserved5648MiB。loss不要求短程下降，不作性能解释。

joint可训练参数44158638，其中UAV321605；这些是当前模型工厂/门的实际计数，非全模型权重/推理成本。分组记录为各参数梯度L2范数的和，不是合并向量L2范数；仅接线审查。

已补常规forward委托到同一loss接口，最终源版本在`compact_joint_foundation_gate_v2`重跑同样门并通过；前台逐项核对complete及所有源码SHA与当前工作树一致、共同基础SHA相同，3次AMP更新均scale512且全部目标组末步梯度非零。两次运行后续loss轨迹并非逐位一致，不宣称CUDA训练复现；本门证明可训练性和来源，正式重复性须由完整训练评价。

## 正式运行前仍需完成

1. 新路径FP32正式推理/强制250实例与源输入干预门；不得沿用冻结SAT前缀保护结论。
2. 新运行器完整保存/恢复模型、optimizer/scaler、Python/NumPy/Torch/CUDA/DataLoader generator及采样位置；独立核定续训样本/增强顺序。
3. 固定完整训练预算、val选权/终局test规则、相同共同初始化的纯SAT/正确RGB-D/常量RGB/无实测深度对照。新紧凑源改变了无效dropout的RNG消费，不能复用旧探针轨迹。
4. 全3060参考平面缓存完成并核验身份；没有该缓存的样本应报错，禁止自动读回实测深度。
5. 完成旧14臂研究审计、梯度方向及端到端成本检查后，按证据确定正式结构。显式几何CNN是否值得保留，仍不是已证实结论。

用户要求的是各模态有实际且合理的任务作用；当前工程门只证明可追踪的输入链和可优化的联合链，不证明每项独立AP贡献或训练稳定性。


## 运行器与验收推进

新增 `scripts/probes/ue_compact_joint_run_20261024.py` 和 `configs/ablations/ue_compact_joint_protocol.json`，产物根 `/data/wangcheng/checkpoint/ue_compact_joint_20261024`。已实现 init/train/test，尚未启动长训练。四臂 satellite/rgbd/constant_rgb/reference_plane 共享新鲜 foundation/新头初始化；每臂单GPU batch1累积8、100epoch/26700更新，AdamW LR5e-4 wd.05、warmup100、cosine至.1、clip.5、AMP512、无EMA。每轮train实际2136图/267更新、两个新worker、连续保存的generator、不逐epoch重设种子；完整459val FP32/max250按combined Mask AP严格较大选best；完训后一次selected-best test。v1 test开发暴露限制保留；不声称与原4卡单流轨迹相同。

`data/ue_instance_targets.py` 只从卫星GT框生成目标，替代旧pilot.Data顺带读实测x的路径。`scripts/analysis/ue_compact_joint_runtime_gate_20261024.py` 通过四臂首val实际FP32推理、RGB-D强制250掩码（峰值reserved3230MiB）、reference_plane禁止EXR/实测缓存读取下8图累积更新、五RNG/模型恢复和两worker的8图增强SHA/顺序一致。顺序门使用预定16图子集，不称全数据评估。

`scripts/analysis/ue_compact_joint_optimizer_resume_gate_20261024.py` 补实际非空AdamW moment、AMP和模型经磁盘保存/故意改动/正式restore后的逐值恢复，再完成8图累积到第2次真实更新，scale512。不声称CUDA训练轨迹逐位复现。

全3060参考缓存已完成（10.42GB）。`scripts/analysis/ue_compact_joint_input_audit_20261024.py` 重核全部npz实际SHA、对应camera SHA、构建源/manifest及123/208/2实例三图的heat/reg/valid逐值等价。`scripts/analysis/ue_compact_joint_engineering_acceptance_20261024.py` 绑定全部门、共同初始化及当前源码，engineering_complete已通过。此前清单的推理/恢复/参考缓存实现和检查已完成；正式训练、科学证据复核、梯度方向及全链路成本仍待完成。不把首图随机初始化指标当AP结论。

### 2026-10-24 实际运行与最终分析入口

- `scripts/probes/ue_compact_joint_owner_20261024.py`：按工程门与科学启动收据核对冻结源码，调度四训练＋四个完训后test任务。显存预算每任务6160MiB、每卡余量2048MiB，不设置人工共租槽上限；失败不按AP提前终止健康进程，也不盲目重试。控制面`execution_complete.json`不代替最终科学验收。
- `scripts/analysis/ue_compact_joint_amp_retry_gate_20261024.py`：强制一次非有限梯度，512→256，同8张增强输入重算，loss逐值一致且AdamW仅一次实际更新。
- `scripts/analysis/ue_compact_joint_launch_review_20261024.py`：绑定当前科学启动证据与源码，决策为配对联合训练；不是用户审批或收益认证。
- `scripts/analysis/ue_compact_joint_paired_20261024.py`：只读最终预测。固定8项比较，分别报告合并路径及条件化原路径的净收益、RGB内容差和实测深度差，再报告两个控制臂净收益。必须完成100epoch和test，复核每轮样本顺序、val-best及更早平局选权、CP字节SHA、共同初始化和冻结源码后才分析。Mask/bbox/Boundary2%/1%均做500次同图配对bootstrap、seed20261021；另报AP75、COCO尺寸组AP及AR250点值。复用已有匹配与加速累积，每个估计器前三次对原COCO参考校准，共24次。元数据明确val-best，未沿用旧工具的last标记。边界宽度是图像对角线比例，512²下2%/1%分别约14/7像素。

`paired_analysis_plan.json`预先保存8项比较；该工具不启动推理、不更换权重或训练结构。完整数值报告必须等待本轮实测，图像bootstrap不能证明训练重复性。最终还需要实际checkpoint参数/optimizer/冻结范围审计、固定输入干预与联合权重成本核对，不能单凭本工具的通过标记认定所有模态均有收益。


### 2026-10-24 固定检查点内容诊断与等待中的收尾任务

`scripts/analysis/ue_compact_joint_rgb_intervention_20261024.py`固定已完成100轮并按val选中的RGB-D best权重，只改五视角RGB；常量为编码器归一化均值，donor为同一test顺序整体平移459//2张，五视角一起替换，保留目标相机/深度字段/对应索引/锚点。donor只读对应PNG，不读donor深度。逐图记录donor映射与实际PNG SHA；结束核对模型状态与CP不变，原始评价写`raw/`，只有全部身份/459图核对后才写顶层complete。它不训练、不重新选权、不代表物理一致RGB-D场景；两个诊断必须在四臂正式训练/test完成之后。

`scripts/analysis/ue_compact_joint_finish_owner_20261024.py`先核对训练owner实际/proc起始tick，再等待其8项完训/test完成收据；没有完成而owner句柄消失则停止并要求工程恢复核查，不重启训练。随后按显存/主存准入执行8项配对分析及2项固定RGB干预，CPU每任务预留4GiB、主机留8GiB；GPU每任务6160MiB、每卡留2048MiB，空闲可共租。源码及命令独立归档`posttraining/`，失败让已有子任务结束后留待核查。收尾complete只认证这10项工程产物，不认证全部科学目标。固定RGB干预的配对区间、实际权重冻结/optimizer审计及最终成本仍需前台继续核对；不能据收尾PID存在宣称已经完成。


### 首个正式epoch检查点（观察快照）

纯卫星E1已完成267次更新及459val，Mask AP .6767106223、bbox AP .6752616101，随后实际进入E2。本次直接读取`epoch1.pth`，334个非空AdamW状态step均267、AMP scale512，五RNG键含loader_generator，合同/2136唯一训练图/epoch边界一致；收据`satellite_epoch1_checkpoint_inspection.json`。这只是单臂E1工程核查，不比较未完成的UAV同轮结果，不证明最终联合增益或全部冻结范围审计。收尾owner274663已确认实际存活并等待真实训练owner269383；当前没有提前执行test或最终收尾分析。


最终配对入口的独立数值核定已完成：使用旧研究固定459图预测重算Mask/bbox/Boundary2%/1%四个点值，与原完整报告逐项绝对差<1e-12；12次身份/随机/重复单图累积均对原估计器校准通过。未完成的本轮研究被前置检查拒绝。收据`paired_analysis_implementation_check.json`绑定实际新脚本SHA。这是工具验证，不是新的联合test结果；正式两臂报告仍需24次校准与完整合同检查。


### 2026-10-25 实际检查点冻结与optimizer审计

新增`scripts/analysis/ue_compact_joint_checkpoint_audit_20261025.py`，CPU构造相同架构/可训练范围，构造期torch.load仅允许通用SAM2；随后显式读取本轮共同初始化与指定实际CP进行审计，不把“分析读取训练CP”与“模型基础初始化”混淆。构造所得439个联合可训练张量、44158638参数与启动前工程门完全一致。按实际optimizer参数组顺序绑定参数名，核对moment形状/有限性与逐参数step，并要求始终使用的RPN分类权重step等于全局成功更新数；条件未用参数允许缺optimizer状态并公开名称。全部冻结参数与共同初始化逐值比较；源/卫星backbone的持久buffer也不得变化。非持久buffer不在CP内，明确列为无法从CP审计，不能伪称全部buffer已核定。

`--epoch 1 --arm satellite`实际通过：323个冻结参数张量逐值不变，334个参数持有step267的AdamW状态且均有更新；12个原SAM IoU/obj-score预测头参数无optimizer状态，名称全部公开，不算334个已更新参数的一部分。本轮代码并未给这些既有头新增训练目标；该观察不是UAV分支未消费。三个非持久buffer为预处理mean/std及原no-mask参考，CP不含它们。第一次分析程序误把非持久buffer当CP键而停止，原源码归档`checkpoint_audit_debug/initial_audit_source.py`，修正只读审计后v2通过，训练进程及数值源码未改变。

缺省`--epoch 1`需要四臂E1都存在，比较相同epoch的2136图顺序；`--final`必须四臂100轮结束，审计各自best和last实际权重、选中轮次/CP SHA及每个实际optimizer状态，仍需结合最终配对预测审计。不能拿首个单臂通过当作四臂或最终收益通过。


### E1四臂已完成：早期验证观察（非最终收益）

|臂|Mask AP250×100|bbox AP250×100|Mask AR250×100|
|---|---:|---:|---:|
|satellite|67.671062|67.526161|73.728972|
|rgbd|66.470323|70.464583|72.465287|
|constant_rgb|68.494030|69.031791|73.786382|
|reference_plane|65.474436|68.664690|72.471963|

这里是共同基础初始化后的第一轮459val，尚无本轮test，不能称稳定净收益。正确RGB-D相对SAT的Mask−1.200739pp而bbox+2.938422pp；相对常量RGB的Mask−2.023707pp；相对参考平面的Mask+.995887pp。前者表明早期两个任务方向不一致；后两项也不能据单轮优化阶段断言RGB无用或深度已成立。不据它们提前停臂或调参。

`checkpoint_audit/epoch1_satellite_rgbd_constant_rgb_reference_plane.json`四臂审计通过：SAT323个/各UAV353个冻结张量逐值不变；SAT334个/各UAV427个实际AdamW状态step均267；四臂同轮2136唯一训练图顺序相同。三个UAV臂无任何uav.*参数缺optimizer状态。正确RGB-D/参考平面427个可训练张量改变，常量RGB426个改变（有optimizer状态不要求每个张量值必变）；12个无状态者仍仅为原IoU/obj-score头。接线和更新证据不代替输入内容净价值。

新增`scripts/analysis/ue_compact_joint_epoch1_paths_20261025.py`：在下轮val覆盖前，将E1四臂预测/评价/逐轮记录独立快照并绑定CP SHA。分别重算原路径与合并路径Mask/bbox/标准尺寸AP/AR；原路径直接取合并输出前缀的完全相同score，使差异仅为追加候选，逐图核对前缀bbox/label/RLE与新增数量。合并点值必须复现原E1指标<1e-12，459图GT/ID跨臂一致。它是不同联合训练模型的早期验证分解，不是同权重内容干预或最终科学验收；没有提前读取test、增加边界loss或改采样。


### E1原路径／新增候选分解已完成

`epoch1_path_diagnostic/complete.json`四臂459图全部点值回放与GT/前缀身份核定通过，快照和分析源已归档。下列Mask均AP×100；分母为459图，每图原路径/追加数量均为预测数而非正确数。

|臂|原路径Mask|合并Mask|新增候选Mask增量pp|原预测均数|追加预测均数|
|---|---:|---:|---:|---:|---:|
|satellite|67.671062|67.671062|0|142.800|0|
|rgbd|66.429224|66.470323|+.041099|134.991|115.009|
|constant_rgb|68.470919|68.494030|+.023111|114.832|135.107|
|reference_plane|65.453291|65.474436|+.021146|133.198|116.802|

RGB-D原路径bbox AP70.420883、合并70.464583；相对纯卫星67.526161的早期bbox提高主要已在原路径，而原路径Mask相对纯卫星低1.241838pp。三UV臂当前几乎填满250推理上限，新增候选在E1的Mask增量很小。它把后续诊断重点指向原路径共享表示/任务优化与新增掩码的成本收益，不证明几何错位或某loss冲突已经成立。

正式化需分别评估内部交互、直接训练监督和推理新增候选，不能用“UAV分支整体有梯度”替115个额外mask解码背书。当前不改100轮推理合同、输出上限或aux权重；最终固定CP已预留原路径及合并评价，后续按完整证据决定是否需要保留候选推理消费。E1不同模型已发生不同优化，不能把原路径差直接当同权重UAV因果伤害，也不能用bbox提高替代分割目标完成。


### 2026-10-25 联合E1固定函数梯度方向

新增`scripts/analysis/ue_compact_joint_loss_direction_20261025.py`，在本轮RGB-D E1真实CP上、8个与旧诊断相同的固定train索引，FP32 eval、无增强/dropout/optimizer更新。检测项固定四个RPN/RoI分类框回归loss，分割项为mask/decoder-tail/shape-prior；三次VJP来自同一前向，main梯度按det+seg相加。按卫星LoRA/neck/检测头/mask解码器及UAV几何/源LoRA/交互/回写等组拼接参数向量计算L2与点积，不以loss数值大小猜影响。结束模型/持久buffer逐值SHA不变，optimizer仍无状态、参数.grad均空；FP32实际peak reserved10870MiB。完整协议、32个图像/比较组合的分组原始数据（8图×4种比较）在`joint_loss_direction/epoch1`，源码归档。

|比较及组|左梯度范数均值|右梯度范数均值|每图余弦均值|负余弦图数/8|
|---|---:|---:|---:|---:|
|main / aux，writeback|1.593855|.861135|−.021419|6|
|main / aux，geometry|1.992299|8.255155|+.158824|1|
|main / aux，source LoRA|.306093|1.055176|+.009413|4|
|main / aux，satellite LoRA|5.785986|2.266652|+.002745|3|
|det / seg，writeback|.480657|1.359242|+.239920|1|
|det / seg，source LoRA|.109079|.241396|+.400193|2|
|det / seg，satellite LoRA|1.742360|4.895937|+.280673|2|
|seg / aux，writeback|1.359242|.861135|−.026421|7|

这些是8个固定eval函数的平均范数/平均每图余弦，非训练随机梯度分布、非合并向量余弦，也不是AdamW后实际步长。不能从aux几何梯度较大直接认定AP伤害；writeback对seg多为轻微负相关但平均接近正交，det/seg主要共享组平均为正，当前不支持普遍强烈对抗的解释。aux对neck/原mask解码器/原检测头的梯度均零，符合它从block4另读实例头的结构；卫星LoRA的aux梯度非零但小于main。后续保持训练合同，按相同函数复查E3/E10，不根据本表调loss权重。

新增`scripts/analysis/ue_compact_joint_gradient_owner_20261025.py`：确认RGB-D训练PID269430及/proc起始tick真实存在后等待E3/E10的epoch JSON和CP全部落盘，按14GiB allocator cap＋512MiB上下文＋每卡2GiB余量准入运行同一固定驱动。owner287046已核定存活，两个诊断在等尚未生成的CP，不是有就绪任务被人为限制。驱动与owner字节归档，失效不盲重启。它只跑离线8图函数，不修改训练/验证/test或选权；科学结论仍需前台核对。


### 2026-10-25 收尾v2：固定内容区间、最终身份与真实权重成本

收尾v1 PID274663仅在等待训练owner269383，实际/proc子进程列表为空；接替前已核对28份数值源不变、原10条命令/marker逐项相同，并归档旧launch/state/plan到`posttraining/control_history/pre_v2`。仅终止该等待控制面，没有终止或恢复任何训练子进程。`scripts/analysis/ue_compact_joint_finish_owner_v2_20261025.py`新owner291123已实际存活，权威状态为`posttraining/state_v2.json`、启动身份为`owner_v2_launch.json`；v1文件保留历史，不能据旧PID不存在认定训练失败。

v2保留原10项，增加4个固定CP RGB区间、1个最终CP审计、1个实际输入字节审计、3个真实联合权重成本case，共19项。等待当前真实训练owner完成原8训练/test任务后才启动，新的只读脚本未改变任何数值训练源/初始化/数据顺序/增强/预算/选权。新增来源连同旧统计内核独立归档`posttraining/source_v2`，任务计划为`plan_v2.json`。失败保留原始日志及未完成产物，不盲重试；已运行健康任务继续完成。

|文件|实际职责及边界|
|---|---|
|`scripts/analysis/ue_compact_joint_fixed_rgb_paired_20261025.py`|只读官方RGB-D正确test与固定CP常量/donor test，按combined/original_path各报告correct−intervened的Mask/bbox/Boundary2%/1%及500次配对区间。校验CP完全相同、GT/ID一致、目标几何保留、donor映射及2295个PNG SHA，24次原累积校准。它不执行新推理、不选择CP；物理不一致donor仍只作诊断。|
|`scripts/analysis/ue_compact_joint_final_inputs_20261025.py`|完训后按冻结SHA256SUMS逐项重哈希全部原始archive文件及3060 native＋3060 anchors＋3060 plane实际npz，绑定四臂合同和元数据身份，不仅核计数。只哈希标签文件，不解码标签或评价AP；不认证物理标定真实性。|
|`scripts/analysis/ue_compact_joint_cost_20261025.py`|按实际val-best权重，三个独立CUDA进程分别测纯SAT、完整RGB-D、同RGB-D只跳过追加候选解码。各case固定3张val、每图warmup一次、各3次正式重复；源编码/交互、原检测/掩码和RLE均执行，实例头在original-only中仍前向。原路径/完整路径保留前缀预测必须逐值SHA相同。|
|`scripts/analysis/ue_compact_joint_finish_owner_v2_20261025.py`|资源准入执行19项和最终成本前缀等价检查，保留原10命令。新固定区间只依赖相应干预完成；最终CP/字节审计与其它统计可并行。三成本case在全部其它16项结束后、同一张无其它计算进程的GPU依次测量，避免把共租争用当模型延迟。该隔离只限成本测量，训练/功能分析继续允许共租。|

成本wall包含eval图像/GT读取、cached geometry与PNG IO，不含离线缓存生成/模型构造/来源审计/预测hash；另报prepared_inference_wall剔除输入阶段，CUDA事件分解可能含host空隙。显存为单独case进程的allocator绝对峰值，包含模型与瞬时输入，context/库分配不在allocated/reserved内；3×3中位不是置信区间或吞吐benchmark。GPU无其它进程条件在每次测量前后核对，同卡同精度/输入以便比较。不能将旧冻结E73时的成本替代真实联合权重成本。

以上新工具目前已通过语法和19项依赖/命令保留核定，正式输入尚未产生，不能宣称这些最终统计/成本已成功完成。`posttraining/complete_v2.json`未来只证明工程产物完成，仍保留`final_scientific_review_still_required=true`。前台最终还要逐项核对实际区间/内容差、初始BUILD_COMPLETE与缓存绑定、成本合理性、单训练seed及固定rig的归因限制；本轮目标未完成。

E2四臂val观察也已产生：Mask SAT73.0874、RGB-D70.0818、常量RGB75.2438、参考平面72.6691；bbox分别72.8883/72.7024/75.5438/73.4249。完整RGB-D尚未优于控制臂；E1的bbox提高在E2未保持。记录该负向早期事实，不拿它提前终止约定100轮，也不将梯度接通说成实际净收益。


后续尚需闭环的独立数据前提：`UE_NATIVE_SOURCE_AUDIT_20261022.md`和旧`UE_STAGE_RETROSPECTIVE_20261016.md`明确区分矩阵回环/文件来源与真实RGB-depth影像共注册；MRQ RGB与SceneCapture depth即使姿态/FOV元数据一致，也不能据此认证像素对应。当前数值/来源审计没有补出独立物理真值。下一项数据检查应直接使用原始train RGB/depth的可辨结构及多视角观测，先确认该前提，不据E2负向结果直接判数据错或加模型容量。该事项属于最终科学审查，收尾19项工程通过不能替代它。


### 2026-10-25 E3首次同轮正向点与梯度复查

四臂E3全459val已完成，Mask AP×100为SAT75.262199、RGB-D78.041882、常量RGB77.291045、参考平面75.985319；bbox分别74.226914/77.751233/77.412654/74.699571。RGB-D相对SAT/常量RGB/参考平面的Mask点差依次+2.779683/+.750837/+2.056563pp。三项同为正是首次早期信号，未做该轮test或bootstrap，不以此选结构、提前停臂或认定训练稳定性。

同一8train函数的E3梯度检查已实际完成、CP/模型不变：det/seg在writeback/source LoRA/satellite LoRA的平均余弦+.176696/+.173427/+.107731；main/aux分别+.008224/+.089469/+.024260，geometry+.127089。E1 seg/aux回写的轻微负向，在E3均值+.006353、3/8负，没有持续强烈对抗证据。主任务/source LoRA均值范数.309494，aux.714021；geometry1.853143/7.706758。仍为FP32 eval固定函数，不等同实际AdamW影响或AP原因；E10相同复查继续等待真实CP。

新增只读`scripts/analysis/ue_retained_rgb_read_audit_20261025.py`逐点验证真实缓存源像素/深度/投影，并比较原始颜色均值/最近观测面过滤/最近锚点/dense warp。8图668301候选无字段重排或view索引错误，后方>1m占6.13%；硬过滤未改善8图原色代理，当前不为此加新门控。完整分母/结果与不能外推的范围见[原始影像对应报告](UE_RAW_RGBD_CORRESPONDENCE_20261025.md)C节。原数值代码/候选/训练合同未变。


### E4同轮验证：内容增量仍在波动

四臂E4完整459val Mask AP×100为SAT77.831728、RGB-D78.141304、常量RGB78.297509、参考平面77.317681；bbox分别75.037069/74.754613/78.128449/76.156845。RGB-D Mask相对SAT/常量RGB/参考平面点差为+.309576/−.156205/+.823623pp。E3正确RGB高于常量的点差在E4转为微负，没有形成稳定RGB增量结论；相对SAT/参考面的正点值也不是test或训练重复性证据。原100轮预算/val-best规则和全部匹配对照保持，未按E3高点提前停训或换结构。


### E5同轮验证：RGB与深度点差再次为正

四臂E5完整459val Mask AP×100为SAT77.973858、RGB-D79.333529、常量RGB78.822960、参考平面77.778058；bbox分别76.917084/78.814047/71.704102/75.384002。RGB-D的Mask相对SAT/常量RGB/参考平面为+1.359671/+.510569/+1.555471pp。保留E4正确−常量微负的历史，因此E5是再次出现正向点，不能称稳定结论。所有臂继续原100轮，未据当前点值调权或选结构；最终test、图像配对区间、单seed限制及真实成本仍待产物。


### E6同轮验证及全部早期轨迹

四臂E6完整459val Mask AP×100为SAT78.783620、RGB-D77.759572、常量RGB78.343966、参考平面78.352069；bbox分别76.460776/78.667377/78.639568/77.901960。RGB-D Mask相对SAT/常量RGB/参考平面为−1.024047/−.584393/−.592497pp，三个点差均负。E5的三项正向点未在E6保持，不能把早期最佳点当稳定结论。

已把当前全部共同完成轮次E1–E6逐项汇总至`matched_validation_observation.json`，包括Mask/bbox/AP75/AR250和每份epoch JSON实际SHA，不挑正向轮次。该汇总不是选权规则，不计算test，不以跨轮平均替代最终预先固定的val-best权重。100轮训练、E10同函数诊断、正式完训test及19项收尾继续原合同。


### 2026-10-25 11:20：E10诊断闭环、E20同轮观察

`joint_loss_direction/owner/complete.json`及E10实际CP/驱动SHA、8条记录均已前台复核。原owner287046正常结束，不能因/proc PID缺失将其认作故障；该控制面尚存state里的waiting_or_running只是最后一次轮询快照。E1/E3/E10共24个固定train函数观察完成，未发生optimizer更新/模型状态改变。E10 main/aux范数均值（writeback .554136/.157177、geometry1.434936/2.554051、source LoRA .304180/.310904、satellite LoRA2.169645/.621598）；main/aux余弦分别+.009401/+.101820/+.069001/+.018282，seg/aux回写+.005934、4/8负，det/seg回写+.069649。没有跨三个anchor持续强烈对抗证据；范数是8图FP32 eval原始梯度，不能解释为AMP/AdamW实际份额或AP因果。

`checkpoint_audit/epoch10_satellite_rgbd_constant_rgb_reference_plane.json`实际四CP审计通过：冻结SAT323/UV各353张量逐值保持共同初始化；所有334/427个非空AdamW状态均step2670；UV没有任何新增可训练张量缺状态。既有12个无目标SAM IoU/obj-score头状态例外仍公开，非持久buffer不在CP核验范围，保持原审计边界。

|共同epoch|SAT Mask×100|RGB-D|常量RGB|参考平面|
|---|---:|---:|---:|---:|
|10|80.381714|81.465672|81.908186|79.725601|
|15|80.247531|82.609538|83.037943|79.024674|
|20|82.471580|83.510195|83.107310|82.128094|

E20 RGB-D相对SAT/常量RGB/参考平面Mask点差为+1.038614/+.402885/+1.382100pp；bbox分别80.475749/81.450268/80.740270/79.792877。E10/E15正确RGB仍低于常量，不能只展示E20。全部已共同完成E1–E20逐轮指标、每份epoch JSON SHA和2136图顺序一致性已汇总`matched_validation_observation.json`。不跨轮平均替代预定选权，不以中期val替代最终test或训练重复性。

当前实际训练owner269383与收尾v2 owner291123仍存活，无失败记录；完训后原19项科学命令保持。E10诊断不再等待/重启；后续只需等待训练100轮、正式test与既定收尾，再进行最终科学审查。
