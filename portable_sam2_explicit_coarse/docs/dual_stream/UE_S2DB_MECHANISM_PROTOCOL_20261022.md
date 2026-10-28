# S²DB-Net 机制借鉴探针（2026-10-22，预先固定）

本轮是机制借鉴，不是论文复现。官方仓库 <https://github.com/EdgedEnd/S2DB-Net>
当次检查只有 README 和架构图，没有模型及训练代码。新代码、缓存、权重和评价独立保存；
旧 v1 探针继续运行。下列合同不根据本轮 test 结果调整。

## 固定合同

EXP-0017 E73 完整卫星基座冻结并保持 eval；仅新增模块及 UAV LoRA 可训练。
保留原 decoder、UDPR、coarse 64 和 conv_s0 注入入口，保留全图高分辨率特征按 RoI 重复的原路径。
数据 2142 train /459 val /459 test，train 训练，val 诊断，test 主要评价。
10 epoch，每轮267次更新，共2670；batch1/accum8，AdamW 1e-4/wd.01，warmup50，cosine到.1倍，
裁剪.5，seed44，E5–E10 种子1044000+epoch。训练AMP、评价FP32，无EMA，maxDets250。
原高斯噪声p.5/std.02及ColorJitter p.5(.15,.15,.10,.02)不变，UAV无空间增强。
固定E10 last，不选test最佳轮次。每轮实际2136张，其余6张按当轮shuffle尾部丢弃。

底座：`/data/wangcheng/checkpoint/ue_minimal_single_20261021_EXP-0017/checkpoints/best_model_epoch73.pth`。
底座文件SHA256：`dd9106f4afe8ba72936316d7efa9621ca0c05d24836fa28c72ea322bed84ace1`。
完整卫星参数及缓冲区状态SHA256：`c6921c30416595fb994facd3d4d927408be9bdd636399fd669cfbe5968329536`。
数据：`/data/wangcheng/dataset/CitySample_SatUAV_TrainMinimal_v1_20261021`。
产物：`/data/wangcheng/checkpoint/ue_s2db_probe_20261022`。
缓存：`/data/wangcheng/cache/ue_v1_s2db_geometry_v2`。

## 第一阶段：10臂

|臂|读取|残差|
|---|---|---|
|A|fixed|normalized|
|B|local|normalized|
|C|fixed|concat|
|D|local|concat|

每臂 correct、constant RGB 训练；C/D 另有 zero 聚合特征训练，共10臂。A重新训练。
constant 是编码器输入固定预训练均值，几何不变；zero 是聚合后乘0，几何支持及卫星query保留。
所有臂加载同一完整 `common_init.pth`，包含未激活的新模块；新增模块在独立RNG中初始化。
每个训练进程建模后重置seed44，避免多余模块初始化改变训练数据/LoRA dropout随机流。

保留128²候选格、最多8候选/格及2候选/视角、原6维几何。新缓存将 source_uv、source_z_m、
satellite_uv、satellite_z_m 随所有排序、去重、过滤传递；旧index/geom逐值断言相等。
每候选二维偏移MLP70→32→2，末层0，tanh每轴±1源特征像素（±4 RGB像素）。
从原离散token中心双线性读取，一图一次，各RoI复用。偏移外观保留源归一化，不重写锚点几何。
越界/无效深度/非有限特征回退原有效锚点；无锚点不注入。
有效深度突变保留为外观上下文，离线统计阈值 abs(z_shift−z_anchor)>max(1m,.05*z_anchor)。
此干预不能找回无锚点区域或远处丢弃的证据。

C/D在相同RoI128²网格读取query32与UAV32，各自无affine LN后拼接，
Conv3×3 64→32 /GELU /Conv1×1 32→32 /通道无affine LN。
A/B沿用Linear32→32及无affine LN。全臂α初始0，输出LN后乘支持掩码，转接收dtype再相加。
C/D是联合内容及局部卷积干预，不能仅归因为拼接。与原生几何探针不同，此处消费RGB编码、LoRA和候选注意力。

## 第二阶段：预先固定D，不按test排名切换

aux_weight {0,.1} × {correct,constant,zero}。前三格严格同合同复用第一阶段D；
有辅助三格从相同common_init重新训练，无热启动已训练分支。因此通常13个独立训练。
辅助头Conv3×3 32→32 /GELU /Conv1×1 32→1，融合前目标128²格聚合、上采样512²。
仅训练期使用。头不直接读卫星特征；注意力和偏移仍受卫星query条件化，称为“卫星query条件下的UAV对齐预测”。
辅助路径重新计算偏移/聚合，query输入detach；主路径不变。
目标为卫星实例掩码并集；监督掩码为有效保留锚点实际投影的唯一卫星像素，而非整个4×4格。
这是观测支持代理，不是完整遮挡认证。BCE有效像素均值，空支持为0。
总loss=原掩码及UDPRloss+.1*BCE；不增加边界loss。

## 固定评价、归因及离线分析

所有E10检查点做correct/constant/cyclic-donor RGB推理。donor只换RGB、保留原几何，输入不一致性必须披露。
报告test MaskAP250/AP75、BoundaryAP2%/1%、COCO尺寸AP、bbox及召回。
Boundary IoU与mask IoU取min，内边界宽度按整图对角线比例取整，512²约14/7px。
500次同图配对bootstrap，seed20261021，名义95%区间。
读取效应B−A/D−C，融合效应C−A/D−B，交互D−C−B+A，所有内容训练对照及辅助六格差。
净增益参考MaskAP+.002且区间下界>0；完整报告更小信号。
单训练seed、图像间相关及多路线探索限制保留，bootstrap不证明训练重复性。

离线记录偏移幅度/每轴≥95%上限比例/接受率/锚点回退/深度突变，分母有效锚点；
注入/基座RMS沿原latest-call定义；内容干预预测变化。
固定bbox匹配IoU≥.5关联GT/DT，匹配独立于掩码；公开未匹配实例。
修改质量：GT内外2px形态学边界、腐蚀后内部、GT+匹配bbox并集外扩8px内剩余背景，
纠错及新增错误分别计数，公开各区域像素分母。
预定观测覆盖组<.5/[.5,.8)/≥.8（实际锚点像素与GT并集交集/GT并集），
COCO尺寸组，图像GT密度≤50/51–100/>100；不用test分位数定阈值。
辅助E1/E3/E10使用固定train实例数四分位代表4图，离线计算共享UAV参数主/辅梯度范数及余弦，
无AMP缩放、无裁剪，零范数null。辅助IoU仅在有效像素计算，不能替代实例AP。

## 工程门与运行纪律

长训练前验缓存来源/单位/token对应/回环、零α原始logits、无支持、零偏移读取、
越界/孔洞/突变/RoI边缘、AMP、LoRA/偏移/融合梯度、卫星参数及缓冲区SHA、正式几何融合250实例资源。
续训恢复Python/NumPy/Torch/CUDA及DataLoader generator，逐轮保存实际样本顺序并检验恢复回放。
旧runner的generator未恢复缺口只在新runner修复，不修改旧运行任务。
所有训练源码快照/初始化指纹/数据身份固定；源码变更后不自动兼容旧检查点。
仅调度空闲GPU，不抢占现有任务；四主臂先启动，其后匹配对照；不按AP提前停止。
新增采样和梯度诊断优先离线，不膨胀逐步日志，不进入loss/门控/选模。
最终分别判定接线、真实内容依赖、相对卫星净收益，公开不可判断项。

工程门修正：投影坐标落在格边界附近时，float32缓存会跨格舍入，因此锚点元数据以float64保存，观测监督像素由此确定；模型局部采样输入才转float32。旧index/geom仍逐值保持。

常量训练对照的主评价输入为constant，其他臂为correct；每个检查点仍额外执行三种真实输入干预，
mode=correct始终读真实RGB，不由训练content覆盖。所有干预完整数值单列，不能将常量输入误报为正确RGB。
新runner完整恢复随机状态，验证真实train数据顺序及噪声/颜色增强前缀回放；CUDA grid_sample反传
可能有原子累加非确定性，因此不声称跨GPU训练或续训浮点轨迹逐位一致。
