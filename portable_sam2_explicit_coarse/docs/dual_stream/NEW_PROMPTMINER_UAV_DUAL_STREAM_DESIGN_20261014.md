# 基于 project/new 的 UAV 双流模型设计草案

2026-10-14。只读核对项目 `/home/wangcheng2021/project/new`，提交 `95026ae`，工作区无未提交改动。本文件是设计建议，尚未修改该项目或验证双流 AP。

## 0. 已确认的首版设计约束

- 候选主模型采用 PAFPN P3/P4 与 SAM2 stride16 image embedding 两个融合入口；仅 FPN 入口作为关键对照。
- UAV 深度在训练和推理中均为必要输入；深度贡献消融后置。
- 已有分割标注仅在卫星视角；UAV 与 BEV 没有分割标签。首版设计不以额外 UAV/BEV 分割标签为前提。
- 正式双流直接从通用 SAM2 预训练权重初始化并联合训练，不依赖 UE 单流检查点或先训练 UE 单流的流程。UE 单流独立训练作对照。
- 卫星与 UAV 使用相同 SAM2 编码器结构和通用预训练初始化，各自拥有独立 LoRA；五个 UAV 视角共享一套 UAV 编码器。首版开放 UAV LoRA，不将冻结策略作为重点消融。
- 首版独立构建 UAV BEV：BEV query 只依据世界坐标、UAV 特征、相机参数和深度。卫星特征在之后的两个融合入口参与门控。
- 首版单层 BEV 保留由 UAV 深度与 K/E 得到的观测高度和置信度辅助信息，供 BEV → 卫星的屋顶投影修正使用；不恢复多高度 BEV。每格可能聚合多个高度，高度辅助对齐是近似，具体聚合方式须结合样本检查。
- 卫星参与 BEV 构建的路线留作后续尝试，具体记录见 §6，尚未决定实施。

## 1. 单流的真实结构

源代码根：`/home/wangcheng2021/project/new/portable_sam2_explicit_coarse`。

`SAM2 Hiera-B+ LoRA → RSSAM2PAFPN → RPN → bbox RoI / mask RoI`。
mask RoI `[N,512,14,14]` 经 CSPM 预测 64×64 粗掩码，从中生成 2P2N 点、框、dense canvas，经 PromptEncoder 到 SAM2 MaskDecoder，再经过 UDPR 的 K=64 不确定点残差修正。

`detection/anchor_detector.py::extract_feat` 返回四条特征：`x`（PAFPN）、`image_embeddings`（stride16，256ch）、位置编码、`high_res_features`（stride4/8，256ch）。PAFPN 是 stride4/8/16/32/64 的五层 512ch。RPN 与 bbox/mask RoI 使用 PAFPN；SAM2 decoder 使用另一路 image/high-res 特征。RoI head 的额外正弦 PE 在其 loss/predict 内加入，实际位于 RPN 之后，不能根据注释误判它已进入 RPN。

CSPM 当前固定 `roi_only`，不使用 image embedding 的视觉 context。点挖掘是 no_grad；正式 dense 配置 `detach_input=True`。因此：

- FPN 融合可用 RPN/bbox/coarse 辅助损失训练 UAV 分支，并通过更好的框和粗提示改善最终掩码。
- 仅 FPN 融合时，最终 SAM2 mask / UDPR loss 不能经粗提示直接回传到 UAV 分支。
- 把 UAV 注入 SAM2 image embedding 可增加最终 mask / UDPR loss 到 UAV 分支的直接梯度通道。
- PromptEncoder 的冻结不等于输入梯度被禁止；这里阻断粗头梯度的是 dense detach 与点挖掘 no_grad。

依据：`cspm/mask_head.py::_forward_coarse`、`mask_head_helpers.py::_shape_prior_to_prompt_mask`、`shape_prior_injector.py::forward_roi`、`detection/anchor_roi_head.py::_apply_coarse_loss`。

## 2. 推荐主结构与对照

UAV 五视角 RGB 由共享的 UAV encoder 编码；用 K/E 的相机感知位置编码和 camera-Z 深度约束，执行 CVT 风格跨视角注意力，输出 `B_bev [B,256,Hb,Wb]`、世界 XY 网格描述、有效观测 mask 与支持度。深度用于注意力定位/筛选和置信度，特征仍由注意力聚合。卫星与 UAV 使用相同通用 SAM2 预训练初始化，冻结原始骨干权重，各自训练独立 LoRA；UAV 的 CVT 与融合模块也可训练。五视角共享 UAV encoder 权重，可分块执行控制显存。首版不以卫星特征生成 BEV query。

世界 BEV 不要求覆盖整张卫星图。以每样本卫星 K/E、BEV 原点/轴向/米每格和原图到模型输入的仿射矩阵，采样出卫星坐标下的 UAV 特征。输出特征必须保持完整卫星网格大小；无支持/越界区域的 UAV 残差为零。首版单层 BEV 附带由深度观测得到的高度及置信信息，供投影时修正屋顶相对地面参考的偏移；具体采样/投影方法仍需实现前核定。ground-plane warp 是参考对齐，不能当作屋顶精确投影；一格一个代表高度也不能精确描述屋顶、立面同时存在的情况。先可视化测量实际错位及高度聚合误差，再确定修正强度或局部采样邻域，不恢复多高度 BEV 体。

两个入口共用同一份 BEV 与几何对齐模块，分别做通道投影和门控：

1. **检测和提示入口**：`x = neck(...)` 后、RPN 前，融合 P3/P4（stride8/16）。RPN、bbox RoI、CSPM 均读取融合后的特征。P2 的高分辨率注入随后按对齐与小建筑结果检验。
2. **掩码视觉入口**：stride16 的 `image_embeddings` 选择后、重复到每 RoI 之前，加入 256ch UAV 残差。低分辨率 SAM2 图像通路直接消费 UAV；高分辨率卫星特征先作为边界细节来源。此入口不直接改变 RPN 输入，但联合训练时可经共享卫星主干间接影响检测。

融合公式：`F' = F + α · M · sigmoid(G([F,Proj(U),C])) · Proj(U)`。
`M` 为几何/实际观测有效性，`C` 为支持度，`G` 为轻量空间门控；投影后做适当归一。先与对齐相加对照。小非零 α 可让 UAV 分支从第一步收到梯度；若选零初始化，应实测其启动过程，避免同时把投影输出也置零导致整个残差支路不启动。门控不预先等同于可靠性，必须用错位/错配对照验证。

## 3. 为什么先保留 CSPM 与 UDPR

更好的融合 RoI 特征可改善粗掩码，从而改善正负点和 dense 提示；这正是当前单流已有的 UAV 信息入口。coarse loss 是此入口的重要监督。先不取消 detach 或改挖点规则，因为那会同时改变单流已验证的梯度契约。

UDPR 继续修正 decoder 的不确定像素，不在尾部重复建另一套 UAV 检索。掩码视觉入口使其通过 decoder 上采样特征间接消费 UAV。若仅观察到尾部增益，需要另作 tail on/off 归因；不能将尾部得分变化直接归因于 BEV 本身。

## 4. 代码接入边界

建议增加 `uav/encoder.py`、`uav/cvt_bev.py`、`uav/bev_sat_align.py`、`uav/fusion.py`，新增 `detection/dual_detector.py` 继承单流检测器。显式通过 `_extract_dual_feat(batch_inputs,batch_data_samples)` 构造融合特征，由双流的 loss/predict 接入原两阶段流程；不在模块属性上保存跨调用的可变 UAV 缓存。首轮不必修改 bbox head、mask head 的前向签名或 MaskHeadOutput。

现有训练链路不支持多模态输入，需新增 UE RGB-D dataset/collate，并让训练、验证、独立推理都传递 RGB、深度、valid、UAV K/E、卫星 K/E、world/BEV 描述与仿射。大张量放 data payload，标识/坐标描述放 metadata。当前 loader 只接受 whu_coco/vhr10_coco，collate 只收卫星和 GT；不能只改模型而丢失 UAV 数据。

卫星和 UAV 的 resize/flip 分别记录图像仿射，投影时使用各自变换；深度几何量不随像素缩放而缩放。当前单流 dataset 执行翻转但未记录 flip 信息，双流首轮可关闭空间增强，随后用带仿射的增强恢复。裁剪/缩放不能改变世界坐标，也不能用 clamp 把 BEV 越界点拉回边界。

WHU 静态配置是 1024 输入与 64×64 SAM embedding。应建立独立 UE 单流配置，再派生双流配置；若选择 512 输入，SAM embedding 是 32×32，PE 图像/embedding 尺寸、固定 pad、监督尺寸等需同时一致；coarse 64×64 是每 RoI 尺寸，可以保留。`final_mask_coordinate_mode=roi_local` 在当前代码是最终监督/粘贴契约，不代表 decoder 的 image embedding 已被 RoI 裁剪。UE 首轮可采用 WHU 的单类/roi_local 机制，并在同一 UE 单流配置下比较双流。

新增 UAV/CVT/fusion 参数需要独立 optimizer 分组和梯度范数遥测；沿用现有 UDPR 裁剪机制，检查新增分支是否挤占卫星基础参数的裁剪预算。遥测建议：BEV valid/support、每层 gate、残差/卫星范数比、各损失对 UAV 分支的梯度范数、coarse IoU 与点正确率。

正式训练使用通用 SAM2 编码器、PromptEncoder 与 MaskDecoder 预训练权重；检测头、CSPM、UDPR、CVT 和融合模块按各自规则初始化，不加载训练好的 UE 单流检查点。项目已有 cross-architecture init 和 missing-prefix 白名单，但不将该能力作为正式训练依赖。新增双流配置应有自己的完整指纹；原架构 ID 逻辑不会因 UAV 键自动生成双流语义标识，需要在新扩展中明确登记，保留旧契约冻结区。双流保存后应能独立 strict rebuild/load。

## 5. 最小实验矩阵与判定

| 实验 | FPN 入口 | SAM image 入口 | 主要问题 |
|---|---|---|---|
| A：UE 单流 | 无 | 无 | 同数据同预算的基线 |
| B：FPN 融合 | P3/P4 | 无 | 定位与粗提示是否可利用 UAV |
| C：SAM 图像融合 | 无 | stride16 | 直接视觉通路是否改善掩码 |
| D：两入口 | P3/P4 | stride16 | 两种消费路径是否互补 |

首轮独立训练 A/D，使用相同空间划分、有效 batch、训练步数和选模规则，先验证双流系统是否改善完整卫星实例分割。B/C 单入口消融后置。D 是候选主模型，结果好坏由实验决定。无需首轮同时添加 bbox RoI 融合、mask RoI 融合和边界模块。

主指标是 mask AP/AP50/AP75；报告 bbox AP、proposal AR@K，固定同一 GT box 或同一 proposals 的掩码质量、coarse IoU 和挖点准确率。GT-box 条件评测是诊断，不当作标准端到端 AP。按真实 UAV 支持度、物体尺寸、拥挤度分组，使用空间隔离的划分。

对有收益的模型检查正确/平移/跨场景错配/零 UAV 残差。零/错配的推理干预用于输入依赖诊断，存在分布变化，不能单独排除容量与优化因素。首轮 A/D 可支持“配对 UAV RGB-D 双流系统改善 UE 卫星可见建筑分割”；若论文要排除容量因素或证明 UAV RGB 独立于几何的贡献，再增加支持度-only、几何-only 或等容量重训对照，不将这些作为首版前置条件。FPN 分支无最终 mask 梯度是当前设计事实，不能误判成整体不可训练；coarse/检测梯度是否足够，须先做真实梯度检查，后续由 B 与 D 比较。

已确认 D 为候选主模型，B 为关键消融；训练的正式起点遵守 §0，不从 A 的 UE 检查点热启动 D。

## 6. 后续路线：卫星参与 BEV 构建（暂不实施）

首选备选方案是轻量卫星条件化 query：保留世界 XY BEV 网格，将卫星低分辨率特征通过相机几何采样到各 BEV 位置，以 `Q = Q_position + Proj(F_sat_to_bev)` 调节查询。五视角 UAV 提供 key/value，K/E 与深度约束注意力采样区域，得到由卫星条件化的 UAV BEV，再进入现有两个融合入口。卫星只影响查询时，输出 value 聚合主要来自 UAV；避免默认将卫星特征直接作为 BEV 输出残差。覆盖之外回退到位置查询，实际 UAV 支持度仍独立计算。该方案是基于 CVT 的拟议扩展，并非原 CVT 已包含的卫星模块。

另一条备选路线是 SG-BEV 风格的卫星预测建筑区域软引导：使用推理时由卫星自身预测的建筑区域，指导 UAV 特征的空间组织/对齐。SG-BEV 处理街景立面特征的重分布，UAV 可观察屋顶，适用性需重新验证；不能直接照搬，不用真值掩码作为推理引导，也不以卫星预测硬掩码排除潜在 UAV 证据。

参考：[CVT, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Zhou_Cross-View_Transformers_for_Real-Time_Map-View_Semantic_Segmentation_CVPR_2022_paper.html)；[SG-BEV, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Ye_SG-BEV_Satellite-Guided_BEV_Fusion_for_Cross-View_Semantic_Segmentation_CVPR_2024_paper.pdf)。

后续若尝试卫星条件化 query，需比较独立/条件化 BEV，在相同训练预算下检查 mask AP 与正确/错配 UAV 响应。卫星判断可能影响跨视角选点，须检查是否抑制卫星漏检区域的 UAV 证据，以及是否主要重复利用卫星信息。

## 7. 无 UAV/BEV 标签时的训练信号

卫星实例标签经已有的实例框/掩码目标生成，提供 RPN 分类与回归、bbox 分类与回归、CSPM coarse、最终 mask 与 UDPR 损失。拟议总损失保留这些项，不增加要求 UAV/BEV 分割标签的目标。

- 检测与 coarse 路径：卫星目标 → RPN/bbox/CSPM loss → 融合 P3/P4 → UAV BEV → CVT → UAV LoRA。
- 最终掩码路径：卫星目标 → final mask / UDPR loss → SAM2 decoder → 融合 image embedding → UAV BEV → CVT → UAV LoRA。
- 点挖掘 no_grad 与 dense detach 仍然阻断 final mask → coarse 提示的直接梯度；它们不阻断第二个 image embedding 入口。
- 深度与相机参数作为几何输入缩小对应关系的不确定性，无需学习深度预测。它们不是 UAV/BEV 分割标签，也不提供新增像素语义监督。

结论限于设计：上述两条可微路径足以建立端到端训练，不存在必须先有 BEV 标签才能训练的缺口；新增模块尚未实现，不能声称已经测得有效梯度或 AP 增益。信号是否足够强取决于观测支持、屋顶对齐、门控/投影的尺度与实际梯度，且模型可能主要依赖卫星通路。

实施后先检查各损失对 UAV/CVT/fusion 参数组的梯度（不要求 LoRA 每个张量从第一步都有非零梯度）、有效区域残差范数与 gate；用少量高覆盖样本的拟合检查确认有信息通路，再看空间隔离验证集的正确/错配 UAV 响应与 AP。避免将门控和投影同时全零初始化，避免新增分支挤占原训练的梯度裁剪预算。若检测梯度明显强于掩码梯度，先检查损失尺度和两入口门控，而不因缺少 UAV 标签直接添加辅助任务。

## 8. 设计收束与待核定的工程参数

用户已明确首版的核心设计倾向：独立深度约束 CVT BEV、单层高度辅助对齐、FPN + SAM2 image embedding 双入口、独立且开放的 UAV LoRA、通用预训练起点联合训练、仅已有卫星实例标签监督。后续卫星条件化 BEV 路线已记录，但不进入首版。

剩余项应以环境/数据证据核定：BEV 实际物理范围与米每格、网格分辨率、模型输入尺寸及五视角执行显存、深度约束注意力的邻域实现、高度聚合与到卫星网格的采样方法。核定时遵守独立 BEV 与卫星覆盖不强求一致、完整卫星预测输出、无支持区域零残差的约束。上述参数未确定不等于核心架构还需新增模块。

2026-10-14 工程核定完成：32 点几何审计、10 点高度代理检查及 RTX 3090 一次前向/反向资源探针。首版默认 512² 输入、512m×512m 的 128² BEV、stride8+stride16 UAV 源特征、8m 局部邻域/每视角最多8候选、2层8-head注意力、单层高度置信辅助的正向投影与两个融合入口。详见 [工程核定](UAV_DUAL_ENGINEERING_SPEC_20261014.md)；这不是正式模型已实现或 AP 增益已验证的结论。
