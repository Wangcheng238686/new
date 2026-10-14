# UAV 双流首版工程核定

日期：2026-10-14。单流源项目 `/home/wangcheng2021/project/new`（核对提交 `95026ae`）。本轮完成数据几何审计与无参数更新的资源探针，尚未实现正式双流或验证 AP。已确认的研究边界见 [设计记录](NEW_PROMPTMINER_UAV_DUAL_STREAM_DESIGN_20261014.md)。

## 1. 核定依据

从当前可读配对目录中，按索引顺序分 32 层、每层固定种子抽 1 点，核对 32 组卫星和 160 个 UAV 深度文件，零读取错误。数据路径和冻结索引见 [审计](probes/uav_dual_engineering_20261014/geometry_audit_height.json)；这不是训练划分或数据版本裁决。卫星与 UAV 均为 512×512；卫星相机在样本中均为正俯视、2000m 高、fx/fy=2844.44，地面参考视野约 360m×360m。UAV fx/fy≈787.91，轨道半径约 300m，逐视角高度有变化。

宿主机有 4×RTX 3090，每卡 24576MiB。当前沙箱不暴露 GPU；宿主机资源探针已实际执行。当前 Python 环境有 cv2、SciPy、Torch 2.8.0+cu128、mmdet 与 peft；没有 OpenEXR Python binding，但现有 cv2 能读取 EXR。实测 EXR 在 cv2 中返回 float32 BGR 三通道，B/G 为零、R 为 camera-Z；读取 `[...,2]`，禁止对三通道求均值。

## 2. 首版默认参数

机器可读草案：[engineering_config.json](probes/uav_dual_engineering_20261014/engineering_config.json)。其中 collision band、门控尺度等是实施初值，尚未用正式双流验证。

| 项目 | 首版核定 | 依据 |
|---|---|---|
| 卫星/UAV 输入 | 均为 512×512 | 原始分辨率，无需放大到 1024 才能接 SAM2 |
| 世界 BEV | 512m×512m，128×128，4m/格，256ch | 32 点中保留 UAV 观测比例中位数 98.79%，最小 97.42%；所有已投到卫星内的采样点均在此范围 |
| BEV 原点 | UAV 共同目标点；缺失时使用 UAV 相机 XY 中心 | 不由卫星图的 footprint 定义，不要求两者覆盖一致 |
| UAV token 特征 | stride8 细节 + 上采样 stride16 语义，统一 256ch | 64×64 token 网格，兼顾细节与深层 LoRA 的梯度通路 |
| 跨视角注意力 | 2 层、8 heads、FFN 512 | 固定米制 query、相机/深度几何编码；无卫星 query 条件 |
| XY 候选邻域 | 半径 8m，每视角最多 8 个候选，总共最多 40 | 卫星地面参考 query 区域的邻域支持率中位数 98.47%；最差样本仍有缺口，不补成全覆盖 |
| 注意力执行 | query 每次 1024 个，checkpoint；五视角一起编码 | 资源探针已验证可行，必要时按视角分块 |
| 观测高度 | 单层辅助图；2m 几何分桶，选最高且有 ≥3 个观测的桶，桶内取均值 | 不把屋顶/立面的所有 Z 混合平均；这只是几何统计，不构建多高度语义体 |
| BEV → 卫星 | 高度辅助的细像素几何足迹，再归并至融合网格 | 不在粗 FPN 格判断硬遮挡；不创建全分辨率密集特征图 |
| 融合入口 | P3/P4（stride8/16），SAM image embedding（stride16） | 首版 SAM 高分辨率分支保留卫星细节 |
| 融合初值 | 每入口 α=0.05，非零投影，空间门控结合几何置信 | 保持小残差且让 UAV 分支启动；此值是工程初值 |
| 每 GPU batch | 首版 1 组卫星+五视角 UAV，AMP+GradScaler | 保守保留正式几何/门控、DDP 与真实候选数的余量 |

512m 的选择使 BEV 与卫星约 360m 地面视野明显不同。384m 也能容纳本次投进卫星的点，但其 UAV 总观测保留比例中位数只有 90.90%、最小 58.85%；512m 能保留更多 UAV 周围上下文。640m 的保留比例中位数 99.98%，若仍用 128 格会增大米每格；因此首版选 512m，额外远处观测仍可作为图像背景，BEV 外残差不注入卫星。真实采样下范围参数必须可配置，不能硬编码旧的 236.5m。

## 3. 几何和注意力的实现约束

所有深度与矩阵平移先统一为米；camera-Z 反投影用 `Z * K^-1[u,v,1]`，不归一化射线。CPU 以 float64 构建局部坐标，传入 GPU 后用 float32 计算几何。若局部原点为 o，world-to-camera 的局部平移为 `t_local = t_world + R @ o`，不只把点减原点而忘记同步矩阵。世界 X 为 BEV 列轴、Y 为行轴；卫星图的旋转和方向由 K/E 处理，不额外手动转图。

深度必须拒绝非有限和非正值；不作每图 min/max 归一化。UAV RGB 做一次 BGR→RGB 和与骨干一致的标准化，禁止重复翻转通道。首版保持原分辨率，关闭空间 resize/flip/erase 增强；之后若启用，必须更新原图→模型像素仿射，深度数值本身不随 resize 缩放。

每个 stride8 图像 patch 的几何锚点使用真实观测像素，不能双线性平均边界两侧深度。本次每视角 patch 中 world-Z 的 P90−P10 超过 2m 的比例中位数为 53.43%；camera-Z 的梯度本身不能用于判断高度混杂，因为平坦地面在斜视图中也有深度斜率。首版允许高度变化大的 patch 增加一个上表面锚点，两锚点可读取相同图像 token；先保留候选数量上限 8/视角，不扩张至全局注意力。

候选以真实世界 XY 邻域形成，relative XYZ、视角、有效性和高度置信进入注意力偏置/位置编码。先以高度软条件选信息，不用硬 Z 阈值把所有立面特征排除。无候选 query 必须避免 all-masked softmax 的 NaN，且对应 UAV 融合残差为零。物理网格原点/轴向/米每格不可训练；学习 query 特征不改变其世界位置。

128² query 对五视角 64² token 的全局注意力共有约 3.36 亿对/头/层；40 候选上限只有约 65.5 万对。显式储存全局 8-head FP16 logits 约 5GiB/层（高效 attention kernel 可以不显式储存，但计算量仍大）。首版用稀疏候选，在 CPU/data worker 预计算几何索引；不缓存 LoRA 可训练编码器的图像特征。

## 4. 高度修正与置信度

正俯视透视相机也有高度引起的径向缩放：`u-cx ∝ (Y-Yc)/(H-Z)`、`v-cy ∝ -(X-Xc)/(H-Z)`。本次 32 点中，各样本 ground-reference 投影误差 P95 的中位数为 7.75px，范围 2.92–33.69px；P99 最大 42.23px。因此不能因 pitch=-90 就忽略高度。

在其中 10 点做独立高度对齐代理测量：同一 XY、只替换代表 Z 后，各样本误差 P95 的中位数从 6.87px 降至 2.13px。再加上 4m 格中心量化，P95 中位数为 3.81px，即约 0.48 个 P3 单元、0.24 个 stride16 单元。这支持先向低分辨率融合入口注入，暂不直接影响卫星高分辨率边界路径。

但存在一个高层建筑场景，修正后的 P95 仍为 16.53px，不能声称一格一高度已精确对齐。上表面高度桶占该格观测至少一半的参考点，各样本 P95 中位数约 0.21px，最大约 0.37px；这部分占参考点比例中位数约 55.61%，结果只对该子集成立。证据：[height_alignment_confidence.json](probes/uav_dual_engineering_20261014/height_alignment_confidence.json)。

据此将上表面观测占比、跨视角支持、最近 XY 距离和高度离散度作为 C_align。首版对高度混杂区域软降权，不硬性宣称整格都是屋顶。一个可用初值是 `C_height = clip(2 * upper_cluster_fraction, 0, 1)`，再结合 view/距离置信与可学习门控；未观测/越界保持零残差。分支必须输出这些量的日志和对齐可视化。

2026-10-14 对抗复核修正：BEV cell 在本格代表高度上投影四角，形成512卫星像素平面的有效小足迹，仅在细像素同位置进行 camera-Z 近表面冲突筛选，再把通过的稀疏条目归并到 stride8/16 特征网格。不能把同粗格的不同像素视为同一射线遮挡，也不能只向四个细像素投格中心后把空洞比例当面覆盖。不建立256通道全分辨率密集canvas。2m带只作为细像素竞争初值，feature均值、mass/coverage与confidence分开处理；对feature可回传梯度，观测几何本身不要求梯度。当前没有卫星深度，仍不证明真实卫星可见性。完整规则和退化方案见 [对抗复核](ADVERSARIAL_DESIGN_REVIEW_20261014.md)。

以上高度代理参考来自 UAV 点云中每个卫星像素最近的观测点，卫星标签只用于选取诊断前景；不把标签作为模型几何输入，也没有新的 UAV/BEV 分割监督。

## 5. 显存与梯度资源探针

探针使用当前单流检测器、独立 UAV SAM2 编码器、两层随机稀疏候选 attention 和两个残差入口。两边均从通用 SAM2 预训练初始化；一次前向+反向，无 optimizer.step，无 UE 单流热启动。GT 来自一个 133 实例样本，RGB 与候选索引为随机占位，BEV→卫星仅用 resize 模拟张量尺寸；它是资源/接线探针，不是正式 CVT、几何实现或性能实验。

| UAV 源特征 | 峰值 allocated | 峰值 reserved | 非零 UAV LoRA 梯度张量 | 解释 |
|---|---|---|---|---|
| 仅 stride8，AMP，五视角一起编码 | 5110.6MiB | 5534MiB | 12 | 很多深层 LoRA 没有参与该特征路径 |
| stride8+stride16，AMP，五视角一起编码 | 5047.0MiB | 5240MiB | 52 | 语义尺度加入后深层 LoRA 可受监督 |

最终探针 29 个 attention 参数张量有非零梯度，全网络无非有限梯度；loss finite。两套 encoder 每套可训练参数 1,492,528，其中 LoRA 1,061,424，其余是单流已有的 SAM2 neck 适配参数；原始 Hiera 权重冻结。此处继承单流 `modules_to_save=['neck']`，不是全量骨干微调。

总可训练参数约 50.91M，AdamW 双状态估算额外约 388MiB；未执行 optimizer.step，未测稳态 DDP/真实几何版本。一次前向反向耗时约 12.33s，不当作训练吞吐承诺。内存测量证明首版尺寸有实现余量，初始仍用每卡 batch1；正式模型建立后再测一次完整峰值，而不是根据探针直接扩大 batch。记录：[resource_probe_multiscale_amp.json](probes/uav_dual_engineering_20261014/resource_probe_multiscale_amp.json)。

## 6. 单流接入与评测配置

建立 UE 单流 512 配置，再派生双流：PromptEncoder image_size=512、embed_size=32，shared image PE 与固定 padding 都同步到 512，最终 mask 监督尺寸同步，coarse 每 RoI 64² 和 UDPR K=64 保留。当前 roi_local 是最终监督/粘贴方式，不表示 SAM decoder 的 image embedding 被 RoI crop。两入口必须传递完整卫星空间特征，沿用这个契约。

32 样本的可见实例数量中位数 61.5、最大 242，有 9 张超过 100。post-NMS `max_per_img` 首版改为 300，单流/双流都采用同样设置；标准 COCO AP 仍用 maxDets=[1,10,100] 作主指标，再单独报告 [1,10,300] 的拥挤场景诊断。仅增大模型输出上限不会自动解除标准 AP100 的计数限制，不把 AP300 混称标准 AP。RPN proposal 数量先保持 1000。

继续只用卫星实例目标的 RPN/bbox/coarse/final mask/UDPR 损失；不额外引入 UAV/BEV 分割目标。UAV/CVT/fusion 分组管理优化器和梯度遥测，主任务与 coarse 对 UAV 的梯度分开诊断。新增大量 attention 参数的范数不应挤占基础网络/UDPR 既有裁剪预算。训练数据划分需以物理区域隔离，并检查卫星/UAV 覆盖重叠与重复建筑；不能使用逐帧随机划分判断泛化。

## 7. 实施验收边界

工程默认已足以开始正式模块实现。实现顺序：UE 多模态 payload/坐标契约 → 多尺度 UAV 特征与稀疏 CVT → 单层高度辅助投影 → 两入口融合 → 独立训练/推理与 checkpoint 契约。

必需检查：空邻域/越界零残差与 NaN、厘米/米和增强仿射、高度前后对齐图、各损失到 CVT/LoRA 的梯度、正确与错位 UAV 响应、正式显存峰值。资源探针只验证尺寸与可微接线，32/10 点几何审计只验证抽样几何。剩余性能风险主要是高层混杂高度和不可证明的卫星可见性，不是缺少 UAV 分割标签。

2026-10-14 对抗复核：实施前还须明确 candidate/observation/height_valid 的区别、唯一token去重、全图embedding到roi_local输出契约，并解耦当前代码中的模型输出cap和AP maxDets。完整清单见 [对抗复核](ADVERSARIAL_DESIGN_REVIEW_20261014.md)。
