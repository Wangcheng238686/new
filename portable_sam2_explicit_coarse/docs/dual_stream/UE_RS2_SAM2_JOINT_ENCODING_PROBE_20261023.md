# RS2-SAM2精读与J1编码器内部交互探针

状态：**设计已实施并启动两臂训练**，见[实现与启动](UE_J1_IMPLEMENTATION_20261023.md)。本文下方为实施前固定设计；两个独立子代理rs2_paper_close_read、rs2_uav_probe_design完成正文/图表与实际代码复核，主代理复读图2/3、公式、表3/4及官方实现。

## 1. 证据身份

- 用户PDF：`/home/wangcheng2021/.codex/attachments/6a0709bd-9c32-4f53-b13f-e6d31c4f1413/RS2-SAM2.pdf`。
- PDF SHA256：9b08cdc33521000b04c4b4da8535e4be23244710ad11ec8a4c124bb24d952fea；arXiv:2503.07266v4，2026-01-15。
- 作者代码：https://github.com/rongfu-dsb/RS2-SAM2 ，核对commit `7321e75c16ec43f0e26391be494d0c69f9bcffb4`；只读副本 `/tmp/rs2-official-20261023`。
- 官方关键文件：models/sam2/modeling/backbones/hieradet.py（Adapter_cross、MultiScaleBlock.forward）；lib/evf_sam.py（冻结范围）；models/sam2/modeling/sam/mask_decoder.py（编码器外BC）。

## 2. 论文实际机制

**Union encoder与SAM2内部交互必须区分。** 第3页图2/Feature Extraction：额外BEiT-3读224²图像和文本，联合编码生成视觉patch、文本tokens和CLS；SAM2另读1024²图像。不是把BEiT-3等同于SAM2。

**内部BL**：第4页图3/式1–2在Hiera编码块内部加入低维双向cross-attention与MLP adapter。原SAM2视觉特征和语言降维后交互，再残差恢复视觉维度并继续后续编码层。文本按alpha_t=.2与原文本混合，持续向后传；额外MLP adapter的alpha_i=.5。官方Large为48个block，逐block插入，不只是stage出口。

作者代码顺序是visual←text，然后text←更新后的visual，并把新text传到后续block；公式写法不能代替该实际数据流。

**外部BC**：编码完成后，用原union文本再与高层SAM特征交互并乘性融合。因此BHFM不全等于内部adapter。

**MPG**：独立CLS/BEiT视觉tokens生成伪mask与稀疏提示；不属于内部BL。首版J1保留现有CSPM/UDPR，不新增MPG/BEiT-3/文本边界loss。

**冻结**：图示与代码均冻结SAM2 image encoder原参数，开放内部adapter；官方lib/evf_sam.py先开放整个visual_model，再冻结image_encoder，故prompt encoder/mask decoder仍可训练；BEiT-3全量可训练。论文不能证明冻结我们全部消费者也足够。J1冻结原模型是可归因的新假设，不是复现其训练范围。

## 3. 消融证据边界

第7页，RefSegRS test mIoU：

| 表3设置 | mIoU |
|---|---:|
| baseline（SAM2+union） | 36.64 |
| +文本边界loss | 38.63 |
| +边界loss+MPG | 60.20 |
| +边界loss+BHFM | 68.71 |
| 完整 | 73.90 |

| 表4设置 | mIoU |
|---|---:|
| Linear | 68.19 |
| 单向Uni | 70.10 |
| 双向Bi | 73.90 |
| 去掉编码外BC | 67.61 |
| 去掉编码内BL | 61.08 |

支持图文指代分割中的编码内部交互及层间持续传递有用；不证明单个入口、跨视角几何、冻结decoder、实例AP或我们数据上有相同增益。没有我们所需的常量RGB容量对照/多seed。文本边界loss式4通过句向量生成每图标量权重，不是提供空间化文字边界标签。

## 4. J1唯一主要问题

**在编码器内部，先让UAV候选读取卫星上下文，再让卫星读取更新后的候选，能否产生优于同结构常量RGB的完整检测/分割收益？**

借用BHFM的低维交互、视觉残差、继续编码思想；首版使用单个入口、局部几何候选和闭合往返路径，是我们的任务适配，不声称完整复现RS2-SAM2。

## 5. 实际入口与候选

EXP-0017是Hiera Base+：stages=(2,3,16,3)，stage_ends=[1,4,20,23]。在trunk.blocks[4]输出插入，512²输入时NHWC `[B,64,64,224]`。返回替换张量同时进入该stage输出与blocks[5]..[23]；不能套论文Large的层号/通道。

stride8及后续特征受影响；已产生的stride4 lateral不直接改变。需实际追踪SAM FPN/检测neck依赖，不能笼统称全部尺度已经联合编码。

UAV沿用五视角编码器+LoRA，源P0连续坐标读取按既有L1解析合同，降维32。camera-Z EXR R、cm→m、相机矩阵等仍用已有合同。深度只限定对应候选和提供几何属性，不是新增深度预测头。

每个64²目标token确定地拼接2×2个128²旧候选格，每格8个，共最多32个候选。保留index/source_uv/anchor/valid，不resize候选图、不重新排序/top8、不静默去重；重复率离线报告。四格重组是新入口的工程适配，不把它与旧后端探针混为单因素比较。

这些是编码器全图token，使用图像坐标和实际pooling footprint，不经过proposal bbox；与S1-R的roi_local mask格是不同合同。一次2×2 pooling的名义格点中心由stride4中心0/4/8…变为8j+2，但并非精确3D点。实际支持bin、投影anchor与token名义中心须分别记录，解析边界/footprint门通过后才实施，避免再次用张量形状推断空间语义。

## 6. 闭合双向交互

卫星低维特征S_j，候选U_ji，卫星3×3邻域N(j)，维度32、首版单头：

1. `U'_ji = U_ji + .2 * Attn(Q=U_ji, K=S_N(j), V=S_N(j))`。
2. `Z_j = Attn(Q=S_j, K=U'_j+GeomK, V=U'_j+GeomV)`。
3. `S_high_j_new = S_high_j + supported_j * W_out(Z_j)`，输出恢复224维。

邻域固定相对位置编码、边界mask、无支持最终严格零。reverse必须有多个有效卫星K，避免单K softmax恒为1。更新U'立即被第二步读取，故反向支路有损失路径；不能并行算两方向后丢掉U'。

该顺序与作者逐层visual→text再传后层不同，是单入口闭合往返的适配。存在卫星→候选→卫星的旁路，常量对照不可省。

W_out零初始化，不叠加零alpha阻断；原MLP/选点/残差限制/损失不变。第一步上游梯度为零是初始化现象，后续必须验证两个attention方向与有效UAV LoRA都有非零梯度。

## 7. 训练范围与梯度

训练UAV LoRA、source/geometry映射、双向attention和224维输出映射。原卫星全部参数（包括已有LoRA、SAM FPN、检测neck/头、CSPM、decoder、UDPR）冻结；不同时添加N2/S1后端入口。

原参数冻结但融合点以后必须自动求导。旧runner整段no_grad(model.extract_feat)不能沿用；block4前可无梯度，插入后blocks5..23/FPN/检测/分割均保留对输入的梯度。原模型eval保持buffer与dropout不漂移，adapter单独train；checkpoint策略另测。

仍使用现有检测、coarse、最终mask和tail目标，不新增辅助边界loss。由于特征提前改变，检测/提示loss也可能连到adapter，应按实际loss分项记录离线梯度，而不是预设只有mask/tail两项。

## 8. 最小矩阵与判据

- 基座：冻结EXP-0017完整推理，无新训练。
- J1-correct：正确RGB+目标几何。
- J1-constant：同结构/初始化/优化/数据顺序，常量RGB+相同目标几何。
- 正确CP的constant/donor RGB完整推理：固定权重依赖诊断，保持目标深度/姿态。

首轮只2条训练，10epoch/2670更新、相同逐epoch种子与既有优化协议，固定E10 last、nonEMA、coarse64、maxDets250。如果后续开放卫星LoRA，必须加同范围纯卫星继续训练对照；当前原模型全冻结无需训练空臂。

主要判据：correct对基座的净MaskAP及对constant训练的内容差，配对图像区间；bboxAP/AR250/BoundaryAP共同报告。常量臂保留UAV几何/支持，不能说它是完全无UAV。只有两臂不能声称双向优于单向；若内容信号成立，再补同入口Uni正确/常量唯一因素对照。

检测框/分数/数量允许随融合变化；只要求相同图像顺序/GT/数据身份，必须新建完整COCO配对评估，不能沿用旧analysis的bbox/scores精确不变断言。需要500次相同图像索引重采样、单seed/开发test限制；可辅以固定基座框诊断分解检测与掩码贡献。

## 9. 实施前必过门

1. 严格重建、W_out=0及disabled下FP32/AMP精确回放全部预测；非零映射无支持恒等。
2. 实测block4形状、后续block输入确为更新tensor；支持域/候选重组/连续源UV解析测试。
3. 注入前后FPN各层依赖核对，确认预期不变层与变化层；不误用roi_local变换。
4. 实际损失到forward/reverse attention、UAV LoRA、W_out梯度；原参数/buffer指纹不变。
5. 最大实例训练反向和250实例FP32推理显存实测；不能套旧尾部约3GiB。现卡上共租任务多，未测不准入。
6. 独立source/runner/manifest，冻结现有运行队列代码；不将本设计描述成已运行实验。
