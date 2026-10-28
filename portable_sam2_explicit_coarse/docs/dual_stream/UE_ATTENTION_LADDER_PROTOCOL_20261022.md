# 归一化attention：六方向独立探针合同

用户授权首发L1/L2/L3，GPU1/2/3各一个额外任务；GPU0不追加，剩余全部按槽释放排队，避免OOM。共同L0归一化attention基准及全部常量训练对照都在固定队列，不根据新test排名或val增益决定是否运行。

## 固定范围及方向

归一化32维单头原attention，原8候选（每视角最多2）、原候选排序/去重、6维几何、支持范围及conv_s0单入口保持。六方向各只改变下列机制，不组合，不重复本机已有learnedoffset/localconv/aux/纯几何NN-difference探针。

| 方向 | 唯一机制干预 | 归因限制 |
|---|---|---|
| L0 | 本轮fresh归一化共同基准 | 旧unit仅历史参照，不继承其trained权重 |
| L1 | 真实source_uv在源特征格固定连续双线性读取 | 不是可学习偏移；不改变候选；必须实际patch lattice解析核定 |
| L2 | 视角内attention后按固定视角序拼接5×32→32 | 新投影容量增加，必须匹配常量RGB训练 |
| L3 | 增加可学习scalar logit的严格零value空候选 | 输出LN后保留非空质量，确保拒绝不被LN消除；不加gate网络 |
| L4 | 外观cosine与独立线性几何score相加 | 改变几何query交互，不叫更精确几何；V保留原几何 |
| L5 | 原logit加固定卫星camera-Z近表面软偏置 | tau=1m固定，不用不同UAV相机z比较；不是真可见性 |
| L6 | 原UAV分支不变，解码第一层token→image cross-attention Q/V各rank4 LoRA | 其余SAT保持冻结，correct/constant均加相同LoRA，单独资源门 |

共L0–L6各correct/constant两臂，14独立训练；共同初始化和共同数据流。常量RGB为预训练均值，正确深度/位姿/候选保留，不称彻底无UAV。
L6 LoRA alpha4、dropout0、B零初始化，A独立RNG不改变adapter/data流；只加指定投影，不解冻整decoder。

## L1坐标必须核定

现存anchor保存源整数像素中心(u,v)，不再将整数token索引冒充连续位置。
实际Hiera PatchEmbed conv7s4p3，格i的卷积采样中心为源整数4i。首stage无pool，FPN0只有1×1侧向，无topdown重采样。
因此源u对应连续feature index=u/4；align_corners=False的规范grid为(u/4+.5)/64−1=(u+2)/256−1。
必须由实际PatchEmbed中心delta核与线性ramp测试证明，而非凭经验加半像素。记录边界border策略，不改支持/候选。先按L0对token作无affine LN，再连续读取，避免顺便改变源归一化。
这是特征采样lattice合同，不认证真实世界像素配准，也不修改正在运行的旧协议。

## L3拒绝的数值合同

零value空候选仅影响softmax，不引入空间gate。原输出Linear+LN如果直接作用于带空候选的pool，LN会近似消除整体概率缩小，因此规范化条件非空pool后，将非空概率质量乘回最终残差；output bias也在该缩放内部。
null倾向极强时实际残差趋零，而非仅注意力统计看似拒绝。无支持仍原式恒等。
null权重/有效候选/拒绝率只离线observe采集，不加逐step日志。拒绝donor更成功不等于正确输入净收益，须配套内容训练比较。

## 数据、训练与评价

沿v1最小归档train2142/val459/test459，完整卫星EXP0017E73，coarse64、maxDets250、原scores、非EMA。除L6指定新LoRA外SAT原参数/buffer冻结。
复用已审计s2db_anchor_metadata_v2_fp64缓存，原index/geom与legacy逐值一致，source_uv/satellite_z随排序/过滤传递，不重采样重排候选。缓存来源审计身份记录在existing_cache_audit_identity.json。
训练10epochs、2670updates、batch1accum8、AdamW1e−4 wd.01、clip.5、warmup50 cosine到.1、AMP scale512。
本轮统一E1..10 seed1044000+epoch，seed45共同参数、seed44训练，实际2136图顺序/完整RNG含loadergenerator每轮存档。旧实验随机流不同，不当单因素控制。
固定E10 last，不按test选epoch/结构。SAT原增强不变；不加新loss、源空间增强或额外模态。
全部14臂主test输入依训练content显式correct或constant；correct7臂额外constant/donor（RGBonly）干预。另14主val诊断。
7项同结构真实−常量训练内容比较是主要归因；14对卫星、6方向对L0及固定检查点输入依赖单列。Mask主指标，两种Boundary辅助，500次配对图像bootstrap，单seed/空间相关/多重探索名义区间限制保留。
本轮test为开发评价，正式泛化需独立区域。禁止将六个各自涨点的模型当六份内容有效证明。

## 显存与排队

L1/L2/L3先各GPU1/2/3一个新slot，GPU0禁止；各自门通过且实际余量足够才train。后续常量控制/L0/L4/L5/L6按固定队列，必须等本owner slot释放，禁止再叠第三个本轮任务。
各方向真实最高GT训练正反向、250RoI FP32分块分别测峰值allocated/reserved，L6不得借用冻结decoder资源数。
计入全部其他owner的当前占用，并保留显式准入余量；外owner异步增长无法绝对保证，但不通过减少实例/batch隐瞒OOM。OOM/退出记录并停该任务，不自动重试无限占卡。

## 完成标准

工程门不等于训练完成/有收益。实际14CP E10/2670、完整合同/source/init、每轮样本顺序/冻结状态、全部459图GT/检测scores一致、所有计划内容比较与指标核对完成后才生成最终报告。
L6 adapter拥有LoRA参数、hook只作用指定decoder投影；原SAT state SHA仍应保持原值，新LoRA单列可训练与checkpoint，不可错误标成“全decoder冻结”。

产物根`/data/wangcheng/checkpoint/ue_attention_ladder_20261022`；模型、runner、schedule、collect均独立文件，不修改旧运行任务。
