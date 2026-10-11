# SURFBEV-B0-V2-20261011：无泄露划分上的卫星单流锚点

## 路由与准入

本 run 是**基座/训练动力学型**重建：换成无跨集物体与视野重叠的 v2 划分后，
旧 EXP-0005 权重不能热启动新 B0，且冻结旧模型不能验证新划分下的训练动力学；
数据隔离门通过后直接全量。从公开 SAM2 Base+ 预训练权重初始化，不加载任何 UE 任务权重。
EXP-ID 为 `EXP-0006`；用户要求开发阶段只跑一个 seed，故只建 seed 44 锚点。

## 不可变数据与增强契约

- raw accepted-pair manifest SHA `f15cd8f5b8d10a69d721546d6cfa1f2cce58af2861d57b1357c5e9ff1e1fe62a`；
  v2 split SHA `17f49430b15037c8da25c63a1bf7fe4d510cf5ff153263d2feba62c4393441d5`。
  train/val/test/excluded = 694/100/151/255；原始严格可见屋顶实例标签。
- 数据增强 **none**：不做随机水平/垂直翻转、噪声、擦除、颜色扰动、mixup、mosaic、
  随机裁剪/尺度或随机 UAV 视角选择。仅原生 512² RGB 双线性固定 resize 到 1024²，
  clean ID 最近邻固定 resize；固定归一化与必要固定尺寸 padding 不计随机增强。
- B0 严禁 UAV 输入；训练与验证加载器均以 v2 manifest 选择场景。

## 四要素与训练协议

- 主指标：v2 封存 test 上 `segm AP@[.50:.95]`。伴随：bbox AP、segm/bbox AP75、
  COCO small/medium/large 分层 segm AP、预测数与 GT 实例数；val 同项仅作过程监测。
- 选择口径：seed 44、100 epoch 的 **last** checkpoint，标准 test 推理一次；
  `best` 只存盘，不凭 val/test 峰值替换 last。旧 v1 AP 不作同分布对照。
- 显著性：单 seed 开发锚点只报点估计，不声称跨 seed 稳健性；后续 UAV arm 应与此
  last 权重、同 v2 test 图逐场 paired bootstrap（10,000 draws，seed 20261012）比较，
  正式结论仍需独立训练重复。B0 本身不设置 AP 越过旧 v1 的门。
- 数据/接线门：manifest 与 raw payload 重算一致；train/val/test 任意两组共享可见
  global ID、400m 双轴视野缓冲冲突、RGB SHA 重复均为 0；v2/旧 v1 模型指纹不变、
  差异仅数据划分与明确的无增强契约；无 UAV 路径读取；任一失败不得启动。

训练配置：RSPrompter/SAM2 单流 1024²；GPU0–3 DDP，每卡 batch=1、累积=2，
常规 global effective batch=8；AdamW lr=5e-4、wd=.05、warmup 100 optimizer steps、
cosine；100 epoch、val 每轮一次、无早停、EMA 关闭。694 train 场的
DistributedSampler `drop_last=True` 每轮舍 2 场，每卡 173 mini-batch，87 optimizer steps
（末组 global batch=4）。随机打乱仅影响采样顺序，不是输入数据增强。

## 遥测与解释边界

记录 commit/tag、数据和 split SHA、模型指纹、seed/GPU/world size、所有增强概率、
每轮 sampler 数、学习率/损失、val segm+bbox/AP75 和 GPU/NaN 状态。
这些过程遥测只证明任务按契约运行，不能替代 last 标准 test 推理；单 seed B0 仅为
后续冻结 P1 的锚点，不是 UAV 增益证据。
