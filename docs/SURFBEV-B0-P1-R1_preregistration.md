# SURFBEV-B0/P1-R1 预注册

## 路由与不可变数据

新 raw 数据基座加模块路线：先 B0 基座，再 P1 冻结信息门；P1 通过前禁止 H1 训练。
唯一输入清单为 `accepted_pairs.jsonl`，SHA256 为
`f15cd8f5b8d10a69d721546d6cfa1f2cce58af2861d57b1357c5e9ff1e1fe62a`。
`surfbev_b0_split.json` 固定 train/val/test 为 949/100/151 scenes（63/9/9 cell）；test
是 gx=2..4、gy=-7..-5，val 是 gx=-7..-5、gy=-3..-1。标签是 strict visible roof-top
raster，绝不从 `instances.json` 补全不可见实例。

## B0

卫星单流 RSPrompter/SAM2、1024x1024、100 epoch、seed 44/45。512x512 原始卫星 RGB
双线性上采样，strict instance-ID 以最近邻上采样后提取 mask/bbox；UAV 深度和相机仍按原始
512x512 坐标解释。每 seed 的 last checkpoint
是唯一口径；禁用早停并保存 epoch100 的 `last_model_epoch100.pth`。val 仅监控，test 在协议冻结后各执行一次。主指标 test segm AP；伴随 bbox AP、
segm/bbox AP75、面积三分位 segm AP、预测数与 GT 数。B0 不读取 UAV 文件。
面积三分位仅用于描述 test，按 test strict mask 的实际像素面积计算 1/3 与 2/3 分位边界，
在 COCOeval 的每个固定面积段分别汇总 AP；不用于选择权重或阈值。

### B0 双卡执行修订（2026-10-11，启动 EXP-0003/0004 前）

用户要求停止两条已运行的单卡任务，改为每条 seed 使用双卡。EXP-0001/0002 在 epoch 1
结束后按进程组停止，保留日志与 checkpoint，但未完成预注册的 100 epoch，不能作为 B0
test 证据。新 EXP-0003/0004 从同一预训练 SAM2 权重**重新开始**，不载入旧 run 的
epoch-1 checkpoint；模型、数据、val/test 指标与 last 选择口径均不变。此项为基座训练
动力学/并行执行变更，不是新的 UAV 模块增益实验；P1 仍须等待新 B0 seed 44 last。

两条新 run 分别固定 GPU0,1 与 GPU2,3，DDP world size=2，每卡 batch=1、梯度累积=4，
global effective batch=8，学习率仍为 5e-4、AdamW weight decay=0.05、100 optimizer-step
warmup 后 cosine、100 epoch、每轮完整 val、关闭早停与 EMA。当前 DistributedSampler
`drop_last=True` 对 949 个 train scene 每轮舍去 1 个，使每卡 474 个 mini-batch，仍为
每轮 119 次优化步；末个累积组每卡 2 个样本、global batch 4。这与单卡每轮 949 个
mini-batch 不是逐样本严格等价，故新 EXP-ID 独立登记。实验主指标、
伴随指标及 test 封存规则仍按上文 B0 四要素执行。启动前须通过双卡 loss/梯度同步、
采样数量、val 仅 rank0、契约 echo 及显存烟测；训练中监测 non-finite、两 rank 存活、
GPU 显存和学习率步数。训练遥测不能替代 test AP 结论。

### B0 四卡执行修订（2026-10-11，启动 EXP-0005 前）

用户进一步决定开发阶段只保留一个 seed，使用全部四张 GPU。因此双卡 EXP-0003/0004
在正式启动前取消，编号保留而不复用；两卡短烟测只证明 DDP 接线，不计正式 run。
唯一新 run 为 EXP-0005：seed 44，GPU0–3，DDP world size=4，每卡 batch=1、梯度累积=2，
典型 global effective batch=8。DistributedSampler `drop_last=True` 从 949 train scenes
每轮舍去 1 个，每卡 237 个 mini-batch、119 个优化步；末组每卡 1 个样本，global
batch=4。仍从预训练 SAM2 权重重新开始，原单卡 epoch-1 checkpoint 不续训。其余
100 epoch、AdamW/学习率与调度、val/test 隔离、last 唯一选择口径及 B0 指标四要素不变。
单 seed 只能建立开发锚点，不作为跨 seed 稳健性或统计显著性的证据；后续若要主张
UAV 增益，必须与同一 B0 权重/数据/推理协议作配对对照，并补独立重复或其他不确定性
评估。正式启动前需四 rank 短烟测、模型指纹与数据哈希回归、独立审计 PASS。

## P1

固定 B0 seed44 last checkpoint；卫星网络冻结，独立预训练 UAV encoder eval/no-grad。DeviceZ
按 `z=10/DeviceZ` 解码、C2W 回投、satellite W2C 投影及 satellite z-buffer。每实际 P3 cell
输出 12 个训练集 0.5/99.5-percentile 固定高度 bin 的 UAV descriptor、height mean/std、view
count、z-buffer residual 和 missing mask。只用 train 拟合等容量线性读出，val 一次固定 L2 与
F1 threshold，test 一次封存。三臂为 SAT、SAT+correct、SAT+90-degree-XY-sham。

主指标是每 scene roof-boundary F1；伴随是 boundary-error AUROC 与实例 mask-IoU 预测相关。
每个比较使用 10,000-draw scene-paired percentile bootstrap、seed 20261012。P1_PASS 要求：
被动 harvest parity；correct/sham 点数和容量相同；主及两项伴随的 correct-SAT、correct-sham
CI 下界全部 >0；coverage/z-buffer/residual 无契约异常。否则 P1_FAIL 并关闭接线。
