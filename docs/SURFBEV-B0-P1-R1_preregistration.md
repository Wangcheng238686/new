# SURFBEV-B0/P1-R1 预注册

## 路由与不可变数据

新 raw 数据基座加模块路线：先 B0 基座，再 P1 冻结信息门；P1 通过前禁止 H1 训练。
唯一输入清单为 `accepted_pairs.jsonl`，SHA256 为
`f15cd8f5b8d10a69d721546d6cfa1f2cce58af2861d57b1357c5e9ff1e1fe62a`。
`surfbev_b0_split.json` 固定 train/val/test 为 949/100/151 scenes（63/9/9 cell）；test
是 gx=2..4、gy=-7..-5，val 是 gx=-7..-5、gy=-3..-1。标签是 strict visible roof-top
raster，绝不从 `instances.json` 补全不可见实例。

## B0

卫星单流 RSPrompter/SAM2、512x512、100 epoch、seed 44/45。每 seed 的 last checkpoint
是唯一口径；val 仅监控，test 在协议冻结后各执行一次。主指标 test segm AP；伴随 bbox AP、
segm/bbox AP75、面积三分位 segm AP、预测数与 GT 数。B0 不读取 UAV 文件。

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
