# SURFBEV-B0 四卡启动独立复核（EXP-0005）

VERDICT: PASS

独立审计对象：clean worktree `/tmp/surfbev-b0-exp-0005`，HEAD
`1908085e4593f21b57c685654cd7ee607bda8bb4`；`EXP-0005` tag 指向该提交，
`git status --porcelain` 为空。wrapper `bash -n` 通过，审计方实跑
`B0_PREFLIGHT_ONLY=1` 退出码 0；回显 seed 44、GPU0–3、world size 4、每卡 batch 1、
累积 2、典型 global batch 8、100 epoch、last 保存、模型指纹与 pair/split 两哈希。
正式输出目录启动前不存在，与旧单卡工件隔离，且不使用 resume。

DDP loss 经 wrapper forward 进入 reducer，不再绕过梯度同步；
`DistributedSampler(drop_last=True)` 对 949 train scenes 每轮舍 1 个，每 rank 237
mini-batch、119 次优化步，末组 global batch 4。完整 val 仅 rank0 执行，其余 rank
经 Gloo 控制广播等待；仅 rank0 保存 checkpoint。模型 config 不变，四臂指纹见
`SURFBEV-B0-DDP4-harness.md`。

独立审计核对四卡短烟测日志 `/tmp/surfbev_b0_ddp4_smoke_20261011_a.log`：
4 rank 加入 Gloo，3 mini-batch/rank、2 optimizer steps，rank0 val 2 scenes，
`last_model_epoch1.pth` 保存（452,597,991 bytes），无 traceback，正常退出。
grad-stride warning 仅为潜在性能风险；训练 loss 日志只代表 rank0 本地均值；
两张验证图像的 smoke AP 不计结论。单 seed 不证明跨 seed 稳健性。

本 PASS 仅准许按 EXP-0005 协议启动完整 B0；不能证明训练收敛或 UAV 增益。
