# SURFBEV-B0 1024 正式启动独立复核

VERDICT: PASS

复核对象：训练代码提交 `be19b5eae5eca5788fec222065a14564e0d59907`；干净 detached worktree `/tmp/surfbev-b0-exp-0001-0002`。`EXP-0001`、`EXP-0002` 两个 tag 均指向该提交。两次 `B0_PREFLIGHT_ONLY=1` 实跑退出码均为 0，分别锁定 seed 44/GPU2、seed 45/GPU3；输出目录互斥且启动前均不存在。

独立审计核对了四臂 config：旧三臂指纹与 `SURFBEV-B0-1024-harness.md` 一致，B0 与 WHU1024 的 model 字典逐键相等。raw reader、loader、test-only 均按 1024 契约运行；B0 不读取 UAV。wrapper 拒绝额外参数，强制 clean worktree、pair/split 哈希、模型指纹、`SAVE_LAST_MODEL=1`、100 epoch、early stopping 关闭。preflight echo 的 commit、EXP-ID、seed、GPU、超参、数据协议和输出目录一致。

实施方 GPU2 单样本 loss/backward 烟测峰值 reserved 10.81 GiB；审计方未重复该动态测试。preflight 在沙箱内出现 CUDA init warning，因其只作静态门，不代表训练 GPU 验证；正式启动后须核对 CUDA 实际迭代与日志，无迭代不得视为启动成功。

审计边界：PASS 仅允许按预注册协议启动 B0，不证明训练收敛或 AP 增益。test 必须等 last checkpoint 产生后一次封存推理。
