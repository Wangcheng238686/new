# EXP-0001 独立复核记录（a3 K扫参 wrapper，两轮）

## 第一轮（2026-09-15，agent_afdf0c29）
VERDICT: FAIL。发现：CRITICAL——wrapper GPU 守卫逻辑反转（awk END{print c+0} 恒输出，
[ -n ] 恒真，任何机器状态下拒绝启动）；MINOR——汇总表不兼容锚点旧命名、seed45 目录
误并作 K=64 行、K=0 未拒绝。其余全部通过：默认不变性（a3 臂仅两行变更，K64 默认
与历史逐字节一致）、K 变体单变量（config diff 仅 num_points）、matrix300 从头接线
（无 INIT/TRAIN_ONLY 泄漏）、锚点目录保护（K=64 跳过）、dclip 臂无扰、DRY_RUN 双态
验证。

## 第二轮（同 agent 复核修复）
VERDICT: PASS。守卫双分支实测（空闲→放行；模拟占用→拒绝）；K 校验拒 0/非数；
汇总表锚点行正确填充（best=best_model_epoch45 0.6930，last 0.6696/0.7190），
seed45 排除，单 K=64 行。无回归。非阻断备注 3 条（nvidia-smi 缺失时 fail-open、
前导零 K、脚注平台值与 last 推理值口径差）已记录不处理。
