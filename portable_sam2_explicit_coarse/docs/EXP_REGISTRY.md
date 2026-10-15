# EXP-ID 登记表（全局唯一，只发不复用；发放时点=进入全量训练）

| EXP-ID | 臂/内容 | commit+tag | 启动时间 | 状态 |
|---|---|---|---|---|
| EXP-0001 | a3-K32 matrix300 从头（UDPR K敏感性首行，A3_NUM_POINTS=32，v1损失） | 见 tag exp/EXP-0001 | 2026-09-15 | done: 平台0.6593，K64−0.0096，掉出平坦带 |
| EXP-0003 | a3-K128 matrix300 从头（决策关键行：argmax在64还是128/覆盖饥饿判别） | 5e26612 | 2026-09-15 | queued：等 b_box（pid 1138350，~07:30释放）后自动拉起 |
| EXP-0002 | a3-K64 matrix300 从头 4×1×2 形态对齐 a3r（machine2；同机同形态配对 + 1×8vs4×2 拓扑A/B，RUN_FORM_SUFFIX=form4x2） | 见 tag exp/EXP-0002 | 2026-09-15 | running |

| EXP-0004 | UE RGB-D dual，seed42，4×1×2，80epoch，generic联合初始化 | 见 tag exp/EXP-0004 | 2026-10-14 | registered：首轮开发训练；状态见训练协议的run_state.json |
| EXP-0005 | UE satellite single，EXP-0004同清单/预算配对，从generic独立初始化 | 见 tag exp/EXP-0005 | 2026-10-14 | queued：同进程队列，dual成功后启动 |

| EXP-0006 | UE dual修复标签清单，627/153，旧EXP-0004首轮因1016未映射ID失败 | 见 tag exp/EXP-0006 | 2026-10-14 | registered：新清单从generic重启 |
| EXP-0007 | UE single与EXP-0006同清单预算配对 | 见 tag exp/EXP-0007 | 2026-10-14 | queued：dual成功完成后启动；旧EXP-0005未启动 |

## 2026-10-15 完成核验

- EXP-0006：80轮/Adam6240完成，best-segm E72 .584958，最终E80 .5819；严格完整验证与干预已归档。
- EXP-0007：80轮/Adam6240产物完成，best-segm E66 .583000，最终E80 .5808；TCPStore监视告警及queue状态过期，正常进程退出码未确认，不伪造returncode0。
- EXP-0004：失败于1016坏标签；EXP-0005未启动，被新清单配对替代。历史登记保留。
- 结论：主分割+0.196个百分点，伴随bbox−0.151；探索性区间跨0，暂无可靠净收益。见 [完整分析](dual_stream/UE_AD_RESULTS_20261015.md)。
