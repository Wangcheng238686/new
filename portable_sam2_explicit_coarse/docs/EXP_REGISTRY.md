# EXP-ID 登记表（全局唯一，只发不复用；发放时点=进入全量训练）

| EXP-ID | 臂/内容 | commit+tag | 启动时间 | 状态 |
|---|---|---|---|---|
| EXP-0001 | a3-K32 matrix300 从头（UDPR K敏感性首行，A3_NUM_POINTS=32，v1损失） | 见 tag exp/EXP-0001 | 2026-09-15 | done: 平台0.6593，K64−0.0096，掉出平坦带 |
| EXP-0003 | a3-K128 matrix300 从头（决策关键行：argmax在64还是128/覆盖饥饿判别） | 5e26612 | 2026-09-15 | queued：等 b_box（pid 1138350，~07:30释放）后自动拉起 |
| EXP-0002 | a3-K64 matrix300 从头 4×1×2 形态对齐 a3r（machine2；同机同形态配对 + 1×8vs4×2 拓扑A/B，RUN_FORM_SUFFIX=form4x2） | 见 tag exp/EXP-0002 | 2026-09-15 | running |
