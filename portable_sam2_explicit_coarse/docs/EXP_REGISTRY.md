# EXP-ID 登记表（全局唯一，只发不复用；发放时点=进入全量训练）

| EXP-ID | 臂/内容 | commit+tag | 启动时间 | 状态 |
|---|---|---|---|---|
| EXP-0001 | a3-K32 matrix300 从头（UDPR K敏感性首行，A3_NUM_POINTS=32，v1损失） | 见 tag exp/EXP-0001 | 2026-09-15 | done: 平台0.6593，K64−0.0096，掉出平坦带 |
| EXP-0003 | a3-K128 matrix300 从头（决策关键行：argmax在64还是128/覆盖饥饿判别） | 5e26612 | 2026-09-15 | queued：等 b_box（pid 1138350，~07:30释放）后自动拉起 |
| EXP-0002 | a3-K64 matrix300 从头 4×1×2 形态对齐 a3r（machine2；同机同形态配对 + 1×8vs4×2 拓扑A/B，RUN_FORM_SUFFIX=form4x2） | 见 tag exp/EXP-0002 | 2026-09-15 | running |
| EXP-0046 | WHU-1024 UBR 协议重跑：UDPR→native UBR（S 形零证据 6625 参数，seed2080），fast150 冻结协议（fp32/4×1×2/seed44/A系增强/best-segm）；config whu1024_ubr_fast150_20261114.py + --native-ubr；预登记 docs/evidence/whu_nwpu_ubr_rerun_protocol_20261114.md；探针门全过（指纹 parity 5d5f78ec/模块一致性/smoke/round-trip/审计 PASS） | 见 tag exp/EXP-0046 | 2026-11-15 | stopped@E22(用户指示)：E22 val segm 0.7001 ≥ 旧UDPR论文run ep98 val best 0.6948 达标后停止；best(E22) test 推理进行中；由 EXP-0048 接棒 |
| EXP-0047 | NWPU VHR-10 UBR 协议重跑：同模型差分，matrix300 冻结协议（AMP/分片val/PE 0.1/decoupled-aux-clip(native_refiner.)/best-composite）；config vhr10_ubr_matrix300_20261114.py；full_image 坐标契约（安装器放宽记录） | 见 tag exp/EXP-0047 | 2026-11-15 | queued：EXP-0046 完成后 4 卡串行发射 |
| EXP-0048 | WHU-1024 UBR fast60 短调度家族：从头重训（非 resume），60ep 余弦全程退火 + 分卡验证；模型差分与 EXP-0046 相同（UDPR关停+native UBR）；wrapper whu1024_ubr_fast60_20261115.sh；里程碑决策见 evidence 预登记文档 | 见 tag exp/EXP-0048 | 2026-11-15 | pending：E22 test 推理完成后发射 |
