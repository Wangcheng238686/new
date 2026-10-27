# G＋UBR首轮四卡短比较

2026-10-27。真实检测器四卡门attempt3已过，先执行独立恢复门，再启动本研究。全量100轮不由本轮自动发射。

## 比较与归因

|臂|上游G|尾部|所回答问题|
|---|---|---|---|
|G_no_tail|正确RGB-D、现有多候选G、实例头均保留|无UDPR，无UBR loss；未使用的UBR参数仍共享初始化|尾部替换的参考|
|G_UBR_geometry|相同|UBR读取正式geometry5|空间尾部＋几何是否有用|
|G_UBR_zero_geometry|相同|仅UBR端口geometry5全零，其他状态/feature不动|真实尾部几何相对匹配容量的效果|

两个UBR臂的几何差异只限尾部，上游几何和正确RGB始终保留，不能用该对比宣称全部UAV RGB/深度必要，也不能将其称纯卫星基线。最后仍需单/双流同预算比较及内容干预。首轮无新boundary loss、不解冻额外模块、不取消实例头、不改候选保护prefix或评分。

## 预算与评价

使用当前通用foundation初始化及共同新头，不加载E73或trained G。原G优化合同：AdamW 5e-4、wd .05、warmup100、clip.5，保持100轮cosine日程的前10轮（非新的10轮cosine）；AMP/EMAoff，四rank batch1×accum2＝global8，每轮既定2136训练图、267次更新，共2670次更新。loader次序逐值核原记录；seed44、每rank generator44+rank。不是旧冻结探针1e-4合同。

固定E10 last，不据test调预算、架构或选权。E1/E10各算完整459val；三个E10均结束后各算完整459test，描述开发暴露测试集。在线报告bbox/segm AP250，记录完整实例预测，Boundary AP和配对误差留离线统一分析，不向训练逐步日志增字段。短预算只用于机制和优化诊断，不以阴性否定全预算方向，不以单种子涨点认定稳定净收益。

每epoch原子覆盖last（phase=after_epoch），另保留E1/E10；恢复严格核model、optimizer、scaler、各rank Python/NumPy/Torch/CUDA/DataLoader generator、完整source/input/order合同。恢复工程门使用真实完整模型状态与显式synthetic AdamW moment，验证序列化和五种next-five RNG精确恢复，不伪称真实下次更新轨迹已经复现。

## 工程与来源

- `rsprompter/uav/ubr_detector_integration.py`：构造前禁tail、原3tuple capture、已选pool、全context及可微active、source段/chunk32一致推理。
- `tests/test_ubr_detector_integration.py`：控制器、来源分段、checkpoint等单元行为。
- `scripts/probes/ue_ubr_runtime_20261027.py`：通用初始严格过滤旧tail键、实际G/payload/旧候选策略、单次特征提取和所有loss在forward。
- `scripts/probes/ue_ubr_detector_gate_20261027.py`：实际四卡零初始化、反传、AMP重试、空rank、64自然匹配RoI、250推理压力。
- `configs/ablations/ue_g_ubr_short_protocol_20261027.json`：本研究固定三臂合同。
- `scripts/probes/ue_ubr_short_run_20261027.py`：四卡训练/评价/恢复工程门，所有loaded本地数值源和config必须匹配已提交Git blob；模型构造后建立实际import闭包，含repo内SAM2 vendor。
- `scripts/probes/ue_ubr_short_owner_20261027.py`：四卡成组、最多两组共租，三训练结束再串行test，不抢占外部任务，失败停止新准入。

每卡资源依据真实峰值6530MiB＋1024余量＋512 context＝8066MiB预约，另留2048MiB；allocator cap9216MiB不是预约量。按四卡共同headroom准入，扣除启动进程尚未实际驻留的预约，不凭单卡空闲启动。每任务独立torchrun rendezvous、source/argv/PID/starttick/log/rc归档，source/receipt/order改动停止新准入。

产物计划`/data/wangcheng/checkpoint/ue_g_ubr_short_20261027`；具体启动与完成状态以owner_state/rank_launches/progress/receipt为准，不把计划写成已经跑完。

## 实际启动记录

源码固定提交`5b719eee64786e0c0e4e88f2d6b4815eed7781b5`。独立四卡恢复门`/data/wangcheng/checkpoint/ue_ubr_resume_gate_20261027/resume_gate/G_UBR_geometry/complete.json`全部通过：模型、synthetic AdamW moment、scaler与五类RNG next-five exact。

owner PID1407943已实际启动两组四卡：G_UBR_geometry launcher1408127（rank1408141/43/48/50），G_no_tail launcher1408128（rank1408137/38/42/44），zero_geometry pending按资源接续。实际study_contract绑定94份本地数值源及上述提交，各rank身份独立核查通过。初始no_tail已完成E1 update150，loss/clip前范数有限、AMP retry0；UBR当时在首25日志窗口，进程活跃无错误，不能将此状态写成已完成训练或已有AP收益。最新状态始终以owner_state/progress为准。

当前UBR整池上下文＋逐实例完整反传明显比参考耗时，应将训练/推理成本与AP一起报告；本轮不在运行中改批次、分辨率或结构来优化速度。每卡实际两组约13GiB，留有余量；第三等待是本轮2组上限及资源合同，不是单卡调度。

## 用户要求立即三组共租（2026-10-27）

用户明确要求显存足够时立即启动等待的zero_geometry。新增`scripts/probes/ue_ubr_short_adopt_owner_20261027.py`独立控制器：严格核旧owner与8rank身份、归档原状态，只终止旧owner（torchrun/rank进程不接收信号），取得同一owner锁后接管存活任务并启动第三个四卡组。数值源/预算/初始化/allocator cap/固定E10评价不变，训练仍绑定5b719ee。

准入改为每组实测最坏6530MiB＋512 context＝7042MiB，保留每卡全局2048MiB余量；移除旧每组额外1024MiB重复缓冲及两组上限，最多三组。实际四卡准入余量约7855–7966MiB，足以放入7042MiB。源与order检查、未兑现预约扣除、失败停止新发射、完整test串行规则继续保留。接管归档及新controller SHA/commit见产物`adoption_three_groups_20261027/policy.json`，实际状态看owner_state。
