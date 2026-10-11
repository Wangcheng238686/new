# SURFBEV-B0 v2/no-augmentation 启动实施独立复核（EXP-0006）

VERDICT: PASS

本结论只针对当前待提交实施的接线与数据隔离；**不是已经满足全部正式拉卡手续**，也不是 AP 或 UAV 增益结论。正式启动仍须先将实施、预注册、清单和本报告提交为干净 commit，登记 commit/tag，并在该提交的干净工作树运行 `B0_PREFLIGHT_ONLY=1 B0_EXP_ID=EXP-0006 bash scripts/ue/run_surfacebev_b0_v2_ddp4.sh`。当前审计时工作树为 dirty，因此没有声称 wrapper 完整 preflight 已通过，也未启动训练。

## 审计范围与实测

- `build_surfacebev_b0_split_v2.py --verify` 从 raw payload 重算成功，清单 SHA256 为 `17f49430b15037c8da25c63a1bf7fe4d510cf5ff153263d2feba62c4393441d5`；train/val/test/excluded 为 `694/100/151/255`。train–val、train–test、val–test 每对的共享可见 `global_instance_id`、400 m 双轴视野缓冲冲突和完全相同 RGB SHA256 均为零。该检查覆盖当前数据契约定义的空间/物体重复，不保证跨城市域泛化。
- `UERawPairDataset` 默认 v2，且同时钉住 pair manifest、v1/v2 split 的 SHA 和 schema；逐 scene 核对 accepted 清单与网格规则，不能用同名篡改清单静默换 split。实际构造默认 v2 的 train/val/test 得到 `694/100/151`；显式 `surfbev_b0_split.json` 得到历史 v1 `949/100/151`。旧单卡与四卡 wrapper 也已显式传 v1，避免默认切换破坏历史复现。
- raw B0 的 `__getitem__` 只读取 satellite RGB 和 strict clean-ID PNG，固定双线性/最近邻 512→1024 resize；`load_uav=False` 分支在 UAV 读取前返回。重复读取同一 train 样本，图像、bbox、scene ID 相同。raw loader 直接使用该 dataset，不实例化通用的随机翻转/噪声/擦除/多尺度数据集；传入的非零 flip/vflip/noise/erase 概率会被拒绝。DDP `DistributedSampler` 的 epoch shuffle 只改变顺序，不改变样本内容。
- trainer 将 v2 文件名传给 train、val 和 test loader；checkpoint 快照记录 split 文件名/SHA 和 `augmentation_contract=none`；raw checkpoint 续训或 test-only 时若 split SHA 不同则拒绝。v2 wrapper 显式传 v2、seed 44、4 卡、100 epoch、last、无 early stop/EMA，并阻止额外 trainer 参数、dirty 工作树和已有输出目录。
- `bash -n` 检查新旧三个 wrapper 通过；相关已修改文件的 `git diff --check` 通过。按 wrapper 环境变量计算的模型指纹为注册值 `36549a00a8282b2b9903d8c07c7cc9ce6f6e8fba3e93d58212de1999b1aa3dab`，与旧四卡 wrapper 相同。EXP-0006 已独立占号；预注册固定主/伴随指标、last 口径和单 seed 解释边界。

## 发现及启动前条件

1. **非阻塞、启动硬前置：** 目前所有相关代码与清单尚未提交，`docs/EXP_REGISTRY.md` 的 EXP-0006 行也明确处于启动前。必须在本报告纳入同一干净提交后打 EXP-0006 tag，并补登记 commit/tag 和 preflight echo。不得在当前 dirty 工作树直接启动全量。
2. **解释边界：** v2 把原 v1 train 的 255 场排除，并保留既有 val/test；它修复已量化的同物体和视野重叠，但不是外部城市或真实世界测试。v1 的约 0.8 val AP 不可作为 v2 AP 基线。
3. **随机性边界：** “无数据增强”不等于完全确定的训练优化：sampler 仍随机打乱，模型训练还可能有优化随机性。当前实现符合预注册的“输入无随机增强”。

若提交后 clean preflight、tag 或运行时 echo 任一失败，本 PASS 自动失效并暂停拉卡；修复后复核关键分支。
