# UE 遥测与全量启动前独立审核

日期2026-10-14。独立只读审核者：implementation_training_review，未参与本次编码。
实施前审核基础见 `docs/dual_stream/INDEPENDENT_IMPLEMENTATION_REVIEW.md`。
本次仅新增detached观测与字段字典，没有配置键、前向或训练策略变更。

## 审核者最终回复全文

**VERDICT：PASS（只读实现、字段字典及启动协议审核）。** 未发现新增P1/P2；无需因本次遥测变更阻止正式训练。

- 新统计均脱离梯度，未参与融合输出、损失或裁剪。梯度日志对应真实独立裁剪组，读取时间为AMP unscale后、裁剪前。
- DDP在epoch尾统一字段集合，缺失字段补零sum/count，各rank执行一致collective；未发现新增死锁路径。
- 字典已明确跨rank逐forward有限值均值、零支持的0仍计入分母、训练峰值不含随后验证，以及AMP跳步时旧EMA/scheduler仍按尝试推进。这些说明与源码一致。
- `--early-stopping-patience 0`确实禁用早停；两臂同manifest、seed、80epoch与预算，双流先跑、单流顺序执行，协议合理。`--max-nonfinite-batches 3`会在第三个跨rank确认的非有限loss batch终止，属于失败中止。
- 新smoke源码覆盖跨rank缺失字段、非有限值分母、AMP尝试/跳步计数、融合输出与梯度逐位等价及零支持。实际测试尚在执行，本审核不将其写成已通过。

一处非阻断文档补充：当前暂存区与工作树字典仅写“末组可能只有1个mini-batch”，尚未明确本协议的628样本 → 每rank157个microbatch → 每epoch79次更新尝试、末次全局batch4。建议补入该具体口径。

## 主代理闭环与实际校验

上述具体预算已补入字段字典与首轮训练协议。两rank gloo smoke完成，缺失key跨rank分母、NaN排除、AMP尝试/跳步计数全部PASS。GPU AMP smoke完成：新增遥测启用后融合输出与所有参数梯度逐位等于原计算，覆盖区分母和零支持全部PASS。

初版校验发现覆盖区gate统计使用半精度求和造成统计舍入差；新增supported字段改为detach后FP32归约，再验全部通过，正式forward未改变。已通知审核者。

校验入口：`scripts/smoke/check_ue_telemetry.py`，CPU用2-rank torchrun，GPU用 `--gpu`。
实际校验记录：`/tmp/ue_uav_experiments/20261014_ad_seed42/telemetry_checks.json`。
语法和 `git diff --check` 通过。原架构合同/旧推理重建文件保持冻结；新增字段不改变模型指纹。

## 判读限制

零支持样本的supported均值0仍进入epoch均值；AMP尝试计数不同于成功Adam更新数；训练allocator峰值不同于全机显存。机制遥测不能替代A/D AP100和伴随bbox指标。

## 启动接入修正

首次四卡正式入口在第一批前报错：trainer device为字符串，epoch遥测读取device.type失败。未有参数更新；队列停止，single未启动。修正UEEpochTelemetry入口用torch.device统一输入类型，smoke改用字符串复验；保留失败日志 `/tmp/ue_uav_experiments/20261014_ad_seed42/failed_dual_pid737627/train_EXP-0004_pid737627.log`。原提交/tag不改写，重启提交以追加retry1 tag记录。

### 修正后独立复核全文

**VERDICT：PASS。** 构造器先执行`torch.device(device)`，再访问`.type`及显存API，修复了trainer传入字符串导致的启动错误，也继续支持已有`torch.device`输入。CPU smoke改为字符串，覆盖此次触发条件。

修正未改变forward、梯度、DDP集合通信或统计分母。保留失败日志并使用新commit与retry tag重启，符合实验追踪要求。本次仅完成只读复核，未重新执行测试。

主代理实际两rank字符串device复验PASS，跨rank有限值/缺失字段分母与AMP尝试计数通过。
