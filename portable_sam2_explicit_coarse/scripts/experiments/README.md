# 全量预算实验入口

用户约定（2026-10-28）：后续所有全量预算任务都必须提交一个可一键启动的 wrapper。不得只保存临时 shell 命令。

每个 wrapper 必须提供：

- 版本化协议、共同初始化、数据与样本顺序身份；
- 独立运行源码、不同臂目录/配置/检查点/日志/锁；
- `plan / queue / status / train / resume / eval`，并发准入与拒绝重复启动；
- 同臂严格续训、模型/optimizer/scaler/五类RNG恢复；
- 明确验证选权和测试规则，禁止按测试结果选权或修改预算；
- 源码提交/输入SHA/启动命令/进程身份记录和完成审计。

## Native A/B/C 100轮

```bash
./scripts/experiments/ue_native_full100.sh plan
./scripts/experiments/ue_native_full100.sh queue
./scripts/experiments/ue_native_full100.sh status
```

`queue`一次登记三臂，在当前归因分析全部结束后自动完成新协议四卡恢复门，再按显存共租训练。每个实验四卡，不降单卡。重复queue对活owner幂等；死owner需显式 `recover`，它检查旧launcher和rank身份、完成产物以及同臂last，不盲目重复训练。

```bash
./scripts/experiments/ue_native_full100.sh recover
# 仅在owner不活跃、前置分析与恢复门通过、四卡内存足够时允许单臂操作：
./scripts/experiments/ue_native_full100.sh train --arm A
./scripts/experiments/ue_native_full100.sh resume --arm B
./scripts/experiments/ue_native_full100.sh eval --arm C
```

默认root：`/data/wangcheng/checkpoint/ue_g_ubr_native_full100_20261028`。可用`--root`建立另一份独立研究，绝不覆盖已有合同。`UE_EXPERIMENT_PYTHON`只控制前端解释器，实际子任务解释器路径记录在控制器。队列源码固定为首次prepare时的已提交Git工作树`root/runtime_source`；后续主开发目录变化不会污染在训任务。不要在runtime_source编辑代码。

每臂100轮26700更新，fresh共同基础初始化，新的固定100轮共同顺序（seed44+epoch），每轮完整459val；验证MaskAP严格更高才更新best，平局保较早轮。三训练完成后各自best在完整459test评价。历史10轮仅参考，不作本轮初始化。

故障的半成品eval或恢复门不会自动删除/覆盖；先查看日志，显式归档该失败目录后使用recover。同臂checkpoint合同不匹配直接拒绝。
