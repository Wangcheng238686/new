# SURFBEV-B0 1024 四卡启动 harness（EXP-0005）

目标臂为与 `EXP-0001/0002` 同一卫星-only B0 model、同一 1,200-pair 空间切分，
但改成单 seed 44、GPU0–3 的四卡 DDP。原单卡两 run 在 epoch 1 后按用户指令停止；
它们的 checkpoint 不续训。曾拟的 EXP-0003/0004 双卡正式 run 未启动。

模型 config 未改。沿用 `SURFBEV-B0-1024-harness.md` 的四臂解析检查：B0 与
WHU1024 model 字典相等，指纹均为
`36549a00a8282b2b9903d8c07c7cc9ce6f6e8fba3e93d58212de1999b1aa3dab`；
旧 A1 与 dual 指纹分别为
`6f6f10dedf0de9347bb8ec22fadacdf461c5440beb10bdc7a6337ebb043f5538` 和
`7e3ce8fa3466ac6936b8a6f270ac7be022ccf4605bc0e9e79351f43956816f90`。
本轮 trainer 仅修 DDP 训练调用：经 `model(..., mode="loss")` 进入 DDP forward，
而非跳过 reducer 调 `model.module.loss()`；单卡 loss 路径不变。

短烟测（非正式全量，`/tmp/surfbev_b0_ddp4_smoke_20261011_a.log`）：
四 rank 均加入 Gloo 控制组；NCCL 训练以每卡 batch 1、accum 2 完成每 rank 3 个
mini-batch、2 次优化步；rank0-only val 处理 2 个样本；正常保存
`/tmp/surfbev_b0_ddp4_smoke_20261011_a/last_model_epoch1.pth`，launcher exit 0，
四 GPU 显存回落。DDP 的 grad-stride warning 只提示可能影响性能，未造成错误。
烟测的两样本 AP 无统计意义，不得当作 B0 成绩。

正式完整训练的 `DistributedSampler(drop_last=True)` 每轮在 949 train scenes 中
舍去 1 个，每 rank 237 个 mini-batch，梯度累积 2，119 optimizer steps/epoch；
通常 global effective batch 8，末组为 4。val 由 rank0 在完整 100 scene 上评估，
test 151 scene 在 last checkpoint 完成后才封存推理。遥测中 rank0 loss 是 rank0
本地均值，不等于四 rank 全局训练 loss；是否提升以标准 val/test 而非该损失判定。
