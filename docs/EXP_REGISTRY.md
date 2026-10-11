# 全量实验登记

| EXP-ID | 臂 | seed | 数据与选择口径 | 状态 |
| --- | --- | ---: | --- | --- |
| EXP-0001 | SURFBEV-B0-raw1024-single | 44 | 1200-pair strict raw，空间 949/100/151；100 epoch，last | 用户要求改双卡，epoch 1 后终止；非完整 B0 结果 |
| EXP-0002 | SURFBEV-B0-raw1024-single | 45 | 同一 split 与协议，独立 seed；100 epoch，last | 用户要求改双卡，epoch 1 后终止；非完整 B0 结果 |
| EXP-0003 | SURFBEV-B0-raw1024-DDP2 | 44 | 曾拟双卡 0,1；每卡 batch 1、累积 4 | 用户改为单 seed 四卡，启动前取消；编号不复用 |
| EXP-0004 | SURFBEV-B0-raw1024-DDP2 | 45 | 曾拟双卡 2,3；每卡 batch 1、累积 4 | 用户改为单 seed 四卡，启动前取消；编号不复用 |
| EXP-0005 | SURFBEV-B0-raw1024-DDP4 | 44 | GPU0–3；每卡 batch 1、累积 2、global batch 8；100 epoch，last；从头训练 | 2026-10-11 16:03 启动，四卡训练中 |

两条记录的 pair manifest SHA256：`f15cd8f5b8d10a69d721546d6cfa1f2cce58af2861d57b1357c5e9ff1e1fe62a`；split SHA256：`894df4561b166d37d06d033a6fc326223062384b402f730591b4c76d41f926d5`。测试集只在协议冻结与 last checkpoint 产生后运行标准推理。

训练代码提交：`be19b5eae5eca5788fec222065a14564e0d59907`，两枚同名 EXP tag 均指向该提交。独立审计见 `docs/reviews/SURFBEV-B0-1024-launch-review.md`。日志与 checkpoint：

| EXP-ID | 日志 | checkpoint 目录 |
| --- | --- | --- |
| EXP-0001 | `/data1/wangcheng/checkpoint/surfbev_b0/launcher_logs/EXP-0001-seed44.log` | `/data1/wangcheng/checkpoint/surfbev_b0/surfacebev_b0_raw1024_s44_EXP-0001` |
| EXP-0002 | `/data1/wangcheng/checkpoint/surfbev_b0/launcher_logs/EXP-0002-seed45.log` | `/data1/wangcheng/checkpoint/surfbev_b0/surfacebev_b0_raw1024_s45_EXP-0002` |

EXP-0001/0002 的训练代码提交与 tag 均为 `be19b5e`，保留 epoch-1 `last_checkpoint.pth`，不删除、不用于新 DDP run 续训，也不作为 test 结果。EXP-0003/0004 未启动、无权重。

EXP-0005 训练代码提交/tag：`1908085e4593f21b57c685654cd7ee607bda8bb4` / `EXP-0005`；
独立审计：`docs/reviews/SURFBEV-B0-DDP4-launch-review.md`；
日志：`/data1/wangcheng/checkpoint/surfbev_b0/launcher_logs/EXP-0005-seed44-ddp4.log`；
checkpoint 目录：`/data1/wangcheng/checkpoint/surfbev_b0/surfacebev_b0_raw1024_ddp4_s44_EXP-0005`。
