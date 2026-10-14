# 全量实验登记

| EXP-ID | 臂 | seed | 数据与选择口径 | 状态 |
| --- | --- | ---: | --- | --- |
| EXP-0001 | SURFBEV-B0-raw1024-single | 44 | 1200-pair strict raw，空间 949/100/151；100 epoch，last | 用户要求改双卡，epoch 1 后终止；非完整 B0 结果 |
| EXP-0002 | SURFBEV-B0-raw1024-single | 45 | 同一 split 与协议，独立 seed；100 epoch，last | 用户要求改双卡，epoch 1 后终止；非完整 B0 结果 |
| EXP-0003 | SURFBEV-B0-raw1024-DDP2 | 44 | 曾拟双卡 0,1；每卡 batch 1、累积 4 | 用户改为单 seed 四卡，启动前取消；编号不复用 |
| EXP-0004 | SURFBEV-B0-raw1024-DDP2 | 45 | 曾拟双卡 2,3；每卡 batch 1、累积 4 | 用户改为单 seed 四卡，启动前取消；编号不复用 |
| EXP-0005 | SURFBEV-B0-raw1024-DDP4-v1 | 44 | GPU0–3；每卡 batch 1、累积 2、global batch 8；原拟 100 epoch，last | 2026-10-11 用户要求切换无泄露 v2，epoch 58 后定向终止；v1 存在跨集建筑/视野重叠，不作为独立泛化证据 |
| EXP-0006 | SURFBEV-B0-raw1024-DDP4-v2-noaug | 44 | v2 split SHA `17f49430b15037c8da25c63a1bf7fe4d510cf5ff153263d2feba62c4393441d5`；694/100/151；GPU0–3；无随机增强；100 epoch、last；从预训练 SAM2 全新开始 | 2026-10-12 09:13 完整完成 100 epoch；best segm/mAP 0.7977、last epoch val segm/mAP 0.7952 / bbox 0.7782；test 未开封；现为 DUAL 探针谱系锚点 |
| EXP-0007 | SURFBEV-B0-512native-sealed-DDP4-v2-noaug | 44 | 同 v2 split 与 pair manifest；**原生 512 输入协议**（无上采样、GT 原生 512）；架构=封存 a3sgm 新身份 `ue-whu-a3sgm-512-v1`；GPU0–3；每卡 batch 1、累积 2、global 8；无随机增强；100 epoch、last；从预训练 SAM2 全新开始 | 预注册 `docs/EXP-0007_preregistration.md`（原生 GSD claim 协议，EXP-0006 为算力对照）；2026-10-14 启动 |

EXP-0001 至 EXP-0005 的 pair manifest SHA256：`f15cd8f5b8d10a69d721546d6cfa1f2cce58af2861d57b1357c5e9ff1e1fe62a`；v1 split SHA256：`894df4561b166d37d06d033a6fc326223062384b402f730591b4c76d41f926d5`。EXP-0006 使用同一 pair manifest、v2 split。测试集只在协议冻结与 last checkpoint 产生后运行标准推理。

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
EXP-0005 停止时训练日志最后完整记录为 epoch 58/100；best 和 last checkpoint 保留，
未运行封存 test。其后旧实验结果仅作历史诊断，不充当 v2 单流锚点。

EXP-0006 预注册见 `docs/SURFBEV-B0-V2-20261011_preregistration.md`；代码提交/tag 为
`a954c384f227a62b71c36a8e15d40465f0b9434b` / `EXP-0006`，独立审计见
`docs/reviews/SURFBEV-B0-V2-launch-review.md`。在干净 detached worktree
`/tmp/surfbev-b0-exp-0006` 的 `B0_PREFLIGHT_ONLY=1` 已回显正确 commit、指纹、
v2 split、无增强和超参。2026-10-11 23:22 用户要求直接启动，GPU0–3 仍有其他用户
进程各占约 28GB，本 run 在剩余显存上共租。四 rank 已连通并执行首批前反向；显存余量
需持续关注。日志：`/data1/wangcheng/checkpoint/surfbev_b0/launcher_logs/EXP-0006-seed44-ddp4.log`；
checkpoint 目录：`/data1/wangcheng/checkpoint/surfbev_b0/surfacebev_b0_v2_raw1024_noaug_ddp4_s44_EXP-0006`。
启动时 torchrun PID 349998，四 rank PID 350073–350076；test 仅在 last 完成后按协议推理。
