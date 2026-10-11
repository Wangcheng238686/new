# 全量实验登记

| EXP-ID | 臂 | seed | 数据与选择口径 | 状态 |
| --- | --- | ---: | --- | --- |
| EXP-0001 | SURFBEV-B0-raw1024 | 44 | 1200-pair strict raw，空间 949/100/151；100 epoch，last | 2026-10-11 15:32 启动，GPU2，训练中 |
| EXP-0002 | SURFBEV-B0-raw1024 | 45 | 同一 split 与协议，独立 seed；100 epoch，last | 2026-10-11 15:32 启动，GPU3，训练中 |

两条记录的 pair manifest SHA256：`f15cd8f5b8d10a69d721546d6cfa1f2cce58af2861d57b1357c5e9ff1e1fe62a`；split SHA256：`894df4561b166d37d06d033a6fc326223062384b402f730591b4c76d41f926d5`。测试集只在协议冻结与 last checkpoint 产生后运行标准推理。

训练代码提交：`be19b5eae5eca5788fec222065a14564e0d59907`，两枚同名 EXP tag 均指向该提交。独立审计见 `docs/reviews/SURFBEV-B0-1024-launch-review.md`。日志与 checkpoint：

| EXP-ID | 日志 | checkpoint 目录 |
| --- | --- | --- |
| EXP-0001 | `/data1/wangcheng/checkpoint/surfbev_b0/launcher_logs/EXP-0001-seed44.log` | `/data1/wangcheng/checkpoint/surfbev_b0/surfacebev_b0_raw1024_s44_EXP-0001` |
| EXP-0002 | `/data1/wangcheng/checkpoint/surfbev_b0/launcher_logs/EXP-0002-seed45.log` | `/data1/wangcheng/checkpoint/surfbev_b0/surfacebev_b0_raw1024_s45_EXP-0002` |
