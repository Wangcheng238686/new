# 全量实验登记

| EXP-ID | 臂 | seed | 数据与选择口径 | 状态 |
| --- | --- | ---: | --- | --- |
| EXP-0001 | SURFBEV-B0-raw1024-single | 44 | 1200-pair strict raw，空间 949/100/151；100 epoch，last | 用户要求改双卡，epoch 1 后终止；非完整 B0 结果 |
| EXP-0002 | SURFBEV-B0-raw1024-single | 45 | 同一 split 与协议，独立 seed；100 epoch，last | 用户要求改双卡，epoch 1 后终止；非完整 B0 结果 |
| EXP-0003 | SURFBEV-B0-raw1024-DDP2 | 44 | 曾拟双卡 0,1；每卡 batch 1、累积 4 | 用户改为单 seed 四卡，启动前取消；编号不复用 |
| EXP-0004 | SURFBEV-B0-raw1024-DDP2 | 45 | 曾拟双卡 2,3；每卡 batch 1、累积 4 | 用户改为单 seed 四卡，启动前取消；编号不复用 |
| EXP-0005 | SURFBEV-B0-raw1024-DDP4-v1 | 44 | GPU0–3；每卡 batch 1、累积 2、global batch 8；原拟 100 epoch，last | 2026-10-11 用户要求切换无泄露 v2，epoch 58 后定向终止；v1 存在跨集建筑/视野重叠，不作为独立泛化证据 |
| EXP-0006 | SURFBEV-B0-raw1024-DDP4-v2-noaug | 44 | v2 split SHA `17f49430b15037c8da25c63a1bf7fe4d510cf5ff153263d2feba62c4393441d5`；694/100/151；GPU0–3；无随机增强；100 epoch、last；从预训练 SAM2 全新开始 | 2026-10-12 09:13 完整完成 100 epoch；best segm/mAP 0.7977、last epoch val segm/mAP 0.7952 / bbox 0.7782；test 未开封；现为 DUAL 探针谱系锚点 |
| EXP-0007 | SURFBEV-B0-512native-sealed-DDP4-v2-noaug | 44 | 同 v2 split 与 pair manifest；**原生 512 输入协议**（无上采样、GT 原生 512）；架构=封存 a3sgm 新身份 `ue-whu-a3sgm-512-v1`；GPU0–3；每卡 batch 1、累积 2、global 8；无随机增强；100 epoch、last；从预训练 SAM2 全新开始 | 预注册 `docs/EXP-0007_preregistration.md`；2026-10-14 09:43 启动、14:29 完成 100 epoch（last segm 0.7522 / bbox 0.7309）。**数据污染**：训练窗口内 49 train/15 val 场景 GT 被 componentclean_v2 修复管线改写（见 `docs/DATA-INTEGRITY-20261014_incident.md`）；作为锚点作废，由 EXP-0009 冻结快照重训替代 |
| EXP-0008 | SURFBEV-B0-512dual-sealed-DDP4-v2-noaug | 44 | 同 EXP-0007 协议逐项对齐；唯一变量=冻结 UAV 支路+零初始化门控 P3 融合（身份 `ue-whu-a3sgm-512-dual-v1`，指纹 `4ccb42b8…`）；PI 豁免筛查层直接双臂 A/B（预注册登记）；遥测四类+UAV-off 消融 | 预注册 `docs/EXP-0008_preregistration.md`；启动被 wrapper 数据复验正确拦截（完整性事故），未训练一行；由 EXP-0010 冻结快照替代，代码与审计谱系直接继承 |
| EXP-0009 | SURFBEV-snap512-sealed-DDP4-v2-noaug | 44 | 协议与 EXP-0007 逐项一致；数据根=版本化冻结快照 `/data2/wangcheng/dataset_frozen/ue_raw_hero_uav_pairset_20261014_frozen`（manifest 指纹校验）；原生 GSD 协议不变 | 事故后替代 run；待修复管线稳定后自动启动 |
| EXP-0010 | SURFBEV-snap512dual-sealed-DDP4-v2-noaug | 44 | 同 EXP-0009 + 冻结 UAV 支路与零初始化门控融合（EXP-0008 谱系）；A/B 对 EXP-0009 | 2026-10-15 21:54 完成 100 epoch（last 0.7550/0.7317）；拉通落判 **DUAL_HARM**（双指标显著负，on−off 不显著）；后续量纲缺陷由 EXP-0014 修复验证 |
| EXP-0011 | SURFBEV-snap512dualv2-sealed-DDP4-v2-noaug | 44 | 同 EXP-0010 唯一变量=融合层从仅 P3 扩为 **P2+P3 双层**（`target_levels=[0,1]`，逐层独立 proj+零初始化门控，分层遥测；身份 `ue-whu-a3sgm-512-dualv2-v1`，指纹 `d26f1f97…`）；与 EXP-0010 并行共租四卡；A/B 对 EXP-0009 | PI 决议 2026-10-15：检验"融合点上移至 RPN 小目标主力层"假设；审计 PASS（tag `EXP-0011`→`3555a61`）；2026-10-16 00:42 完成 100 epoch（last 0.7550/0.7327）；拉通落判 **DUAL_HARM**（segm 显著负；`docs/DUAL-AB-LADDER-20261016_conclusion.md`） |
| EXP-0012 | SURFBEV-snap512duallora-sealed-DDP4-v2-noaug | 44 | 同 EXP-0011 唯一变量=**UAV 编码器 LoRA**（use_lora=True，r=16/α=32/qkv+proj/neck，+1.06M 可训；身份 `ue-whu-a3sgm-512-duallora-v1`，指纹 `50f6b097…`）；与 0010/0011 并行共租 | 检验"冻结特征质量是瓶颈"假设；启动前修复两处静默梯度断点（rasterizer @no_grad 装饰、out 装配原地写）；审计 PASS（tag `EXP-0012`→`06366e6`+勘误 `97ceb5c`）；2026-10-16 05:04 完成 100 epoch（last 0.7555/0.7349）；拉通落判净差不显著（CI 均含 0；USED_BUT_NULL 语义，δ 算子勘误见拉通结论） |
| EXP-0013 | SURFBEV-snap512dualv2l0-sealed-DDP4-v2-noaug | 44 | 同 EXP-0011 唯一变量=融合层从 P2+P3 双层改为**仅 P2 单层**（`target_levels=[0]`；身份 `ue-whu-a3sgm-512-dualv2-l0-v1`，指纹 `277a95dd…`）；与 0011/0012 并行共租；A/B 对 EXP-0009 | PI 决议 2026-10-15：EXP-0010（仅P3）判负后补齐 {P2}×{P3} 因子表做层级归因；预注册 v2.3；审计 PASS（`docs/reviews/EXP-0013_implementation_review.md`，tag `EXP-0013`→`9ab3798`）；2026-10-15 22:50 启动，epoch 1 遥测单层键正常（gate_steps=174，bev_coverage 0.898）；2026-10-16 07:09 完成 100 epoch（last 0.7581/0.7326）；拉通落判 **DUAL_HARM**（bbox 显著负）；v2.3 因子矩阵：P2 单层 ≈ P2+P3 |
| EXP-0014 | SURFBEV-snap512dualnorm-sealed-DDP4-v2-noaug | 44 | 同 EXP-0010 唯一变量=**BEV 逐通道定标**（冻结 buffer `bev_scale`，s_c=1/max(rms_c,0.05)，train 全量 694 场 harvest，sha `f7c3c5fb…`；身份 `ue-whu-a3sgm-512-dualnorm-v1`，指纹 `473d1ede…`）；A/B 对 EXP-0009，机制对照 EXP-0010 | PI 决议 2026-10-16：DUAL-DELTA-DECOMP 证实的量纲缺陷（语义通道贡献 0.04%）修复臂；预注册含 v2.3 有效开度算子勘误（δ_L 分母被 cm 通道污染）；审计 PASS（`docs/reviews/EXP-0014_implementation_review.md`，tag `EXP-0014`→`e193045`）；2026-10-16 01:41 启动（launcher PID 3989266，契约 echo 齐全），epoch 1 遥测正常（gate_steps=174，bev_coverage 0.958，gate_absmax 3.7e-3，delta_p3_ratio 6.2e-4） |
| EXP-0015 | SURFBEV-snap512dualv2l0lora-sealed-DDP4-v2-noaug | 44 | 同 EXP-0013 + uav_encoder **use_lora**（r=16/α=32/qkv+proj/neck，+1,492,528 可训，与 0012 同配方）+ trainer 侧 **fusion 免 weight decay**（`--fusion-no-wd 1`，gate/proj 单独成组 wd=0、lr 同 other；身份 `ue-whu-a3sgm-512-dualv2-l0-lora-v1`，指纹 `5c7e1afc…`）；A/B 对 EXP-0009 | PI 决议 2026-10-16（不换路线）：DUAL-WHY-NULL 门控扫描 0012 k=2 峰值 +0.40（门欠开）+ wd 0.05 关闭力；组合臂（vs 0013=+LoRA+wd豁免，vs 0012=+wd豁免−L1）；预注册 `docs/EXP-0015-0016_preregistration.md`；冒烟通过；审计 PASS（`docs/reviews/EXP-0015-0016_implementation_review.md`，tag→`cf00f7a`）；2026-10-16 11:11 启动、17:14 完成 100 epoch（last 0.7576/0.7335）；八臂拉通落判：**双指标无显著差异（segm 精确持平 +0.02 [−0.63,+0.64]，全梯队首次）**，on−off 双指标首次同显著（+0.38/+0.34），vs 0013 segm +0.73 显著 |
| EXP-0016 | SURFBEV-snap512dualembed-sealed-DDP4-v2-noaug | 44 | **image_embeddings 注入位**（`target_levels=[]`+`embed_fusion`，262→256 零初始化门控加到 SAM2 掩码解码器稠密条件场 stride-16/32²；冻结 UAV 编码器）+ trainer 侧 **fusion 免 weight decay**；身份 `ue-whu-a3sgm-512-dualembed-v1`，指纹 `12a63439…`；A/B 对 EXP-0009 | PI 决议 2026-10-16（不换路线）：拉通分析确认 BEV 到不了掩码合成主通路，本臂把注入移到合成点（单注入位，high_res 留作消融）；组合臂（vs 0010=注入点+wd豁免）；预注册同 0015 文档；冒烟通过；审计 PASS（同 0015 文档）；2026-10-16 11:11 启动、17:12 完成 100 epoch（last 0.7557/0.7334）；八臂拉通落判：**DUAL_HARM**（segm −0.62 [−1.19,−0.05]、bbox −0.75 [−1.30,−0.21]）；on−off bbox 逐位恒 0——注入位纯度从结局侧获证（embedding 只达掩码合成）|
| EXP-0017 | SURFBEV-snap512dualv2l0lora-fk50-sealed-DDP4-v2-noaug | 44 | 配置/身份同 EXP-0015（`ue-whu-a3sgm-512-dualv2-l0-lora-v1`，身份复用循 0007/0009 先例）；唯一变量=**融合延迟解冻**（`--fusion-unlock-epoch 50`：gate 值钉零至 ep50 末，第 51 epoch 起自由；锁定期 UAV 支路含 LoRA 梯度恒零）；A/B 对 EXP-0009，单变量对照 EXP-0015 | PI 决议 2026-10-16：0015 预注册解读条款兑现（训练动力学变量）；预注册 `docs/EXP-0017_preregistration.md`（机械核验线：锁定期注入比恒 0/解锁行/解锁后>0，冒烟 3-epoch 已验证）；2026-10-16 待审计放行 |
| EXP-0018 | SURFBEV-warm-r1g-full-DDP4-v2-noaug-cotenant | 44 | WARM-R1G 全量（筛查链结论建议）：EXP-0015 基座 + gate_init 0.001（身份 `ue-whu-a3sgm-512-warm-r1g-v1`）+ `--init-from EXP-0009 last` + `--freeze-base`（LoRA 精确集可训）；100ep 单余弦无重启；DDP4×batch1×累积2 = global 8（与阶梯逐值一致），与 EXP-0019 共租四卡并行；A/B 对 EXP-0009；test 封存至双 seed last | 预注册 `docs/EXP-0018-0019_preregistration.md`（11:40 修订：DDP2 初版 ~4ep 终止作废，改 DDP4 共租）；2026-10-17 11:47 启动、16:37 完成 100ep（s44 last 0.7577/0.7384；末代 delta_p3_ratio 0.121）；落判见 `docs/EXP-0018-0019_conclusion.md`：中间态（s44/合并池 bbox CI>0 谱系首次，幅度 +0.06~0.10 未近 DUAL_GAIN 门；test 开封 parity） |
| EXP-0019 | SURFBEV-warm-r1g-full-DDP4-v2-noaug-cotenant | 45 | 同 EXP-0018 唯一变量=seed 45；DDP4 共租四卡并行；seed 一致性条款见预注册 | 同 EXP-0018 文档；2026-10-17 16:37 完成 100ep（s45 last 0.7590/0.7384；末代 delta_p3_ratio 0.110）；落判同文档：中间态（双指标全正、CI 含 0） |
| EXP-0020 | SURFBEV-warm-r1c-full-DDP4-v2-noaug-cotenant | 44 | WARM-R1C 全量（R1C 筛查结论建议）：R1G 配方 + 描述子通路替换（CVT 注意力 + 统计组反投影，身份 `ue-whu-a3sgm-512-warm-r1c-v1`，模块审计 FAIL→修复→PASS @896b07c）+ warm 冻结 + 100ep 单余弦；DDP4 共租与 EXP-0021 并行；A/B 对 EXP-0009，臂间对照 EXP-0018/0019；test 已有 single 锚、双 last 齐后再开封本臂 | 预注册 `docs/EXP-0020-0021_preregistration.md`；2026-10-17 19:05 启动、22:33 宕机（ep54）后 23:08 续训（调度重启伪影登记）、2026-10-18 02:27 完成 100ep；落判见 `docs/EXP-0020-0021_conclusion.md`：中间态（合并池 bbox +0.079 CI>0、segm +0.055 含 0；臂间 vs 0018 不显著 R1C≈R1G；全量阶梯：模板结论成立且 s44 全恒定≈真实——通路塌缩为静态偏置） |
| EXP-0021 | SURFBEV-warm-r1c-full-DDP4-v2-noaug-cotenant | 45 | 同 EXP-0020 唯一变量=seed 45；DDP4 共租并行 | 同 EXP-0020 文档；22:33 宕机（ep53）后 00:03 续训、2026-10-18 03:30 完成 100ep；落判同文档（segm +0.077 [−0.008,+0.165] 边缘） |


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
