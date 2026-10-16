# 工件说明：编号实验日志 + 数据集描述（2026-10-16）

目的：另一台机器拉取本仓库即可做训练曲线/遥测对比；大体积产物（权重、
逐场 A/B 数据、数据集本体）留在本机数据盘，通过下方访问映射直接取。

## 本机访问映射（分析机可直接 rsync/scp）

- 主机：`wangcheng@10.169.218.118`（局域网 IPv4；本机另有 IPv6 出口）
- 权重（各臂 `last_model_epoch100.pth`，约 1.08GB/个）：
  `/data2/wangcheng/checkpoint/surfbev_b0/surfacebev_v2_512*_s44_EXP-00{09..16}/`
- 完整训练日志（0015/0016 收线后以本机为准再拉一次）：
  `/data2/wangcheng/checkpoint/surfbev_b0/launcher_logs/`
- 拉通分析工件（summary.json / 逐场 rows / 身份清单）：
  `/data2/wangcheng/checkpoint/surfbev_b0/probes/DUAL-AB-LADDER-20261016/`
  （另有 DUAL-GATE-SWEEP / DUAL-REDUNDANCY / DUAL-DELTA-DECOMP 等子目录）
- 复现评估所需的 SAM2 预训练：
  `/data1/wangcheng/pretrained/sam2_hiera_base_plus.pt`
- Python 环境：`/data2/wangcheng/envs/cvt2/bin/python`

## 数据集（仅描述，不入库）

**开发/实验用冻结快照**（EXP-0009 起全部臂的唯一样本来源）：

- 路径：`/data2/wangcheng/dataset_frozen/ue_raw_hero_uav_pairset_20261014_frozen`
- 来源：`ue_raw_hero_uav_pairset_20261011` 生产集的逐文件一致拷贝固化
  （2026-10-14 数据完整性事件后建立；事件报告
  `docs/DATA-INTEGRITY-20261014_incident.md`）
- 接受清单 `accepted_pairs.jsonl` SHA256
  `f15cd8f5b8d10a69d721546d6cfa1f2cce58af2861d57b1357c5e9ff1e1fe62a`；
  v2 空间防泄漏拆分 `surfbev_b0_split_v2.json` SHA256
  `17f49430b15037c8da25c63a1bf7fe4d510cf5ff153263d2feba62c4393441d5`；
  快照清单指纹 `7b914d9a…`（`<snap>_manifest.json`）
- 规模：train 694 / val 100 / test 151（255 场景因清洗排除；test 至今未开封）
- 内容结构：每场景 = 卫星 512² 原生 GSD 图 + GT（单类建筑实例 bbox/mask）+
  五视角 UAV 512² RGB（`view_yaw_*` 子目录）+ DeviceZ 深度 + 双侧相机
  内外参（cm 世界坐标）；契约见 `portable_sam2_explicit_coarse/data/ue_raw_pair_dataset.py`
- **生产集仍在写入，勿直连**：`/data1/.../ue_raw_hero_uav_pairset_20261011`
  为渲染管线活跃目录（/data1 磁盘已满 95%），实验一律走上面的冻结快照。

## 日志清单与完整性注记（artifacts/launcher_logs/）

| 文件 | 状态 |
| --- | --- |
| EXP-0001/0002-seed*.log | 早期单卡 run（用户终止于 epoch 1 后），仅存档 |
| EXP-0005-seed44-ddp4.log | v1 split 时代，ep58 定向终止（数据泄露后作废） |
| EXP-0006-seed44-ddp4.log | 1024 v2 锚点，完整 100 epoch |
| EXP-0007-ddp4.log | 512 原生首跑，完整（后被判定数据污染作废） |
| EXP-0008-ddp4.log | 启动被完整性守卫正确拦截（124B 存根，本身是证据） |
| EXP-0009-ddp4.data1-partial.log | 单流基线，/data1 副本至 ~ep61（完整线被链式重启截断） |
| EXP-0009-ddp4.data2-stub.log | 124B 重启存根；两份合起来即全部现存 stdout |
| EXP-0010…0014-ddp4.log | 完整 100 epoch，含逐 epoch DUAL 遥测行 |
| EXP-0015/0016-ddp4.log | **进行中快照**（拷贝于 2026-10-16 ~12:50，约 ep22）；终值以本机原文件为准 |

EXP-0009 的终值（last segm 0.7593 / bbox 0.7374）恢复自其封存 checkpoint
的逐 epoch history（`history.val_metrics[-1]`），方法交叉验证见
`docs/reviews/DUAL-AB-LADDER-20261016_result_audit.md` 第 2 节。

各臂身份链（指纹/基线 ID/协议契约行 `EXPxx_CONTRACT`）见各日志头部与
`docs/EXP_REGISTRY.md`。
