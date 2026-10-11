# SURFBEV-B0-SPLIT-V2-20261011：无跨集可见物体/视野重叠划分

## 范围与路由

这是 B0/P1 的**数据契约修订**，不启动训练、不改变原始 1,200-pair 数据，也不覆盖
`surfbev_b0_split.json`。基座数据划分改变后，旧 EXP-0005 不能充当 v2 的 B0 锚点，
其 val/test 数字不得与 v2 的数字直接比较。新 B0 属基座/训练动力学型，数据审计门通过后
才另立训练 EXP-ID；P1 仍需等待 v2 B0 last checkpoint。

## 先验规则与四要素

- 唯一输入为 `accepted_pairs.jsonl` SHA256
  `f15cd8f5b8d10a69d721546d6cfa1f2cce58af2861d57b1357c5e9ff1e1fe62a`。
  v1 中的 val/test 场景集合原样保留；其余原 train 场景逐一判定保留或 `excluded`。
- 1200 张卫星相机均为 512²、200 mm 焦距、36 mm sensor、2000 m 高度、正俯视；
  Z=0 的地面视野宽度为 360 m。采用 400 m 的保守双轴间隔：若两相机中心同时满足
  `|ΔX|<400m` 和 `|ΔY|<400m`，则它们不得分属 train 与 val/test。
- 从 strict clean-ID raster 中取**实际可见** local ID，再用同场 `instances.json` 映射
  `global_instance_id`；任一可见全局 ID 不得跨 train/val/test。不可见候选不参与此门。
- 主指标/硬门：三对 split 的可见全局 ID 交集均为零，且任意跨 split 场景对的
  400 m 双轴冲突数均为零。伴随：保留/排除场景数、每 split 可见实例数、输入清单
  哈希、原图 SHA256 跨 split 重复数、相机契约一致性。
- 选择口径：上述规则一次确定，不按模型 AP、图像内容难度或未来 probe 结果调整。
  显著性：完整 1,200 场精确枚举，非抽样，无 bootstrap；任何主指标非零为 `INVALID`。
  `PASS` 仅表示该定义下划分隔离，不能证明任务难度、域泛化或模型增益。

## 执行与不变量

生成 `surfbev_b0_split_v2.json`，为全部 1,200 scene 写唯一 `split`：
`train` / `val` / `test` / `excluded`；排除原因逐场记录。旧 v1 清单保持不变。
数据适配器显式选择 manifest；v2 训练、验证、测试 checkpoint 快照均记录文件名与 SHA256，
续训/测试时不允许无意更换 split。验证时独立从 raw 标签与相机文件重算全部交集，
且确认 v1 与 v2 的 val/test 场景集合逐项相同。

## 生成结果与使用（2026-10-11）

- v2 manifest：`portable_sam2_explicit_coarse/data/ue_raw_hero_uav_pairset_20261011/surfbev_b0_split_v2.json`，
  SHA256 `17f49430b15037c8da25c63a1bf7fe4d510cf5ff153263d2feba62c4393441d5`。
- 场景：train 694、val 100、test 151、excluded 255；全部 1,200 场各出现一次。
  排除原因计数：共享可见建筑 ID 253 场、400 m 视野缓冲 227 场；同场可有两个原因，
  因此两数相加不是 excluded 数。
- strict raster 可见 GT 实例出现次数：train 38,566、val 2,792、test 9,632；
  对应不同可见全局建筑数分别为 1,727、211、464。
- 完整枚举的 train/val、train/test、val/test 三对：共享可见 global ID = 0，
  400 m 双轴缓冲冲突场景对 = 0，完全相同 RGB SHA256 = 0。所有 1,200 场相机契约一致。
- 原始 RGB、clean-ID PNG、`instances.json`、相机内外参均未改动；清单含源数据指纹。
  `python portable_sam2_explicit_coarse/scripts/ue/build_surfacebev_b0_split_v2.py --verify`
  会从原始文件重新构造并逐字节对照现存清单，不覆盖文件。
- B0 训练/测试入口缺省使用 v2；旧 EXP-0001/0002/0005 wrapper 显式指定 v1，
  以保护历史复现。新 run 仍应显式传 `--raw-split-manifest surfbev_b0_split_v2.json`
  作为 wrapper 契约。数据适配器钉死两个 manifest SHA，
  checkpoint 的 split SHA 与请求 split 不同则拒绝续训/测试。

本结果是数据隔离门 `PASS`，**不是**任何 AP 或 UAV 信息增益结果。旧 EXP-0005
仍属于 v1，不得从其 last checkpoint 在 v2 上打分并称作 v2 基座。
