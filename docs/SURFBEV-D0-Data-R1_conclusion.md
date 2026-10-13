# SURFBEV-D0-Data-R1：原始配对数据读取与相对几何探针

**状态：PASS_FOR_READER_AND_RELATIVE_GEOMETRY_ONLY。** 这不是 AP 实验，也不是
DeviceZ 米制深度的最终标定。它只放行原始标签 reader 的实现，以及以 DeviceZ 建立
跨视角相对表面几何的后续数据准备；模型训练仍未授权。

## 预注册口径

- **路线：** 数据契约只读探针，临时编号 `SURFBEV-D0-Data-R1`。
- **数据：**
  `ue_raw_hero_uav_pairset_20261011/accepted_pairs.jsonl`，SHA256
  `f15cd8f5b8d10a69d721546d6cfa1f2cce58af2861d57b1357c5e9ff1e1fe62a`。
- **主读数：** `satellite_instance_id_clean.png` 的非零 local ID 是否均属于
  `instances.json`；以及五相机矩阵的 `W2C @ C2W` 自洽性。
- **伴随读数：** 在 120 个按场景名排序的确定性样本上，采用生产元数据给定的
  RGB packed decode，扫描 `z_camera_cm=A/device_z` 的
  `A∈{1,2,5,10,20,50,100}` cm；每 scene 汇总跨视角重投影深度绝对对数残差的中位数。
- **选择口径：** 预定尺度网格的全量结果；不按图像、特征或标签挑选。
- **判定：** 所有 1,200 场景均应可读取且无 unexpected raster ID；相机自洽误差
  `<=1e-9`；几何尺度只允许作为候选，不得声称绝对米制深度。

## 结果

| 项目 | 结果 |
| --- | --- |
| manifest 场景 / 成功标签读取 | 1,200 / 1,200 |
| unexpected raster ID 场景 | 0 |
| label raster 形状 | 全部 512×512 |
| 标签前景像素中位数 | 127,417.5 |
| `max|W2C @ C2W-I|` | `2.91e-11` |
| 最优 DeviceZ 候选 | `A=10 cm` |
| `A=10` 的 scene-median 残差中位数 / p90 | `0.001125` / `0.003507`（约 0.11% / 0.35%） |
| `A=5` / `20` 对应中位数 | `0.1561` / `0.1069` |

`A=1,2 cm` 没有跨视角落在目标视场内的支持点；这本身不能作为它们优于或劣于其它
尺度的证据。`A=10 cm` 与相邻可支持候选之间有明显差距，说明当前 decode、K 与
外参可以构成非常一致的相对几何。该 120-scene 几何子样本是场景名排序前缀，仅覆盖
9/81 个空间 cell；不同尺度的投影支持数也不同，且当前没有 target z-buffer 遮挡过滤。
因此它不是全空间代表性的尺度比较，更不是严格的跨视角匹配验证。

## 能证明与不能证明

**能证明：** 原始 strict raster ID 可以直接产生卫星实例 mask/bbox，而不需由多边形
近似；五路 UAV 的序列化矩阵彼此可逆；数据中存在与 `z=A/device_z`、`A≈10 cm` 相容的
相对表面几何候选。因此可以着手编写 raw reader 和相对几何的数据准备流水线。

**不能证明：** 这不能证明 DeviceZ 的绝对深度正确，更不能证明 UAV 会提升任何
检测或分割 AP。交叉视角一致性可以被共同的尺度或渲染契约错误所满足。依照
`SURFBEV-20261011`，仍需由同 FOV/分辨率/packing 的已知 10/30/60/100/150m 平面渲染
验证 UE `ConvertFromDeviceZ`：每距离中心 80% 像素中位相对误差 `<=1%`、任一点
`<=2%`，才可放行米制 SurfaceBEV。

另有 503 个场景存在 JSON 声明但 raster 中不可见的实例；这被严格 producer 契约允许
（选入范围但完全被遮挡），reader 应只以 raster 可见 ID 监督，绝不可为这些实例伪造
mask。当前验证的是 `visible raster ID ⊆ declared JSON ID`；reader 实现时还须显式断言
JSON 的 declared local ID 无重复。

## 工件与复现

- 实现：[probe_surfacebev_data_contract.py](/home/wangcheng/project/new/.worktrees/dual-stream-dev/portable_sam2_explicit_coarse/scripts/ue/probe_surfacebev_data_contract.py)
- 汇总：[summary.json](/home/wangcheng/project/new/.worktrees/dual-stream-dev/portable_sam2_explicit_coarse/outputs/SURFBEV-D0-Data-R1/summary.json)
- 每 scene 标签：[per_scene_labels.jsonl](/home/wangcheng/project/new/.worktrees/dual-stream-dev/portable_sam2_explicit_coarse/outputs/SURFBEV-D0-Data-R1/per_scene_labels.jsonl)
- 每 scene 几何：[per_scene_geometry.jsonl](/home/wangcheng/project/new/.worktrees/dual-stream-dev/portable_sam2_explicit_coarse/outputs/SURFBEV-D0-Data-R1/per_scene_geometry.jsonl)

复现命令：

```bash
python portable_sam2_explicit_coarse/scripts/ue/probe_surfacebev_data_contract.py \
  --max-scenes 1200 --geometry-scenes 120 \
  --out portable_sam2_explicit_coarse/outputs/SURFBEV-D0-Data-R1
```

## 独立审计

独立审计已完成，结论为 **CONDITIONAL PASS**：仅放行 raw reader 与相对几何候选，
不放行米制 SurfaceBEV 或模型训练。审计发现与本结论一致，并要求 D0 的最小补件为：

1. 10/30/60/100/150m 受控平面绝对标定；
2. 扩展 PNG decode、矩阵逆与 declared-ID-unique 检查到全部 1,200 场景；
3. 用覆盖 81 个空间 cell 的分层样本（或全量）重跑几何扫描，报告共同支持样本；
4. 加 target-view depth-consistency/z-buffer 遮挡过滤后，才允许将它作为 SurfaceBEV
   的几何输入。

审计人未参与该 probe 的实现或设计；审计结论已在本文件完整吸收。
