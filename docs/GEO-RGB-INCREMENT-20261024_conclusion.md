# 已验证接口上的 RGB 增量检验：落判（2026-10-24 晚）

对照预注册 `docs/GEO-RGB-INCREMENT-20261024_preregistration.md` 第一部分。
全部为本地训练权重（seed44、共享 common_init、同预算 2670 更新、
pilot 训练循环逐字复用——模块替换法，锁定源未动），条件评估协议与
复现轮完全一致（强 SAT 前缀 66.0875、FP32、maxDets250、参考记录）。

## 结果（test 459，segm Mask AP ×100）

| 配置 | Mask | 判定 |
| --- | ---: | --- |
| 强 SAT 前缀 | 66.0875 | 参考 |
| 几何-only（复现轮基线） | 66.5546 | 净 +0.4672 [CI +0.297,+0.616] |
| **GR-real（fields 真 + RGB 真）** | 66.5401 | 净 +0.4526 [CI +0.284,+0.550] |
| **GR-constRGB（fields 真 + RGB 均值）** | 66.4910 | — |

配对 bootstrap（500/seed20261021）：

| 预注册门 | 结果 | 判定 |
| --- | --- | --- |
| **RGB 内容门**（GR-real − GR-constRGB，Mask CI 下界>0） | **+0.0490 [−0.036, +0.102]** | **FAIL——CI 含 0** |
| 接口净门（GR-real − 前缀 ≥ 几何净−0.1pp） | +0.4526 ✓（门槛 0.3672） | 过 |
| RGB vs 几何-only 头对头 | **−0.0146 [−0.156, −0.002]** | **几何-only 反而显著略优** |

四指标细目（GR-real − GR-constRGB）：Mask +0.049 [−0.036,+0.102]、
bbox +0.004、Bnd2% +0.009 [−0.036,+0.103]、Bnd1% −0.049 [−0.036,+0.100]
——全部含 0 或近零。RGB vs 几何-only：Mask/Bnd2%/Bnd1% 的 CI **上界
<0 或触 0**（加 RGB 反而轻微伤害，可能为容量稀释/训练噪声）。

## 判定

**RGB 外观在已验证接口上无可兑现增量（在单因素直投形式下）。**

1. 预注册 RGB 内容门失败：真 RGB 相对常量 RGB 的增量 +0.049pp，
   CI 含 0——不能拒绝"增益为零"；且点值仅为几何内容差（+0.247）的
   1/5。
2. 头对头：几何+RGB 不优于几何-only（四指标 CI 上界 ≤+0.0007，
   Mask/Boundary 上界 <0）——直投 RGB 不带增量还略负。
3. 这与远端结论的保留判断一致（"UAV RGB 旧路线失败因素尚未完全
   拆开；本结论不宣称 RGB 无信息"）：本检验只否证**z-buffer 胜者
   像素直投 + 同一 CNN**这一最小形式，不否证 CVT 式跨视角聚合。
   按预注册条款，CVT 变体保留为后续臂（复杂表示 vs 直投的增量
   归属），但先验下调。

## 附：新几何-only 复验读数

本轮 GR 双臂的任何50@250 = 13176/13124（real/const），均高于复现轮
几何-only 的 12895——**架构扩容（5→8 通道）本身提高了候选召回**，
但经接口消费后的 AP 增益不变（+0.45 vs +0.47）：召回增益发生在
接口 250 名额与去重之后不产生 AP 的区段，内容差仍由几何场承载。

## 工件

- 权重：`checkpoint/ue_geometry_rgb_pilot_20261024/{geometry,constant}/last.pth`
- 条件评估：`checkpoint/ue_geometry_rgb_conditional_20261024_localweights/{geometry,constant}/test/`
- 配对 CI：`checkpoint/geo_rgb_cis/{rgb_content,rgb_net,rgb_vs_geo}.json`
- RGB 缓存（z-buffer 胜者源像素，fields 逐位对齐已验证）：
  `cache/ue_v1_rgb_geometry/`（3060 点，native512_rgb_v1）
