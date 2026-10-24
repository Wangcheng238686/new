# 预注册：已验证接口上的 RGB 增量检验 + 联合训练最终形态（2026-10-24）

依据：远端结论 `UE_UAV_EFFECTIVE_DIRECTION_VERDICT_20261024.md` §5
（"RGB/CVT 的后续增量可在这个已验证接口上检验"；"最终双流从共同
初始化联合训练仍是后续开发任务"）与本机复现（
`docs/GEO-DIRECTION-REPRO-20261024.md`，net +0.4672 / content +0.2468
全部 CI>0）。先预注册后编码。

## 第一部分：RGB 外观增量（单因素 vs 几何-only）

**接口不动**：实例候选接口、直接实例监督、冻结掩码路径消费、
强 SAT 前缀与参考记录全部沿用；唯一变量 = 几何头输入加 UAV RGB。

- **输入扩展**：投影 RGB 三通道——镜像 native 缓存的 z-buffer 数学
  （同一最近表面胜者），胜者源像素的 UAV RGB；同一 support 门控。
  新缓存 `native512_rgb_v1`（fields4+rgb3+support 同一 npz，独立
  目录，不触碰已复现缓存）。
- **网络**：pilot.Net 逐层规格不变，唯 stage-a 输入 5→8 通道；
  新脚本 `ue_geometry_rgb_pilot_20261024.py`（不动锁定源）。
- **臂（同预算 10ep/2670 更新/同顺序，seed44，共享 seed45 式
  common_init）**：
  - **GR-real**：fields 真 + RGB 真；
  - **GR-constRGB**：fields 真 + RGB 换 train-only 均值（同 support，
    吸收容量因素）。
  - 对照基线 = 本机几何-only 重训（已存，net +0.4672）。
- **判据（全部先于 test 收口）**：
  1. **RGB 内容门**：GR-real − GR-constRGB 逐图配对 bootstrap
     （500/seed20261021）Mask CI95 下界 >0——否则 RGB 外观在该接口
     无可兑现增量（远端结论的保留判断成立），落判记录；
  2. **接口净门**：GR-real vs 强 SAT 前缀 CI 下界 >0 且不低于
     几何-only 净（+0.4672）−0.1pp；
  3. 伴随：bbox/Boundary 双口径、新增实例计数（对齐远端
     new-instance 口径）。
  4. CVT 式跨视角注意力暂缓：投影 RGB 直入是 RGB 内容假设的最小
     单因素形式；若过门再立 CVT 变体（复杂表示 vs 直投的增量归属）。

## 第二部分：联合训练最终形态（探索性首跑）

- **形态**：共同初始化联合训练——卫星分支从封版单流初始化
  （或共同随机初始化，二选一在冒烟后定）+ 几何/RGB 头，端到端，
  训练含卫星任务损失与几何头直接监督；推理沿用实例候选接口。
- **首跑预算**：10ep 筛查（同 batch/优化器族），kill 门：
  卫星前缀质量崩塌（val Mask 较 E73 前缀掉 >1pp 即停）。
- **判据**：联合形态 vs（冻结 E73 + 几何头）的同协议对照；
  仅在第一部分落判后启动，防止算力空烧。


## 第一部分落判（2026-10-24 晚，`GEO-RGB-INCREMENT-20261024_conclusion.md`）

**RGB 内容门 FAIL**：GR-real − GR-constRGB = +0.049pp [−0.036,+0.102]
（CI 含 0）；头对头 vs 几何-only 四指标 CI 上界 ≤+0.0007（Mask/Bnd
上界 <0）。**z-buffer 直投 RGB 的最小形式被否证**；CVT 变体按条款
保留但先验下调。容量注记：8ch 使 any50@250 召回 +281（13176 vs
12895）但接口消费后 AP 不变——召回增益落在 250 名额/去重后不产生
AP 的区段。