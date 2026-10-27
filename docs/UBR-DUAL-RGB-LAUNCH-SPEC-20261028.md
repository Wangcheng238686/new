# UBR 双流 +RGB 臂启动规格（EXP-0030 / UBR-DUAL-RGB-20261028-REAL）

## 0. 背景与决议

PI 初衷：最终双流应消耗 UAV 全部传感（**RGB + 深度 + 内外参**）。当前在跑的 EXP-0029（dual real）只用了深度+内外参（几何图路径），RGB 缺席——那是证据链的刻意终点，但与初衷冲突。调和依据：远端 candidate_detail 矩阵 test 判决中，**原双流 compact 的 RGB 通路内容门 CI 阳性（+0.89pp [+0.64, +1.20]）**，是唯一 test 阳性的 RGB 消费形态（P0 源编码器 → 每格等权均值 → block4 注入）；判负的是 Q 式逐候选注意力读取（−2.92pp）与本线冻结时代的直投/特征注入。**决议：加发 +RGB 臂，采用唯一阳性形态；当前 dual(D+pose) 臂保留为对照臂，不杀。**

## 1. 臂定义

```
EXP-0029 dual(D+pose) 全部保留：
  几何图 → 几何头(202,581, λ_geo=1.0) + ⑤端口(UBR 45ch) + 推理候选消费(r(s)/0.25/250)

新增 RGB 消费路径（compact 同构，逐位移植）：
  UAV RGB ×5视角 ─→ 源 LoRA SAM2 编码器(r16/α32/dropout.05, 基础ctor权重)
                ─→ P0 金字塔层(256ch) ─→ rgb_project ─→ 32d 描述子
                ─→ 候选源锚点采样(FP64 连续读取, ≤8候选/格, ≤2/视角, 原排序)
                ─→ 每格等权均值(reduce_rgb = mean)
                ─→ block4 位零初始化残差注入（独立残差路径）
```

关键设计决定：
- **只移植 RGB 支路**，不移植 compact 的几何 CNN（我们的几何消费已有三个位：候选/⑤端口/几何头，再引一路几何会双重消费、混淆归因）。因此本臂测试的是"**P0+等权均值+block4 注入这一 RGB 消费形态移植进统一架构后的增量**"——compact 的 +0.89 是该形态可行性的证据，不是本臂必然为正的保证；本臂自己的门才是判决。
- 注入残差**零初始化**：初始前向必须与 EXP-0029 逐位一致（恒等门，见 G2）。
- 源编码器入 optimizer 的 backbone 组待遇（LoRA @5e-4/wd.05），与卫星主干 LoRA 同组规则。

## 2. 移植来源（SHA 锁定）

| 组件 | 来源（远端 codex/uav-bev-dual，commit 于启动时钉死） | 适配 |
|---|---|---|
| 源编码器 | `rsprompter/uav/rgb_p0_encoder.py` | 原样 |
| 等权均值/连续采样 | `rsprompter/uav/attention_ladder.py` 的 `sample_source_continuous`/`masked_weights`；`compact_joint.py` 的 `reduce_rgb`(mean 分支) | 原样 |
| block4 注入语义 | `rsprompter/uav/compact_joint.py` `CompactInteraction` | 仅 RGB 支路，独立零初始化残差 |
| 源锚点/候选索引缓存 | 远端 FP64 锚点缓存 + manifest（`data/ue_candidate_evidence.py` 同源） | 路径与 SHA 写入合同 |
| 初始化纪律 | 只加载通用 SAM2 foundation ctor（与卫星主干同一文件），`actual_torch_load_paths` 仅允许该文件；`task_weights_loaded=False` | 断言 |

本线落点：`rsprompter/uav/ubr_dual_rgb_integration.py`（新建）+ `configs/ablations/ue_ubr_dual_rgb_512.py`；运行器复用 `ue_ubr_reseal_dual_20261027.py` 谱系。

## 3. 合同要素（RESEAL_CONTRACT_V4）

- tag `UBR-DUAL-20261028-REAL`，commit + 全量 source_hashes（含移植文件与远端来源 SHA）。
- 协议与 EXP-0029 逐字段一致：seed 44、batch 1×accum2、100ep、lr 5e-4/wd.05/warmup100、AMP、无 EMA、val 每 epoch 四卡分片、FP32 canonical AP250、512 输入、nproc 4。
- `geometry` 块照抄 EXP-0029（缓存 `native512_height_dx_dy_viewcount_v1`、λ_geo=1.0、r(s)/0.25/250 不变）。
- 新增 `rgb` 块：encoder 配置、P0 投影维度、锚点缓存路径+SHA、注入位（block4）、初始化文件清单、`rgb_content: real`；预留 `rgb_content: constant` 分支（train-only 逐通道均值、同 support 掩码——第四臂零成本切换）。
- `dual − single` 差异枚举更新为 {⑤端口, loss_geometry, RGB 注入路径} 三项。

## 4. 工程门（fail-closed，全过才准启动）

| 门 | 内容 |
|---|---|
| G1 骨架封印 | seed-44 双构建对比：除源编码器/rgb_project/注入适配器外，全部张量与 EXP-0029 构建逐位相等 |
| G2 恒等门 | 零初始化下，本臂前向与 EXP-0029 前向输出**逐位一致**（残差未激活 = 纯几何双流）；随后单次真实更新有限、遥测齐全（RGB/描述子统计、每格 support、残差 RMS） |
| G3 资源冒烟 | 峰值 ≤22 GB/卡；epoch 时长预算：比 EXP-0029 慢 ≤35%（5 视角编码器的代价）；记录实测值入合同 |
| G4 合同封印 | val 分片对齐继承（已过）；合同 v4 + 审计文档落盘 |
| G5 常量分支 | `rgb_content: constant` 单元级演练（为第四臂备好，不启动） |

## 5. 预注册（四要素）

1. **主指标**：test 459、固定 E100、MaskAP/bboxAP + Boundary2%/1%，配对图像 bootstrap（500 次，seed 20261021）。
   - 主比较：**dual(+RGB real) − dual(D+pose)**（EXP-0029，同协议同构）。
   - 必报分层：密度（≤50 / 51–100 / >100 实例图；test 为 74/197/188——val 无 >100 图，**判决只在 test**）、面积（small/medium/large）。
2. **选择口径**：固定 E100，不以 val 挑轮；val 仅作训练健康监控。
3. **判定**：
   - **RGB_WINS** = MaskAP 差 CI 下界 >0 → 启动第四臂 +RGB(constant) 做 RGB 通路内容差，过门后 RGB 进最终形态（三模态全用，初衷达成）；
   - **RGB_NULL** = CI 含 0 → 最终形态维持 D+pose，论文如实记录"RGB 在 compact 形态有效、在统一架构无增量"；
   - **RGB_HARM** = CI 上界 <0 → 同上并记录干扰证据。
   - test 全程只在全部臂锁定后读一次。
4. **限制声明**：单 seed、E100 前缀无（全预算）、开发暴露 test 口径沿用现行合同；不以非零梯度/attention 非均匀/残差非零认定成功。

## 6. 排队与资源

- **启动槽位：等 EXP-0028-R2（单流再封版）或 EXP-0029 任一收线后**。禁止三 job 共租：当前两 job 已占 ~29 GB/卡，第三 job ~17–19 GB 会顶到 46+/48 GB，val 峰值期 OOM 风险不可接受。
- 预计：实现+门 ~半天；排队后与剩余一臂共租运行 ~2–2.5 天收线。
- 遥测哨兵：每 epoch 记录源编码器显存/耗时占比；若 epoch 慢于预算 35% 超 2 个 epoch，回查编码器批处理（5 视角应合并批过）。

## 7. 明确不做

- 不做 Q 式逐候选注意力读取（test CI 阴性 −2.92）；
- 不做 RGB 直投 ⑤端口/几何头输入（冻结时代已否证，且本臂变量已足够）；
- 不移植 compact 几何 CNN（几何双消费）；
- 不改 r(s)/0.25/250/FP32 契约任何一位；
- 第四臂 +RGB(constant) 仅在 RGB_WINS 后启动。

## 8. 工件

- 本规格：`docs/UBR-DUAL-RGB-LAUNCH-SPEC-20261028.md`
- 对照臂：`/data2/wangcheng/remote-lthpc/checkpoint/ue_ubr_dual_real_20261027/`（EXP-0029，在跑）
- 单流基线臂：`/data2/wangcheng/remote-lthpc/checkpoint/ue_ubr_reseal_r2_20261027/`（EXP-0028-R2，在跑）
- 远端证据：`/data/wangcheng/checkpoint/ue_candidate_detail_ddp_20261026/analysis/test/summary.json`（Q/G/QG 判决）与 compact 四臂轨迹
- 移植源：远端 `rsprompter/uav/{rgb_p0_encoder,compact_joint,attention_ladder}.py`、FP64 锚点缓存
