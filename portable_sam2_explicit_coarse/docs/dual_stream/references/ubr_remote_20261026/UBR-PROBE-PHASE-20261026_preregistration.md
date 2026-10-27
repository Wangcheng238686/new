# UBR 探针阶段预注册：门0 snap / 级1 固化参照 / 级2 加法探针（2026-10-26，启动前）

依据 `docs/UBR-UNIFIED-ARCHITECTURE-DESIGN-20261026.md` §五 阶梯前三级。骨架门自检：三级均直接服务单一问题（UAV 信息对卫星侧实例分割是否有帮助——本阶段聚焦其分割/边界半边）；判据全部校准于内容门与配对 CI；无支线目标；不动协议契约与会议证据链。

## 资产清单（已定位，SHA 于级1 落盘）

| 资产 | 路径 | 身份 |
|---|---|---|
| 旧封版 A | `/data2/wangcheng/checkpoint/citysample_v1/citysample_v1_512single_defaug_ddp4_s44_EXP-0025/best_model.pth` | EXP-0025 v1best（epoch 由 checkpoint 头钉死于级1） |
| 旧封版 B（E73） | `/data2/wangcheng/remote-lthpc/checkpoint/ue_minimal_single_20261021_EXP-0017/checkpoints/best_model_epoch73.pth` | 接口证据宿主（66.09 强SAT带） |
| 边界评估工件 | `/data2/wangcheng/checkpoint/surfbev_b0/probes/CITYSAMPLE-BOUNDARY-AP-20261021/` | §一弱点量化来源（exp0025_v1best_test.json 等） |
| 几何缓存 | `/data2/wangcheng/remote-lthpc/checkpoint/ue_geometry_mechanism_20261022/`（native512 版本串于级1核对） | 5ch 几何图 |
| 数据集 | `/data1/wangcheng/dataset/CitySample_SatUAV_TrainMinimal_v1_20261021`（2142/459/459） | **test 全程不读** |

## 门0 SNAP-20261026（零训练，≈0 成本）

- **问题**：冻结掩码边界处，几何图内是否存在可吸收的边界证据（headroom）。
- **机制**：冻结 EXP-0025 v1best 在 **val 459** 推理；对每个实例的掩码边界像素沿外法向 ±5px 搜索高度梯度 |∇h| 局部极大值（限 support=1）贪心吸附；重建掩码后重算指标。
- **四要素**：主指标 B-AP_l(7px) 与塌落率，伴随 AP_l（禁单指标）；选择口径=无训练、单次确定性（seed 20261021 采样处固定）；显著性=real−sham 与 real−constant 逐图配对 bootstrap（500 draws，seed 20261021）；判定门：**ΔB-AP_l(real vs 不吸附) ≥ +1.0pp 且 real−sham CI 下界 >0 且 real−constant CI 下界 >0.5pp 差** → 级2 放行；real≈sham≈constant（差 <0.3pp）→ 几何图是瓶颈，§四暂停，先修 z-buffer。
- **对照**：sham（±5px 随机吸附位）、constant（train-only 均值图，同 support 纪律）。
- **遥测三件套**：判读边界=零训练只能证"证据存在性"，不证可学习性；标定校验=B-AP 评估器先在既有 val/test 工件上复现数字再上岗；机制遥测=吸附位移直方图、边界 support 覆盖率（不替代结局）。

## 级1 FREEZE-REF-20261026（≈0 成本）

- 双参照归档：两 checkpoint 拷入 `frozen-refs/`（SHA256 + epoch 身份钉死）；val 459 冻结评估（segm/bbox mAP + AP75 + B-AP 全套）双参照落盘 manifest。
- 门：工件齐全、评估器 parity 通过（复现 20261021 工件中可对照数字）。

## 级2 UBR-ADDPROBE-20261026（加法探针，探针预算）

- **构型**：冻结旧封版 A（**含其内部 UDPR**，加法语义）真实 train 推理输出（掩码 logits/解码器特征/SDT/邻居）上独立训练 UBR；serial 串联。UBR 模块为生产层实现 `rsprompter/uav/ubr.py`（工程可读、SDT/带掩码/常量图为可测纯函数，k∈{6,8,12,16}）。
- **臂**：UBR(real) vs UBR(constant)（support=1 处 train-only 逐通道均值、洞处零）；可选第三臂 UBR(zero端口) 作单流条件健全性。
- **四要素**：主指标 B-AP_l(7px) 与 Mask AP，伴随 Boundary1%/2%、AP75；选择口径=**固定 E10**（禁 best 挑选）；显著性=逐图配对 bootstrap 500 draws seed 20261021；判定门（§4.9 原文）：**B-AP_l ≥ +1.0pp 且 AP_l 回退 ≤0.1pp 且输出层内容差 real−constant ≥ +0.5pp（均 val 459）** → 级3（再封版）放行；任一不过 → §四按预注册回退（UDPR 保留、UBR 双流专用路线复议）。
- **已知限制（预先声明）**：加法构型 ≠ 最终替换构型；证据迁移沟由级3 填补。
- 预算：UBR ~200K 参数、缓存特征训练，单卡探针级（小时级×2-3 臂）。

## 完整性契约（fail-closed）

启动前 wrapper 断言：双 checkpoint SHA、几何缓存版本串、数据集 splits/annotations SHA（沿 EXP-0024/25 预注册清单）、源码已 commit 且工作树 clean、输出目录不存在。门0/级2 均只读 val；test 仅在阶梯级3/级4 收线时按预注册口径读。

## 工件

门0 `scripts/probes/ue_ubr_snap_20261026.py`；级2 训练/评估 `scripts/probes/ue_ubr_addprobe_20261026.py` + 生产模块 `rsprompter/uav/ubr.py`；产物根 `/data2/wangcheng/remote-lthpc/checkpoint/ue_ubr_probe_20261026/`。

---

## 修订记录（2026-10-26 晚，用户批准 A+B）

### 门0 v1 结果与第三结局认定
- v1（常量臂有 bug：support 洞边缘平滑假梯度，见审查与实跑双重确认）：nosnap 63.49/AP_l 53.07；real 47.86/44.26；sham 42.17/39.30；constant 42.13/37.36。
- 配对：real−nosnap **−15.19pp** CI[−15.56,−14.83]；real−sham **+5.66pp** CI[+5.39,+5.92]；real−constant +5.51 CI[+5.24,+5.80]。
- 按预注册门（real−nosnap ≥ +1.0pp）**判 FAIL**；但失败分支（信息不存在→修 z-buffer）不成立——real−sham 干净为正证明高度梯度含边界判别信息。认定为**第三种结局：信息存在，零训练逐像素贪心吸附机制破坏性**（±5px 独立抖动破坏边界形态）。
- 流程违规记录：v1 以未提交的向量化版本发射（出处补录 34fc924，认账）。

### 修订 1（A）：门0 v2 重跑
- 常量臂改为全域常数（梯度可证为零 → 与 nosnap 逐位一致；修复 83472f2，带洞 identity 已验证）。
- 仅审计目的：预期 real−nosnap 仍为负、constant==nosnap 逐位成立、real−sham 与 v1 一致。v2 结果为门0 的最终记录。

### 修订 2（B）：级2 立项依据增补
- real−sham +5.66pp（CI 下界 +5.39）= 几何图中边界信息存在性的直接证据。
- 零训练吸附的破坏性 = "信息在、收割机制不行"——**学习型空间连贯消费者（UBR）是收割该信息的必要条件**，其带内"保持原判"平凡策略天然规避吸附式破坏。
- 级2 仍按原四要素门执行（B-AP_l ≥ +1.0pp、AP_l 回退 ≤0.1pp、内容差 ≥ +0.5pp），发射等 v2 记录落盘。
