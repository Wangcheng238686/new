# UBR v1.1 显存/速度优化与再封版臂重启方案（2026-10-27，PI 决议：当前臂不可接受，立即执行）

## 0. 背景与决定

EXP-0028（UBR-RESEAL-20261027，`ue_ubr_reseal_20261027.py`）实测：峰值显存 35–37 GB/卡（48GB 卡，余量薄）、13.2 min/epoch（训练 11.2 + val 2.1）、100ep ≈ 22h，且照此实现 1024 输入不可行（UBR 激活 ~4×）。**PI 决议：停下当前臂，按本方案实现 v1.1 后重启。** 本方案所有代码锚点来自当前实现（`rsprompter/uav/ubr_train_integration.py`），执行层优化与模型容量变更分开列，等价性口径单独声明。

## 1. 瓶颈定位（代码级，来自 `ubr_train_integration.py::compute_loss_ubr`）

| # | 问题 | 位置 | 后果 |
|---|---|---|---|
| P1 | **全批 padding 到最大 patch**：`_pad_batch`（L305–309）把全部 ~64 实例 pad 到本 batch 最大窗口（大建筑可近 512²） | `_pad_batch` | 激存爆炸主因：64×512²×(32+1+1+1)ch |
| P2 | **UBR 训练路径显式 FP32**：`up[...].float()`（L289）+ 全程 float32 conv | L288–301, L313 | 激存/带宽 ×2 |
| P3 | **逐实例 Python 循环**：每实例一次 CPU 窗口切片、一次 SDT、一次 `F.interpolate`、一次 H2D 拷贝（`.to(dev)`）、一次 512² clone 断言 | L272–303 | 启动开销/主机瓶颈，GPU 间歇空转（实测 util 0–41% 凹槽） |
| P4 | **全量 512² 可微 paste**：`_do_paste_mask` 对全部 N 实例贴到 512² | L245–247 | 计算+激活，非首要 |
| P5 | **val 单卡 rank0**：459 图 FP32 串行 | 启动日志 `rank0_only=1` | ~15% epoch 时间 |

## 2. 优化项

### A 组：执行层（不改模型定义，数值中立——全部执行）

**A1 分桶批处理（对 P1+P3，速度与显存双主杠杆）**
- 实例按窗口面积排序分 3–4 个桶（如 ≤96²、≤192²、≤384²、>384²），桶内 padding，逐桶一次前向。
- 替换 L269–309：`feats/logits/sdts/nbrs/gts/bands` 收集后按桶重组；`ubr_refiner` 调用从 1 次全量变 ≤4 次桶批量。
- 带内输出逐位不变（padding 在带外，损失由 band 掩码）；小实例不再陪大建筑付 512² 的账，典型记忆节省 3–6×。

**A2 组装向量化（对 P3）**
- `window_around/feat_window` 全 numpy 逐实例 → 向量化为批量张量运算；GT/邻居裁剪合并为每次 H2D 单批拷贝（pinned memory）；逐实例 `F.interpolate` → 桶内一次 interpolate；L279–281 的全图 clone 断言改为 debug 开关（默认关，或向量化一次校验）。

**A3 UBR 路径半精度（对 P2）**
- UBR conv 包进 `torch.autocast('cuda', dtype=torch.float16)`；SDT 用 fp16 存储；**损失计算（focal/dice）保持 FP32**（从 fp16 logits 上转型计算，标准做法）。激存/带宽 ~2×。

**A4 分块反向（对超大桶兜底；默认可关）**
- 实现：逐桶 `bucket_loss.backward(retain_graph=True)`（末桶不 retain）。注意语义：UBR 图与主 mask 损失共享上游（z′→probs512），retain 的是小体积共享上游（z′ [N,128²]、probs512 [N,512²] ≈ 67MB），UBR 自身激活随每桶 backward 释放。梯度累加数学等价。
- 若 A1+A3+B1 后显存已达标，此项可不启用（保留代码路径+flag）。

**A5 验证分片（对 P5）**
- 照搬远端成熟模式：459 图按 rank 分片推理（FP32、`torch.no_grad`）→ CPU Gloo `gather_object` → rank0 恢复原序后 canonical AP250。val 2.1 → ~0.6 min/epoch。必须验证与单卡 val 逐位同值（远端已证明该模式可行）。

### B 组：模型容量（定义变更，需写入新合同并全臂统一）

**B1 容量下限：32ch × 6 conv 块**（设计规格"32–64ch、6–8 块"的下限；力学是 UBR 的价值来源，容量不是瓶颈）
- 显存/速度再 ~2.4×。
- ⚠️ 约束：**自此以后单/双流全部臂统一 32ch×6**（逐位同构要求）；当前臂作废正好是统一窗口。band_k 不变。

**B2 UBR 块 gradient checkpointing（flag，512 默认关、1024 必开）**
- `torch.utils.checkpoint` 包 `ubr_refiner` 的块；UBR 激存再 ~8×，代价 UBR 前向重算（epoch +~10–20%）。

### C 组：明确不做

patch 降采样/截断（毁 native 主张）；带内稀疏卷积；val 改 AMP（违反 FP32 canonical 契约）；动 DataLoader/RNG 合同；动 ⑤端口/损失形式/带宽 k 语义。

## 3. 预期预算

| 配置 | 峰值显存/卡 | epoch 时长 | 100ep |
|---|---|---|---|
| 当前（实测） | 35–37 GB | 13.2 min | ~22 h |
| A1–A5（执行层） | ~18–22 GB | ~7–8 min | ~12–13 h |
| A 组 + B1(32×6) | **~12–16 GB** | **~5–6 min** | **~9–10 h** |
| 再开 B2（1024 场景） | +~2–3 GB 基座 4× 后仍 ≤20 GB | +10–20% | 1024 可行 |

1024 移植时一并钉死：band_k 与 32px 外扩为像素单位 → 1024 下 k、margin 随分辨率 ×2（或改声明为物理单位，二选一写入合同）。

## 4. 重启程序（EXP-0028 → EXP-0028-R2）

1. **停当前臂**：kill 训练进程树（wrapper bash → torchrun → 4 rank），保留 `launch2.log` 与 `ue_ubr_reseal_20261027/` 整目录**原样归档为作废记录**（含 ep1–13 json/pth，供 v1.1 等价性回放对照用），新输出根换名（如 `ue_ubr_reseal_r2_20261027`）。
2. **实现 A1–A5 + B1 + B2 flag**（预计 ~150–250 LOC，集中在 `ubr_train_integration.py`、`ubr.py`、reseal runner 的 val 路径）。
3. **工程验收门（fail-closed，全部通过才准重启）**：
   - G1 执行等价回放：v1.1 以**旧容量**（关 B1/B2）对同一固定 batch（取自作废臂 ep1 的前 3 个 batch，含 RNG 回放）跑 3 次更新，各 loss 分量与作废臂记录相对误差 < 1e-3（AMP 容差；A5 val 分片另做 459 图全量逐位对齐）。
   - G2 容量切换冒烟：B1 开启后同 batch 单更新，loss 有限、遥测键齐全（UBR/instances、band_nonempty_frac、focal、dice）。
   - G3 显存/速度遥测：smoke 数 epoch 实测峰值 ≤20 GB、epoch ≤7 min（含 B1 时 ≤6 min）。
   - G4 冻结参数 SHA 不变（prompt 栈/PE/主干 LoRA 结构逐位）；`source_hashes` 全量 bump，新合同 `RESEAL_CONTRACT v2` 声明：执行层变更数值中立、容量 32ch×6 为新统一定义。
4. **重启**：沿用原 wrapper（EXP 协议、seed 44、DDP4、审计流程不变），输出根换名；启动后先看 ep1 详细损失（对照作废臂 ep1：loss_ubr=0.0312, focal=0.0877, dice=0.2240 量级——B1 生效后数值会合理偏移，但应同数量级、无 NaN）。

## 5. 监控哨兵（重启后）

- 每_epoch 详细损失行已有；新增每_epoch 峰值显存与 epoch 秒数落盘。
- 复发检测：GPU util 若仍出现 >5% 采样落在 0–20% 区间 → A2 组装向量化不彻底，回头查桶循环。
- 显存爬升（epoch 间 monotonic 增长）→ 碎片/泄漏，考虑 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（若未启用）。

## 6. 工件

- 作废臂：`/data2/wangcheng/remote-lthpc/checkpoint/ue_ubr_reseal_20261027/` + `..._launch2.log`（归档不动）
- 代码锚点：`rsprompter/uav/ubr_train_integration.py`（compute_loss_ubr L220–327、_pad_batch L138、window_around L85）、`rsprompter/uav/ubr.py`、`scripts/probes/ue_ubr_reseal_20261027.py`（runner）、`configs/ablations/ue_ubr_single_512.py`
- 远端 val 分片参照：`scripts/probes/ue_candidate_detail_ddp_20261026.py`（rank 分片 + Gloo gather + canonical AP250）
