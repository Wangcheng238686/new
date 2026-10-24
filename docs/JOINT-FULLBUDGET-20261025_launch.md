# 从头联合训练全量预算启动记录（2026-10-25 晨，PI 裁决后）

PI 四项裁决：第三臂不做常量（目标=尽快做出优于单卫星的联合模型）；
超参维持 5e-4/1e-3/0.05/λ0.1；**直接全量预算（不设 10ep 裁决屏）**；
UAV RGB 必须保留叙事（路线=CVT 跨视角聚合增量臂，预注册待写）。

## 启动清单

1. **joint-100ep 全量（DDP2，卡 0+2）**：`ue_joint_geo_sat_ddp_20261025.py`
   （新写：配对数据集把卫星 I/O 移入 worker——单进程脚本的 ~2.6h/ep
   瓶颈；per-rank batch 4×2 = 全局 8 与屏幕臂等效；单次合并反传一次
   DDP 同步；find_unused_parameters 容纳 RSPrompter 条件头；rank0
   验证/存档）。scratch 自 SAM2 ctor、geo common-init seed45、超参按
   裁决口径。冒烟 3.76s/batch → **16.8 分/epoch → 100ep ≈ 28–30h**
   （明晚收线）。gate 双 rank PASS（有限梯度+峰值 14.5GB/卡）。
   ep1 实测 11.2 分钟、loss 0.864。
2. **10ep 屏幕双臂继续（卡 1/3，单进程）**：joint-scratch（ep5，
   loss 0.777）+ sat-scratch 基线（ep3，loss 0.607）。今天下午跑完
   → 让出卡 1/3 → 同 DDP 引擎拉起 **sat-only-100ep 公平基线**
   （`--no-geo`，同预算同协议）。
3. **对照锚**：E73 封版 66.09（强 SAT 前缀）；接口增量 +0.47 复现
   在案；公平从头基线由第 2 步提供。

## 收线判据（全量双臂齐后）
- 双指标净门：joint-100 vs sat-100，逐场配对 bootstrap CI>0；
- 边界指标：B-AP（2%/1%）伴随；
- 内容差：常量几何臂（PI 已裁暂缓——收线后再决定是否补）；
- 收线后条件评估走已验证接口（候选消费）。

## RGB 叙事路线（待预注册）
CVT 式跨视角聚合（非 z-buffer 直投）并入几何头输入通道，几何接口上
10ep 增量探针；直投已否证（+0.049 CI 含 0），CVT 为唯一存活形态。

工件：`checkpoint/ue_joint_scratch100_20261025/`（protocol.json 含
全量预算裁决记录）、`/tmp/joint100.log`。
