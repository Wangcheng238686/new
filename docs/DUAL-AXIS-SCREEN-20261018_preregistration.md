# DUAL-AXIS-SCREEN 预注册：R1CH 统计分箱加宽 + R1Q 注意力修复（2026-10-18 凌晨）

依据 HANDOFF（2026-10-18）与 `docs/DUAL-WARMSTART-SCREEN-20261017_conclusion.md`
第 7–8 节。两条轴不混臂、各自 30ep 热启动筛查（协议同 warm 链：warm 冻结、
LoRA 精确集、单余弦、DDP4 batch1×累积2、fusion 免 wd、seed 44、last 口径、
val 100 逐场配对 bootstrap 10k/seed 20261014、test 不开封）。

## 臂 A（方向一）：WARM-R1CH（已实现，只欠冒烟）

- 配置 `configs/ue_whu_a3sgm_512_warm_r1ch.py`：R1C + height_edges_cfg
  4 分箱 [-1e9, 8m, 16m, 32m, 1e9]，统计 6→24ch（bin-major），BEV=152。
  修复史 fdb39a3/5dd9d9b。
- **判据（handoff 原文）**：对 R1C 30ep（segm +0.135 [+0.019,+0.254]）的
  臂间配对比较——(i) 双指标点估计不低于 R1C（臂间 Δ ≥ 0）；(ii) 过筛查门
  （on−off 双指标 CI 下界 >0 且 on−single segm 点估计 >0）；(iii) 机制预期：
  阶梯活统计边际（R1C 的 C2−C4 = +0.088）应增长（筛查后跑 C2/C4 记录）。
- **mAP_s 防护门（全臂适用）**：逐场 segm_s（COCO AP_small）配对，
  vs R1C 回退 ≤ 1.0 点。

## 臂 B（方向二）：WARM-R1Q（新臂，本预注册先于编码）

R1C 配方 + 两处同改（缺一都有失败证据：纯几何查询=模板解 C0−C2=0；
无监督的注意力输出不被消费）：

1. **查询内容化**：BEVDescAttention 查询 = cat([几何嵌入 q_geom(xy),
   sat_proj(卫星 P2 该格特征)])。卫星特征取注入前的冻结基座输出
   （extract_feat 里 x[t]，requires_grad False，显式 detach，无梯度回流），
   下采样到 64×64 注意力网格，加 Linear(512→geom_dim) 投影（可训）。
   q_proj 输入维 geom_dim→2×geom_dim。几何接线（坐标框架/符号/缩放）
   过与 R1C 同级的实现审计。
2. **逐场景辅助监督**：复用 R1B bev_head 机器（hidden 64，weight 0.05），
   **输入锁定为注意力输出图**（128ch@64×64 上采样至注入分辨率；配置
   bev_head_source='attn'，默认 'bev' 保持 R1B 位不变），目标为本场景
   GT 建筑并集足迹，损失 BCE + Dice（等权求和后乘 bev_aux_weight 0.05；R1B 纯 BCE
   保留为历史臂行为）。不加任何附加件（方差正则/温度/对比项禁用）。

- 配置 `configs/ue_whu_a3sgm_512_warm_r1q.py` = R1C + 上两键；
  身份 `ue-whu-a3sgm-512-warm-r1q-v1`。
- **双重验收门**：
  - 筛查门：同臂 A (ii)；
  - **机制门**：阶梯 C0−C2（注意力输出换 train 均值图）的 segm
    CI95 下界 > 0——注意力输出必须显著掉点才算修复成功；机制门不过
    则不采信为成功（判"更好的模板"），按 handoff 硬砍条款处理。
- mAP_s 防护门同臂 A。

## 共同条款

- 评估行新增逐尺寸字段（bbox_s/segm_s 等，向后兼容），防护门用
  segm_s；
- 收线时 0020/0021 的正式判定与全量阶梯复核先于两臂判读；
- 方向三（EXP-0022 起发号）按 handoff 决策树：两臂过门→A+B 组合筛查；
  仅 A→A 全量；B 机制门败→硬砍描述子通路（纯统计 BEV 全量）；双败→
  60ep 余弦+40ep 5e-5 平台续训试验。

## 工件

- R1CH/R1Q 筛查：`launcher_logs/WARM-R1C{H,Q}-ddp4.log`、
  `probes/WARM-SCREEN-20261017/rows_warm_r1c{h,q}_*.jsonl`
- 全量阶梯复核：`rows_warm_r1c_full_s4{4,5}_c{1,2,4}.jsonl`

---

## 落判记录（2026-10-18 09:50）

### 臂 A（WARM-R1CH）：筛查门 **FAIL**，方向一关闭

| 对比 | segm | bbox |
| --- | --- | --- |
| vs EXP-0009 | −0.015 [−0.115,+0.088] | +0.056 [−0.046,+0.165] |
| 臂间 vs R1C@30 | **−0.151 [−0.266,−0.035]**（显著劣化） | −0.032 [−0.133,+0.067] |

- 预注册判据 (i) 不满足（segm 点估计大幅低于 R1C）、(ii) 筛查门不过
  （segm CI 含 0 且点估计负）。多高度分箱把 6ch 统计扩到 24ch 反而
  摊薄/噪声化了 30ep 制度下唯一有效的统计内容——与全量阶梯"活统计
  边际也在消失"的读数方向一致。
- mAP_s 防护：R1C@30 rows 生成于 per-size 补丁前，无 _s 可配对——
  登记为口径缺口，防护由臂间 segm 主指标承担（已显著劣化，无需
  _s 佐证）。后续臂的 rows 均带 _s 字段。
- 决策树状态：A 败。B（R1Q）成为唯一活轴；若 B 双门过 → B 单变量
  全量（EXP-0022）；若 B 机制门败 → 硬砍描述子通路；双败 → 调度假设
  臂。R1Q 训练中（ep3/30，aux 0.057 正常）。
