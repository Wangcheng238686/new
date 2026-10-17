# EXP-0020/0021 预注册：WARM-R1C 全量双 seed（2026-10-17 晚）

依据：WARM-R1C 30ep 筛查（`docs/DUAL-WARMSTART-SCREEN-20261017_conclusion.md`
第 7 节）——注意力描述子通路给出全谱系最强 segm（+0.135 [+0.019,+0.254]，
CI>0），相对同配方 R1G 提升 +0.046（臂间不显著但方向一致）；PIXEL-CLS
A-PASS 的臂级兑现。协议逐字沿用 EXP-0018/0019（见
`docs/EXP-0018-0019_preregistration.md` 全部条款，含 DDP4 共租注记、
test 封存条款、seed 一致性条款），唯一替换 = 配方换为 WARM-R1C。

## 臂定义

| EXP-ID | 臂 | seed | GPU | 世界大小 |
| --- | --- | --- | --- | --- |
| EXP-0020 | SURFBEV-warm-r1c-full-DDP4-v2-noaug-cotenant | 44 | 0–3 共租 | DDP4 |
| EXP-0021 | SURFBEV-warm-r1c-full-DDP4-v2-noaug-cotenant | 45 | 0–3 共租 | DDP4 |

- 配置 `configs/ue_whu_a3sgm_512_warm_r1c.py`（身份
  `ue-whu-a3sgm-512-warm-r1c-v1`，与筛查臂同指纹）；模块实现经
  FAIL→修复→增量复核 PASS（reviews 文档追加节，commit `896b07c`）
- warm 初始化 + 冻结（LoRA 精确集 + fusion 135,120 + bev_attn 95,488）；
  100ep 单余弦无重启；batch 1×4ranks×累积 2 = global 8、87 步/epoch
- 数据契约同 0018/0019（冻结快照双哈希）；test 封存至双 seed last

## 四要素（与 0018/0019 相同，仅列差异）

1. 主指标：同（每 seed + 合并池 val 100 逐场配对 vs EXP-0009 rows）。
2. 伴随：(a) on−off 消融；(b) 门遥测 + **注意力健康遥测**（step-0
   delta_p3_ratio 应 ~0.28，末代开度对照 0018/0019 的 0.11-0.12）；
   (c) parity 逐位抽查（卫星侧冻结不变性）。
3. 选择口径：last_model_epoch100.pth。
4. 判定门：DUAL_GAIN / DUAL_PARITY / DUAL_HARM / 中间态，原文同
   0018/0019 预注册；**预注册对比臂**：EXP-0018/0019（通路替换的臂间
   配对检验，segm/bbox 各报）；seed 不一致 → 判读降级条款同。
5. test 开封：双 last 齐后单流+双 seed 各一次（沿用 0018/0019 已开的
   rows_single_test.jsonl 作锚）。

## 预期与落空条款

- 预期位：segm 合并池 CI>0（30ep 已单 seed 过线，余量足）；bbox 合并池
  过线概率中高（30ep 点估计 +0.088，合并收窄 ±0.07）。
- 若 segm 合并池过线且 bbox 含 0：判"通路对 segm 特异"，R1D 并联臂
  （保留栅格化描述子）成为下一变量。
- 若双指标合并池均过线：DUAL_GAIN 幅度条款（≥+0.5）仍大概率不满足——
  预注册此点：幅度以逐项报告呈现，门槛未达不掩方向结论。
