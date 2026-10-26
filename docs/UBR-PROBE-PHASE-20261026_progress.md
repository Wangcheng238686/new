# UBR 探针阶段进展（滚动记录）

## 级1 FREEZE-REF-20261026 — 完成（2026-10-26）
- 双参照归档：`/data2/wangcheng/remote-lthpc/checkpoint/ue_ubr_probe_20261026/frozen_refs/`（SHA256SUMS + manifest.json）。
- EXP-0025 v1best（header ep71，SHA 2fcae2ff…）：val 标准 AP 0.8690 / AP_l 0.9321；边界7px AP 0.6349 / **AP_l 0.5307**。
- E73 = EXP-0017 best_model_epoch73.pth（header ep72/文件名73，SHA dd9106f4…）：val 标准 AP 0.8529 / AP_l 0.9259；边界7px AP 0.6046 / **AP_l 0.5236**。
- 两参照弱点签名一致（边界7px AP_l ≈ 0.52–0.53），UBR 靶点跨封版稳定。
- 评估器 parity：18/18 指标对 20261021 归档逐位复现（n_pred 42664 一致）——标定校验遥测项满足。

## 门0 SNAP-20261026 — 运行中
- snap 探针于 GPU3 运行（向量化后预计 ~1h），结果待判。
- 独立审查：UBR 模块审查 FAIL→2 bug 已修复带回归测试（commit 8b24f69）；snap 脚本审查进行中（发射授权：用户）。
