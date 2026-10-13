# SURFBEV-DISCONNECT-QA-20261012 独立审计

## VERDICT

**R3 PASS：形态计数和抽样实现可信。** 语义上的“两栋误合并”仍不是自动计数结果，不得把形态候选直接说成确认错标。

## 独立复算

审计员只读复算全部 794 个 train/val 场景、40,745 个 v3 实例：train 任意非连通 1,364、M2 58、strong 27；val 分别为 230、30、25。独立得到的 88 个 M2 scene-local 候选集合与 R3 `candidates.csv` 完全一致；52 个 strong 不变。所有 794 场的来源/v3 SHA 与 manifest 一致，train/val 的 global ID 无交叉；已知正例 `19064360d2c1ae7c`（local 16、global 1874325323）落在 strong，第二大块 4,642 px、占 49.9247%、间距约 32.98 px。

R3 候选表与初版逐字节一致，说明 R1/R2→R3 仅修抽样，不改事先固定的候选定义/全量计数。R3 的 32 条可视化样本对应 32 个不同 global ID，全部 train/val、全部来自候选，已知正例的**整个** global ID 被排除；固定种子重放顺序及每个 global ID 的代表场景选择完全一致，32 张图与 CSV 对齐。17 个 strong global ID 排除已知正例后，余下 16 个全部入样；19 个 weak global ID 中抽到 16 个。

## 审计过程中的修正

- 初版抽样跨 strong/weak 重复 4 个 global ID，且已知正例的其他场景混入随机样本：形态计数 PASS，语义比例估计 FAIL。
- R2 消除重复和正例混入，但先抽 scene-local 行再去重，会偏向在更多场景出现的 global ID，且一个 global ID 可跨层：仍不得用于比例估计。
- R3 先按 global ID 定层、选代表场景、再均匀抽 global ID，通过复核。R1/R2 原产物保留作审计轨迹，不覆盖。

非阻断建议：脚本未来可增加已知正例 metadata global ID 等于硬编码 global ID 的断言。此次产物已经由独立复算验证对应关系。
