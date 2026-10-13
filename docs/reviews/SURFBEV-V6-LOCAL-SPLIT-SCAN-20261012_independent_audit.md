# V6 图内误并 R3：独立审计

**VERDICT：PASS，限于只读候选清单。标签语义及发布门未通过。**

独立复算 M3 manifest 的 train 694、val 100 场，确认扫描 40,784 实例，输出 58 个 M2（28 strong）、25 个窄颈候选。R3 的 `disconnected.jsonl` 与 `neck.jsonl` 与 R2 字节 SHA 相同。25 个窄颈原掩码均恰好一个 8 连通域，R3 代码亦显式要求此条件。

对全部 794 张 train/val RGB 重算 SHA，与 `rgb_inputs.jsonl` 逐项匹配；对未命中池按先场景、后实例的固定 SHA 规则独立排序，前 12 个场景/local ID 与 `nonhit_random.jsonl` 完全一致。脚本在 split 判断后才读取影像；输出无 test/excluded；没有标签写入。

解释限制：58 个 M2 中 43 个触边，28 strong 中 13 个触边。7/10 个基于 UE 全局 ID 的分组只是去重复审图索引，不代表物理楼栋数或可靠空间对应。旧版到新版的 local ID 会变化，同 `(scene, local ID)` 差集不能解释为已修复数。R3 PASS 不代表任何候选可自动拆分，也不代表已构建完整可训练清洗数据集。
