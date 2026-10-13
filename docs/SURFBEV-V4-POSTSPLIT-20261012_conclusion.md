# SURFBEV-V4-POSTSPLIT-20261012：已确认误并实例的后处理

## 结论

R3 数据侧拆分通过独立审计，可作为**候选派生标签版本**。本轮只处理用户视觉确认的 global ID `1874325323`：在 17 个 val 场景把同一 ID 的两栋建筑分成两个实例；6 个无法可靠拆分的 excluded 场景保留原样。其他 1,177 场景不变，train/test 完全不变。没有重渲染，也没有用模型侧 `IGNORE`。

R1 因裁边质心漂移在写标签前停止；R2 因 ID 冲突审计失败并保留为失败产物；R3 修复 local/global ID 接线后重新物化、复审通过。全量 1,200 张派生 PNG、逐场哈希和 child 映射见 `portable_sam2_explicit_coarse/outputs/SURFBEV-V4-POSTSPLIT-20261012-R3/`。前后可视化见其中 `visual_review/index.html`；独立审计见 `docs/reviews/SURFBEV-V4-POSTSPLIT-20261012_independent_audit.md`。

## 使用边界

R3 的 `manifest.jsonl`/`v4_png` 是权威标签；原始 `instances.json` 仍是来源元数据，不能直接解释拆分后的两个 child。当前 B0 reader 仍读取原始 `satellite_instance_id_clean.png`，**尚未切换到 v4，也未启动训练或评测**。如后续要使用，应另立接线和实验注册门，显式读取 R3 manifest 并校验哈希。因这次只改 val 标签，任何前后 AP 差异都是评测真值变化，不能解读成模型学到了新的实例分离能力。

其余 16 个 strong 候选尚未经语义确认，不自动批量拆分；若要扩展到全数据，应先固定可推广的判据并用分层视觉抽样审查，而非把两个连通块一概当成两栋建筑。
