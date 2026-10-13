# SURFBEV-V4-POSTSPLIT-20261012 独立审计

## 判定

R1 在写出前因裁边质心漂移未过几何门；R2 的像素拆分虽通过，但 12/17 个新 local ID 撞上同场 `instances.json` 中不可见实例的声明 ID，且一个 child global ID 超出 signed int64，判定 **FAIL**，不得接入 reader。

R3 独立只读复审 **PASS（仅限数据后处理产物契约）**。R3 不意味着训练 reader 已接入，也不证明模型 AP 提升。

## 全量复核证据

- v3 manifest SHA-256：`8d3da5addd302451337954191dccbf1b76f588a0790f1073415e30439015922f`；R3 manifest SHA-256：`49b7f9268db3f767cef4d5cd56265d674da59c54808ca9650140d2d3c5e1884f`，均与实际文件一致。
- 1,200 个唯一场景与 1,200 张 PNG。val 为 17 `SPLIT`、83 `UNCHANGED`；excluded 为 6 `HOLD`、249 `UNCHANGED`；train 694、test 151 全部 `UNCHANGED`。
- 仅原 parent local 的 65,759 个像素改变 ID；原实例前景并集不变，非 parent 像素逐位不变；源 PNG、v3 PNG 哈希一致。
- R3 的 17 个新 local ID 与原始 raster 和同场 `instances.json` 全部已声明 local ID 零冲突。child global ID 为 `4417111715389972016`、`6313126716928380807`，均为正 signed int64，且与 1,200 场原始 global ID 无交集；child 仅出现于 val。
- 独立复算 23 个 parent 场景的门：9 个完整 val 场景与 8 个裁边 val 场景符合 `SPLIT`；6 个 excluded 场景不符合面积/余块门，维持 `HOLD`。完整视图最大中心半径 6.228 m、最小归属优势 67.286 m；无判定不一致。
- `visual_review/full/` 的 23 张前后图覆盖全部 `SPLIT`/`HOLD`；已抽看用户确认样例、裁边样例和 HOLD，图文一致。

## 接线限制

只清理用户确认的 1 个 global ID；其余形态相似候选未被推定为误标。R3 的 `manifest.jsonl` 和 `v4_png` 是派生标签权威；原始 `instances.json` 是来源元数据，不包含 child 映射，不能当作 R3 标签真值。当前 `UERawPairDataset` 仍读取原始 PNG，尚未接入 R3。train/test 标签不变，不能用本次修正主张训练增益或 AP 提升。
