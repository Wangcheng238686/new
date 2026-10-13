# SURFBEV-MASK-QA-20261012-R1 独立审计

**VERDICT: PASS（仅限分层抽样与人工复核包的首次定版）。** 不代表任何标签已由人工
确认错误，也不代表清洗标准已被人工证实或数据集已完成清理。

审计人独立只读复核了协议、脚本、R1 产物；未参与抽样设计或实施。初版报告曾指出：
单套意见栏不能执行双人盲审，且各层短缺原因未写入结果。实施者据此补稳定
`sample_id`、独立 A/B 表、裁决表、预选双审子集与逐层去重诊断；关键分支和产物
再次复核后给出上述 PASS。

## 复核事实

- 512² 原生 ID raster 的 7 层互斥阈值、优先级和协议一致；脚本在 `test`/`excluded`
  分支 `continue` 后才读取 ID PNG 和 `instances.json`，test 仅解析 manifest/split
  元数据，RGB/ID/实例内容不参与标准制定。
- R1 的 `sample_manifest.csv`、`reviewer_A.csv`、`reviewer_B_blind.csv`、
  `adjudication.csv` 分别有 113/113/23/23 行。`sample_id` 各表唯一；B 与裁决表的
  ID 集合恰等于 `summary.json` 预先列出的 23 个双审 ID；所有 ID 均属于 113 例
  样本。`sample_id=SHA256(scene:local:global)[:16]` 逐行复算一致。
- 113 例分别来自 113 个不同 scene/global ID；train/val 为 82/31，无 test/excluded。
  A/B 意见栏及裁决栏为空，未预填任何人工结论。
- 126 目标仅取得 113 的缺口已显式归因：train `sparse` 可选独立 global ID 10，
  少 2；val `thin` 互斥层人口 0，少 6；val `sparse` 可选独立 global ID 1，少 5。
  val 的细长 mask 存在于更高优先级的微小/小面积层，`val/thin=0` 不等于无细长建筑。
- 协议明确：形态标记只是复核线索，不能自动删除；富集样本不能直接估总体错标率；
  双审与冲突裁决后才能冻结清洗规则。

## 边界与后续

本次未验证任何人工判读质量。所选 ID PNG 的 SHA256 已逐样本记录；RGB 与
`instances.json` 当前只记录源路径，未逐文件哈希钉死，若源 payload 可能变动，
复核前应补 provenance。该点不阻断本次抽样，但限制长期复现性。不得以本次 PASS
宣称数据标签已经清洁，也不得据此触发重训或打开封存 test。

工件：

- 协议：`docs/SURFBEV-MASK-QA-20261012_protocol.md`
- 实现：`portable_sam2_explicit_coarse/scripts/ue/sample_surfacebev_mask_qa.py`
- 输出：`portable_sam2_explicit_coarse/outputs/SURFBEV-MASK-QA-20261012-R1/`
