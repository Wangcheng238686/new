# A0 双位置 Oracle 结果

> **计量更正（2026-09-09）**：本页最初 canvas 工件的 `run_manifest.json` 没有
> `dataset.num_classes`。`bootstrap_paired_map.py` 因而回退到单类合同，下面最初记载的
> canvas bootstrap CI **不得作为十类 NWPU 结论引用**；原始十类推理点估计仍保留为定位信息。
> 在相同 A0 checkpoint、同 130 张 image ID、并复用带十类合同的独立几何审计 manifest 后，
> `learned canvas, alpha=current → alpha=1` 的正确十类 500 次 image-paired bootstrap 为
> 0.666929→0.669332，Δ=+0.002403，95% CI [-0.001495,+0.006008]，不通过稳定收益门。
> 因此不能据旧 CI 宣称“全开 dense gate 已被证明有效”；如继续 canvas 路线，必须完成逐实例
> Oracle 与同数量随机 selector 对照。

- Canvas（support-matched GT, alpha=1）：0.666929→0.688200，Δ=+0.021271；这是原始十类
  inference 点估计。其随页原始 100 次 bootstrap CI 属上述失效单类合同，不再引用。
- Tail GT Oracle（无 prompt/token 改动，K/ROI=64/256/1024）：0.709425/0.733393/0.755542；这是 GT 上限，不是可部署结果，尚未导出 tail bootstrap records。
- 结论：优先训练冻结 A0 的 PointRend-style decoder-tail classifier；CanvasRenderer 为次选；P2 不作为首个信息源。
