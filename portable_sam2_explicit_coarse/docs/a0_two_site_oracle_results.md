# A0 双位置 Oracle 结果

- Canvas（support-matched GT, alpha=1）：0.666929→0.688200，Δ=+0.021271；130 图、100 次 paired bootstrap CI [+0.013998,+0.026354]。
- Tail GT Oracle（无 prompt/token 改动，K/ROI=64/256/1024）：0.709425/0.733393/0.755542；这是 GT 上限，不是可部署结果，尚未导出 tail bootstrap records。
- 结论：优先训练冻结 A0 的 PointRend-style decoder-tail classifier；CanvasRenderer 为次选；P2 不作为首个信息源。
