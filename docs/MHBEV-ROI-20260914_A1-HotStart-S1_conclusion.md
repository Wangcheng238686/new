# MHBEV-ROI-20260914 — A1 HotStart S1 结论（负结果）

**状态：FAIL / 关闭 A1；不得进入 A2 或全量训练。**

## 预注册口径与工件

- 路由：模块叠加型 A1 热启动筛查；仅训练 `a1_p2.{proj,gamma}`。
- 锚点：`ue_single_c5v2_control_tr128_va259_s44_r1/best_model_epoch86.pth`。
- 数据及选择口径：UE val 全部 259 scenes / 5,937 instances；两个 seed 的
  `last_checkpoint.pth` 做标准推理。
- 主指标：`segm/mAP_s`；伴随：`bbox/mAP_s`、`segm/mAP`、`bbox/mAP`。
- 筛查门：两个 seed 的最后 5 次验证均值须有
  `Δsegm/mAP_s >= +0.015`、`Δbbox/mAP_s >= 0`、两个全局 mAP 均不低于
  -0.002，且两 seed 方向一致。

## 正式 last 推理结果

锚点为 segm/bbox `mAP_s = 0.0809998 / 0.0901865`，segm/bbox `mAP =
0.731209857 / 0.720871204`。

| seed | Δ segm/mAP_s | Δ bbox/mAP_s | Δ segm/mAP | Δ bbox/mAP |
|---:|---:|---:|---:|---:|
| 44 | +0.001097 | +0.000589 | -0.000163 | -0.000090 |
| 45 | -0.001243 | -0.004432 | -0.000881 | +0.000414 |
| 均值 | **-0.000073** | **-0.001922** | -0.000522 | +0.000162 |

两个 seed 的 `segm/mAP_s` 方向相反，且均值不但远低于 `+0.015` 门槛，还略为
负值；`bbox/mAP_s` 均值也为负。因此尽管全局 mAP 守门未失守，A1 筛查门失败。

checkpoint 内的最后 5 次验证也一致落判：seed 44 的 `Δsegm/mAP_s` 为
`+0.001104`、`Δbbox/mAP_s` 为 `+0.000598`；seed 45 为 `-0.001395`、
`-0.004389`。这既不满足主门，也不满足双 seed 同方向约束。

## 落判与归档

1. 关闭 A1 方向的本轮后续工作；不实现或运行 A2/A3，也不进行 A1 全量训练。
2. 此结论不能解释为 UAV 无信息，只能说明：在冻结单流检测器的条件下，当前
   P2、z=0 几何 residual 没有产生稳定且足够大的小目标增益。
3. 若未来重启，应另立项并重新经过冻结探针；候选假设是让 UAV 信号更直接服务
   小 proposal recall/ROI separation，而不是放宽本次门槛或挑选 best checkpoint。

工件：

- seed 44 checkpoint：`/data1/wangcheng/checkpoint/worktrees_dual-stream-dev_work_dirs/ue/a1_p2_z0_hotstart_r2_s44_tr128_va259/last_checkpoint.pth`
- seed 45 checkpoint：`/data1/wangcheng/checkpoint/worktrees_dual-stream-dev_work_dirs/ue/a1_p2_z0_hotstart_r2_s45_tr128_va259/last_checkpoint.pth`
- 正式推理：`portable_sam2_explicit_coarse/outputs/MHBEV-ROI-20260914/A1-HotStart-S1/s44_last_standard/` 与 `s45_last_standard/`
- 独立结果复核：`docs/reviews/MHBEV-ROI-20260914_A1_hotstart_review.md`
