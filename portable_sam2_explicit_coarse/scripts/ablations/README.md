# a3sgm 训练入口（瘦启动器）

2026-10 a3sgm-only 清理后，本目录只有两个正式入口；模型形态全部在静态配置里
（`configs/a3sgm_*.py`，架构指纹与论文 checkpoint 的训练期契约逐字节一致），
脚本只负责机器环境、数据/调度策略与发射形态。

| 脚本 | 数据/监督契约 | 发射形态 |
|---|---|---|
| `whu1024_a3sgm_roilocal_full_fast.sh` | WHU-1024 全量、roi_local + standard loss、PE 冻结、fast150（150ep、每 ep 验证） | fp32、NPROC×1×accum=有效 8 |
| `vhr10_a3sgm_matrix300.sh` | NWPU VHR-10、full_image + roi_balanced_dice、PE 下采样栈 0.1×LR、matrix300（300ep、每 5ep 验证） | AMP、4×1×2、分片验证 |

## 论文消融臂（P / B / P+B / P+B+M）

matrix300 入口用 `ABLATION_CONFIG` 换 delta 配置（`_base_` 加减键，见
`configs/ablations/`），并自设 `RUN_TAG`：

```bash
ABLATION_CONFIG=configs/ablations/vhr10_point_only.py RUN_TAG=vhr10_matrix300_p \
  bash vhr10_a3sgm_matrix300.sh
```

## 续跑

`RESUME_OWN=<last_checkpoint.pth> bash <入口>.sh`（透传为 `--resume-from`）。

历史 arm 字典（p/b/pb/a0/a1-a5/r3/ksweep 等 22 臂的 env 管线）已随清理删除；
其结果记录见 `docs/evidence/box_prompt_dependence_vs_contribution.md`、
`docs/evidence/p2v2_matrix300_four_arm_machine2_results.md` 与两张顶层追踪表，
复现历史臂请回退到清理前提交 add7482。
