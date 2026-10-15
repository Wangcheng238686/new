# UE 单流/双流首轮训练预注册

日期：2026-10-14。用户明确授权直接启动训练；沿用已确认的通用权重联合初始化，
不走UE单流热启动。正式实现与真实梯度/DDP/严格重建验收已通过。此次是固定当前完整样本的
80轮开发对照，不等待所有生产样本完成，也不把当前清单当最终数据版本或独立test。

## 实验身份与入口

- EXP-0004：dual，先执行；EXP-0005：single，dual成功完成后同一队列顺序执行。
- 两臂commit相同，分别以 `exp/EXP-0004`、`exp/EXP-0005` tag钉死。
- 单一模型入口：`bash scripts/ue_rgbd.sh <arm> train`，队列只调度该入口。
- 运行目录：`/tmp/ue_uav_experiments/20261014_ad_seed42`；队列PID、各臂PID、日志完整文件名和状态在 `run_state.json`。
- 清单：`/tmp/ue_uav_experiments/20261014_ad_seed42/dataset/manifest.json`，SHA256 `424c943596e8f8096fe75ca51b9a2b22a5b12eb38ba68a2ac834ca5fd329d047`。
  628 train / 153 validation；视锥包络隔离，跨集合global建筑ID重叠0。
  后续生产的新样本不动态加入本次训练。共享不可训练几何cache：`/tmp/uav_dual_dev_data/cache`。
- 工件与日志在运行目录的 `<arm>/`；权重、日志、数据清单和绘图不进入git。

## 固定训练协议

4张RTX3090，GPU0/1/2/3；每rank batch1、累积2，常规有效batch8。
628样本 / 4rank = 每rank157 mini-batch，无 sampler 补齐；最后累积组1，实际全局batch4。
每epoch79次更新尝试，80epoch共6320次。两臂均禁用早停以匹配预算；中途代码bug或同步非有限loss累计3次则失败关账，队列不再推进。

512²输入；AdamW LR2e-4、weight decay0.05；卫星/UAV LoRA倍率0.1，CVT/fusion1，
mask decoder0.05；warmup100尝试步，其后沿用原cosine；AMP初始scale512。
基座/UDPR/UAV/CVT/FPN/SAM按既有独立分区max_norm0.5裁剪。
seed42；每epoch完整validation；EMA decay0.999，更新间隔1，epoch5起用EMA评测。
不计算第二次validation loss。保存best-segm、best-bbox、每epoch覆盖的last_checkpoint与最终last。

AMP跳步由新遥测记录；现有scheduler/EMA计数按尝试推进。若发生明显持续跳步，先排查数值，
不把尝试次数冒充Adam实际更新数。运行期间记录训练进程显存；UE渲染保留运行。

## 指标四要素及结论限制

1. 主指标：完整卫星实例 `segm/mAP`（COCO AP100）。伴随：bbox AP100、segm/bbox AP75，
   AP300以 `*_max300` 单列。模型最多输出300，不改变主AP100。
2. 选模：各臂独立best validation segm AP100；主比较这两个best，额外报告最后5轮validation均值和last，
   不用bbox-best替换主选择，也不隐藏bbox退化。
3. 不确定性：本轮单seed、只有validation选模，没有独立test和种子噪声标定。
   仅报告开发方向与完整曲线；不预设未经标定的数值显著线、不宣称显著收益。
   论文/泛化判断需独立空间test、image-paired bootstrap 95% CI，及多seed核对训练波动；
   validation bootstrap不能消除选模偏差，不能冒充test结论。
4. 机制判读：按 [DEBUG_FIELDS](../DEBUG_FIELDS.md#4-ue-uav-双流遥测2026-10-14) 使用支持、门控、残差、
   attention、梯度、AMP和显存。非零注入/梯度只证明消费路径；AP提升仍须A/D独立训练比较。
   正确/zero/shift干预用于后续依赖诊断；容量、深度贡献消融后置，不冒称本轮完成归因。

## 启动命令模板（两臂仅arm与工件目录不同）

```bash
NPROC_PER_NODE=4 CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/ue_rgbd.sh dual train \
  --data-root /tmp/ue_uav_experiments/20261014_ad_seed42/dataset \
  --ue-manifest /tmp/ue_uav_experiments/20261014_ad_seed42/dataset/manifest.json \
  --ue-geometry-cache /tmp/uav_dual_dev_data/cache \
  --checkpoint-dir /tmp/ue_uav_experiments/20261014_ad_seed42/dual/checkpoints \
  --early-stopping-patience 0 --max-nonfinite-batches 3
```

## 启动前审核

既有实现审核见 [独立审核记录](INDEPENDENT_IMPLEMENTATION_REVIEW.md)。
此次遥测补充的完整独立判决与校验记录落盘至 `docs/reviews/UE_TELEMETRY_PRETRAIN_20261014.md`。

## 标签失败后重启（EXP-0006/0007）

EXP-0004在第一轮中因1016 clean掩码ID68未登记而失败，single未启动；日志保留于 `/tmp/ue_uav_experiments/20261014_ad_seed42/failed_dual_pid739553/`。原清单781个执行样本（628+153）的标签审计仅该点失败。新清单排除该点，627 train/153 val，hash `495841b2174242c87661e6b40dba2ddcca13d64b13798116efc8c933c679ca59`，原清单另存不覆盖。两臂从generic重新初始化；仍4卡×1×2、80轮，每rank156、每epoch78尝试（总6240），训练DistributedSampler(drop_last=True)每epoch丢弃3个样本，不补齐；shuffle使被丢弃样本随epoch变化。主指标、seed、LR及选模规则不变。新运行EXP-0006（dual）/EXP-0007（single），新commit/tag独立登记，不复用原实验数字。
