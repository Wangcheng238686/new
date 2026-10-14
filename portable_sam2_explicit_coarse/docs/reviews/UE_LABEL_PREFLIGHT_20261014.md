# UE 标签失败修正与重启审核

## 失败证据

EXP-0004首次实际更新同步通过后，在第1轮50mini-batch之后因1016clean掩码ID68无映射退出。日志保存 `/tmp/ue_uav_experiments/20261014_ad_seed42/failed_dual_pid739553/train_EXP-0004_pid739553.log`；无epoch checkpoint，未启动single。旧manifest、label_audit.json保留在同目录。生产原件不修改。新实验EXP-0006/0007用新manifest从generic重新初始化。

## 独立审核全文

VERDICT：PASS。此次标签验证抽取与执行清单排除未发现 P1/P2；不代表训练已完成。

审核发现与证据：

- `data/ue_labels.py:10` 保留原映射检查：拒绝重复或非正 local ID，拒绝 clean mask 中无映射的可见 ID；额外要求实例图为二维整数。函数不修改实例图，不将未知 ID 清为背景。
- `data/ue_rgbd_dataset.py:92` 在生成目标前调用共享验证。有效样本仍按原 `np.unique(ids)` 顺序生成每实例 mask、包围框和单类标签，目标构造未改变。
- `tools/prepare_ue_rgbd.py:58` 在读取相机、深度及计算空间范围之前验证标签；异常进入明确的排除记录。
- 独立读取新旧真实 manifest：训练集仅删除 1016，无新增；验证集 153 个 ID 完全一致；新训练集为 627。`split_audit` 完全一致。
- 独立核对空间合同：所有训练样本范围上界严格小于原 cut，所有验证样本范围下界严格大于原 cut；两组 global building ID 集合交集为 0。此次排除保持原空间隔离。
- 独立调用共享 helper 检查 manifest 的全部 1057 个 inventory points：仅 1016 失败，明确报告未知 ID 68。该点已从所有执行 split 排除，其余 inventory points 通过。

边界：

新 manifest 的 `points` 仍保留 1016 的 inventory 记录；实际训练、验证由 `splits` 选择，故不会读取该点。这是保留原空间边界的精确排除清单，不能描述为已重新生成并过滤全部 inventory。

采样协议已按现有代码纠正：627 个训练样本、4 rank、sampler `drop_last=True`，每 rank 156 个样本，每 epoch 78 次累积更新尝试；训练每 epoch 丢弃 3 个样本，验证不丢弃。

## 主代理校验

共享helper全780执行样本通过，1016明确拒绝ID68。Python语法与git diff --check通过。验证报告 `/tmp/ue_uav_experiments/20261014_ad_seed42/label_fix_verified.json`。有效样本mask/box顺序计算不变，双臂空间合同不变。正式重启后继续观察第一轮训练与完整validation，而非仅以首次update判断稳定启动。
