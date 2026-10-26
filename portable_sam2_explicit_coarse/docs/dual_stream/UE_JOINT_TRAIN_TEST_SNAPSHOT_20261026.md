# 四臂 val-best 权重的 train/test 补充评价

## 用户要求与评价边界

2026-10-26 用户要求四臂在训练集和测试集上评价，按原验证集选出的权重，在 GPU2 排队。
使用各臂当前的 best_model 副本，不等所有臂100轮结束；这是用户明确授权的补充评价。
原100轮训练、val选权、完训test及19项收尾的数值代码和产物不改。

快照建立后不再追随 best_model 更新。每臂记录选中epoch、验证Mask AP、当时已完成预算、
checkpoint SHA、训练合同和选中epoch记录SHA。四臂预算可能不同，不能标成同预算最终比较。
训练集指标是已见样本上的拟合诊断，不能替代测试泛化；单训练种子的同图bootstrap也不能
证明训练重复性。不得按本轮test结果更换结构、选权或训练超参数。

## 实现与产物

- 脚本：`scripts/analysis/ue_joint_snapshot_eval_20261026.py`。
- 运行根：`/data/wangcheng/checkpoint/ue_joint_train_test_snapshot_20261026`。
- `prepare`：稳定复制四份best；检测训练并发写入，核对CP内epoch/合同/best、逐轮val的最高分及更早平局规则。
- `eval`：复用原Runtime与core.inference_outputs；训练集2142张、测试集459张，均关闭增强、FP32、eval/inference_mode、maxDets250。
- 每图保存合并预测和original_path的框、分数、RLE；25图写一次独立进度文件，不新增逐步训练日志。
- 评价前后核对参数与持久buffer摘要、无参数梯度、optimizer状态为空，保存实际导入项目源码及SHA。
- `owner`：8项GPU2串行评价，先四臂test，再四臂train；显存free至少8192MiB才启动（6144工作预算＋2048余量），不抢占共租任务，不自动重试失败。
- `analyze`：各split齐备后独立CPU任务，与后续GPU评价可并行。报告合并Mask/bbox/Boundary AP2%与1%、原路径Mask/bbox；合并Mask AP做500次同图bootstrap，seed20261021，首3次对冻结原COCO累积校准。
- 比较正确RGB-D/常量RGB/参考面各自相对SAT，以及正确RGB-D相对两项内容对照。所有图像ID和GT须相同；Mask/bbox重算须复现评价点值。
- `manifest.json`绑定训练源28份、补充评价和统计源、Git HEAD、固定权重与预算；`evaluations/{arm}/{split}`保存预测/指标/完成收据；`analysis/{split}.json`保存最终配对结果。

这不是复跑训练，也不是E60检查点评价。E60无独立CP，不能把当前best的结果写成E60。
完整结果须等8项评价和两个统计收据，owner进程存在或有非零梯度不代表收益成立。

## 命令

在 `portable_sam2_explicit_coarse` 目录，使用 `/data/wangcheng/envs/cvt2/bin/python`：

```bash
python scripts/analysis/ue_joint_snapshot_eval_20261026.py prepare --root /data/wangcheng/checkpoint/ue_joint_train_test_snapshot_20261026
python scripts/analysis/ue_joint_snapshot_eval_20261026.py owner --root /data/wangcheng/checkpoint/ue_joint_train_test_snapshot_20261026
```

`owner`为子进程显式设置CUDA_VISIBLE_DEVICES=2；运行时Torch设备0对应物理GPU2。
准备、单图验收与正式队列状态以外部运行目录实际收据为准，不在本文预写完成。
