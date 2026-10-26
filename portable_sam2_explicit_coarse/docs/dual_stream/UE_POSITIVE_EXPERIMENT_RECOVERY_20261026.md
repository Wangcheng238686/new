# 两条正向实验线的恢复与归档

## 实验身份

1. `ue_instance_interaction_20261024`：固定E73、实例监督内部交互，14臂固定E10；正向RGB internal test Mask净差+0.5240pp，正确训练对常量+0.1937pp。原27份源码快照、三个初始化、14个E10权重、预测及统计仍存在。仅冻结基座开发结论，不是共同初始化完整联合训练。
2. `ue_compact_joint_20261024`：通用SAM2共同初始化的四臂100epoch联合训练，E60 val RGB-D86.5553、SAT85.5318、常量86.1068、参考面83.7995。原29份源码快照、共同初始化和四臂E50等固定权重仍存在；E60无独立权重，不能直接加载重评。该实验尚未全部完成，E60不是最终结果。

## 恢复实现

`scripts/analysis/ue_positive_experiment_recovery_20261026.py`仅CPU文件审计/复制，不导入模型、不启动训练、不改变运行源或原数据。它逐值核对启动声明SHA、原source快照与当前代码，复制到全新归档目录，复制后再次核对源/目标SHA；源在复制期间变化则报错。保存两组源码、小型身份文件、实例14臂E10与3初始化、冻结E73、关键正确/常量test预测和统计、四臂E50与共同初始化及已落盘epoch记录。联合rolling last/best、进行中预测不复制。

```bash
/data/wangcheng/envs/cvt2/bin/python scripts/analysis/ue_positive_experiment_recovery_20261026.py --output /data/wangcheng/checkpoint/ue_positive_recovery_20261026
```

输出`recovery.json`绑定每份资产SHA、当前代码提交及外部数据身份；数据/原始缓存/通用foundation不入Git、不完整复制。当前pip freeze仅为归档时环境，不能倒推启动时环境。归档位于同一磁盘，不是异地备份。完整训练逐位复跑、干净检出复跑及模型重新评价尚未执行。

## Git与复用

原数值代码多数已在c9b6d60按实际加载依赖补录；本次补齐原快照中尚未跟踪的owner、验收、统计与协议文件，保持每一份字节不变。无需退回f0b20e9，也不将当前Q/G当作历史正向模型。归档后以`recovery.json.git_commit`检出，核对外部输入，再在新目录做复评/复跑，原目录保持隔离。

实例模型可用原runner `eval --family rgb --route internal --content geometry --rgb correct --seed 44 --split test --input geometry --checkpoint 原epoch10.pth --out 新评价目录`重新评价；需同时保留原root的初始化、原E73路径、数据和缓存及通用权重。新训练须遵守当前每实验四卡要求；原单卡合同的精确重跑与四卡新对照是不同实验，本次不启动任一种训练。联合E50可核对来源/恢复状态，若要重新继续训练先做真实恢复连续轨迹专项，不能将E50当E60。
