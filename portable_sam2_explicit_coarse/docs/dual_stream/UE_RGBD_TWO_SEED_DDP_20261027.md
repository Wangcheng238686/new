# 原 RGB-D 模型：44/45 两seed四卡全预算复现

用户分配四卡预算，要求两条RGB-D实验共租全部四GPU。本轮每seed为独立torchrun world4任务，
不是每seed仅两卡；允许两个组共租GPU0–3，按实际显存余量准入，不抢占既有任务。

## 固定合同

- 原CompactJoint/InstanceInteraction模型，正确五视角RGB、实测深度、内外参对应；不加Q/G新结构。
- seed44、45，各100epoch/26700次成功AdamW更新，每轮2136/2142图；有效batch8=4卡×batch1×累积2。
- LR5e-4、wd.05、warmup100更新、cosine到.1倍、clip.5、AMP初始scale512，关闭EMA。
- 保留原卫星Gaussian p.5/std.02及ColorJitter；UAV无空间或额外颜色增强。
- 全459val每轮FP32，按combined Mask AP250最高选权，平局保留较早轮；各seed满100轮后固定best推理全459test一次。
- 每轮保存完整不可变epochN.pth，last/best为原子替换的硬链接，E60也有完整独立权重。约两百份770MiB级完整CP，预算约154GiB，源/预测另计。

## 随机性与初始化

仅加载通用SAM2基础权重；不热启动E67/E73或任何遥感任务权重。
每seed从头创建卫星任务头、UAV几何/交互/实例头；UAV源初始化fork seed=2001+seed。
seed44初始卫星摘要与原单卡共同初始化相同，seed45不同。构造期实际torch.load受基础权重白名单限制。

每seed用独立CPU Generator(seed)，连续预生成100个randperm(2142)的前2136个ID。
每全局block8按连续2图分给rank0..3；无padding、重复或跨rank实例遗漏。
rank训练随机流和loader generator初始seed为1000×seed+rank，此后连续保存/恢复。
epoch边界检查点保存四rank各自Python/NumPy/Torch/当前CUDA/DataLoader generator，以及模型/优化器/AMP。

因此这不是原单卡轨迹的逐位复现：DDP归约、任务loss的分布式归一化、rank增强/Dropout随机流和新全局图序均有不同。
本轮只有RGB-D两seed，不能单独认定跨seed净增益；同DDP/同预算纯卫星对照仍是另一个实验需求。
原v1开发暴露测试集、固定rig/单城市分布和仅两训练种子的限制保留。

为复用原Runtime，私有初始化仍保存其父协议镜像（仅覆盖seed）；其中loader/accumulation等旧单卡文字不是本轮执行合同。
外层新protocol明确覆盖四卡分片、每rank累积2和四rank恢复；原Runtime仅用于模型构造、输入、优化器数值及LR调度。

## 文件与运行

|文件|用途|
|---|---|
|`configs/ablations/ue_rgbd_two_seed_ddp_20261027.json`|两seed全预算合同与随机种子规则|
|`scripts/probes/ue_rgbd_two_seed_ddp_20261027.py`|prepare/finalize/gate/train/eval；复用已验证四卡更新/归约/全图评价内核|
|`scripts/probes/ue_rgbd_two_seed_owner_20261027.py`|两组四卡共租调度、完训best测试、独立进程/来源绑定|

根：`/data/wangcheng/checkpoint/ue_rgbd_two_seed_ddp_20261027`，每seed独立子目录。
`prepare`保存两套通用初始化、全100轮图序、实际模型构造导入源码快照及当前环境包版本。
每seed的真实四卡gate涵盖强制AMP回退、4次成功更新、副本逐字节一致、冻结参数、非空moment/五RNG恢复、FP32及强制250掩码。
`finalize`要求所有被记录项目源码与Git HEAD逐字节一致，再绑定两seed工程收据和环境SHA。
正式owner每组按两gate中最大reserved峰值＋1024MiB预留，各卡另留2048MiB；现存组尚未分配的峰值也扣入预算。
无硬编码的单组并发上限，无失败盲重试；异常保留完整日志和CP，需专项审查才能恢复。

```bash
python scripts/probes/ue_rgbd_two_seed_ddp_20261027.py prepare
torchrun --standalone --nnodes=1 --nproc_per_node=4 scripts/probes/ue_rgbd_two_seed_ddp_20261027.py gate --seed 44
torchrun --standalone --nnodes=1 --nproc_per_node=4 scripts/probes/ue_rgbd_two_seed_ddp_20261027.py gate --seed 45
python scripts/probes/ue_rgbd_two_seed_ddp_20261027.py finalize
python scripts/probes/ue_rgbd_two_seed_owner_20261027.py
```

使用cvt2 Python；torchrun显式CUDA_VISIBLE_DEVICES=0,1,2,3。运行/完成状态以外部收据和真实PID为准，不把工程gate当作AP复现成功。
