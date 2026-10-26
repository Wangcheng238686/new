# Q/G四卡DDP实施与复现

本文件取代本轮单卡启动安排。用户明确要求每个实验四卡训练、余量足够时多实验共租、不够则排队，并做好Git管理。已停止只有早期更新的四个单卡新进程及其owner；原源码、日志、状态、初始门保留在`ue_candidate_detail_20261026`，`ddp_transition_requested.json`和`stopped_for_ddp.json`记录原因。不将其续接成DDP轨迹。此前原100轮四臂属于已启动的既定合同，未改其源码或中途转移。

## 结构与公平对照

Q/G模型与几何单位沿用[机制合同](UE_CANDIDATE_DETAIL_20261026.md)，不再调整。新训练初始化仍为共同通用SAM2/new heads，绝不从E73或其它已训练任务权重热启动。

新增四卡运行器使用`torchrun --standalone --nproc_per_node=4`和DDP。每卡batch1/累积2，全局有效batch8，每轮2136唯一图/267成功更新，10轮2670。每8个全局训练ID按连续2个分给每rank，重组后逐值核对既定顺序。rank数据种子44+rank，worker/generator固定规则；不同结构/内容条件共享同一四卡合同。

DDP原生任务归一化和随机流不保证等价于单卡，因此本轮重跑四卡原双流正确/常量RGB以及纯卫星基线。矩阵为Q/G/QG各正确/常量6条、QG参考面1条、原双流正确/常量2条、纯SAT1条，合计10条。单卡旧E10只保留历史参考，不充当四卡主比较。全组LR仍用原100轮日程前2670步、无EMA、原增强/损失/UDPR/coarse64/NMS/打分和250上限不变。

## 分布式训练、评价与恢复

- 前一个microbatch用DDP.no_sync，第二个同步；loss除本rank累积2，由DDP平均4rank梯度。
- 任一rank前向非有限则共同停止；梯度溢出必须所有rank一致，恢复相同两张已准备图、buffer和rank RNG并降低scale，最多重试4次；不跳过成功更新预算。
- checkpoint保存unwrapped模型、非空optimizer/scaler以及4rank各自的Python/NumPy/Torch CPU/当前CUDA/DataLoader generator。只在epoch边界恢复；每rank只初始化自己的CUDA RNG，避免get_rng_state_all额外占用其它卡上下文。
- 每轮val按rank分片，Gloo收集CPU预测，rank0恢复原459图顺序后按FP32 canonical AP250评价；不平均四份AP。固定E10，不按val/test挑轮次。
- 全部10条训练完成后，所有锁定臂完整test；QG正确权重再做常量/donor RGB。原路径/合并路径、配对bootstrap及困难分布分析沿用前述合同；比较只使用相同四卡执行设置。

四卡门覆盖Q/G/QG、原双流和纯SAT五种路径：不同rank不同密度样本、强制同步AMP backoff及同输入loss精确回放、3次真实更新后的参数字节SHA全rank一致、冻结参数不变、FP32分片推理、每rank强制250实例解码、故意破坏模型/moment/generator后的恢复、全部五类RNG逐值恢复，再实际完成第4次更新。首个门在保存时局部变量与RNG函数同名导致停止，失败源码/日志保留`engineering_debug/attempt1`；修复后重跑，不将失败标成通过。

## 四卡实验组共租

每个训练/推理任务均启动4rank，占用GPU0–3。调度器按五种实际门的最大reserved加1024MiB作为每rank工作预留，设备另留2048MiB。新组只有四张卡都满足预算才准入；启动尚未分配显存的rank也预扣预算。没有固定并发组数限制；显存不足时留在队列，不停原任务。不同组使用独立随机rendezvous端口。失败停止新增调度，保留其它健康组执行。

## Git与产物

新增源文件逐项：

| 文件 | 职责 |
|---|---|
| `configs/ablations/ue_candidate_detail_ddp_protocol.json` | 四卡矩阵、batch/预算、随机流、评价/恢复合同 |
| `scripts/probes/ue_candidate_detail_ddp_20261026.py` | 四rank真实训练/验收、分片评价、rank RNG恢复 |
| `scripts/probes/ue_candidate_detail_ddp_owner_20261026.py` | 四卡组资源准入、排队/共租、完整实验DAG |
| `scripts/probes/ue_candidate_detail_ddp_analysis_20261026.py` | 只比较四卡E10结果，配对效应与难例指标 |
| `scripts/probes/ue_candidate_detail_repro_20261026.py` | 实际本地import清单、Git blob逐字节匹配、环境/初始化/数据/顺序身份 |

根目录`/data/wangcheng/checkpoint/ue_candidate_detail_ddp_20261026`。源码提交到当前`codex/uav-bev-dual`分支；只纳入本实验和实际依赖的源/配置/文档，不使用git add .，不提交checkpoint/日志/缓存或其它未关联修改。`reproducibility.json`绑定实际Git commit，要求全部加载的本地Python文件和声明数值源字节等于该commit；未关联dirty文件仅列出而不暗中纳入。

数据/预训练权重不进Git。归档需同时保留：

1. 原v1数据、实测native/anchor缓存、独立参考面缓存及各自checksum manifests；路径和SHA由contract/repro记录。
2. 原通用SAM2 foundation；`reproduction_inputs`另复制共同初始化文件/JSON和10轮训练ID顺序JSON，并核对来源SHA。初始化文件不是任务热启动权重。
3. `environment/pip_freeze.txt`、GPU型号/UUID/driver、Python及torch/mmcv/mmengine/mmdet/peft版本；compiled CUDA扩展需使用相容环境，不承诺仅pip freeze即可重建轮子。
4. frozen_source、manifest、job_plan、rank_launches、逐轮sample_order、各checkpoint及评价/分析产物。

复现时先checkout `reproducibility.json.git_commit`，还原上述输入及路径；新输出根目录必须与旧产物分离。先运行五种`gate`（均四卡），再运行`repro prepare`，最后启动owner。例如：

```bash
python -m torch.distributed.run --standalone --nproc_per_node=4 scripts/probes/ue_candidate_detail_ddp_20261026.py gate --variant both --root /data/wangcheng/checkpoint/NEW_RUN
python scripts/probes/ue_candidate_detail_repro_20261026.py prepare --root /data/wangcheng/checkpoint/NEW_RUN
python -u scripts/probes/ue_candidate_detail_ddp_owner_20261026.py --root /data/wangcheng/checkpoint/NEW_RUN
```

首条示例需分别替换variant为query/detail/both/baseline/satellite完成五门。显式续训同样用torchrun4，`train --variant ... --content ... --resume .../last.pth --root ...`；控制面失败需先审查原state/退出码，不能重复启动owner或盲目重跑覆盖。

工程门/首批更新不证明输入内容净收益。科学目标仍待完整10条固定实验、全部test/内容对照/困难分组及后续全预算/成本复核。

## 2026-10-26 独立审查补充

审查报告与门记录归档在运行根 `review/independent_review.md`、`review/froc_correction_gate.json`。未发现必须停训的确认DDP错误；93份源与提交逐字节一致，两组各4个真实rank和五类工程门已独立核定。

确认离线指标问题：原冻结分析的FROC逐检测前缀允许拆开同分组。新增独立 `scripts/analysis/ue_candidate_detail_froc_ties_20261026.py`，按完整同分组端点计算合法score阈值，并允许空预测。它沿用原贪心匹配，仅替代FROC，不改AP、计数、训练数值源或原产物。独立watcher命令：

```bash
python -u scripts/analysis/ue_candidate_detail_froc_ties_20261026.py --watch-root /data/wangcheng/checkpoint/ue_candidate_detail_ddp_20261026 --owner-pid 904665
```

权威修正产物在 `analysis_corrected_froc/{val,test}`，覆盖10个匹配评价及QG正确训练权重的constant/donor干预，共24项；绑定预测、身份、checkpoint和修正源SHA。同分反例、100组无tie等价及真实旧val459图的图序反转一致性已核对。修正器作为后续独立提交，不改训练提交c9b6d60。

结论限制：G是保留旧CNN的增量旁路，不能据净AP声称旧冗余已解除；Q仍限8候选/32维，常量RGB可含位置特征，必须靠匹配内容差及差中差判断真实内容价值。中心格碰撞、3×3峰值抑制、NMS及新增候选未二次分类保持原样。10epoch单seed仅筛查机制，正式化仍需全预算、重复性、成本及逐GT净纠错验收。恢复门验证状态恢复和后续更新，但尚未验证真实epoch连续路径与恢复路径逐值等价；首次实际续训前需完成该专项，不打断健康任务。
