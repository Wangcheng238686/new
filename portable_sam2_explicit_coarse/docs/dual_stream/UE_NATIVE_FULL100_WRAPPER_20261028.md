# Native A/B/C 全100轮排队合同

用户授权排队A/B/C全量100轮，后续全预算任务必须一键wrapper并隔离不同臂。统一入口为`scripts/experiments/ue_native_full100.sh`，操作说明见同目录README。

## 文件和责任

| 文件 | 实际职责 |
|---|---|
| `scripts/experiments/ue_native_full100.sh` | Bash安全路径/解释器/线程设置，一条命令转控制器 |
| `scripts/experiments/ue_native_full100_control.py` | CPU准备、固定Git运行工作树、持久队列、依赖/资源准入、重复启动/恢复审计、单臂入口 |
| `scripts/probes/ue_ubr_native_full_run_20261028.py` | 固定100轮四卡训练、100轮共同orders、五RNG保存/恢复、验证最佳/测试、arm锁/rank清单 |
| `configs/ablations/ue_g_ubr_native_full_protocol_20261028.json` | 固定预算、初始化、优化器、数据/监督/选权与评价合同 |
| `scripts/experiments/README.md` | 本次操作及后续所有全预算wrapper必须遵守的规范 |

## 模型和数值合同

A无native头；B同头但尾部RGB和geometry归零；C正确尾部证据。上游全部G与正确RGB-D，B不是纯卫星。原模型接线保持；新头FP32计算岛，其他AMP，512主mask/coarse64与实例头loss保持，无UDPR/全图UBR/新边界loss。

共同generic common_init、Gseed2060/head2080；每臂从E1完整到E100，不能把旧E10任务权重作为初始化。4rank×batch1×accum2=8，每轮2136张/267更新，总26700；LR.0005/wd.05/warmup100/100轮cosine→.1/clip.5、seed44、AMP512、无EMA、maxdet250。每轮完整459val，按segm/mAP strictgreater选best，平局较早。所有三训练完成后各自best跑完整459test，不据test选权/改预算。

100轮样本次序在prepare阶段一次冻结：训练2142个ds.samples原顺序，torch.randperm(2142, manual_seed(44+epoch))[:2136]，每轮同一global8块每rank连续2张。新100轮顺序合同与旧短探针不同，因此不是E10状态无损延长；三臂始终共同同序。manifest与101份顺序文件绑定SHA，重复准备核字节一致，拒覆盖/路径逃逸。

## 隔离与启动

独立root=`/data/wangcheng/checkpoint/ue_g_ubr_native_full100_20261028`，A_nohead/B_zero/C_correct各自contract/last/best_model/保留epoch/验证记录/进程锁，检查点严格含arm/source/data/order/world/phase。原子last后best复制中断可从last的best状态修复。训练后验证阶段保存Python/NumPy/Torch CPU/当前CUDA/DataLoader generator五类rank状态；gate是epoch0合成恢复机制，不冒称实际验证或next-update等价。

prepare创建固定提交的独立Git工作树root/runtime_source，完整import来源和wrapper/配置均核commitblob，禁止在运行树改动。入口优先加载该工作树的SAM2并断言实际sam2模块路径，防editable install逃逸到主开发树。主开发目录后续新代码不改变在训来源。

queue持久owner等待当前analysis_native_20261028三份完成标记（所有配对/固定框/干预完成），然后新full恢复门、三臂train、最后serial eval。等待不读取指标作预算决策。每臂四卡，显存够可3组共租；资源预留A5988/B8596/C8890MiB，cap5600/8200/8500MiB，每卡额外512MiB。实时扣减未兑现预留。当前工程峰只提供准入依据，不保证每一场景内存上界。

活owner重复queue幂等。dead owner需显式recover，核launcher与rank starttick，活进程拒二次启动；已完成任务核固定100轮与权重SHA后跳过，未完成训练只能恢复自己的last。gate/eval半成品保留，需人工审计并显式归档；不静默覆盖。manual train/resume/eval也核前置、同root恢复门、所有四卡余量与同臂锁。

## 工程范围

100轮入口独立新增，原短研究/归因源码不改。CPU顺序/拒覆盖/协议/CLI、Git重定位及来源核验与真实排队状态需验收。GPU恢复门在前置完成后执行，排队成功不冒充100轮训练已启动/已完成或收益已验证。

## 已实际排队

运行源码提交`e8fe406`，不可改工作树`ue_g_ubr_native_full100_20261028/runtime_source`；105份运行来源与100轮orders准备成功。owner PID1576874，状态`waiting_for_analysis`，仅等待`analysis_native_20261028/attribution_queue_state.json`完成；恢复门/三训练/三测试共7项pending，无100轮训练进程抢跑。重复wrapper queue已实际验证幂等。

独立审查发现并修复SAM2 editable闭包、gate身份绑定不足、manual入口绕过准入问题；CPU负向核验拒旧git/错arm/错cap/损坏CP与失败依赖。独立运行树CPU导入和源码一致性审计记录root_queue_acceptance.json。GPU full恢复门仍排在前置之后，不将当前CPU排队验收写成GPU100轮验收。

## 用户授权提前GPU共租（2026-10-28）

用户明确表示卡已空，可共租。新增前端wrapper `release`：仅在全任务pending且owner waiting_for_analysis时核PID/starttick/argv，停止空闲调度器，归档原manifest与state，去掉分析完成依赖，再用原固定数值源码恢复调度。当前旧owner1576874退出、新owner1578097接管；不停止分析进程、不修改e8fe406运行树，也不改实验合同/种子/预算/测试准入。source/hash和移除依赖原因记control_overrides。正式GPU恢复门仍先于三组train。
