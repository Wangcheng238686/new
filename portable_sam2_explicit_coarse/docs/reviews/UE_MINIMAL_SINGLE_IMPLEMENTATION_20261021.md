# UE minimal single 独立实施审查（2026-10-21）

VERDICT: PASS (implementation / release) — 新runner与修复后的wrapper实施审查通过，无代码阻断；正式算力启动仍以运行准入门完成为条件。

## 审查身份与范围

审查者 /root/minimal_training_review；只读审查新增 ue_minimal 路由，未修改项目代码，未启动 GPU 工作。审查时工作树未提交、主代理仍在补 smoke。独立运行 CPU smoke，输出 independent_cpu_smoke.json。

## 已核实

- configs/ablations/ue_minimal_single.py相对封版ue_rgbd_single.py仅模型test_cfg.rcnn.max_per_img从300改为250；继承coarse64，SAM/CSPM/UDPR保持。
- data/ue_minimal_dataset.py独立解码COCO RLE（包括uncompressed counts list），按每实例保留掩码，不降为单ID图；检查面积/形状，和split顺序逐项一致。训练集2142，稠密场景208实例。
- train增强Gaussian p=.5/std=.02*255/clip uint8，随后RGB uint8 ColorJitter参数与用户要求一致，再转回BGR。val/test不增强。CPU smoke验证增强不改变稠密场景目标。
- 四卡batch2/acc1有效batch8，DDP sampler drop_last+loader drop_last产生每rank267个batch/epoch，共2136张/epoch。warmup100 optimizer steps、100epoch、LR5e-4、backbone与decoder multiplier1、关闭EMA在wrapper显式设置。
- 新路由val maxDets=[1,10,250]，模型max_per_img250在预测前应用；AP自定义上限重算。独立CPU 220实例校准AP=1.0通过。AR@250标签在训练与推理均已补齐。
- 训练用train/val；test仅文件hash参与身份，不被本次smoke解码或评估。协议明确val-segm选择best，test最终一次报告。
- 新格式按embedded model_config重建，经architecture fingerprint和通用SAM checkpoint SHA检验，禁止allow_nonstrict；恢复走原strict load及新增data/training/augmentation/evaluation snapshot validator。
- 原四UE臂配置文件、模型源码、optimizer未修改。新增训练/数据loader分支由dataset==ue_minimal隔离；老数据格式仍走原路由。
- 未增加重遥测；协议公开新旧差异，未把新数据结果当成旧基线涨分。

## 正式启动前阻断项（尚需主代理提供证据）

1. 完成四卡真实DDP短程训练/验证/strict恢复与推理闭环；独立CPU smoke不覆盖DDP和训练checkpoint身份。单卡GPU smoke已看到落盘，但不代替DDP门。
2. 完成旧四臂全臂harness及旧/新resolved model diff检查，不能只依据静态审查宣称指纹未变。
3. 补齐docs/EXP_REGISTRY编号、architecture guide同步、commit+tag，确认干净commit，再拉起全量。
4. 核对实际启动argv无覆盖固定协议的额外参数，保留最终resolved args/git身份与磁盘空间检查。

## 非阻断发现与解释边界

- wrapper允许尾部CLI覆盖epochs/lr/batch/ema/max-val-batches等。它支持smoke，但固定协议不是不可绕过的CLI约束；本次正式argv需审计，无需为此大改旧wrapper。
- minimal dump-env在train分支设置TEST_MAX_PER_IMG=250之前输出，因此可能null；实际配置和val强制250没有跑错。若用dump-env充当完整契约，应补minimal专用值和增强身份。
- archive_identity哈希dataset/split/annotation与SHA256SUMS清单，不重新校验所有RGB字节。它防标签/索引身份漂移，但不能单凭该函数保证图像内容未变。建议正式启动前至少核对本次train/val卫星RGB清单SHA，避免全量UAV EXR校验的无关IO；记录该边界。
- resume validator未显式比较model_config/runtime SHA，但原architecture验证和strict state load仍在，恢复加载完整旧权重。未来允许跨代码改变非参数行为时应强化；本次同commit恢复不构成阻断。
- eval helper沿用既有GT bbox面积作为size bucket area。与封版一致，应在报告说明，不能称标准mask面积size分桶。

## 独立CPU证据

`/tmp/ue_minimal_single_20261021/independent_cpu_smoke.json`：fingerprint 3ff1179dbc3957dae8e93fdcd85207e14cb7ce06a7511d74c3ff9887d42ddace；train2142；max instances208；image IDs2464/1234；target preservation true；AP250 calibration1.0。


## 二次复核：runner与固定协议修复

复核对象scripts/run_ue_minimal_experiment.py及同轮wrapper/data/infer修复；只读，无新增CPU/GPU smoke。

- wrapper minimal train仅允许data-root/checkpoint-dir/resume-from；epochs/max-train-batches/max-val-batches只在UE_MINIMAL_SMOKE=1时放行。EMA、LR、batch、累积、验证间隔等不能尾部覆写。minimal run runner强制UE_MINIMAL_SMOKE=0并设置4卡，防止smoke参数污染正式入口。
- minimal TEST_MAX_PER_IMG=250提前到dump-env之前，早期环境记录与train一致。
- resume新增比对model_config/runtime_config，连同原strict load与架构验证，解决首次报告中恢复身份缺口。同DDP拓扑恢复可行；跨单卡/DDP恢复因runtime control backend不同会有意拒绝。
- runner全量sha256sum -c SHA256SUMS.txt检验实际归档字节（包括test标签的字节校验，不解码、不用于选模），成功之后才train；dirty工作树立即拒绝。
- train子进程非零退出会失败，不进入test。训练完成后只解析best_model.pth，原训练实现仅依据val segm/mAP严格新高维护此链接；不会使用bbox best或last替代。
- selected checkpoint resolve为具体文件并计算SHA256，先写SEALED_SATELLITE.json，再启动test。test调用固定weights=model/split=test/batch1/score-thr0，无根据test反馈重选权重的回路。
- 独立新run-root要求exist_ok=False，可防同目录覆盖或意外重复test。此约束意味着中断后需人工决定恢复流程；不是自动断点恢复runner。
- 推理AR@250标记已修，eval contract写入导出manifest。
- 观察到4卡DDP smoke日志完成训练与验证，并生成last和bbox best检查点；该短程segm AP=0故没有best_model链接，runner遇到此情况会明确失败、不对test偷偷换选模标准。正式训练出现正AP后才会建立best_model。

当前无新增代码阻断。待主代理完成strict infer/resume、全臂harness、完整归档校验、注册/commit/tag后可启动正式run；本次PASS不替代这些运行门。此前非阻断事项中wrapper覆盖、dump-env、resume身份和图像字节校验均已有实现修复。保留既有bbox面积size分桶解释边界。
