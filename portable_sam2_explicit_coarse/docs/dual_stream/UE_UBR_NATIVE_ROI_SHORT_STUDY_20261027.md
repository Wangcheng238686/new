# Native RoI 三组四卡短实验

用户2026-10-27授权接好训练并并行启动。A_nohead无新头、B_zero同头但仅尾部RGB/几何归零、C_correct正确证据；上游均为相同G及正确RGB-D，B不是纯卫星。

## 实际接线

- `scripts/probes/ue_ubr_native_run_20261027.py`明确复用冻结旧short engine的train/save/restore/五类RNG/样本顺序/accum2方法，在本进程绑定native Runtime、receipt核验、source inventory及原`core.inference_outputs`评估。旧数值文件不改。
- `configs/ablations/ue_g_ubr_native_protocol_20261027.json`固定fresh共同generic初始化（非E73或trainedG热启动）、Gseed2060、head2080，10轮2670更新、每组四卡每rank1图累积2/global8，LR.0005/wd.05/warmup100/原100轮cosine前10轮、seed44、AMP/新头FP32、无EMA、maxdet250。
- 原train每轮2136个既定ID（不重排/不补样），E1/E10完整459val，epoch边界last并额外E1/E10。固定E10，不选val-best；三组完成后统一完整459test。
- 原512主mask监督/coarse64/.1/检测/实例loss保持，无UDPR/globalUBR/额外boundaryloss。
- 共享根合同排除arm、tail端口与allocator cap，外层每臂检查点完整保留；不能因资源cap不同拒绝同一数值合同。

## 续训与调度

`scripts/probes/ue_ubr_native_owner_20261027.py`绑定已提交源码、实际import闭包含vendoredSAM2、工程/资源/恢复receipt和逐轮顺序SHA。每组四卡，不降单卡。预留按A/B/C独立实测峰值及上下文，实时扣除尚未兑现预留；显存不足等待、已在跑任务不抢占。全部训练完成才顺序测试。owner失败停止新准入并保留健康子任务，不盲目重启。

发射前真实四卡resume gate核模型、optimizer非空synthetic状态、scaler及每rank Python/NumPy/Torch/CUDA/DataLoader generator五种流；该门仅恢复机制，不冒充真实next-update轨迹相等。源码先提交后发射，owner给子进程`UE_UBR_NATIVE_SOURCE_COMMIT`锁定数值提交，后续仅文档提交不改变此身份。

本实验是固定10轮机制比较，不代表全量收敛；A/B/C比较新增容量和联合尾部RGB-D证据贡献，不是完整RGB必要性证明。测试集已暴露，仅描述性开发评价，不据结果改预算/选权。
