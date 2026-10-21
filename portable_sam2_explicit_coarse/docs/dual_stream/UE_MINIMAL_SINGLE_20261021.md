# 固化单卫星基座：2026-10-21 训练协议

用户授权：先训封版单卫星；固化后再开发双流。未实现/启用新UAV分支。

## 固定协议（训练前登记）

- 数据：`/data/wangcheng/dataset/CitySample_SatUAV_TrainMinimal_v1_20261021`，review99 v3用户验收修订，2142 train /459 val /459 test。COCO独立RLE唯一标签权威；保留实例重叠，禁止降为单ID PNG。卫星目标区域隔离口径，不声称全UAV输入地理隔离。
- 模型：继承本机封版 `a3sgm_whu1024.py → ue_rgbd_single.py → ue_minimal_single.py`；512原生，coarse64不变，SAM/CSPM/UDPR不改。max_per_img=250。通用SAM2初始化，无UE/WHU热启。
- 训练：4×3090，每卡batch2、累积1、有效batch8；seed44，100epoch，AdamW各主可训练组LR5e-4/wd.05，warmup100 optimizer steps、cosine、globalclip .5、AMP初始scale512；EMA禁用，禁早停。与远端4090对齐优化配方，但数据/增强/coarse64/选择best/AP250为本次显式差异，不能称完全复现实验。
- 增强仅训练RGB：Gaussian p=.5，std=.02×255，clip转uint8；随后RGB空间RandomApply(ColorJitter(brightness=.15,contrast=.15,saturation=.10,hue=.02),p=.5)，再转回BGR供原preprocessor处理。两个增强独立触发；不改标签、不翻转、不旋转、不缩放。val/test完全关闭。
- sampler与loader均drop_last，2142点/4rank→每rank535点→267个batch；每epoch实际2136点，6点随机轮换丢弃，无补重复；267updates/epoch，总26700。有效batch相同不保证随机轨迹逐位等价。
- 训练只读取train/val标签。归档身份会hash test标签文件但不解码、不计算其指标、不用于选择。
- 主指标val segm AP@[.50:.95]，maxDets=[1,10,250]；伴随bbox/AP50/AP75/size/AR。新route训练/推理的AR@100旧键专门更名AR@250，不改旧臂。
- 选权重：完整val每epoch，最高segm AP250（严格大于，平局保留较早者）；另保存bbox best与last。不得根据test更换权重。
- 完训后对选定best-segm运行一次459图test，严格model权重，maxDets250，导出预测/GT/DT/运行身份。test仅终局报告；无对照涨分主张，单seed/val选模局限明确。不沿用旧0.008噪声阈值证明新数据显著性。

## 实现与遥测

新增 `data/ue_minimal_dataset.py`（独立RLE解码/形状面积验证/固定split/增强/归档hash）、`configs/ablations/ue_minimal_single.py`、`scripts/smoke/check_ue_minimal.py`。统一入口 `scripts/ue_rgbd.sh minimal train|infer`。现有UE四臂默认行为不变。

不增加每步重遥测。沿用loss、梯度clip/AMP、峰值显存、CSPM/UDPR字段（口径见DEBUG_FIELDS）。新增运行身份是配置/数据SHA/增强参数/评估maxDets，不能代替结果验证。主记录 `/tmp/ue_minimal_single_20261021/`，正式检查点 `/data/wangcheng/checkpoint/ue_minimal_single_20261021_EXP-0017/`（启动前检查空间）。

启动门：独立实现审查、旧臂指纹与环境隔离检查、RLE/增强保标签检查、220目标AP250校准、真实稠密场景batch2前后向与预测、4卡DDP短程smoke。门未过不得启动正式run。所有smoke不读取test图像。


## 实施验收（启动前）

独立实施审查PASS：[完整报告](../reviews/UE_MINIMAL_SINGLE_IMPLEMENTATION_20261021.md)。
真实训练最大208实例图2464+图1234的batch2，AMP反向与更新有限，峰值6741.6MiB。
合成220目标AP250校准=1.0；增强前后独立实例掩码/框一致。4卡2update+2val smoke正常exit0，
DDP同步偏差0。验证短程AP接近0不作为性能结论。严格推理验证2图成功，AR@250标签与合同正确。
归档36734文件的sha256sum -c全部通过exit0（只做字节完整性，不读取test结果）。
旧4臂模型合同与封版一致；minimal模型仅max_per_img从300改250，coarse64及所有参数结构不变。
固定协议CLI覆写攻击被拒绝，污染env与清洁env dump一致。跨单卡/DDP恢复被身份检查拒绝，
同4卡恢复另核验（结果见启动门JSON）。未修改冻结architecture_contract.py/inference/model_rebuild.py。

EXP-0017正式入口：

```bash
bash scripts/ue_rgbd.sh minimal run \
  --data-root /data/wangcheng/dataset/CitySample_SatUAV_TrainMinimal_v1_20261021 \
  --run-root /data/wangcheng/checkpoint/ue_minimal_single_20261021_EXP-0017
```

runner单进程顺序执行：全归档字节预检→100ep train/val→SEALED_SATELLITE.json→test。
状态写run_state.json；train/test日志带runner PID。若子进程失败不进入下一阶段，不自动换权重。
