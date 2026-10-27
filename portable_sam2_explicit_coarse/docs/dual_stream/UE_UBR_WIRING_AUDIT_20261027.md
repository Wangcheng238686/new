# UBR坐标接线审计与已锁定合同

2026-10-27。用户要求子代理沿Git历史审慎解决接线问题。历史调查、适配器实现和独立复审分别执行，主代理核源码、实际测试与产物。**交付是已验证的坐标适配器和明确的监督合同，尚非完整检测器P1验收。**

## 1. 历史结论

- `6dc6aa4`：全图稀疏前景监督曾背景塌缩，转roi_local。
- `49670c5`：native decoder网格按RoI解释，GT proposal crop、推理bbox paste。
- `eef70ca`：UDPR保持28²原生RoI主mask监督，另设tail监督；部署阈值.4。
- `ddb9524`：先sigmoid再bbox概率paste。

最终captured act2 32ch特征与raw mask logits共享native网格。输入曾重复全图feature，不足以将输出认定为图像坐标。适配器对两者使用相同bbox采样格；不能只paste mask却全图resize特征。该映射延续旧latent合同，不声称特征具有理想的物理标定。先前S1混淆输出网格的退休证据是工作区产物，未编造Git提交。详见[历史审查](reviews/UE_UBR_COORDINATE_HISTORY_20261027.md)。

## 2. 已实现

`rsprompter/uav/ubr_coordinate_adapter.py`：

1. 浮点xyxy bbox、像素中心+.5、align_corners=False、零padding，与真实FCN贴图一致；不提前取整或裁框。
2. mask先sigmoid再paste，32ch特征直接用同一grid paste。显式支持scale_factor、rescale、非方形canvas；输入bbox不原地修改。
3. 有限logit epsilon=1e-6；FP32 roundtrip门容差1.1e-6。过小不可表示epsilon拒绝。
4. 原概率独立保存；部署时UBR未改变logit的位置恢复原概率，零残差包括.4 tie的binary/uint8恒等。**此恢复仅推理，训练直接监督连续refined_logits。**
5. band和neighbor二值状态应从保存的p>=.4生成，避免反logit舍入移动threshold tie。UBR输出在图像网格，不二次bbox paste。
6. CPU遵循FCN skip_empty的bbox±1范围，GPU遵循完整paste；低分辨率大框的插值halo上游本就有后端差异，不伪称CPU/GPU互相逐位一致。

适配器不自动处理instance排序：调用端必须先选mask通道，保证logit/feature/bbox/候选ID一一对应。不修改原SAM、mask head、ROI head或UDPR源。

## 3. 明确修订监督合同

全图512² UBR输出禁止送入旧roi_local loss_and_target后缩至28²，这会与GT crop错位。也不用不可逆paste→crop回环。

固定目标为：检测四项各1＋原native mask1＋coarse .1＋UBR .1＋实例头1；实例头内部logwh .1。UBR focal gamma2+Dice只在其预测边界带、assigned全图GT上计算；删除旧UDPR margin，不增新loss。**原主mask不直接监督UBR**，这是对早期架构描述的公开修订。decoder经原主loss与可微UBR路径受训。

context保留当前P1候选评分/保护prefix政策，来自完整pre-UBR预测pool且detach。active UBR直接选同pool真实预测，bbox IoU>=.5分配最大IoU GT（平手GT原序），同GT允许多预测；score降序/ID升序最多64/图，不加GT预测候选、不采用低于阈值强配。GT只分配监督，不建邻居、排序或NMS。空正预测时连参零loss，图等权分母不省空图。active必须可微重新解码，不能复用no_grad上下文tensor。

该规则避免GT正RoI和预测ID之间的自排除猜测，但引入显式UBR监督人口选择；不能宣称与旧tail总loss数值等价。只监督匹配正预测不能证明已训练新增误检拒绝能力，补框共享RoI确认仍留P2。独立意见与裁定见[复审](reviews/UE_UBR_WIRING_INDEPENDENT_REVIEW_20261027.md)。

## 4. 已执行证据

- 主代理CPU组合测试：26项（adapter10、UBR8、context8）。覆盖真实FCN概率/二值/uint8、fractional/clipped框、非等比rescale、empty/invalid、特征landmark、sigmoid顺序反例、exact/adjacent threshold、梯度、epsilon反例。
- 四卡适配器门：每rank分别实际FP32、FP16、BF16输入，CUDA autocast，真实FCN对照。全部binary/uint8恒等、logits与feature梯度有限非零；最大FP32反变换误差 9.999997701e-07，低于1.1e-6。
- 产物：`/data/wangcheng/checkpoint/ue_ubr_adapter_gate_20261027/complete.json`。新源与gate各SHA记录在每rank产物。它是合成适配器检查，不是DDP训练/完整模型/正式几何/AP门。

运行：

```bash
/data/wangcheng/envs/cvt2/bin/python -m pytest -q tests/test_ubr_coordinate_adapter.py tests/test_unified_boundary_refiner.py tests/test_ubr_prediction_context.py
/data/wangcheng/envs/cvt2/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 scripts/probes/ue_ubr_adapter_gate_20261027.py --output /data/wangcheng/checkpoint/NEW_UNIQUE_ADAPTER_GATE
```

## 5. 完整P1仍须实施的接线与门

新opt-in decoder capture/mask head/ROI head与G wrapper尚未接入。后续必须保证UDPR不实例化、extraPE一次、capture逐调用局部保存、不由context覆盖训练输出；固定train/eval/dropout模式与RNG恢复。geometry5与返回canvas同坐标，satellite缺省geometry=0。

真实同权重tail-disabled基线对比raw logits/embedding/coarse/det scores，零UBR最终binary/score精确恒等；移除训练过的UDPR本身不可能与旧UDPR结果恒等。真实可微active decoder梯度、首步out与更新后hidden梯度，以及完整pool context+最多64 active反传四卡峰值尚待测。已过的250个顺序组件推理不替代此门。当前没有新增AP结论、没有正式训练启动、没有改动旧实验。
