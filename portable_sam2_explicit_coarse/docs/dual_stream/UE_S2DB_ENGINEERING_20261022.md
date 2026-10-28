# S²DB机制探针：工程验收（2026-10-22）

## 已完成

- train/val共2601缓存新建，原index/geom逐值不变；独立核对原文件SHA、源token/实际像素、camera-Z单位及投影回环。
- 最大锚点元数据投影复算误差 `1.7053025658242404e-13 px`。
- 发现并修正float32投影元数据在格边界舍入跨格的问题：缓存投影改存float64，监督掩码先由float64确定，模型读取输入再转float32。
- CPU：零偏移读取等价、孔洞/越界回退、有效深度突变保留、两种融合无支持恒等、RoI边缘有限值。
- 续训：Python/NumPy/Torch/CUDA及DataLoader generator均恢复并回放；真实train的4张前缀顺序及高斯/颜色增强逐值一致。
- A/B/C/D及D_aux均通过真实冻结基座GPU门：FP32/AMP零α原始logits逐值恒等、无支持非零α原始logits恒等、零偏移GPU逐值等价；新增LoRA/偏移/融合梯度可达；完整卫星参数及缓冲区SHA不变。
- D_aux确认辅助路径query梯度为零/未使用，主路径保持可训练。
- 正式几何和融合、FP32、强制250 RoI并沿生产32块推理通过，所记录最大已分配显存约2.5 GiB。此数字不是训练峰值，不包含其他进程占用或CUDA保留缓存。
- 新评价封装在完整459张旧test baseline上与旧固定分析逐值对账：Mask AP `.6455098296975914`，Boundary AP2% `.6299859945771743`，Boundary AP1% `.541968930778396`，误差0。

## 训练与结论边界

工程通过只证明接线及数值合同；不代表新方案已有AP收益。
主协议见[固定合同](UE_S2DB_MECHANISM_PROTOCOL_20261022.md)。
10臂首轮及固定D的3个新增辅助训练由独立持久owner调度，完整E10/2670更新后才评价；
预留旧任务GPU直至其释放，不按AP提前停止。所有训练从相同完整common_init初始化。

实际状态与日志在 `/data/wangcheng/checkpoint/ue_s2db_probe_20261022/state.json`、`logs/`；
完整产物完成前不生成成功结论。最终交付为 `complete_audit_summary.json` 及 `FINAL_REPORT.md`。

## 已完成内容配对的正式阶段裁决

见[内容归因裁决](UE_COMPLETED_CONTENT_ATTRIBUTION_VERDICT_20261023.md)。A/B/C/D及匹配常量8臂已独立核对E10、训练记录、预测及配对区间；当前不通过正确RGB提高整体Mask的证据判据，无需等待所有未完臂才能对这8臂下判断。常量控制仍保留UAV几何/支持，不能推成所有UAV无效。
