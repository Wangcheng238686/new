# UBR-RESEAL-20261027 预注册：级3 单流再封版（UDPR→UBR 换装，从头 100ep）

依 `UBR-UNIFIED-ARCHITECTURE-DESIGN-20261026.md` §五 级3 与 `UBR-PROBE-PHASE-20261026_preregistration.md` 阶梯。骨架门自检：直接服务单一问题（UBR 统一换装后的单流基线不劣于旧封版——最终形态双流主张的前提）；判据全部预注册；无支线；会议证据链由本实验的验收门保护。

## 路由判定
基座/训练动力学型：UDPR→UBR 换装改变训练架构本身，热启动无定义（旧权重含 UDPR 参数，无法对应加载）。冻结诊断级证据 = 级2 加法探针（三门全过，val+test 双考场）。**直接全量**。

## 实验定义
- **新臂**：单流统一架构 = 旧封版骨架逐位同构，唯二差异：`decoder_tail_refiner` 不实例化（删代码非开关）；UBR(off)（端口⑤全零，163,537 参数）以训练内方式接入——`loss_ubr`（带内 k=8 focal BCE+Dice，逐实例等权，λ=0.1）随主干联合训练，梯度经①–④回流 decoder；推理时带内重分类生效。
- **对照（旧封版）**：EXP-0025 v1（同数据集、同协议），主对照 last-vs-last，best-vs-last 作参考读数。
- **协议**：与 EXP-0025 预注册逐字相同——seed 44、DDP4 每卡 batch1×累积2（全局8）、lr 5e-4、wd 0.05、warmup 100、cosine、100ep、默认增强（hflip 0.5）、val 459 全量逐 epoch、test 全程不读。硬件宿主允许 4×3090（lthpc，compact 收线后）或 4×4090 co-tenant；DDP 语义与 4090 原跑等价（梯度平均同构）。单卡累积8 为降级备案，须在 wrapper 声明后方可使用。

## 四要素
1. **主指标（三指标双盯）**：val 459 segm mAP、bbox mAP、segm **AP75**（UDPR 会议贡献集中在 AP75，换装不许丢）。伴随：B-AP_l(7px)/B-AP(0.02)、AP_s/m/l、逐 epoch 曲线。
2. **选择口径**：新臂 last epoch100；旧封版 last（ep100，已有 last_model_epoch100.pth）；参考读数 old-best(ep71)。
3. **显著性**：val 459 逐图配对 percentile bootstrap，500 draws，seed 20261021（拉通口径）。
4. **判定（预注册）**：
   - **RESEAL_ACCEPT** = segm mAP、bbox mAP、segm AP75 三指标 新−旧(last) 的 CI 均含 0 → 统一架构成为新封版（铸造新指纹），级4（双流最终形态）放行；
   - **RESEAL_NULL** = 任一指标 CI 含 0 但 |mean|>1.0pp → 记录"数值上略差但统计等价"，按 PI 裁决；
   - **RESEAL_FAIL** = 任一主指标 CI 上界 < 0（显著劣化）→ 换装失败，回退"UDPR 保留 + UBR 双流专用"分裂构型（设计反对的形态，作为已预注册兜底）；
   - B-AP_l 预期显著为正不作门（机制增益方向已知）；AP75 显著为正同样只作加分记录。

## 完整性契约（fail-closed，wrapper 断言）
新臂 config 对旧封版 config 的 diff **恰为** {decoder_tail_refiner_cfg=_delete_, UBR 接入键}；旧封版 fingerprint（2fcae2ff…）不变；数据集 splits/annotations SHA 沿 EXP-0024/25 清单；源码 commit 且树洁、EXP-ID 发号打 tag；独立子代理实施+接线审计 VERDICT: PASS 落盘 `docs/reviews/` 后方可拉卡；输出目录不存在。

## 工件
训练器 `scripts/probes/ue_ubr_reseal_20261027.py`（模块替换式复用生产 trainer，UDPR 删除 + UBR 训练内接入）；配置 `configs/ablations/ue_ubr_single_512.py`；产物根 `/data2/wangcheng/remote-lthpc/checkpoint/ue_ubr_reseal_20261027/`（lthpc 宿主时为 `/data/wangcheng/checkpoint/` 同名）。
