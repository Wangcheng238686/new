# RGB压缩/消费机制变体：共租前缀实验

用户授权在余量足够时共租运行。原四臂100epoch训练及其source/初始化/选权保持原协议。本轮独立文件与根目录，不导入任务E73权重作为训练初始化。

## 首轮三个单改动变体

| 名称 | 唯一机制改动 | 保留的限制 |
|---|---|---|
| view_preserve | 原先8候选直接平均改为按5个固定视角分组求和/总候选数，拼成160维，再线性160→32 | 视角内部仍汇总；输出仍32维，并非保留全部候选到SAT query。新增5120参数。 |
| spatial_preserve | 融合后固定2×2平均改为pixel_unshuffle后128→32线性压缩 | 输出仍64²/32维，仍有降维；新增4096参数。检验可学习压缩，不等同提高空间分辨率。 |
| no_sat_copy | uout forward返回output×0，使updated_u和实例头ur中的反向卫星value复制同时归零 | SAT query、head_satellite与几何仍在，不能叫无SAT捷径/纯RGB模型。reverse部分梯度严格零但参数仍按AdamW处理。 |

每方向均有正确RGB与常量RGB训练，共6条。两种带参数的变体初始化复现原数值功能：视角权重按原候选数加权，缺失/不均匀视角也适用；空间权重每子格1/4。新增模块独立fork RNG，不扰动共同权重与数据初始化。

## 共同基础与预算

- 原 `ue_compact_joint_20261024/common_init.pth` 的通用SAM2及共同新头初始化，不从单流任务检查点热启动；seed44。
- 10epoch，每轮2136图、267次成功更新、accum8，总2670次；沿用原100epoch学习率日程的前2670步，lr5e-4/wd.05/warmup100/clip.5/AMP512、同溢出重试规则、无EMA。
- 每轮样本顺序必须与原compact对应正确/常量臂的epochJSON逐值一致，否则失败停止；同原Gaussian/ColorJitter增强、UAV无增强。
- 每轮完整459图FP32验证，存当前值；固定E10用于比较，不按本轮最佳epoch挑变体，不调用test。
- 原compact正确/常量E10检查点及纯卫星E10另做同口径val复评；训练不重跑基准。
- 每变体正确训练E10再做常量/跨半个val顺序donor RGB干预；只换RGB，保留目标几何和对应。输入反事实物理不一致限制保留。

这只回答同前缀预算下早期RGB学习差异，不等同100epoch最终优劣；单seed、开发val与多路探索限制保留。

## 主分析

分别在原卫星路径与合并路径上计算bbox、Mask、Boundary2%/1%。

```text
内容差 C_v = AP(v正确训练) − AP(v常量训练)
差中差 D_v = C_v − C_原compact
净变化 = AP(v正确) − AP(原compact正确/同轮纯卫星)
同权重依赖 = AP(v正确输入) − AP(v常量或donor输入)
```

500次同图配对bootstrap、seed20261021，GT/图像顺序严格核对。差中差使用同一次抽样的四份AP联合计算，不相减两个区间。单臂上涨不能证明RGB保留改善；新增参数的优化/归纳偏置混杂不会被常量对照彻底排除。no_sat_copy训练消融与推理关闭干预的含义不同，改善也不能自动证明原路径有害。

## 已通过的工程门

CPU：视角加权初值最大误差1.1921e-7；空间池化及复制后的完整interact/head误差0；无支持写回恒等；no_sat_copy反向梯度为0且前向SAT-query读取有梯度。

GPU：训练最大实例图208实例，三个方向各3次真实accum8更新，新增参数/源LoRA有更新，FP32正式combined推理通过。峰值reserved：view5690MiB、spatial5656MiB、no-copy5654MiB。完整原始报告在 `gates/<variant>/gpu_gate.json`。

调度：单进程torch allocator7800MiB上限；空闲≥11000MiB准入，含3200MiB以上上下文/增长余量；本owner每卡最多1个任务。原四臂约6–7GiB/卡，当前余量足够。6训练先启动4个、另2排队，不终止其他owner。失败停止新增任务，正在跑的任务自行收尾，保留退出码和日志。

训练遥测沿用原loss/preclip norm/AMP retry与epoch输出节奏，progress每25updates；variant/content仅身份字段。无逐步全候选/attention分布统计，无新loss或门控遥测负担。

## 文件与产物

- 模型：`rsprompter/uav/rgb_mechanism_variants.py`。
- 训练/门/评估：`scripts/probes/ue_rgb_mechanism_run_20261025.py`。
- 共租调度：`ue_rgb_mechanism_owner_20261025.py`；纯卫星参考：`ue_rgb_mechanism_satellite_reference_20261025.py`。
- 配对差中差：`ue_rgb_mechanism_analysis_20261025.py`，原/合并路径分别计算，先写Mask/bbox再Boundary；沿用已校准向量化bootstrap。
- 根目录 `/data/wangcheng/checkpoint/ue_rgb_mechanism_20261025`，manifest锁定数值及driver SHA、共同初始化SHA；state/job_plan/logs/contract保留任务与身份，`analysis/summary.partial.json`及`summary.json`为阶段/完整结果。

在全部训练/评估尚未完成前，不将资源门的3次更新或训练loss作为收益证据。

启动核验：owner548328；view正确548674/GPU2、view常量548675/GPU0、spatial正确548676/GPU3、spatial常量548677/GPU1存活。四条均已输出首25次成功更新，loss/norm有限、AMP retries为0；原四臂owner仍running、failed为空。额外真实增强DataLoader的前16个ID已与原E1逐值核对一致，记录 `loader_prefix_gate.json`；完整每epoch顺序仍由训练端门控。数值及driver源已复制到 `frozen_source/`，后续禁止覆盖活跃source。
