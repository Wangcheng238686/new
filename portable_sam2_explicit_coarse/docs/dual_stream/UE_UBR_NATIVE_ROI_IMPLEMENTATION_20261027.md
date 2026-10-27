# Native RoI evidence refiner 实施记录

2026-10-27。按用户最近两轮确认实施，由核心、集成、独立审查子代理分工，主代理重跑CPU与复核四卡产物。当前停训状态保持，本轮只工程开发，不启动固定训练预算。

## 一次解码的实际路径

```text
五视角RGB/深度/相机 → 现有G候选读取与旧上游交互（原数学不动）
  ├─ reduce_rgb原返回值 → 保存prefusion aligned32 + candidate valid.any
  ├─ 原geometry5 → 保存当前图证据
  └─ 卫星主干/FPN → 原检测/RoI采样、coarse64与SAM提示
      → 原SAMdecoder一次 → native128 logits+act2 feature32
      → 同RoI采样alignedRGB32/geometry5 → native residual head
      → refined native logits
          ├─ 原插值至512² + 原assigned RoI crop主mask loss
          └─ 原sigmoid→bbox probability paste→实例预测
```

主检测损失/实例发现头及loss仍沿原路径；无独立UBR loss，无SDT、neighbor、active重匹配、额外SAM解码。UDPR和旧globalUBR都不实例化。

## 文件职责

| 文件 | 实际职责 |
|---|---|
|`rsprompter/uav/native_roi_refiner.py`|显式RoI中心/stride/origin采样，RGB weightedbilinear、geometrynearest；70→32→3DW/PW/GN/GELU块→zero1 residual|
|`rsprompter/uav/ubr_native_integration.py`|NativeEvidenceG复用原RGB聚合、NativeMaskHead前后局部context、NativeDecoderShim临时act2capture保3tuple、NativeRoIHead空rank连参|
|`scripts/probes/ue_ubr_native_runtime_20261027.py`|共同foundation+strict过滤旧tail键，A无头/B零端口/C正确；新head seed2080 fork不扰shared RNG，不构旧globalUBR|
|`scripts/probes/ue_ubr_native_gate_20261027.py`|真实4rank sameweight恒等、3AMP工程更新、主mask独立梯度、原consumer250压力、A/C计时与全repo源快照|
|`tests/test_native_roi_refiner.py`|原点反例、valid/NaN/独立support、nearest/weighted读法、非零梯度/概率detach/empty行为|
|`tests/test_ubr_native_integration.py`|单decoder3tuple、异常清理、非法imageID、空RoI连参、prefusion缓存数学和梯度|
|`configs/ablations/ue_native_roi_development_contract_20261027.json`|开发合同/三臂/坐标/监督/待验收状态，无预算发射授权|

## 坐标与通道

输入70ch=decoder32+RGB32+geometry5+detachprob1。decoder/RGB独立无affine通道LN；geometry保持height/100、dxdy/10clip±5、viewcount/5、support0/1，不误称AGL/物理坡度。几何support只屏蔽几何，RGBsupport来自保留候选valid.any；无证据仍允许卫星修正，不给最终delta乘支持掩码。

RoI native中心映射x=x0+(j+.5)(x1-x0)/W，y同理。RGB是floorUV/4桶中心4j+2/stride4，独立于旧CNN/Hiera名义原点；geometry是floorUV栅格像素中心j+.5/stride1。二者各自显式grid_sample align_corners=False；RGB有效加权分母阈值1e-6，invalid先where清除（防NaN×0），图像外额外清零；geometry nearest保持高度跳变。框/imageIDs detach，不向离散候选框伪造几何梯度。

## 监督与实例身份

实际配置mask target512，native128沿原loss插值到512，GT为assigned RoI crop。新头在原maskhead返回前替换logits，主mask权重1直接训练native头，coarse .1/检测四项/实例1及内部logwh .1不变。删除旧tail loss，禁止以全图GT直接resize比较native RoI。正例身份沿原sampling_results，不再预测pool IoU分配，train/eval各自原maskhead调用一次。

旧float roi_img_ids必须验证整值再long传sampler。每call maskhead try/finally清context；decoder temporaryhook finally移除。证据缓存随encode/clear更新，不detach，不再注册一份UAV模块。训练异常时调用者仍须执行rt.clear以释放本图source graph。

## 初始化与检查点

A不安装native头；B/C共同新头seed2080。其余从相同common_init generic状态严格加载，只退役UDPR键；Gseed2060与source LoRA保持。不加载trained G用于正式初始化。shim包装原maskdecoder后，state前缀从mask_decoder.*变mask_decoder.base.*，必须构造完整native结构后严格恢复新cp；不能把旧state strict=False静默掩盖。runtime的load_engineering_G仅供工程门显式旧键迁移、role记录，不是热启动训练授权。

## 测试范围与局限

主代理重跑新CPU15项通过（8核心＋7集成）；独立审查另核混合FP16/BF16输入的恒等与梯度。真实四卡attempt1遇到gate记录层错误：parity observe留下tensor，空rank不触发decoder时JSON写旧tensor，非模型loss错误；已保留并退出自有gate。attempt2清记录并让mainmask-only其余loss以0连接DDP，数值模型源未改。

工程门的3次更新为每rank1图、global4且无累积，不冒充正式global8预算；250框为同一真实形状框重复的资源fixture，非自然250不同预测/漏检评价。A/C绕过refiner计时使用同模型同RNG同图，单批计时不是收敛速度/AP。需要另有训练配对才判断尾部容量与真实内容收益。候选压缩、bbox限定外扩、上游G混合的限制仍在。

## 混合精度审查与修复

attempt3证明末层自然更新且在optimizer，但scale32时隐藏梯度为零。attempt4按正式init_scale512执行，首步overflow回退256，后两步隐藏层有非零梯度、副本一致；其后独立mask诊断因gate遗漏scaler失败，不能称为结构断接。该诊断已改为与训练相同scale/backward/unscale、无optimizerstep。独立CUDA同权重自然两步实验将问题定位为FP16隐藏激活反传下溢。

最终只把新增6625参数头整体设为FP32 island，输入cast保持可微；末端delta转回原logits dtype相加。原decoder/LoRA等上游仍AMP，loss权重/分母、零初始化不变。只将末层FP32不够，因为梯度再转回隐藏half激活仍会下溢。跨回上游half激活仍有舍入风险，不能由新头梯度非零断言真实RGB内容有收益。所有失败attempt保留，最终门必须绑定最终source。

## 最终工程验收与主代理复核

最终产物：`/data/wangcheng/checkpoint/ue_ubr_native_gate_20261027/attempt5/complete.json`（四rank passed），同目录保存81份实际import源快照与审查说明。主代理逐份SHA核对当前81源零漂移，另核旧global/fast合同97份原源零漂移；CPU15项通过。

| 验收项 | 实测与范围 |
|---|---|
| 初始化恒等 | 同权重/框/chunk的native raw、概率与二值逐值相等 |
| 解码次数 | 正常64正RoI为同一次SAM解码；首步512→256重试计两次forward，非单次重复解码；empty rank为0 |
| 自然更新 | 四rank3次AMP完整目标工程更新，参数副本SHA一致；首次仅out非零，后续input_proj非零 |
| 主监督可达 | 单独原loss_mask、其余loss乘0连图，scale256：native hidden范数.001221，decoder2.26657，共享UAV LoRA .001181 |
| 资源 | 第二/三成功更新约3.09–3.20/2.53秒；首步约5.66秒含overflow重试；训练reserved最高8558MiB，所有已测case最高8564MiB |
| 250框压力 | 原consumer/chunk32共8次解码，实际消费更新后的logits，maskconsumer约.635–.688秒，reserved最高3654MiB；不含extract_feat |

主mask诊断的共享UAV梯度混有原G路径，不单独证明新增RGB端口贡献，更不证明内容/AP净收益。A/C计时每组仅一次且A先C后，缓存顺序影响明显，不能据此认定新头比无头更快。三次global4工程更新不等于正式global8预算；250框重复fixture不是自然测试集评价。

本轮仅有工程产物，无训练AP，无正式native续训轨迹验收；下一训练运行器必须按新shim的state前缀严格恢复并补恢复门。当前旧任务继续停止。本轮代码及文档形成独立Git提交，未修改原在训源码。
