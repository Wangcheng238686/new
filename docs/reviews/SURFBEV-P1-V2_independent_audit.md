# SURFBEV-P1-V2 独立实现与接线审计（修复复审）

**审计对象：** `docs/SURFBEV-P1-V2_preregistration.md`、
`portable_sam2_explicit_coarse/scripts/ue/probe_surfacebev_p1.py`，及其复用的
`probe_surfacebev_p1_h0.py`、`data/ue_raw_pair_dataset.py`、
`rsprompter/surfacebev.py`。

**审计时间：** 2026-10-12  
**审计范围：** P1 首次定版的修复复审；未运行 P1，未读取 test scene 的 RGB、UAV、
相机或 strict-ID 标签文件。

## VERDICT: PASS — 允许按预注册协议运行一次 P1

前次审计的两个 test 前阻断项均已修复：SurfaceBEV/P3 cell 行序已 fail-closed 验证，
且每个实际处理场景均执行和归档完整的 z-buffer/coverage/residual 遥测门。当前未发现
其他会阻止 P1 开封 test 的实现问题。

该 PASS 只授权冻结信息读出 P1；不构成 UAV 提升 bbox/segm AP 的结论，也不授权 H1。

### R3 — 无梯度推理内存修复：PASS

一次 **无 test** 烟测在 GPU0 因共租显存 OOM 退出；没有执行 `harvest('test')`，因而
没有读取任何 test scene 内容。修复仅为在 `pred_and_p3()`（第 30 行）添加
`@torch.inference_mode()`：该函数本就只运行 eval 状态下的两次标准 `predict` 和一次
`extract_feat`，不反传、不训练也不写 detector 状态。

该作用域与 H0 中已使用的 `@torch.inference_mode()` 被动 P3 parity 方式一致。它不会
改变 B0 输入、预测哈希比较、P3 读取、SurfaceBEV 特征、标签生成、correct/sham 对照、
split 选择或 `--allow-test` 的控制流；仅禁止 autograd 保存中间激活。因此本修复不改变
预注册指标或审计结论，R1/R2 的 PASS 保持有效。

## 复审发现与验证

### R1 — P3/SurfaceBEV row-major cell 对齐：PASS

P3 仍在第 43 行以 `[H,W,C] -> [H*W,C]` 行优先展平。SurfaceBEV 在第 48 行已修为
`[height-bin,H,W,10] -> permute(1,2,0,3) -> [H*W,120]`，因此第 `y*128+x` 行恰为同一
`(y,x)` 的 12 个高度 bin 和 10 个通道。

第 61–64 行增加了 fail-closed marker 检查：若某次代码改动引入 shape 不变的 H/W
转置，脚本会在读取数据、加载模型前退出。复审另以唯一 `(bin,y,x,channel)` 编码验证了
`(0,0)`、`(3,7)`、`(127,126)` 三个位置：每个 P3 row 都只含相同坐标的全部 12 bin。

### R2 — 全 split 几何遥测门：PASS

对 `harvest()` 的 train/val/test 每个场景、correct/sham 两个条件，代码现在均要求：

- correct/sham 原始输入点数相等；
- SurfaceBEV tensor 有限；
- `zbuffer_points > 0`；
- `covered_cells > 0`；
- `zbuffer_residual_mean` 有限。

每场 telemetry 归档完整的 rasterizer 统计字典（包括 input/in-FOV/winner 点、coverage、
residual 及有效视角分位数），而非仅保留 coverage。任一异常会在形成读出特征前失败。

## 其余协议逐项核对

1. **锚点身份：PASS。** 锁定 EXP-0006 seed-44 epoch-100 `last` 的绝对路径、SHA256
   `fc3ead…65145`、零基 epoch=99、run tag、seed；严格 state-dict 加载拒绝键不一致。
2. **split 与 test 开封边界：PASS。** v2 manifest/hash 固定为 694/100/151。构造 test
   dataset 只读取 pair/split manifest；test scene 的图像、UAV、相机和标签仅在
   `harvest('test')` 中被索引，且仅在 `--allow-test` 后执行。此次复审没有读取 test
   scene 内容。
3. **P3 parity：PASS。** 每场 harvest 在 P3 抽取前后分别标准 `predict`，逐字段比较
   bbox/score/label/mask hash；P3 同时强制为 B0 实际的 `512×128×128`。
4. **标签隔离：PASS。** `rasterize_surfacebev` 只接收冻结 UAV descriptor、DeviceZ、
   K/C2W/W2C 和 H0 train-only height edges；strict ID 与 B0 prediction 只在其后的
   boundary/error/IoU readout target 构造阶段使用，不进入 UAV feature 或几何。
5. **correct/sham 与容量：PASS。** 两臂复用同一 descriptor/depth/camera，差异仅是
   world XY 旋转后再投影/z-buffer；共享原始候选数、64 维输出、train cell 采样、随机
   投影、线性模型和优化器。in-FOV/winner 点可以不同，正是 sham 的允许几何后果。
6. **训练、选择、统计与判定门：PASS。** 读出仅在 train 拟合；val 一次从预注册 C 和
   threshold 集合选择所有臂共有值；test 不再选参。151 场 scene-paired、10,000 draw、
   seed 20261012 bootstrap 对 F1/AUROC/Spearman 的 correct−SAT、correct−sham 六项
   差值全部要求 CI 下界 `>0`，否则 P1_FAIL。

## 非阻断记录

- `dh` 是未使用的局部变量。它不影响当前数据流或判定；后续清理时可改为 descriptor
  hash 的显式 telemetry，但不能在本次已冻结协议中改动读出逻辑。
- `--allow-test` 是受控 launcher 的人工开封开关，不能替代本审计。运行命令应引用本
  审计路径、使用新且空的 output 目录，并仅执行一次；test 开封后不得因结果回调 C、
  threshold 或任一读出设计。

## 结论边界

本复审只证明 P1 实现满足开封前的接线和隔离契约。P1 的运行结果只能回答
“冻结 SurfaceBEV 是否含有超出卫星 P3、且依赖正确几何的可读取信息”；即使 P1_PASS，
也仍须另立 H1 热启动筛查，才能检验对卫星实例检测/分割 AP 的影响。
