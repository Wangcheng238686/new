# WHU-1024 全量 A3 制式训练 —— 运行协议与判读口径（预注册）

日期：2026-09-15 深夜发射（首发 fp16）；**2026-09-16 02:10 fp32 重启（现行）**。
本文在出数前固定协议；拍板记录来自 2026-09-15 用户 Q1–Q10 回复。
吞吐/事故/形态选择的完整证据见 §4a–4c。

## 1. 目的与定位

用 NWPU matrix300 上的 **A3 制式**（PBM/D5-B 基座 + UDPR-v1 K64，P2-BRR 关闭）
在 WHU-1024 全量（train 2943，不滤空图）从头训练 150ep，替换论文主表 WHU 行。

**动机（论文一致性）**：ICASSP 正文方法 = CSPM(P+B+M)+UDPR（A3 架构），但现行
WHU 行（76.4/73.4，fast150）来自 `rd_p2` 形态（P2-BRR 开、UDPR 关），P2-BRR 在
正文中从未描述。本跑让 WHU 数字与论文声称的方法第一次自洽。

## 2. 拍板记录（2026-09-15，用户原话口径）

| # | 决定 |
|---|---|
| Q1 | (a) 制式平移，从头训；NWPU checkpoint 不参与 |
| Q2 | A3 全制式（已在另一台机器报告最优）；单独 wrapper，不混入口 |
| Q3 | P2 模块（P2-BRR）不再启用 |
| Q4 | 不设 bbox 门槛：赌 WHU 量够大、两指标都训好（无回退预案） |
| Q5 | 150ep、每 2ep 验证、多套选模（segm-best/bbox-best/composite/last），其余旧口径 |
| Q6 | A 系同款增强（hflip+vflip 0.5 + 多尺度 896–1152 @0.5） |
| Q7 | 2×48G；EMA+AMP 开；有效 batch 8 与每 epoch 优化器步数不变（LR 5e-4 不动） |
| Q8 | 换新架构指纹；历史 ckpt 只能靠 git 找回旧代码跑（已接受） |
| Q9 | a3r/a4 不上（主论文未报告 a3r、机制复杂变量多；a4 留下版本）；承认弱点 |
| Q10 | 直接上，时间紧任务重 |

## 3. 制式契约（architecture_id = `r1_c4_pafpn_coarse_points_box_dense_emb64_udprk64`，与 NWPU a3 同 ID 族）

- pafpn / coarse / points_box_dense / stride16 / dense 源不 detach / PE mask-downscaling 解冻（D5-B, lr_mult 0.1）
- final mask：roi_balanced_dice @ full_image 坐标；segm 评分=detector
- UDPR residual_v1：K=64, hidden=128, point_loss=1.0, delta_logit_max=2.0（不物化 mode 键，保持 v1 历史 ckpt 契约）
- P2-BRR=0；R3/dense-capacity/ROI-SAM/warmup 全关
- 增强五元组 = `_run_vhr10.sh` A 系同款（hflip0.5+vflip0.5+ms0.5 value 896..1152）
- 新架构指纹：`06e71d06f5171a0abaa8dfa9817ad7ed7a7fa8a33ec63365129d760109909957`

## 4. 启动形态与实测

- wrapper：`scripts/ablations/whu1024_a3_pbm_udprk64_full_fast.sh`（入口 `_run_ablation.sh`，契约校验 OK）
- GPU1/2（RTX4090 48G），**2 rank × bs1 × accum4 = 有效 8（fp32/AMP=0，2026-09-16 02:10 起）**；**368 优化步/epoch、总 55200 步，与 fast150 调度逐点一致**
- 显存实测 ~23G/卡（fp32 bs1，余量充足）；fp16 首发（bs2）见 §4b 事故记录
- seed 44 / AMP=0 / EMA tracking-only（赛后 val 对比 raw vs EMA 定 test 出场权）
- 选模四别名：best(segm) / bestB(bbox) / bestC(0.5*bbox/mAP+0.5*segm/mAP) / last；早停 patience=10 次验证（=20ep，fast150 语义）自 E90 起
- 冒烟（小子集）已验证：UDPR 参数组 38217 在训、TAIL 遥测正常（零初始化恒等）、D5-B active、P2BRR 0%、DDP 同步审计偏差 0
- **教训（已入协议）**：from-scratch run 的冒烟必须覆盖 ≥ warmup+150 优化步——首发两次冒烟（15/40 micro）都没碰到 step ~110 的死亡点，导致 NaN 在正式 run 才暴露
- 训练日志：`logs/ablations/whu1024_a3_pbm_udprk64_full_fast_tr1.0_va1.0_20260916_021050_pid2414015.log`（fp16 首发日志：`..._20260915_232026_pid2348404.log`，已废）
- checkpoint：`/data1/.../ablations/whu1024_a3_pbm_udprk64_full_fast_tr1.0_va1.0/`（fp16 废权重归档 `..._nan_amp1_bak_20260916/`）

### ⚠ 4b. fp16 发散事故与 fp32 切换（2026-09-16 凌晨处置记录）

**事故**：首发（2026-09-15 23:20，AMP=1 fp16、bs2×accum2）在 **optimizer step ~110**
（100 步 warmup 满 LR 后 ~10 步）出现首个非有限损失（00:35:11, epoch=1），此后全网
NaN：E2 起所有损失 NaN、trainer 跳步（E6 时 nonfinite 计数 3498）、val DT=0。
权重已废，目录归档 `whu1024_a3_pbm_udprk64_full_fast_tr1.0_va1.0_nan_amp1_bak_20260916/`。

**对照证据**：fp16+lr5e-4 在 NWPU A 系（稀疏，`_run_vhr10.sh:131` AMP=1 契约）与
WHU P2BRR/roi_local 形态（fast150 有 18ep AMP A/B 验证）都稳定；A3×WHU 密度组合下
fp16 从未验证过。**fp16 假设被实验否定**：同制式 fp32 复现（02:10 重启）顺利越过
死亡点（opt step 150/250/350+ 全程 0 非有限）。

**处置**：wrapper 默认改为 **AMP=0（fp32）+ bs1×accum4**（= 8/24 paper run 的原始
micro 形态；368 优化步/epoch 调度不变）。bs1 而非 bs2 的依据见 4c。

### 4c. 吞吐实测与形态选择（分段计时证据）

| 形态 | 实测 | 折算 | 判定 |
|---|---|---|---|
| fp16 + bs2×accum2（首发） | ~9.5-10.9 s/micro | 4.75 s/样本 | ❌ NaN + 反常慢 |
| **fp32 + bs1×accum4（现行，2 rank）** | **1.45 s/micro** | **1.45 s/样本** | ✅ 现行形态 |
| fp32 + bs2（GPU3 单卡分段） | 4.39 s/micro | 2.20 s/样本 | ❌ 每样本更慢 |

fp32-bs2 分段（`SEGMENT_TIMING` 探针，100 micro 平均）：prep 0.283 / fwd 1.600 /
item 0.032 / bwd 2.477 s——**fwd+bwd 占 93%，纯计算主导**，无同步病可修。
独立 bench（`scripts/probes/bench_full_image_targets.py`）测得 targets+support 构造
仅 ~250 ms/micro（bs2 密度）——**早先"每 RoI 同步税"假说被定量否定**，向量化修复
按证据关闭（当前形态上限收益 <10%，不值得动损失路径）。

**现行 ETA**：1471 micro/epoch × 1.45 s ≈ **35.6 min/epoch → 150ep ≈ 3.7 天**
（+ 每 2ep 验证开销，实测后回填；总估 3.7-4.2 天，完赛 ~2026-09-20）。

**E1 实收（fp32 形态，2026-09-16 02:47:33）**：34 min/epoch，train=1.419，全程
0 非有限；**num_rois=0 的空 RoI batch 出现 4 次全部正常通过**（排除了
"WHU 空图 × fi/UDPR 路径 DDP 死锁"假说——该组合从此有正面证据）。

**E2 首验（2026-09-16 03:23）**：val(627) segm/mAP **0.6267** / bbox 0.6864——
仅 2ep 即达到 dev10%/15ep 筛选的水平（~0.62-0.63），学习轨迹健康；四别名 ckpt
全部正常落盘。验证耗时 ~2 min/次（rank0_only bs4），全程 75 次 ≈ 2.5h。
**最终 ETA ≈ 31-34 min/epoch × 150 + 2.5h ≈ 3.4-3.6 天，完赛 ~2026-09-19/20。**

**已否决加速项**：bs4（探针 43.1G/48G，多尺度 1152 密图必炸）；fp16（NaN）；
bs2（每样本更慢）；向量化（收益不足）；OMP 线程（381s vs 388s 无差异）。

### 4a.（历史）首发吞吐记录（fp16 形态，已被 4b/4c 取代）

- fp16+bs2 稳态 ~10.9 s/micro（含预热 14.9），主进程单核 ~99%、GPU 跷跷板；
  当时归因为同步税，**4c 的分段计时已推翻该归因**（正确解释：fp16×bs2 组合病）。
- 指纹不变性：AMP/batch 形态均为启动参数而非 config 键，现行 DRY_RUN 复核
  `model_fingerprint=06e71d06...` 与首发逐字节一致。

## 5. 判读口径（出数前固定）

- 主判读：val(627) 选模四别名中**按预注册规则取 test 出场 ckpt**——
  1. raw vs EMA 先在 val 上对比，胜者出 test；
  2. test 主行 = segm-best（与主表 Ours 行历史口径一致）；bestB/bestC/last 全部落档供消融与附录；
  3. **禁止跨别名取最大值填主表**。
- test 推理：`infer_whu_checkpoint.sh` 同协议（maxDets=100，无 TTA，2220 张）；
  需以 wrapper 同款 env 重建 A3 形态（新指纹）。
- 伴随记录：bbox 全量 12 指标、Boundary AP（附录 E 联动重推）、参数量表（附录 D，
  +UDPR 38217 参数 / D5-B 4684 参数）。
- 噪声标尺：0.008（项目内训练噪声尺度）；单 seed，不做跨 seed 归因声明。

## 7. 种子复刻臂（seed 45，GPU3，2026-09-16 03:27 自主拉起）

主 run（seed 44）之外的同配置独立复刻，用户第三张卡预算授权。**唯一差异 = seed
44→45**；其余逐项同 §3/§4（fp32、bs1、150ep、A3 全制式）。单卡形态
1 rank × bs1 × accum8 = 有效 8，2943 micro/epoch ÷ 8 = **368 优化步/epoch，
与主 run 调度逐点一致**（= machine2 matrix300 的单卡等价形态）。契约校验 OK。

- 目的：matrix300 文档遗留的"bbox 缺口家族归因待 seed 复刻判定"在 WHU 主表数字上
  直接闭环；主 run 出头条，本臂提供跨种子证据。
- 日志：`logs/ablations/whu1024_a3_pbm_udprk64_full_fast_s45_tr1.0_va1.0_20260916_032727_pid2444880.log`
- checkpoint：`/data1/.../ablations/whu1024_a3_pbm_udprk64_full_fast_s45_tr1.0_va1.0/`
- 判读：与主 run 同 §5 口径；跨臂只做"同向/噪声带内"描述性对比，
  **不做 cherry-pick**（两臂各自按 segm-best 出 test，报双双落点）。
- ETA：单卡吞吐实测后回填（初速 ~1.8s/micro，估 5-6.5 天，完赛 ~9/21-23）。

## 6. 下游联动清单（出数后）


- 论文主表 Ours-WHU 行替换 + 消融表是否补 WHU 版（UDPR 增益当前仅 NWPU 证据）
- `WHU1024_EXPERIMENT_TRACKING.md` Ours 节更新（先补登 fast150 的 76.4/73.4 口径）
- Boundary AP 附录、参数量附录、pred.segm.json 重落盘
