# WHU full-image 开发实验结果台账

记录日期：2026-09-06。来源为已完成训练的原始日志，非对话数值转抄。
本次共核对 14 次运行，各有 epoch 3/6/9/12/15 五次 segm/mAP 验证。
日志原先已落盘；本文件补齐 D0、D6 及复跑的汇总，并集中收录 D1–D5 作为对照。
仅记录与审计，不表示启动新实验、修改模型或提交 Git。

## 0. 2026-09-07 暂定结论更新：全验证集平均递增

当前应表述为：**P→PB→PBM 已在 WHU 完整验证集上呈现平均 best 指标递增；
Mask 配方的重复结果提供正面证据，Box 的独立稳定贡献仍需确认。**
不能笼统表述为“消融不成立”；也不能提升为跨种子、全量训练或 P2 增益已确认。
本节更新后文 §3 基于 62 图 dev 指标作出的阶段判读，不删除历史记录。

| 配置 | 全验证集预选 best：各次运行 | 描述性均值 | 次数 |
|---|---|---:|---:|
| P | 0.619496 | 0.619496 | 1 |
| PB | 0.628248 / 0.616028 | 0.622138 | 2 |
| PBM（D5-B） | 0.628565 / 0.629699 | 0.629132 | 2 |

Box 平均增量 +0.002643，Mask 配方平均增量 +0.006994。PBM 包括 dense 与 PE
mask_downscaling 适配，不能全部归因于画布内容。D5-B 是开发筛选候选，尚非独立确认。
这些权重仍只训练了 294 张图、15 epoch；best 由 62 图 dev 预选，再评估 627 图，
不是在 627 图上重新扫描 epoch。数据来源：
[完整评测汇总](/data/wangcheng/checkpoint/portable_sam2_explicit_coarse/refactor_golden/fulleval/summary.json)。
此前审计的 18 项 manifest 均覆盖正确的 627 个 validation ID、15,926 GT，严格加载无缺失。
其中 9 个 last_checkpoint 的 epoch=13（零基，即 E14），不是 E15；不可与训练日志末轮混称。

### 不稳定原因：事实与假说分开

- 已知：同一 seed/子集的 PB 重复，在全验证集 best 仍差约 0.01222；因此不能只归因于
  62 图评估样本少。训练轨迹或权重选择差异也存在，但尚未定位具体随机源。
- 已知：训练约 37 optimizer steps/epoch，总计 555 steps；100-step warmup 约占 18%。
  是压缩开发预算，不等于充分训练；LR 衰减结束不证明统计意义上的收敛。
- 合理假说：294 图的场景多样性有限，可能增加优化敏感性/过拟合，但没有受控的数据量
  对照，不能将“不稳定的主因就是数据少”写成结论。固定子集本身不是两次运行换了数据。
- 合理假说：非确定性数值、随机增强/采样、硬选点阈值与检测筛选可能放大训练差异；
  尚无逐步重放证据，不能只把 cuDNN/AMP 指定为根因。
- 合理假说：同源点/框/粗掩码信息冗余使增量偏小，影响“增量的可见度”；不等价于
  解释全部重复波动。冻结提示开关和误差分解可进一步检验。

验证顺序建议：先固定配方与独立数据划分种子/训练随机种子，再做重复；若专门定位
数据量与更新预算，比较 10% 短训、10% 延长训练、较大子集的匹配更新预算。
固定 epoch 扩大数据会同时增加 optimizer steps，不能将收益全部归因于样本多样性。
以上仅记录建议，未启动训练；增加数据不保证每个消融项增量变大。

## 1. 协议与证据边界

- WHU 10% train / 10% validation：294 张训练图、62 张验证图；15 epoch，每 3 epoch 验证。
- 两卡 DDP、AMP、每卡 batch 1、accum 4，名义有效 batch 8；基础 LR 5e-4、warmup 100 optimizer steps、EMA 关闭。
- full_image + roi_balanced_dice，stride-16 / embedding 64；按验证 segm/mAP 选 best。
- seed 44；复跑是同种子非确定性训练重复，不是独立多种子统计。
- best/末轮均为日志四位小数。曲线包含全部验证点，不将平滑 early-stop 指标当 best。
- 表中未包含 smoke、旧 roi_local 实验或冻结 oracle 干预；不能与这些协议直接混比。
- 本次未重新加载或计算 checkpoint 哈希；下方 checkpoint_dir 为启动日志记录的位置，不能代替权重完整性验证。
- 早期冻结诊断存在已撤回的 v1 结论；以 [v2 报告](p2_dense_frozen_diagnostic_report.md) 的证据边界为准。

## 2. 全部已完成运行

| Run tag | E3 | E6 | E9 | E12 | E15/末轮 | Best | Best epoch |
|---|---:|---:|---:|---:|---:|---:|---:|
| whu_d0fi_point_10p_2gpu | 0.5657 | 0.5228 | 0.5389 | 0.6087 | 0.6008 | 0.6087 | 12 |
| whu_d1fi_pb_control_10p_2gpu | 0.5415 | 0.5059 | 0.5809 | 0.6144 | 0.6078 | 0.6144 | 12 |
| whu_d1fi_pb_mask_alpha010_10p_2gpu | 0.5131 | 0.4978 | 0.5963 | 0.6090 | 0.6083 | 0.6090 | 12 |
| whu_d2fi_full_alpha010_p2b010_d050_10p_2gpu | 0.5224 | 0.5440 | 0.6033 | 0.6029 | 0.6069 | 0.6069 | 15 |
| whu_d2fi_mask_alpha010_10p_2gpu | 0.5247 | 0.5586 | 0.5913 | 0.6133 | 0.6059 | 0.6133 | 12 |
| whu_d3fi_full_p2esc030_d100_10p_2gpu | 0.5294 | 0.5573 | 0.5928 | 0.6155 | 0.6100 | 0.6155 | 12 |
| whu_d4fi_mask_csig_10p_2gpu | 0.5308 | 0.5304 | 0.5941 | 0.6078 | 0.6092 | 0.6092 | 15 |
| whu_d5a_mask_gate050_10p_2gpu_rep2 | 0.5223 | 0.5456 | 0.5984 | 0.6086 | 0.6063 | 0.6086 | 12 |
| whu_d5a_mask_gate050_10p_2gpu | 0.5727 | 0.5477 | 0.5940 | 0.6057 | 0.6037 | 0.6057 | 12 |
| whu_d5b_mask_gate050_pe010_10p_2gpu_rep2 | 0.5399 | 0.5492 | 0.6006 | 0.6162 | 0.6111 | 0.6162 | 12 |
| whu_d5b_mask_gate050_pe010_10p_2gpu | 0.5188 | 0.5455 | 0.5940 | 0.6117 | 0.6097 | 0.6117 | 12 |
| whu_d6fi_mask_gate010_pe010_10p_2gpu_rep2 | 0.5353 | 0.5613 | 0.6019 | 0.6064 | 0.6098 | 0.6098 | 15 |
| whu_d6fi_mask_gate010_pe010_10p_2gpu | 0.5610 | 0.5427 | 0.5957 | 0.6159 | 0.6061 | 0.6159 | 12 |
| whu_d6fi_pb_control_10p_2gpu | 0.5022 | 0.4876 | 0.5554 | 0.6055 | 0.6008 | 0.6055 | 12 |

配方释义：D0=Point；PB=Point+Box；Mask=Point+Box+dense；Full=Mask+P2。
D1/D2 Mask 为 alpha 初值 0.1、PE 冻结；D2 Full 残差帽 ±0.05；
D3 为 ±0.30 与辅助权重 0.20 的组合配方；D4 为 csig 画布。
D5-A=alpha 初值 0.5、PE 冻结；D5-B=同门控+PE 下采样适配（LR 倍率 0.1）；
D6 Mask=alpha 初值 0.1+相同 PE 适配。alpha 为可学习门初值，不是固定终值。
D5/D6 Mask 均关闭 P2。rep2 表示独立输出目录的同配置复跑。

## 3. 2026-09-06 的 dev 阶段裁决（全验证集更新见 §0）

1. **递增矩阵未建立。** Point best 0.6087 落在历史 PB 0.6144 与 D6 PB 0.6055 之间；Box 尚无稳定增益证据，也不能据单次 Point 判为无效。
2. D6 Mask 首跑/复跑 best 为 0.6159/0.6098，末轮 0.6061/0.6098。两次都高于 D6 PB，但复跑没有复现首跑 +0.0104 的 best 差距（复跑为 +0.0043）。
3. D6 Mask 两次平均 best 0.61285；两次 PB 的描述性平均 best 0.60995，差 +0.00290。历史/同期混合且样本极少，不是严格配对多种子显著性结果，不能定义“噪声带”。
4. D5-B 两次平均 best 0.61395，高于 D6 的 0.61285；小门控尚未证明优于大门控。不得宣称 D6 是已确认新基线。
5. D5-B 相对对应 D5-A 两次 best 差 +0.0060/+0.0076，支持 PE 适配值得保留为候选；并不自动证明相对 PB 的稳定收益。
6. D3 末轮训练局部 gain：Dice -0.000658、boundary-F1 +0.001417；旧 Full 不可拼接到新 PE 适配 Mask 上冒充最终消融。
7. Point-only 仍使用检测框作 ROI 几何、粗掩码作挖点源；它只关闭 box/dense 提示输入，不是无检测框或无粗掩码模型。

## 4. 机制与 Box 日志审计

D6 两次末轮的 dense 注入范数比约 6.55%/6.60%，gate 约 0.1090/0.1085；
PE 的 final-mask 梯度和 epoch 参数更新均非零。复跑 PE 梯度 group-L2 RMS
0.03554，更新 group-L2 RMS 0.0003553，10 张量活跃 100%。
这证明路径学习，不证明验证收益稳定。口径见根目录 DEBUG_FIELDS.md。

当前 BOX 日志的 fallback_count、refined_proposal_iou_sum、refined_valid_count
在 proposal-only 路径初始化为零，refined_proposal_iou 因此为零；
不能解释为框与 GT 的 IoU 或 Box 的贡献为零。
PROMPT/use_box_token 在前向生成，但当前打印前缀未包含 PROMPT；
现有日志没有 Box 开关导致最终掩码/AP 差异的因果观测。
本次仅记录该审计发现，未增加字段或改变打印行为。

## 5. 建议下一步（未执行）

- 用现有 dev 规则已选出的 best 与末轮权重，在相同 627 张完整 validation 上统一评估；
  不扫描所有 epoch 重新择优。全验证集评估不是全量训练确认，也不能替代训练复跑。
- 固定同一 Mask 权重与检测/点坐标，只去除 box token、dense 或两者，诊断依赖与冗余；
  冻结干预不替代独立训练消融。
- 若启动 D0 选点系列，先保持点数/其余规则不变，比较第二正点的内部距离排序与安全近边界候选；
  得到候选后必须在 Box+Mask 配方复验，正式矩阵各行使用一致选点策略。
- Mask 配方确定后才做同配方 P2-on/off 对照；不同时改变门控、PE、选点、P2 幅度和损失。
- 以上均为待验证方向，不是收益承诺；未执行结果不得写入正式成绩表。

## 6. 原始证据路径

### whu_d0fi_point_10p_2gpu

- [原始日志](/home/wangcheng2021/project/new/portable_sam2_explicit_coarse/logs/ablations/whu_d0fi_point_10p_2gpu_tr0.1_va0.1_20260906_204123_pid380174.log)
- checkpoint_dir：`/data/wangcheng/checkpoint/portable_sam2_explicit_coarse/ablations/whu_d0fi_point_10p_2gpu_tr0.1_va0.1`

### whu_d1fi_pb_control_10p_2gpu

- [原始日志](/home/wangcheng2021/project/new/portable_sam2_explicit_coarse/logs/ablations/whu_d1fi_pb_control_10p_2gpu_tr0.1_va0.1_20260905_171025_pid1285074.log)
- checkpoint_dir：`/data/wangcheng/checkpoint/portable_sam2_explicit_coarse/ablations/whu_d1fi_pb_control_10p_2gpu_tr0.1_va0.1`

### whu_d1fi_pb_mask_alpha010_10p_2gpu

- [原始日志](/home/wangcheng2021/project/new/portable_sam2_explicit_coarse/logs/ablations/whu_d1fi_pb_mask_alpha010_10p_2gpu_tr0.1_va0.1_20260905_181437_pid1421927.log)
- checkpoint_dir：`/data/wangcheng/checkpoint/portable_sam2_explicit_coarse/ablations/whu_d1fi_pb_mask_alpha010_10p_2gpu_tr0.1_va0.1`

### whu_d2fi_full_alpha010_p2b010_d050_10p_2gpu

- [原始日志](/home/wangcheng2021/project/new/portable_sam2_explicit_coarse/logs/ablations/whu_d2fi_full_alpha010_p2b010_d050_10p_2gpu_tr0.1_va0.1_20260905_181928_pid1432707.log)
- checkpoint_dir：`/data/wangcheng/checkpoint/portable_sam2_explicit_coarse/ablations/whu_d2fi_full_alpha010_p2b010_d050_10p_2gpu_tr0.1_va0.1`

### whu_d2fi_mask_alpha010_10p_2gpu

- [原始日志](/home/wangcheng2021/project/new/portable_sam2_explicit_coarse/logs/ablations/whu_d2fi_mask_alpha010_10p_2gpu_tr0.1_va0.1_20260905_171056_pid1286495.log)
- checkpoint_dir：`/data/wangcheng/checkpoint/portable_sam2_explicit_coarse/ablations/whu_d2fi_mask_alpha010_10p_2gpu_tr0.1_va0.1`

### whu_d3fi_full_p2esc030_d100_10p_2gpu

- [原始日志](/home/wangcheng2021/project/new/portable_sam2_explicit_coarse/logs/ablations/whu_d3fi_full_p2esc030_d100_10p_2gpu_tr0.1_va0.1_20260905_203517_pid1718132.log)
- checkpoint_dir：`/data/wangcheng/checkpoint/portable_sam2_explicit_coarse/ablations/whu_d3fi_full_p2esc030_d100_10p_2gpu_tr0.1_va0.1`

### whu_d4fi_mask_csig_10p_2gpu

- [原始日志](/home/wangcheng2021/project/new/portable_sam2_explicit_coarse/logs/ablations/whu_d4fi_mask_csig_10p_2gpu_tr0.1_va0.1_20260905_203517_pid1718133.log)
- checkpoint_dir：`/data/wangcheng/checkpoint/portable_sam2_explicit_coarse/ablations/whu_d4fi_mask_csig_10p_2gpu_tr0.1_va0.1`

### whu_d5a_mask_gate050_10p_2gpu_rep2

- [原始日志](/home/wangcheng2021/project/new/portable_sam2_explicit_coarse/logs/ablations/whu_d5a_mask_gate050_10p_2gpu_rep2_tr0.1_va0.1_20260906_170518_pid4170496.log)
- checkpoint_dir：`/data/wangcheng/checkpoint/portable_sam2_explicit_coarse/ablations/whu_d5a_mask_gate050_10p_2gpu_rep2_tr0.1_va0.1`

### whu_d5a_mask_gate050_10p_2gpu

- [原始日志](/home/wangcheng2021/project/new/portable_sam2_explicit_coarse/logs/ablations/whu_d5a_mask_gate050_10p_2gpu_tr0.1_va0.1_20260906_105349_pid3483437.log)
- checkpoint_dir：`/data/wangcheng/checkpoint/portable_sam2_explicit_coarse/ablations/whu_d5a_mask_gate050_10p_2gpu_tr0.1_va0.1`

### whu_d5b_mask_gate050_pe010_10p_2gpu_rep2

- [原始日志](/home/wangcheng2021/project/new/portable_sam2_explicit_coarse/logs/ablations/whu_d5b_mask_gate050_pe010_10p_2gpu_rep2_tr0.1_va0.1_20260906_170518_pid4170497.log)
- checkpoint_dir：`/data/wangcheng/checkpoint/portable_sam2_explicit_coarse/ablations/whu_d5b_mask_gate050_pe010_10p_2gpu_rep2_tr0.1_va0.1`

### whu_d5b_mask_gate050_pe010_10p_2gpu

- [原始日志](/home/wangcheng2021/project/new/portable_sam2_explicit_coarse/logs/ablations/whu_d5b_mask_gate050_pe010_10p_2gpu_tr0.1_va0.1_20260906_105349_pid3483436.log)
- checkpoint_dir：`/data/wangcheng/checkpoint/portable_sam2_explicit_coarse/ablations/whu_d5b_mask_gate050_pe010_10p_2gpu_tr0.1_va0.1`

### whu_d6fi_mask_gate010_pe010_10p_2gpu_rep2

- [原始日志](/home/wangcheng2021/project/new/portable_sam2_explicit_coarse/logs/ablations/whu_d6fi_mask_gate010_pe010_10p_2gpu_rep2_tr0.1_va0.1_20260906_204123_pid380175.log)
- checkpoint_dir：`/data/wangcheng/checkpoint/portable_sam2_explicit_coarse/ablations/whu_d6fi_mask_gate010_pe010_10p_2gpu_rep2_tr0.1_va0.1`

### whu_d6fi_mask_gate010_pe010_10p_2gpu

- [原始日志](/home/wangcheng2021/project/new/portable_sam2_explicit_coarse/logs/ablations/whu_d6fi_mask_gate010_pe010_10p_2gpu_tr0.1_va0.1_20260906_182616_pid129381.log)
- checkpoint_dir：`/data/wangcheng/checkpoint/portable_sam2_explicit_coarse/ablations/whu_d6fi_mask_gate010_pe010_10p_2gpu_tr0.1_va0.1`

### whu_d6fi_pb_control_10p_2gpu

- [原始日志](/home/wangcheng2021/project/new/portable_sam2_explicit_coarse/logs/ablations/whu_d6fi_pb_control_10p_2gpu_tr0.1_va0.1_20260906_182616_pid129380.log)
- checkpoint_dir：`/data/wangcheng/checkpoint/portable_sam2_explicit_coarse/ablations/whu_d6fi_pb_control_10p_2gpu_tr0.1_va0.1`
