# A3SGT：UDPR 尾部完全梯度隔离（tailstop）

**状态：已实施、已过本地验证（模块级梯度隔离测试 + 18 臂全量验证 ALL PASS），
等待独立子代理审核后启动 matrix300 四卡训练。**

## 1. 动机（证据链）

a3sg（A0-SG + UDPR-v1）对 a0_sg 的推理结果为：segm +0.012，**bbox −0.022，
composite −0.005**——任何诚实口径下 composite 为负。"UDPR 在 A0-SG 上有增益"
未达成。机制定位：

1. a0_sg 的 bbox 0.7348 是全矩阵最高：梯度隔离（`SHAPE_DENSE_DETACH=1`）切掉
   dense 通路对共享主干的污染，检测器受益（a0 0.7208 → a0_sg 0.7348）。
2. v1 tail 有两条未隔离的梯度通道（`decoder_tail_refiner.py` residual_v1 分支）：
   `feat`（decoder 上采样特征）、`token`、`selected_logits` 均不 detach；且 refined
   logits 直接替换输出，full-mask loss 经 z′ = z + delta 回流主干。
3. 交叉验证：同一 UDPR，在已污染基座 a0 上 bbox −0.002（0.7208→0.7190），在
   干净基座 a0_sg 上 −0.022（0.7348→0.7130）。检测器越干净，被 tail 梯度伤得越重。
4. 训练侧 TAIL 遥测 a3 与 a3sg 几乎逐项相同——隔离没有削弱 tail 本身，−0.0093
   的 a3−a3sg 差距在基座×tail 交互（梯度协同通路被切）。

结论：A4/A5 攻的 gate/输入表示不是病根；病根是 tail 训练梯度打穿主干。

## 2. 设计（三句话）

新 matrix300 臂 `a3sgt`（tag `_a3sgt_pbm_sg_udprk64ts`，模型配置相对 a3sg 的
唯一差异 = `decoder_tail_refiner_cfg.mode: residual_v1_stop`）：

1. **tail 输入全 detach**：`residual_v1_stop` 模式下 `flat_logits`/`selected_logits`/
   `feat`/`token` 全部 detach，返回张量也构建在 detached 网格上——tail 侧任何
   损失（含直接落在 refined 输出上的对抗性损失）都无法到达 decoder/主干。
2. **训练期 full-mask loss 走未细化 z**：`sam2_mask_head` 在
   `_decoder_tail_train_base_logits and self.training` 时返回 base logits，
   base 的训练目标与 A0-SG 逐位同构——bbox 保住 0.7348 是结构性保证。
   eval/inference 仍部署 z′ = z + delta（trainer 验证循环 model.eval() 已核实）。
3. **模块、选择、损失、遥测全部继承 v1**：同一 `mlp` 参数名（a3sg checkpoint
   可换模式加载做探针）、同一 `_v1_point_loss`、同一 TAIL/* 遥测。初始
   delta=0（零初始化末层），初始前向精确等于 A0-SG。

## 3. 判定门（预注册）

- **效用门（matrix300，诚实口径 = last 与平台，不含 best 尖峰）**：
  bbox 相对 a0_sg 在 ±0.005 内（结构性预期）**且** segm ≥ +0.005（即
  composite ≥ +0.005）。过门则"UDPR 在 A0-SG 上有增益"以 tailstop 达成。
- **机制门（训练遥测）**：`TAIL/delta_abs` 从 0 单调增长后稳定；
  `TAIL/selected_bce` 下降；`loss_mask` 轨迹与 a0_sg 同量级（≈0.07–0.08 终盘）；
  `debug_mask_target_fill≈0.018`（fi 签名）。
- **失败判读**：若 bbox 门过而 segm 门不过（post-hoc tail 写入无净价值），
  则证明 v1 的 segm 增益本质是基座协同而非 tail 写入，UDPR 谱系关账；
  若 bbox 门也不过，则隔离路由实现有缺陷，先查工程再下结论。
- 与 a3 的对照（非门、记录用）：a3sgt ≥ 0.6689（a3 平台）则隔离谱系全面胜出。

## 4. 实施与验证记录（2026-09-13）

| 项 | 内容 | 结果 |
|---|---|---|
| 模块 | `decoder_tail_refiner.py`：mode 集、`residual_v1(_stop)` 共用 v1 结构、stop 分支 detach（含返回网格） | py_compile PASS |
| 路由 | `sam2_mask_head.py`：构造期 `_decoder_tail_train_base_logits`；forward 训练期保留 base | py_compile PASS |
| 配置 | `whu1024_baseplus_explicit_coarse.py`：`residual_v1_stop` 分支仅在该模式物化 `mode` 键 | 指纹回归 PASS |
| 接线 | dev.sh `a3sgt` 臂、matrix300/series 白名单、`_run_vhr10.sh` 契约 echo 增 `tail_mode` | DRY_RUN 契约正确 |
| 隔离测试 | `scripts/smoke/test_decoder_tail_stop.py`：对抗损失零泄漏 + v1 对照确实泄漏 + 前向逐位等价 + 零初始等价 | PASS |
| 臂验证 | `verify_p2v2_arms.py` 18 臂全量：结构、污染隔离、指纹、pairwise | **ALL PASS**；a3sg vs a3sgt diff = {mode} 单变量；全部遗留臂指纹不变 |
| 预存缺陷修复 | verify harness expected 里 `detach_dense_input` → `detach_input`（a0_sg 臂 9/12 加入时未对齐，本轮全量跑暴露） | 已修 |

## 5. 边界

- 不改 a3/a3sg/a4sg/a5sg 的任何配置、checkpoint 或历史结果；不动 DCR 文档的
  A4/A5 审计结论。
- 不引入新损失、新遥测字段；K64/hidden128/cap2.0/q-init/协议全部与 a3sg 相同。
- 论文表述：过门前只能称"candidate tail-isolated UDPR"。
