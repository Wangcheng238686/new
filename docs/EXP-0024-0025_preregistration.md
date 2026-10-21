# EXP-0024 / EXP-0025 预注册：清洗后全量数据集上的封版单流卫星基线双臂（2026-10-21，启动前）

PI 决议（2026-10-21）：新纪元首对全量。数据集 =
`/data1/wangcheng/dataset/CitySample_SatUAV_TrainMinimal_v1_20261021`
（CitySample Sat–UAV 最小训练归档，3060 点 train 2142 / val 459 / test 459，
实例 126214/14980/44988；标签修订 `review99_v3_user_approved_20261021`，
COCO 未压缩 RLE 为唯一标签权威；原生 512×512；拆分由数据集方提供并声明
已防泄露）。**test 全程不读**；训练用全量 train，验证选权重用全量 val。

## 模型与协议（封版不变量）

- 模型：封存 a3sgm @512 原生（配置 `configs/ue_whu_a3sgm_512.py`，身份
  `ue-whu-a3sgm-512-v1`，指纹 `5305b315…`）——卫星视角单流，默认不上采样
  （数据原生 512，无 resize）。
- 协议：seed 44、DDP4 每卡 batch 1 ×累积 2（global 8）、lr 5e-4、wd 0.05、
  warmup 100 步、cosine、100 epoch、last 选择口径、amp、无 EMA、禁早停、
  SAVE_LAST_MODEL=1、val 每 epoch 全量 459 场。
- 数据适配：`--dataset whu`（whu_coco 路径）+ env 指向
  `annotations/{train,val}.json`、image subdir 为空；唯一代码改动 =
  `WHUCocoInstanceDataset._decode_mask` 兼容未压缩 RLE（counts 列表先过
  frPyObjects；对既有压缩 RCO/多边形路径零影响）——冒烟已对照归档官方
  reader 逐位验证（20/20 实例 mask 逐位一致；翻转一致性 144/144）。

## 两臂定义（唯一变量 = 训练增强）

| 臂 | EXP-ID | TRAIN_FLIP_PROB | 其余增强 |
| --- | --- | --- | --- |
| A 全关增强 | EXP-0024 | 0 | vflip 0、multi-scale 0、噪声 0、擦除 0（显式置零） |
| B 默认增强 | EXP-0025 | 0.5（trainer 默认） | 同上全 0（默认即仅水平翻转） |

注：旧 ue_raw_surface 格式在 trainer 内强制归零全部增强；本对走 whu_coco
路径不受该归零影响，默认增强 = 仅 hflip 0.5（历史 hflip-only 协议）。val
侧两臂均零增强。增强实现在数据集内（img+mask+bbox 同步翻转）。

## 四要素

1. 主指标：val 459 全量逐 epoch COCO segm/bbox AP@[.50:.95]；两臂对照差
   = EXP-0025 − EXP-0024（last 口径）；伴随：AP50/AP75、面积分层、逐 epoch
   曲线。
2. 选择口径：两臂均 last epoch 100。
3. 显著性：收线后对两臂 last checkpoint 做 val 459 场逐场 scene-paired
   percentile bootstrap（10,000 draws、seed 20261014，沿用拉通工具口径）；
   日常训练曲线读数不构成判定。
4. 判定（预注册）：
   - **AUG_WINS** = B−A 两主指标 bootstrap CI 下界均 >0 → 默认增强成为
     新纪元标准协议；
   - **AUG_NULL** = CI 含 0 → 记录"增强对封版模型无显著差"，以数值高者为
     习惯默认并在预注册外如实标注非显著；
   - **AUG_HARM** = 任一主指标 CI 上界 <0 → 增强有害，关闭增强为标准。
   - 单臂绝对水平不设门槛（本对是基线建立，非增益判定）；卫星基线 =
     两臂中 last 口径综合更优者，双流新纪元全部对照此基线。

## 完整性契约（wrapper fail-closed）

包装器钉死：dataset.json、splits/{train,val,test}.json、
annotations/{train,val,test}.json 的 SHA256、标签修订串、规模断言
（2142/459/459、实例 126214/14980/44988）、干净工作树、模型指纹
`5305b315…`、身份 `ue-whu-a3sgm-512-v1`、输出目录不存在。

## 工件与审计

wrapper `scripts/ue/run_exp0024_ddp4.sh`（全关增强）、
`scripts/ue/run_exp0025_ddp4.sh`（默认增强）；启动前独立审计；结论前结果
审计。两臂并行四卡共租；预计 wall time ~1.5–2 天（2142 图/epoch + 459 场
全量 val/epoch）。


## 补充登记（2026-10-21，收线前）：test 评估协议的输出上限修正

PI 授权的 test 中期 peek 发现双重上限掐住密集场景：**41% test 场景 GT
>100 实例（最大 238、p99=230）**，而模型 `test_cfg.rcnn.max_per_img=100`
与 COCOeval 默认 `maxDets=100` 双双生效；val 最大仅 87（上限从不生效，
训练期监控因此从未暴露）。同权重（EXP-0025 ep44 冻结）单变量消融：

| 协议 | segm AP | AP50 | AP75 | AP_s | bbox AP |
| --- | --- | --- | --- | --- | --- |
| cap100+md100 | 0.5233 | 0.7411 | 0.6191 | 0.4672 | 0.5674 |
| **cap250+md250** | **0.5788** | **0.8460** | 0.6771 | 0.5377 | **0.6325** |

**+5.5 segm / +6.5 bbox，增量集中于小实例与 AP50**（排名 100 之外
的检测几乎全是小建筑）。EXP-0024（ep42）同向（0.488→0.527）。

**收线规约（本节即登记，先于收线）**：EXP-0024/25 的 test 收口及此后
新纪元一切评估，统一 `max_per_img=250` + `COCOeval maxDets=250`
（评估工具 `scripts/ue/eval_citysample_test.py` 已内置；val 上限不生效
不受影响）。模型配置文件不动（保持封版身份），上限在评估工具层施加。
中期参考读数（ep44/42 best 冻结）：0025 test segm 0.579 / bbox 0.632；
0024 test segm 0.527 / bbox 0.556。