# Dense形状／幅度冻结探针：独立静态实施审核

审核者：gradient_design_audit；未参与该runner或协议设计、编码。2026-10-15。

VERDICT：PASS（静态实现允许三图smoke；正式153图准入仍要求smoke及下列manifest补齐通过）。

审阅文件：`/tmp/ue_dense_shape_confidence_probe.py`、`/tmp/ue_uav_experiments/20261014_ad_seed42/final_analysis/dense_shape_confidence/PREREGISTRATION.md`。未运行GPU、未改代码。

## 已确认

1. 三cell固定为原预测P、预测二值形状±4、GT64形状±4；未见训练、optimizer、backward、RPN或bbox/AP运行。所有运行在torch.inference_mode内。
2. 替换位于真实粗头输出；预测二值源由当前原logit>=0确定。GT源来自真实get_coarse_targets，按原chunk顺序取片。两者都经过原transform/paste、PE、dense alpha与support，没有暗改support。
3. 非基线cell在真实mask_decoder入口替换为基线实际sparse，不是统计用补零token；实际2/6token及箱token都包含在原样sparse内。逐目标SHA再次对照历史P源。
4. decoder image embedding、PE、high-res特征逐值核对原features；feature/geometry/RGB前后hash及parameters version、hooks清理有断言。没有改变SAM视觉输入。
5. 基线与G分别精确核对历史PP、PG逐目标IoU及dense SHA；不是只核对平均数。support0目标在两个干预cell最终paste mask逐像素等基线且dense-base norm为0。
6. 三图smoke各cell实际输入sham重复核对mask及sparse/dense，正式不重复sham（null），口径明确。
7. 主比较B-P和G-B预先固定，两个97.5%边际bootstrap及.005均值门，未挑选子组。B-P应解释为硬二值有限幅度／置信分布干预，包含去除形状内部置信梯度，不是纯温度校准。G-B是在P点条件下等±4幅度的源形状对照；GT64仍为oracle。

## 正式运行前的归档补齐

- manifest应写入严格重建所用完整architecture_contract（schema/id/fingerprint），当前审阅版本尚缺；同时保留checkpoint SHA、源脚本/helper/protocol SHA（后者当前已存在）。
- 记录实际dense_prompt_cfg（temperature、clamp、outside fill）和restrict_dense_prompt_to_box；避免仅记录alpha及detach而无法自描述源转换。
- 建议显式校验B实际注入源的二值形状与P原shape相同、有效logits仅{-4,+4}，并核对逐chunk source范围。公式已经满足该语义，显式断言可防未来实现偏移。G源同样校验等幅度。
- 当前snapshot chunk=32；脚本的range和size断言假定chunk>0。当前实验符合，但它不是通用chunk=0 runner。正式保留当前32并记入manifest即可。

## 不能据此作出的结论

静态PASS不替代smoke、完整运行和独立数值审；提示oracle效果不证明解除detach收益，不证明UAV可预测GT形状，不是AP或完整数据集结论。输入值保持仅覆盖当前声明的冻结干预，未验证任何训练资源或梯度数值。

## 定版修正与smoke原工件复核

VERDICT：PASS，准入唯一153图固定三cell运行。没有追加GPU。

定版runner SHA `3a6c98cdd808dc0bf1edbd9b52b2dd6fb5fe1aa787d70c90f2fe67db47319937`与smoke manifest精确一致。已补architecture_contract、trace helper SHA、P/B二值保持与±4值域断言；固定chunk==32。dense原cfg额外字段未单独重复写manifest，但完整strict snapshot/checkpoint SHA及合同已锚定，当前非阻塞。

独立CPU读取`dense_shape_confidence/smoke/`的run_manifest、invariance_checks和三组逐目标JSON：3图287/288/295、45GT、无missing/unexpected keys、无training/optimizer、参数版本不变；真实点全部6token；chunk为13/14/18，对应范围完整。所有声明exact布尔检查为True，三cell输入sham均True。稀疏SHA逐目标精确等P；B源前景计数精确等预测形状；G源前景计数等GT，源二值IoU全部1。每组2例support0的dense-base norm都精确为0，最终mask阴性检查True。

重新求均值：P 0.7930128877433098，B 0.7917017095437373，G 0.8062071131962905。仅作三图复算／接线验收，不解释完整效应方向。smoke不覆盖多chunk或2token，正式继续强制源码内范围、框序与实际输入检查。最终153图结果仍需独立数值审。

## 正式153图原始产物独立复核

VERDICT：PASS，完整固定三cell实施与历史回放不变量通过。本审核复算均值，不替代主代理对bootstrap及预注册门的独立数值审。

未追加GPU。直接读取`dense_shape_confidence/full/`三组4496逐目标JSON、run_manifest、153条invariance_checks及历史factorial的PP/PG逐目标JSON，不依赖作者summary。manifest complete=True，153图/4496出现，参数版本不变，耗时246.3014715秒；脚本SHA与定版一致。

- 153图所有exact布尔检查通过，正式sham字段均为null（没有伪称重做sham）；42图多chunk、总197chunk；实际全部6token，没有2token实际样本。
- 三组逐条(image_id,gt_slot,global_id)、support、actual sparse SHA及baseline sparse SHA精确一致。4496键唯一，不存在错序或重复。
- P与G分别逐目标精确复现历史PP、PG的IoU、actual sparse SHA、actual dense SHA，不是仅平均一致。
- B源64²二值形状的前景计数及coarse binary IoU逐目标精确等P；G源前景计数等GT64且源binary IoU全为1。实际±4值域及B源逐像素binary保持由运行时assert验证，原始coarse大张量未落盘，本次没有声称再次逐像素重算该张量。
- 每组197例support0：dense-base norm=0、paste XOR=0、最终IoU等P，阴性验证通过。P最终改变0例，B4230例，G4231例；改变数不是改善数。

独立复算全均值：P=0.8417685247679701，B=0.8413199689046548，G=0.8573875912505797；B−P=-0.0004485558633159997，G−B=+0.016067622345924373。这些仅为固定GT框、固定视觉特征下最终paste mask IoU。

解释必须收窄：B保持的是源64²阈值0的二值shape；原logits与±4之后仍要bilinear resize/paste和冻结PE，连续场过零边界／dense分布可随量化改变，不能称最终dense形状严格相同或纯置信校准。G−B是在等±4源幅度、原P点、原支持规则条件下由源GT64形状替换带来的平均条件改善，不证明UAV能够生成该形状，也不证明解除detach训练收益。当前均值不支持仅靠预测源±4量化带来改善；正式门仍须照预注册完整区间执行。

## 主代理结果与判定复核

VERDICT：PASS。主代理审查runner/summary，并另以独立脚本直接读取逐GT记录，重算seed1780三contrast的实例等权均值、图像sum/count区间、237建筑等权，均与作者summary差<1e−12；核对原点SHA、所有197support0实例IoU及paste XOR精确不变。脚本DENSE_SHAPE_CONFIDENCE_ROOT_AUDIT.py与对应JSON一并归档。

预注册主门原样执行：预测形状±4−原预测logits均值−.000448556，97.5%区间[−.000923192,+.000023090]，FAIL；GT形状±4−预测形状±4均值+.016067622，97.5%[+.013620570,+.018728376]，PASS。不后选分组放宽阈值。支持当前等幅度来源形状差有改善空间，不支持把GT收益解释为本次二值±4幅度变化本身。不能排除其它校准方案，也不能推断UAV可达、AP或解除detach训练收益。
