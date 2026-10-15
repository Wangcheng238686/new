# 冻结跨视角特征互补探针：独立实施审核

2026-10-15；审核者gradient_design_audit，未参与runner/summary编码。只读审查，没有启动GPU、拟合或改脚本。

VERDICT：静态PASS，允许六图smoke；full281提取及CPU拟合准入建议待smoke原始工件核验后追加。

审阅：`/tmp/ue_complementarity_probe.py`、`/tmp/ue_complementarity_summary.py`、`/tmp/ue_uav_experiments/20261014_ad_seed42/final_analysis/uav_complementarity/PREREGISTRATION.md`。

## 实施核定

1. Satellite provenance正确：neck forward hook立刻clone输出，此时尚未进入post-neck UAV融合。原extra PE和mask RoI extractor继续使用，非从已融合FPN反向剔除UAV。共用E72双流训练权重，**不是独立单流E66**；共享卫星参数曾接受双流损失训练，不能称无UAV训练影响的特征。
2. GT与RoI按原dataset顺序读取；真实coarse target通过head.get_coarse_targets、原GT boxes和slot assignment构建。逐图恢复local instance id的像素mask，与item.gt_masks逐值核对；local/global映射及唯一性可防ROI/建筑ID错序。UAV投影用生产BEVToSatellite至128²，原RoIAlign层spatial_scale=.25，aligned/sampling_ratio等来自原层，和卫星RoI目标框一致。下采样及目标fraction规则与协议一致。
3. 直接粗头rawcoarse在smoke要求逐值等原zero-UAV完整maskhead捕获输出。正式不运行SAM，所测目标是pre-UAV卫星粗掩码在8²的残差可读性，不是最终mask或AP。
4. 数据选择hash('uav-complementarity-v1:'+image_id)，固定128train、全部153val，无按mask质量/误差挑选。train/val样本身份和manifest SHA被summary再次重算；visible global IDs交集要求为空。验证GT只用于fit之后的评估，标准化mu/std、intercept、ridge beta只用train。lambda=.01、投影seed与维度固定，没有验证选超参。
5. Ridge目标y=GTfraction−coarse_prob；预测coarse+residual后clip[0,1]，按每ROI64cell等权算Brier。代码主差controlBrier−SUBrier，正号代表SU更好。IoU为8²占用阈值.5，不能当coarse64或最终mask IoU。
6. Shared39dim（sat32+7geom），各cell额外32slots，总71。SU/SX使用同一投影，SX仅UAV projected map roll(32,32)，保留原q/coverage；控制改变局部appearance对应而非physical camera intervention。SN独立固定imagehash随机32维；S64第二独立sat32。
7. Bootstrap按图聚合ROI sum/count，而非把所有网格行当独立；两主比较SU-SG、SU-SX使用各97.5%区间、Bonferroni两门，均值>=.001且下界>0，两者都通过才overall PASS。support_positive及尺寸组全量描述，不能后选子组过门；建筑等权是伴随指标，不替换预定ROI等权。
8. manifest记录checkpoint、脚本/helper/protocol SHA、合同、矩阵SHA、原图身份与NPZ SHA；summary验证全281、固定列表、源SHA及前后不变量。未发现optimizer/backward、生产模块替换或参数更新。

## 必须保留的解释边界

- SG的32slots为零，活跃维数明显较低，**不是等活跃容量对照**。报告必须如实列active_input_dim。
- S64/SU/SX/SN参数形状一致和相同ridge正则构成预算控制；其协方差与有效自由度不完全相同。若SU只胜SG/SX，而不胜S64/SN，不能宣布UAV相对同预算卫星表征具有优势。预注册主门仍原样执行，不能事后更换主比较。
- 正例仅为当前固定线性读出下UAV信息可及性，可能依赖空间/几何关联；负例仅否定该分辨率、特征投影、线性容量和128train预算，不能否定UAV原始信息。
- EMA E72按本验证集选出、特征编码器曾在全部627train训练，诊断fit128不意味着编码器只见过128图。建筑ID零交集、footprint和origin审计不消除同场景空间相关和选模泄漏。
- 使用GT共同框及GT建立监督是诊断条件，不是部署可用输入；不得据此直接转入正式训练或断言AP改善。
- smoke仅3train+3val，不能声称覆盖全体P2/P3、多chunk或所有几何退化。正式继续断言NPZ形状/有限值、身份、source hash及参数版本；任何异常停止且incomplete，不能用时间预算截断后报完整结果。

## 后续验收

待六图smoke完成，直接读取manifest/NPZ核selected IDs、projection/hash、GT/ROI顺序、shape/有限值、coarse exact、参数版本及source invariants；不替作者改summary，不追加GPU。

## 六图smoke独立原工件验收

VERDICT：PASS，建议准入唯一281图固定特征提取及预注册一次CPU ridge拟合；不包含正式训练授权。

读取smoke/run_manifest和6个NPZ，未使用作者summary替代raw核验。runner SHA `897a178a5d8c54bc1d607c0bdb660485419eaf38fa17b5ff84a2b4128b0758c2`与manifest一致。strict EMA无missing/unexpected；training/optimizer=False；parameter versions unchanged=True。smoke complete=False正确表明不是完整281图，而非运行失败。

实际train图7/15/25对应34/58/103 RoI；validation图287/288/295对应13/14/18 RoI，共240。每个NPZ的文件SHA、image_id/gt_slot顺序、所有数组长度/有限性均通过。sat、sat_extra、UAV、shift、noise均[N,64,32]，geometry[N,64,7]，prob/target[N,64]；prob与target均[0,1]，geometry首列逐值等prob。每图global_id唯一，smoke可见train158栋/val19栋、共享0。

实际RoIAlign spatial_scale=.25、aligned=True、sampling_ratio=0、output14²，卫星512channel、BEV/UAV256channel128²、coarse64²、最终读出8²与协议一致。6图均记录smoke_coarse_exact/source_features_unchanged/payload_rgb_satellite_unchanged/hooks_removed=True；这对应runner源码中的逐值assert，raw coarse大张量未落盘，本次没有伪称重新对大张量计算。UAV与shift特征6图均有数值差异（平均绝对差约.365–.524），未退化成相同appearance对照；这不是质量判据。

全量仍保留preUAV provenance、visible global-ID零交集及空间邻近审计、固定列表/NPZ SHA、active dim与容量控制报告。smoke结果不落性能门，正式若gate fail照实关账，不追加训练或更改lambda/seed。

## 正式281图及读出结果独立复核

VERDICT：PASS（实施／结果归档）；预注册工程支持门overall FAIL，不进入自动训练升级。

未重跑GPU或五组ridge拟合。独立逐个读取281个NPZ、manifest、val_predictions并从原预测重算Brier及四个contrast；验证五cell参数/标准化仅train来源的源码。主代理另行独立重拟合五组和区间；本条不冒充第二次拟合验证。核验输出落盘`/tmp/ue_complementarity_independent_raw_audit.txt`。

### 身份与全cache

- 完整281图，128train/153validation，17117个RoI（train12621、val4496）；提取耗时357.76秒（诊断工件耗时，非部署性能）。parameter_versions_unchanged=True，全部source features、payload/RGB/satellite、hooks不变量通过。
- 281个NPZ文件SHA、image_id/gt_slot顺序、每图global_id唯一、所有特征shape/有限值和prob/target[0,1]均独立通过；geometry首列逐值等coarse prob。固定hash选128train及全153val清单独立重算一致，图ID零交集，无重复图。
- 实际可见train2239栋、val237栋，共享global ID为0；不是把17117次出现误当独立建筑。
- runner/protocol/helper当前SHA与manifest一致；E72 checkpoint路径、manifest SHA、完整architecture_contract与历史strict EMA追踪manifest精确相同，严格加载无missing/unexpected。矩阵seed及shape固定为(512,1811)、(512,1812)、(256,1813)，hash在manifest可追溯。
- val_predictions的truth/baseline/global_id/image_id与按manifest顺序拼接153个原cache逐值相同；五组预测shape均4496×64、有限且在[0,1]。不存在重新按结果挑选val行的证据。

### 从原val_predictions重算

| cell | Brier | 活跃输入维数 |
|---|---:|---:|
| SG | 0.035374335115366036 | 39 |
| S64 | 0.035356366168366786 | 71 |
| SU | 0.03501472213476101 | 71 |
| SX | 0.0352014973519677 | 71 |
| SN | 0.0353746573695728 | 71 |

所有Brier及差值与summary精确到1e-14以内一致。SU相对SG下降0.000359612980605027，相对SX下降0.000186775217206685；相对S64下降0.000341644033605769，相对SN下降0.000359935234811786。SU在这些预定描述控制下均更低，但两主幅度均低于预定.001，不能放宽门。

summary两主97.5%边际区间分别[.0003088726335,.0004188762678]、[.0001445643718,.0002343517015]，区间为正但幅度门失败；静态核对bootstrap用image paired sum/count、seed1790、1000次及.0125/.9875分位数与预注册一致。独立root负责重算区间。本审核没有把统计区间为正偷换成工程PASS。

### 结论边界

有**微弱的、当前固定读出可访问的条件增量信号**，并非没有UAV信息；但未达到预注册工程支持幅度，不足以依据GT dense oracle自动训练P2或解除detach。SG活跃维39、其他71须如实报告；S64/SN在本次表现未消除SU微弱优势，但没有多seed/多投影/独立数据验证，也不是独立单流模型比较。

空间审计按manifest的`footprint_xy_m`计算AABB：selected train-val交叠0对、最近origin距离最小912.5m、中位1352.14m。虽然summary字段名含satellite，当前审核未核producer定义，故不进一步解释为实际卫星图像ground footprint独立。它不消除同场景、编码器见过全部627train、E72验证选模和单seed的限制。8² Brier的改善不能与GT提示oracle最终mask IoU直接比较或换算，不能推导AP增益。

预定小目标子组中SU相对SX及S64的Brier reduction点估计为负，因此总体微弱可读信号不适用于所有尺度；不后选中大目标修改主门。root已另行完成五cell重拟合及CI独立复算PASS，见工件目录`root_numeric_audit.json`；本审核仍保持独立cache/预测复算职责边界。

## 主代理拟合与结果独立重算

VERDICT：PASS（复现与解释）；预注册工程门FAIL不变。主代理未调用作者summary函数，直接读取281原NPZ，重算五cell训练集标准化mean/std、intercept及ridge beta，五组系数差<1e−10；以所得参数重建全部验证预测，Brier/8²IoU/建筑等权及四对照差与summary<1e−12。再按seed1790重算两个主imagepaired sum/count区间和门，与summary<1e−12。脚本UAV_COMPLEMENTARITY_ROOT_AUDIT.py及对应JSON归档，不把复算当作第二seed实验。

所有结果限定冻结双流训练表征、GT框、随机投影32维、8²读出和128train预算。没有证据把它称为独立单流净增益、UAV可达最终mask/AP收益或解除detach有效性。主代理综合另两路后不自动训练；下一步先定位BEV投影/局部形状信息和冻结读出限制。
