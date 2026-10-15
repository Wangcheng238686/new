# 冻结GT点与GT密集提示2×2交换探针

2026-10-15，EXP-0006/0007。仅已有开发子集153验证图、4496次建筑出现、237栋建筑；非独立test。E72 strict EMA、correct UAV、GT共同框。无训练/optimizer/RPN/AP、无幅度搜索，不改生产源码和正式遥测。

## 四组固定干预

先在每图同一组视觉特征上执行原预测P和上轮GT oracle G（同一get_coarse_targets64²，固定前景+4/背景−4；原点挖掘/转换/alpha/support保持），捕获真正送进SAM decoder的sparse（invalid中和后真实2或6token，包含固定box token）与dense（原gate和support后的实际输入）。不得将Observer补0统计副本喂模型。

四组：baseline=(P sparse,P dense)、gt_point_only=(G sparse,P dense)、gt_dense_only=(P sparse,G dense)、joint_gt_prompt=(G sparse,G dense)。对角为live源运行，逐GT终点IoU精确复现上轮gt_prompt_oracle两组；交叉两组用同P上游生成的调用，在decoder forward_pre_hook仅交换actual sparse/dense，其他positional/image/high/flags/kwargs保持。交叉上游仍预测P，不把coarse称GT；点diagnostics必须按actual source映射，非上游P误称G。

GT sparse包括与P逐值相同的最后两个box token，仅点编码和实际token个数可能变化；若2vs6，保留真实序列结构，效应包括slot有效性/长度，不能声称纯坐标效应。GT dense是同有限±4形状经原PromptEncoder/gate得到，包含置信度/形状/分布差异，非任何可达提示的数学上界，也非UAV可达增益。

## 准入/阴性对照

独立静态审查后root仅准入固定3图smoke（287/288/295）；在两个对角分别通过decoder交换原同源sparse/dense，输出paste mask逐像素精确复现live对角。不新增无observer重复。完整153须另外准入且只4组。每图只提取一次视觉特征；源prompt CPU保留每chunk真实张量直至四组结束，之后释放，不保存大张量。

所有chunk严格生产32、GT顺序/boxes精确；decoder image/PE/high/其他args/kwargs哈希四组一致；source sparse/dense值/hash四组结束后不变；P/G box token逐值相同；交叉actual sparse/dense输入逐值对应选定源且逐GT SHA记录。参数_version、FPN/SAM/PE/high特征hash、UAVRGB/深度/几何hash不变、hook恢复。manifest保存脚本/依赖/GTmanifest/prereg/checkpointSHA和git/config/实际阈值.4。

预期197个dense support0目标在P/G源均严格为no-mask基底，gt_dense_only最终paste mask必须与baseline逐像素相同，不能只比IoU；若失败立即停查。此support是32²框内支持，非UAV几何coverage。不修复support，不追加分组条件或阈值。

## 输出schema/主终点与不确定性

逐GT：identity(image_id,gt_slot,global_id,bbox_area,roi_level)、mode、gt_iou、paste_xor_pixels(对baseline)、dense_support_cells、actual_point_source(P/G)、actual_dense_source、upstream_coarse_source、真实sparse_token_count、actual/source sparse/dense SHA、按actual sparse源记录point valid/correct/labels/coords/local_yx、实际dense对no-mask基底L2、coarse_target_fg_pixels。live P/G点diagnostics逐项精确复现上轮；oracle局部粗掩码完美是输入构造，非预测收益。

预注册两主contrast：gt_point_only−baseline、gt_dense_only−baseline，均全4496 GT框paste IoU。1000次paired image sum/count bootstrap seed1770；两contrast同时控制采用Bonferroni，各97.5%区间quantile(.0125,.9875)。探索性headroom门为每contrast均值>=.005且该区间下界>0；所有组必须报告，不按结果后选。交互 GG−GP−PG+PP、GTdense条件下的GTpoint效应(GG−PG)、GTpoint条件下的GTdense效应(GG−GP)及GG−PP仅描述，同步给95%paired区间但不重复充当显著性筛选。

固定描述分组：small/medium/large按bbox面积32²/96²，P2/nonP2，dense support0/非0，237栋建筑先内部平均再等权。报告每组最终IoU/有效点正确率、paste改变数、support0。图像bootstrap仅描述本子集，不覆盖空间相关、checkpoint选择、seed/数据集版本不确定性。不从此探针推断解除detach训练必有收益，也不等同检测/实例AP。
