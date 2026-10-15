# P2冻结跨尺度迁移探针预注册

日期2026-10-15。关联EXP-0006/0007。A类冻结诊断；用户授权补P2探针，禁止自动训练或直接修改生产结构。仅当前冻结开发子集153验证图/4496出现/237建筑，非完整生产集或test。

## 问题和首轮范围

当前P2是FPN最高分辨率stride4，512输入对应128²；其512通道与现有P3融合器接口相容。已有81.34%GT mask RoI分配P2。接口相容不等于语义分布相容；本轮借用原P3融合器的已训练权重，不新增随机参数，不称已训练P2模型。

首轮先固定原模型检测框及score，不让P2改变RPN/bbox/NMS。报告“固定原检测框分数的条件segm AP”和GT共同框IoU，不称完整新模型推理AP/检测收益。全部条件bbox自然不变。若预注册主门通过，才进入单独完整检测探针；本轮不自动训练。

## 固定五对照

固定dual E72 strict EMA，原correct RGB-D的P3/P4融合、SAM embedding、PE、高分辨率特征不变。只改变传给mask head的FPN x[0]，其余层逐值保持。使用不改输出hook捕获正确BEV和原P3融合输入/输出，另在同一目标几何下提取既定donor_a RGB的BEV。

1. baseline：原P2，无新增残差，必须精确复现历史mask/RLE与GT框IoU。
2. native_correct_q：正确BEV用原生产BEVToSatellite投影到P2 128²，原P3融合器f(P2,U128,Q128,C128)−P2得到迁移残差R2，P2+.25R2。预定主比较native_correct_q−baseline。
3. native_donor_q：仅R2改为供体RGB BEV（目标深度/相机及几何不变），P2+.25R2_D。供体用已冻结donor_a映射，不选供体；已有train供体/RGB-D失配局限沿用，不能作净内容因果贡献。
4. upsample_correct_q：原正确P3的实际差分残差R3，bilinear align_corners=False上采样到128²后乘原P2 coverage>0，再P2+.25R3up。是“复制既有低分辨率融合信号”的迁移对照，门控输入和特征统计仍不同，不能当严格纯分辨率单变量。
5. native_correct_one：P2+1R2，只作预定压力对照，不能按最好倍率重选。倍率作用于已含原P3 alpha/coverage/confidence/gate的有效残差，不改learned alpha，不重新训练。

不额外归一化R2，不利用GT选择注入强度/范围。native输入是P2卫星特征，不是P3卫星特征，须明确门控存在跨尺度分布偏移。新残差所有P2无覆盖处严格为0。FPN原输出、PE/high、正确SAM及payload/参数均检查不变，hook调用各一次并移除。

## 指标/不确定性/门

主指标：native_correct_q−baseline GT共同框mask IoU；伴随固定原检测框分数的条件segm AP、各尺寸组、直接P2 RoI组和P3 RoI组、建筑等权。记录边界3px形态学band IoU辅助定位，非标准BoundaryIoU。baseline需与历史逐值相同；非P2 GT RoI的mask输出必须逐像素等于baseline（实际GT层级由官方map_roi_levels获取）；任何不变量失败停止修复。

1000次配对图像sum/count IoU bootstrap seed1740，500次条件segm AP配对图像bootstrap seed1741，重复图像独立位置；缓存点估计和fresh-ID重复普通COCO重算必须精确。区间仅探索性，忽略空间相关、选模、seed方差。全部五对照报告，不从q/one后选主臂。

主迁移门：预定native_correct_q相对baseline条件segm AP≥+.005且其探索性95%下界>0，同时GT框IoU≥+.002且其探索性95%下界>0。通过仅称“冻结跨尺度P2迁移有改善线索”，不宣布P2训练增益；未通过也不能否定新训P2融合器价值。

配对RGB附加门：native_correct_q相对native_donor_q的GT框IoU≥+.001且探索性95%下界>0，条件segm AP≥+.002；仅作配对内容敏感线索，保留RGB-D不一致。

原生细网格附加门：native_correct_q相对upsample_correct_q的GT框IoU≥+.001且探索性95%下界>0，条件segm AP≥+.002；只能支持这种原生投影迁移优于这份上采样迁移，不能证明纯分辨率因果贡献。

只有主门通过才考虑下一份完整RPN/bbox/P2探针；主门失败先归档并复盘表示兼容性，不自动改模型、扩倍率、换融合器或训练。

## 离线记录

记录P2/BEV/原P3尺寸、P3 learned alpha、各新增残差||delta||/||P2||（每图全通道全格L2比）、支持内RMS比、覆盖比例、GT官方RoI层级。机制记录不能替代条件AP/IoU，也不新增生产训练遥测字段。
