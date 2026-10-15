# 等权候选与learned CVT形状读出：最终独立审核

VERDICT：PASS。未发现阻断项。四个预注册主门全部FAIL，两stage均无本轮预算下的互补支持；不得据描述C优于U绕门、追加hidden/channel/seed/steps或宣布生产分割收益。

审核者frozen_probe_audit，未参与提取、小头fit或汇总开发。已读工作树`UE_STAGE_SHAPE_20261015.md`及前序实施/smoke记录。本轮亲自CPU全cache检查、固定权重重放及统计重算；因CPU/GPU异构卷积误差，又经root明确授权在GPU1仅重放四图固定小头，不建optimizer、不反传、不重拟合、不跑原生产模型。只/tmp写工件。

## 全281cache验收

亲自执行独立检查器`independent_cache_check.py`，281份全部SHA、protocol/helper/生产source、strict EMA、身份/形状/finite/range、共同支持与不变量通过。128train/153validation共12621/4496RoI；2239/237 visible建筑互斥，独立从cache重建等manifest。每图GT slot/image/globalid合法、共同支持0的四U/C map为0，geometry公共prob和coverage字段一致，common coverage≤original coverage，全体ROI未后筛。

614.031秒、peak allocated2,316,940,288字节、cache2,879,598,544字节，均在原900秒/20GiB/5GiB预算内。train原256协方差仅12621×256=3230976读出单位，两stage相同；不是独立像素样本或语义信息量。首smoke NonePE hash异常及仅跳过None的修复按前序记录保留，无结果选择。

## 12小头训练契约

拟合manifest/source/protocol/feature/norm/sampletable/12权重及train/val预测SHA全部核对。固定seed1831、AdamW lr.001 wd.0001、batch64、last1000、12头各独立权重，同架构初始化逐值重建吻合记录；独立重生全部1000×64图像均匀再图内RoI均匀采样表，逐值相同。loss日志step为1及100/200…1000，无best或earlystop，脚本逐step一次opt.step，共1000；不冒称独立重复训练验证优化轨迹。

train-only标准化实现与前序timing重算已核：正式norm SHA固定，身份-only val隔离检查先行，无val targets用于归一化/超参/训练；所有12头最后训练完才载入val评估。SG有效变量较少，S64/C/CX/U/UX追加相同32维和71通道预算。主Nonlinear、Linear伴随固定，无后选。

## 固定权重预测重放

预先选择训练首/末图7/1261、验证首/末图287/1226，34/94/13/21RoI，四图全部12个固定小头共48项输出。CPU从每图cache、已保存norm和权重独立重放：最大绝对预测差8.2671642e−5、最大relativeL2 2.3089182e−5、单图Brier最大差1.0393560e−6；未通过审核者初设的较严异构allclose，不能报CPU bitexact。原记录保存`independent_prediction_replay.json`。

为区分异构数值差与实现偏差，root授权GPU1仅固定48输出重放，使用原torch/cuDNN确定性flags、原batch64分块、原归一化/权重。亲自执行独立`independent_fixed_head_replay.py`，四图12头输出**全部bitexact**保存预测；结果`independent_gpu_prediction_replay.json`。无optimizer、反传、fit或新的原网络提取。CPU卷积异构差不构成模型bug或效应门变更，且Brier差量级远低于.001工程幅度。

## 主终点与区间重算

独立从保存validation预测重算全部12个Brier、seed1830的1000次153图配对sum/count bootstrap、四个Nonlinear98.75%CI、四个Linear95%伴随及两架构C−U95%描述，误差<1e−12。所有4496RoI包括91个共同支持零，建筑237；没有用子组或边界替代主分母。

|主比较control−stage|均值|98.75%区间|门|
|---|---:|---|---|
|C vs S64|.00018294709|[.00006489653,.00033167347]|FAIL|
|C vs CX|.00059829454|[.00047859091,.00073704166]|FAIL|
|U vs S64|−.00050990813|[−.00065429200,−.00038193061]|FAIL|
|U vs UX|.00010049637|[−.00002586966,.00021050618]|FAIL|

C两差虽区间下界>0但均小于.001，不能称过门。两stage_support均false；C−U描述正差不能覆盖容量/空间对照门失败。工作树数值/判读与独立重算相符。边界、建筑、尺寸等完整伴随由root另有独立复算；本审核不冒称另跑过所有伴随汇总。

## 结论边界

U已是生产候选slot等权聚合，非无损原tokens；C含query/位置/残差/FFN，不单隔离attention。sameI使读出位置支持匹配，roll仍人为OOD。两stage公共stride16池化、32维随机压缩和固定1000步优化限制可读性；公共卫星multilevelROI14→16与两stage直接ROI16不是全部同sampler。协方差秩和阶段差异不证明不可逆信息损失。

任务为16²占据残差Brier，非最终SAM mask/AP。验证153图已反复开发并参与checkpoint选择，非独立test。图像bootstrap不覆盖空间依赖、重复建筑、选模与训练seed。支持正确关账“本轮等权候选替代未恢复足够互补”的固定设置，不证明CVT保留全部形状信息，也不否定未来匹配预算的新问题。

本PASS只完成上一轮结果审核；新投影尺度probe另有协议、静态审和分级准入，不自动授权其full或fit。
