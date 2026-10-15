# 同CVT投影尺度工程草案（未执行）

## 最小可解释问题

同冻结E72 CVT256/相同geometry entry，先生产投影stride16(32²)与stride4(128²)。粗尺度经固定bilinear(align_corners=False)重建到128²，与细尺度在共同fine支持上用相同RoIAlign16²读出。比较的是“生产池化尺度+固定重建后的形状可读性”，不宣称native32消费者逐值等价或投影造成不可逆语义损失。两尺度仍带相同query/FFN和单4m格表达限制。

## 支持/错位冻结提案

定义fineD为原project_mass*project_quality按128²target累加的分母，V=(fineD>0)，I=V & roll(V,32,32)。C128=bilinear(C32,128)，F128=Fproduction128。先四图C128/F128及各自roll(32,32)后乘同I，再统一spatial_scale1/4、aligned/sampling_ratio沿生产flags、输出16²的RoIAlign。公共quality/coverage取细投影q/c同I，所有六cell同geometry。ROI全部保留，missing0与I减少曝光单列，不后筛主分母。

两个尺度pool都保留全部原entry，不以GT筛source/像素；I是读出/重建后共同mask，不改变源原投影。粗32仍可含mask外子区上下文，fine仅本细格pool，这正是池化+重建轴，不能说所有内容支持完全相同。bilinear也可混邻粗格；smoke保存native32原ROI16与重建128ROI16的差指标（RMS/MAE/relativeL2），仅界定路由差，不拟合新cell或按偏差选路径。

错位固定物理卫星128px两轴，coarse经重建后roll32finecell；与native32roll8并bilinear重建可能在wrap边界不交换，统一以共同fine route为准，不隐称物理误标定。I含循环边界，不提供真实错位因果。

## 复用与新缓存

复用上轮281图严格SHA、128hashtrain/153val、sat32/sat_extra32、prob/target16/boundary_weight/boxes/global_id/image_id/gt_slot逐数组复制。旧geometry含上轮32²intersection不同，不复用；旧C也是native32ROI，不复用。新缓存六cell SG/S64/C/CX/F/FX，共71输入，两个stage与shift同256→32 Gaussian seed1813。公共sat512带原PE、多层ROI14→bilinear16，明确与UAV sampler不同但所有cell同base。不增加原256维度/更高头容量/新seed/非线性层。

## 资源/完整性提案

不需要候选uniform聚合或原256cov，故冻结提取大致不高于上一281614秒（但新128²投影/ROI增加成本，smoke实测后核定），预计281≤15分钟/峰allocated<20GiB/cache<5GiB。只/tmp写缓存，正式模型冻结无optimizer/AP/反向。先3train+3val smoke：correct hookon/off所有feature exact、CVT只读取、两尺度payload/input/params/hook不变、project_source相同/独立D核对、I四图共同、cache旧字段逐值复用、coarse native vs重建route偏差如实报告。烟核不选路由/不看val小头得分；full+fit另准入。
