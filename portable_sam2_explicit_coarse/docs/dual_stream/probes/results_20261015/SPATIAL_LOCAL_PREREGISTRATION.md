# 空间对齐及局部注入质量冻结探针预注册

日期2026-10-15，关联EXP-0006/0007，A类冻结诊断。用户要求先回答P2/全层注入，再继续空间/局部检查；本轮不增加P2模块、不训练、不修改默认权重或配置。

## 范围与固定设置

当前冻结开发子集全部153验证图/4496次建筑出现/237栋建筑；不是完整生产集或test。固定E72 strict EMA及正确RGB-D、FPN、PE/high、原检测框score。每图一次完整特征及RPN/bbox，用只读hook捕获SAM输入E0及原输出E1，有效残差R=E1−E0。

## 六个固定对照

原位identity直接用E1。四方向将R在SAM32²网格上平移±.25特征cell，即卫星512²中的±4像素；正dx/正dy表示输出向右/下移动，grid_sample使用反向坐标(x−dx,y−dy)，bilinear、align_corners=false、padding=zeros，非torch.roll、不wrap。shift4_right/left/down/up全部报告，不后选一个当最终模型。

增加smooth_symmetric：四个方向平移R的算术平均（在原有效覆盖内）。用于区分方向性变化与插值/平滑。所有非identity结果在原SAM覆盖>0处注入，原无支持处保持E0；有效残差本就在无支持处为0。移位可能损失边缘支持，必须承认这是下游残差干预，不是重新投影或相机标定真值。

E0+R因舍入不作为identity，identity用E1以便精确回放。主评价为各方向相对identity以及相对symmetric的GT共同框IoU，伴随完整segm/bbox AP。每图一次RPN/bbox；GT框相同。identity完整框score/mask与历史、GT框IoU必须逐值复现；三类不变数据/特征/参数hash或version检查失败则停。

## 空间数值检查

1. 在与实际grid_sample相同32²大小的合成脉冲/线性斜坡上核对方向和.25cell位移：right应将脉冲权重.75留原格、.25移到右邻，反向/竖直相应；边界不wrap。
2. 相机world↔local/同相机depth反投影再投影roundtrip，使用各图5相机各最多256个有效采样像素（先取256格点再过滤，逐视角记录有效数并要求至少1），报告最大数值残差。它只证明内部几何代数一致，不能证明渲染真值标定、真实像素中心、单表面高度或跨视角语义准确。
3. 现有投影是世界单表面BEV→卫星细像素竞争→按卫星特征stride分桶；不使用通用resize/affine_grid去对齐覆盖范围。核对整数分桶/行列约定并保留有限分辨率/单表面近似的局限。

## 局部质量统计

每个GT框mask记录IoU、3像素方形核膨胀/腐蚀构成的边界带IoU（dilation半径3、erosion半径3）、侵蚀内部FN比例（内部空时null、不入均值）、远外部FP像素/GT面积。都是GT框条件诊断，不是AP或精确的论文boundary-IoU标准。

记录细像素投影支持占GT比例、支持区几何quality均值、SAM coarse覆盖/置信GT均值、边界带支持比例、图像边缘截断、GT面积。由project_mass/quality构成fine图；SAM conf/coverage读取原forward输入并nearest还原到像素用于归属，不是语义前景置信。

关联上一轮半强度−原强度GT框IoU差，与局部支持/置信和边界误差分组。预定fine_support [0,.25),[.25,.75),[.75,1]，size<32²/32²–96²/≥96²，clipped/nonclipped（GT贴任一图像边界）；coarse weak为coverage<.25或confidence<.25，仅描述其GT比例。统计不作因果归因，不用GT生成可部署门控。

## 不确定性和门

1000次配对图像bootstrap seed1730，图像内sum/count保持实例等权；四方向相对identity、相对smooth共8个IoU差，用所有差值的最大绝对bootstrap偏离构成同时探索性区间（delta±q95(max|boot−delta|)），控制本轮方向挑选，不作总体显著性。另对237栋先内部平均再等权。AP报告完整点值，本轮不再重复昂贵AP bootstrap；空间相关/选模/训练seed不确定性仍未覆盖。

若某方向相对identity segm AP≥+.005、GT IoU≥+.002且同时区间下界>0，且相对smooth GT IoU≥+.001且同时区间下界>0，称“存在该下游残差方向偏移的支持证据”，不能称相机标定错误或直接修改默认。smooth相对identity segm AP≥+.005且GT IoU≥+.002则称“对称平滑有改善线索”，不解释为方向性错位。若没有方向满足门，则只说本轮±4px未检出支持，不能说全空间对齐正确。

局部统计只描述与减弱收益/边界误差的关系。所有六cell全部报告，固定检查点，不训练、不扩扫描；后续方向待结果决定。
