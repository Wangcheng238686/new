# GT形状提示oracle独立审核（完整记录）

## 实施静态审核

VERDICT：PASS，准入固定三图smoke；不自动准入153正式运行。

审核者：/root/gt_prompt_evaluator，未参与运行器设计或开发，仅独立编写CPU验收汇总。审核对象：/tmp/ue_gt_prompt_oracle_probe.py及PREREGISTRATION.md，并对照生产prompt helpers、dense transform、shape point miner与前轮trace实现。

发现与核验：

1. ShapeReplacement先于Observer注册。PyTorch同模块forward hooks按注册顺序执行，因此Observer和后续真实point/dense路径均接收实际替换后的GT64有限logits；干预未仅修改统计副本。
2. 替换逐chunk累计offset，退出时验证累计GT数量，保存target ranges。每次mask推理与基线chunk数量一致，并逐GT核对真实boxes顺序；原粗输出concat必须与baseline粗输出严格相同。
3. baseline从历史GT框恢复，框字节SHA、RoI level以及历史4496个最终IoU逐项精确对账。oracle额外断言RoI特征不变、替换logits与GT64*8-4严格相等。
4. 两臂共用单次双流特征提取。特征SHA、原RGB/深度/几何payload SHA以及模型参数_version不变。没有训练、optimizer、RPN、bbox/AP评估，生产代码未修改。
5. 原dense alpha、support、点规则、PromptEncoder、SAM视觉与tail保持原样。support=0实际dense与no-mask基底必须精确相同；该组负结果不能否定更准确提示的所有实现。
6. 三图smoke设计包含observer开关最终mask逐像素回归，以及原粗输出clone的sham：全部追踪张量与最终mask均须精确一致。正式不重复sham，节约GPU。
7. GT局部64²来自生产get_coarse_targets，属于crop/nearest后的条件形状诊断；不是完整512真值，不保证数学上限，不能外推UAV可达增益或解除detach训练收益。联合点+dense不能分别归因。
8. 唯一非阻断文案修正：PREREG原先写oracle coarse IoU=1，需要限定非空GT64。quality实现空union用clamp_min(1)，空target全背景IoU为0。独立汇总将记录empty/full target数量。

三图smoke与正式运行结果待追加。

## 固定三图smoke独立验收

VERDICT：PASS，建议主代理准入唯一153图正式冻结运行；本审核不自行启动GPU。

实际独立运行CPU验收 /tmp/ue_gt_prompt_oracle_summary.py --run-dir .../gt_prompt_oracle/smoke，exit 0，输出smoke_summary.json。验证45个GT出现、三图287/288/295及旧trace GT身份/顺序/框面积/层级逐项一致，基线最终IoU、粗IoU/BCE、support与点坐标/标签/有效性/正确性及实际token数量均精确重放。checkpoint、runner、base helper、trace helper、GT manifest和PREREG SHA独立复算通过（checkpoint以8MB流式读取）。

observer开关与clone原粗输出sham实际均逐像素精确，sham各stage也严格相同。全部显式冻结/清理不变量为true，固定RoI两臂无变化，基线各stage delta/paste/点变化为0，GT oracle输入实际等于固定±4。GT chunk sizes分别[13]/[14]/[18]，target ranges与GT slots严格对齐；smoke未覆盖多chunk，正式已保存逐chunk ranges并将独立核验。

两例support=0时实际dense仍精确等于no-mask基底；统计未将此解释为UAV覆盖。所有点/坐标/标签、粗BCE、阶段统计合法有限。smoke主门明确NOT_EVALUATED_SMOKE，不用于幅度选择、早停或收益主张。固定±4及唯一联合替换不变，等待153完整结果才落判。

## 正式153图独立结果审核

VERDICT：PASS。完整产物与冻结协议一致，预注册条件headroom门通过；仅支持该固定GT提示构造在当前冻结GT框条件下改善最终mask，不支持直接进入正式训练或UAV净增益主张。

实际运行独立CPU汇总，exit 0。验证153图、4496次出现、237栋建筑，42张多chunk图共197个chunk；每图chunk列表严格等于生产32切分，target ranges与GT slots逐项精确。两臂GT身份与历史baseline顺序一致，旧基线IoU/coarse IoU/BCE/support/points/tokens精确重放。所有显式冻结不变量通过；正式off/sham字段为null，表示只在已验收smoke测试，不伪称正式重复测试。checkpoint流式SHA和runner/base/trace/manifest/PREREG SHA复算通过。统计范围仍为开发子集，不是独立测试集。

### 主指标与预注册门

- baseline共同GT框最终mask IoU：0.841768524767971。
- joint_gt_prompt：0.8608314016667015。
- 配对均值差：+0.019062876898730412。
- 1000次image-paired sum/count bootstrap，seed1760，95% percentile区间：[+0.01703446700517864, +0.02130070948717839]。
- 预注册门mean>=0.005且区间下界>0：PASS，无后选cell或修改幅度。
- 建筑内先平均、237栋建筑再等权：+0.020008995920481998。
- 2624次出现改善>0.002，1335次退化<-0.002，存在异质性。

### 所有预定分组（描述，不用于后选主结论）

|组|n|最终IoU差|
|---|---:|---:|
|small|881|+0.013708273923461791|
|medium|2510|+0.01555639987525897|
|large|1105|+0.03129696065449962|
|P2|3657|+0.015899706085564014|
|nonP2|839|+0.03285038066958802|
|dense support=0|197|+0.00956340967093728|
|dense support>0|4299|+0.01949818628320942|

### 提示/机制伴随验证

GT64空目标0、全前景54。oracle coarse IoU=1由输入构造产生，不是学习结果。有效点准确率baseline P1/P2/N1/N2分别为4469/4496、4466/4496、2923/4479、4005/4460；GT替换后为4496/4496、4496/4496、4442/4442、4396/4396。准确率只统计有效槽；有效负点总数下降，不能把更少负点误作无效实验，全前景目标本就不存在局部GT负点。

4494次出现的点发生变化，共17646槽变化；4496个canvas/PE dense发生数值变化，4299个实际dense变化，4494个SAM原始和tail输出变化，4386个最终二值mask改变。两臂RoI全部严格不变。197例support=0的实际dense在两臂都精确等于no-mask基底，其中195例点改变、130例最终mask改变；这不证明修正dense支持规则的收益。主实验联合更改点+dense，不能把总体增益分别归因某一通道。

原预测coarse logits每RoI分位数再平均(min/p05/p50/p95/max)为[-11.0282,-7.7717,3.54488,9.71622,11.30155]，每RoI std均值6.02968。固定oracle +/-4同时改变形状及置信/分布，故不能称为纯形状精度因果效应，也不是数学最优上限；未按该结果调幅度。

### 允许和禁止的结论

允许：当前冻结SAM及其后处理能从这套GT64联合形状提示构造获得条件改善，提示生成/利用方向仍有可利用空间，不能因P2迁移弱就判提示路径无价值。

禁止：不能由oracle推断UAV能生成该提示、解除detach或改粗头训练一定提高、AP一定提高、单/双流净增益已成立、点或dense独立贡献已确认。图像bootstrap未覆盖空间相关、验证选checkpoint、单seed或完整生产数据分布的不确定性。当前GT框条件不计检测误差，部署性能需另行验证。本轮无训练，无RPN/AP，无额外单通道或支持修正实验。

## 主代理对汇总程序和结论的审查

VERDICT：PASS。主代理未编写两个子代理的运行器/汇总程序，逐段审查后要求补齐精确chunk ranges、历史baseline提示字段、scope-dependent sham/off及GT空/全前景验证；补齐并验收通过。

主代理另以直接逐GT记录和独立循环（未调用汇总程序函数）重算两组平均IoU差、237建筑等权、全部八分组均值及改善/退化计数、点有效/正确数、stage到达计数与mean RMS、seed1760的1000次配对图像区间，与summary的差均<1e−12。审计脚本和JSON分别归档为GT_PROMPT_ORACLE_ROOT_AUDIT.py与gt_prompt_oracle_root_numeric_audit.json。源/参数不变的事实由已复核运行不变量支持；未重新运行GPU。

最终判读：提示路径能够利用该GT构造，原P2迁移探针未产生粗质量改善不能归因SAM完全不接收提示。当前值得继续低成本拆分点/dense作用，尚不足以选择解除detach、直接新训P2融合器或扩全FPN。固定±4同时改变形状和置信、GT框与联合两通道限制必须保留。
