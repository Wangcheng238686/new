# Native几何固定权重空间诊断（2026-10-22）

## 目的与固定合同

独立审查后，检验已有native几何E10正信号是否依赖正确局部空间字段。不新训练、不选权重、不改既有运行任务。
固定EXP-0017 E73及native_geometry/last.pth、test459、FP32、原scores、maxDets250。
结果前固定计划及SHA在`/data/wangcheng/checkpoint/ue_native_spatial_diagnostic_20261022/plan.json`。
用户授权显存充足时共卡运行；GPU0/2各串行运行一项，启动时至少8GiB空闲。保留其他训练任务。

## 五种输入

- correct：原几何，须逐图逐预测复现已归档native结果。
- zero：相同检查点、相同观测支持，四字段为0。
- shuffle：在每张图有效支持内联合置换四字段向量，seed20261022+pointID；保留支持及字段联合多重集。
- zero_rms / shuffle_rms：同样干预，逐mask调用将残差RMS匹配正确几何残差，使用同一alpha。仅诊断，不当新成绩。

置换破坏空间自相关及高度差的物理一致性，属于分布外干预；下降只提示依赖，不能单独证明几何语义正确。
若正确空间结构不优于扰动，应收紧逐实例正确几何归因。零字段也有输入分布变化，需与训练zero控制分开解释。
不以本轮结果调alpha/重排检测分数。所有输出与原归档GT、检测框/类别/分数逐图一致。

## 遥测与验收

沿用records.stats末mask调用语义；独立all_call_telemetry.json保存全部mask调用，每调用rois及RMS，不能与末调用统计混淆。
新增离线字段raw_intervention_rms、reference_delta_rms、rms_match_scale、residual_cosine_to_correct、residual_change_rms。
RMS匹配按每调用全部RoI/像素/通道计算，不做每像素归一化；scale完整披露，不裁剪。
固定残差方向余弦和幅度变化辅助解释；完整卫星state SHA前后核对。
全部五种推理完成后，干预对correct及correct对satellite做同一500次图像配对bootstrap，原分析程序不变。
输入变换CPU门已通过：联合向量分布逐值保持、支持/无效0不变、确定性、correct恒等、zero恒零。
运行产物：`/data/wangcheng/checkpoint/ue_native_spatial_diagnostic_20261022`。
状态与最终收益另记，不将启动/工程门通过当结果。

## 已完成推理的点值（完整区间统计仍在运行）

| 固定权重输入 | Mask AP |
|---|---:|
| correct | 64.7307 |
| zero | 64.7610 |
| shuffle | 64.5492 |
| zero_rms | 64.7789 |
| shuffle_rms | 64.5508 |

卫星基座64.5510。correct逐预测复现旧归档；五组均通过卫星state前后SHA、GT及检测输出一致核对。
全部调用均为1500次/40932个RoI；RMS匹配最大相对误差<2e-7。
首轮Mask配对：shuffle−correct −0.1815pp，95%[-0.3040,−0.0737]pp；
zero−correct +0.0304pp，95%[-0.0205,+0.1905]pp。
不能宣称zero显著优于correct，也未证correct优于zero；边界及匹配版本区间未完成。

这削弱“正确局部几何驱动推理收益”的强解释：字段训练对照收益仍成立，
但该权重移除字段未降低整体Mask AP，幅度匹配后仍如此。
待完整统计完成，独立report owner会验收并写RESULTS.md、complete_audit_summary.json及仓库结果报告；
未有这些实际产物前不宣称最终统计完成。日志finish_report.log，失败不会伪造成功报告。
