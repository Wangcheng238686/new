# 阶段证据包（2026-10-16）

本包供独立审查；不预设根因，不启动训练/尺度全量运行。模型生产代码与af76c8e相比git diff为空；当前HEAD5bc45b9，后续提交仅诊断/文档。

|编号|问题|原报告|已有证据/实际状态|
|---|---|---|---|
|01|联合训练及净收益|[UE_AD_RESULTS_20261015.md](/home/wangcheng2021/project/new-uav-dual/portable_sam2_explicit_coarse/docs/dual_stream/UE_AD_RESULTS_20261015.md)|两臂80轮；best mask .584958/.583000，bbox .618454/.619967；区间跨0|
|02|全子集框/排序|[UE_DEV_SUBSET_AUDIT_20261015.md](/home/wangcheng2021/project/new-uav-dual/portable_sam2_explicit_coarse/docs/dual_stream/UE_DEV_SUBSET_AUDIT_20261015.md)|153图4496出现237建筑；GT同框dual .841769 single .844073；NMS损失两臂相近|
|03|RGB消费|[UE_RGB_CONTENT_20261015.md](/home/wangcheng2021/project/new-uav-dual/portable_sam2_explicit_coarse/docs/dual_stream/UE_RGB_CONTENT_20261015.md)|几何固定替RGB donorA/B segm .550637/.549565；correct .584958；依赖非净贡献|
|04|融合路径|[UE_FUSION_ROUTE_20261015.md](/home/wangcheng2021/project/new-uav-dual/portable_sam2_explicit_coarse/docs/dual_stream/UE_FUSION_ROUTE_20261015.md)|RGB差主要SAM；GT框81.34%取未直接融合P2|
|05|强度|[UE_SAM_STRENGTH_20261015.md](/home/wangcheng2021/project/new-uav-dual/portable_sam2_explicit_coarse/docs/dual_stream/UE_SAM_STRENGTH_20261015.md)|SAM减半全量分割约降2.07pp，不支持统一减弱；非gate都正确|
|06|平移/平滑|[UE_SPATIAL_LOCAL_20261015.md](/home/wangcheng2021/project/new-uav-dual/portable_sam2_explicit_coarse/docs/dual_stream/UE_SPATIAL_LOCAL_20261015.md)|±4px/pool干预未过门；roundtrip仅代数自洽|
|07|P2移植|[UE_P2_TRANSFER_20261015.md](/home/wangcheng2021/project/new-uav-dual/portable_sam2_explicit_coarse/docs/dual_stream/UE_P2_TRANSFER_20261015.md)|现成P3fuser借P2冻结迁移基本持平；不是P2重训消融|
|08|提示链传递|[UE_P2_PROMPT_TRACE_20261015.md](/home/wangcheng2021/project/new-uav-dual/portable_sam2_explicit_coarse/docs/dual_stream/UE_P2_PROMPT_TRACE_20261015.md)|变化能传入SAM但coarse质量无稳定改善；197例dense support0|
|09|GT提示条件空间|[UE_GT_PROMPT_FACTORIAL_20261015.md](/home/wangcheng2021/project/new-uav-dual/portable_sam2_explicit_coarse/docs/dual_stream/UE_GT_PROMPT_FACTORIAL_20261015.md)|GTdenseonly+.015619最终共同框IoU，点+.004540；不证明UAV可预测|
|10|幅度/形状|[UE_DENSE_SHAPE_CONFIDENCE_20261015.md](/home/wangcheng2021/project/new-uav-dual/portable_sam2_explicit_coarse/docs/dual_stream/UE_DENSE_SHAPE_CONFIDENCE_20261015.md)|等幅GTvs预测形状+.016068；预测±4不改善|
|11|UAV冻结线性互补|[UE_UAV_COMPLEMENTARITY_20261015.md](/home/wangcheng2021/project/new-uav-dual/portable_sam2_explicit_coarse/docs/dual_stream/UE_UAV_COMPLEMENTARITY_20261015.md)|SU-SG Brier改善.000359613，SU-SX .000186775，均<.001门|
|12|读出与几何排查|[UE_BEV_WEAKNESS_DIAGNOSIS_20261015.md](/home/wangcheng2021/project/new-uav-dual/portable_sam2_explicit_coarse/docs/dual_stream/UE_BEV_WEAKNESS_DIAGNOSIS_20261015.md)|未发现全坍塌/正则失效；测量容量限制保留|
|13|原深度/单表面|[UE_BEV_GEOMETRY_QUALITY_20261015.md](/home/wangcheng2021/project/new-uav-dual/portable_sam2_explicit_coarse/docs/dual_stream/UE_BEV_GEOMETRY_QUALITY_20261015.md)|六图cache exact；所选bin内部z替代p99 .08263px，多表面混合存在|
|14|实际注意力|[UE_BEV_ATTENTION_20261015.md](/home/wangcheng2021/project/new-uav-dual/portable_sam2_explicit_coarse/docs/dual_stream/UE_BEV_ATTENTION_20261015.md)|六图投影query heightmass .796 vs uniform .804，joint .139 vs .126；非旧invalidheight混合口径|
|15|尾部误差关联|[UE_BEV_TAIL_ASSOCIATION_20261015.md](/home/wangcheng2021/project/new-uav-dual/portable_sam2_explicit_coarse/docs/dual_stream/UE_BEV_TAIL_ASSOCIATION_20261015.md)|六图未获得稳定尾部→双流退化关联；非因果|
|16|U/C形状读出|[UE_STAGE_SHAPE_20261015.md](/home/wangcheng2021/project/new-uav-dual/portable_sam2_explicit_coarse/docs/dual_stream/UE_STAGE_SHAPE_20261015.md)|CVT非线性读出优于等权，但4主门均FAIL；共同stride16/32维压缩仍受限|
|17|细投影尺度|[UE_PROJECTION_SCALE_20261015.md](/home/wangcheng2021/project/new-uav-dual/portable_sam2_explicit_coarse/docs/dual_stream/UE_PROJECTION_SCALE_20261015.md)|仅6图240ROI smokePASS，full281+拟合未运行；资源准入停止，不是效果FAIL|

重点原始工件：`/tmp/ue_uav_experiments/20261014_ad_seed42/final_analysis/`，`/tmp/ue_stage_shape_probe/`，`/tmp/ue_projection_scale_probe/`，`/tmp/ue_bev_geometry_quality/`。
读出小头确有拟合，生产网络始终冻结；不要把“无模型更新”误写成所有分析均无拟合。所有probe负门只否定对应受限候选，不是UAV信息上界。
