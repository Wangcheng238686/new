
## COCO-eval 双轨说明

trainer 内置的 build_coco_gt_and_dt/run_coco_eval 与 utils/coco_eval_utils.py 版本有意不同轨：
训练侧 run_coco_eval 含自定义 maxDets 的主 AP 重算（统一 maxDet=100/150 协议下最佳模型选择依赖它）；
训练侧 build_coco_gt_and_dt 支持 rles 快速通路以在验证期及时释放 1024x1024 稠密掩码。
utils 版两者皆无，仅服务于 checkpoint 推理与探针。两侧原始 helper（_mask_to_rle/_xyxy_to_xywh）已收敛为单源。
