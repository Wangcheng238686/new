# Frozen BEV error stratification preregistration
No model fitting, GPU execution, new model, performance gate, or hyperparameter choice. Reuse frozen 281 feature caches and five readout prediction arrays. Existing full validation (153 images, 4496 repeated building appearances) only; train subset 128 images defines all quantile cuts.

Fixed strata: bbox small<1024, medium<9216, large; ROI mean projected coverage train 1/3,2/3 quantiles applied unchanged to val; also exact zero coverage and positive coverage. Original satellite coarse 8x8 occupancy Brier train thirds applied to val. size x coverage cross tabs. ROIs are never selected by GT foreground count.

Cell regions from 8x8 soft target: pure background t<=1e-6, pure foreground t>=1-1e-6, mixed otherwise. Mixed cells are coarse occupancy boundary proxies, not physical fine boundary. Report nROI,nCell, cell weighted mean and withinROI then ROI equal mean, building equal latter. Empty ROI regions excluded with explicit counts.

Contrast is control Brier minus SU Brier for SG/SX/S64 (SN supplementary); positive better. Separate oracle512 IoU headroom from 8x8 Brier; do not divide/convert to utilization. Oracle strata: delta<=0, (0,.002], (.002,.02], >.02 fixed before analysis. Correlate all ROI oracle headroom and controls' Brier improvements using descriptive Spearman. No subset bootstrap or multiplicity claims.

Feature descriptors: per-channel scale is train-wide cell std clipped1e-6 (no regression fitting). Local feature variation is RMS of feature minus its ROI cell mean divided by train scale; separately satellite32 and UAV32. Aligned-shift distance uses UAV train scale. Descriptor Spearman with SU-vsSG and SU-vsSX improvement and coverage/coarse error; no causal interpretation. Diagnostics top10 correctable misses sorted descending fixed oracle improvement with SU<=SG, for future inspection only.

Identity assert (image_id,gt_slot,globalid), existing prediction baseline/truth exact equality to cache, all4496 counterpart oracle targets. Coverage denotes sampled BEV projection support, NOT GT visibility or SAM dense support. No AP/final mask claim. All associations exploratory, correlated appearances; additionally aggregate by237 unique buildings.
