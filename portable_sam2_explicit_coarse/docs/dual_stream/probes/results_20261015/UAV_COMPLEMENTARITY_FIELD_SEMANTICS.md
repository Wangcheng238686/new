# 附加字段语义澄清（不改已运行源码或summary）

summary.spatial_audit.train_val_overlapping_satellite_AABB_pairs 实际使用冻结manifest `footprint_xy_m` 字段的轴对齐框相交对数。此次未重新核对producer对该字段的全部视野定义，正式结论应称“manifest footprint AABB重叠对”，不要径称“整张卫星图AABB重叠对”。数值0本身复算无误。

feature geometry[..., -1] 为投影后ROI采样coverage，支持非零仅按其ROI均值>0定义，不是SAM dense框支持、不是真实GT可见性。
