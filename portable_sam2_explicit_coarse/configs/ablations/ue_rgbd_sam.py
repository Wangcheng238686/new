"""SAM-image-only mechanism control; absent FPN route creates no parameters."""
_base_ = 'ue_rgbd_dual.py'
model = dict(_delete_=True,**_base_.model)
model.pop('fpn_fusion')
