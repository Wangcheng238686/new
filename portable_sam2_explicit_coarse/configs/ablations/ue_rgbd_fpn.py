"""FPN-only mechanism control; absent SAM route creates no parameters."""
_base_ = 'ue_rgbd_dual.py'
model = dict(_delete_=True,**_base_.model)
model.pop('sam_fusion')
