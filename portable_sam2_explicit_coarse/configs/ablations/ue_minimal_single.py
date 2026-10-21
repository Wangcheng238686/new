"""Sealed satellite architecture, native512, coarse64, AP250 protocol."""
_base_ = './ue_rgbd_single.py'
model = dict(test_cfg=dict(rcnn=dict(max_per_img=250)))
