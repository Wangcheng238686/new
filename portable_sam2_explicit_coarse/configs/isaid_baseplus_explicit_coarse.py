"""iSAID variant of the explicit coarse-mask prompt route (15 classes).

Inherits the WHU-1024 Base+ explicit-coarse config and only switches the
detection head to the official iSAID 15 categories (mask head is
class-agnostic via the SAM2 decoder). Dataset wiring is handled by
``--use-isaid-coco`` in train_rsprompter_fusion.py: 800x800 patches are
resized to the 1024x1024 model input by the dataset class.
"""

_base_ = ["./whu1024_baseplus_explicit_coarse.py"]

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=15),
    ),
)
