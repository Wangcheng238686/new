"""NWPU VHR-10 variant of the explicit coarse-mask prompt route (10 classes).

Inherits the WHU-1024 Base+ explicit-coarse config and only switches the
detection head to the VHR-10 10 categories (mask head is class-agnostic via
the SAM2 decoder). Dataset wiring is handled by ``--use-vhr10-coco`` in
train_rsprompter_fusion.py: variable-size images (~500-1000px) are resized
to the 1024x1024 model input by the dataset class. Annotations are the
instance-mask COCO conversion from the Precise Mask R-CNN (IGARSS'19)
release, split 70/30 with seed 44 (see coco_split/).
"""

_base_ = ["./whu1024_baseplus_explicit_coarse.py"]

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=10),
    ),
)
