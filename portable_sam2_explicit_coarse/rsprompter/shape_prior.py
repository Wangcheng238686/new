"""Compatibility facade: implementations live in sibling modules.

ShapePriorInjector  -> rsprompter/shape_prior_injector.py
ShapePointMiner     -> rsprompter/shape_point_miner.py
SmallMaskDecoder / RoIBoxEncoding -> rsprompter/shape_prior_injector.py
Registry names and every external import path are unchanged.
"""
from .shape_prior_injector import (
    RoIBoxEncoding,
    ShapePriorInjector,
    SmallMaskDecoder,
)
from .shape_point_miner import ShapePointMiner

__all__ = ["ShapePriorInjector", "ShapePointMiner", "SmallMaskDecoder", "RoIBoxEncoding"]
