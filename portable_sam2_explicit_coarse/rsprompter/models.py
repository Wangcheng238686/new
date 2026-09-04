"""Compatibility facade: implementations live in sibling modules.

RSPrompterAnchor        -> rsprompter/anchor_detector.py
RSPrompterAnchorRoIPromptHead -> rsprompter/anchor_roi_head.py
Registry names and every external import path are unchanged.
"""
from .anchor_detector import RSPrompterAnchor
from .anchor_roi_head import RSPrompterAnchorRoIPromptHead

__all__ = ["RSPrompterAnchor", "RSPrompterAnchorRoIPromptHead"]
