#!/usr/bin/env python3
"""Lightweight contract test for the readout runner's label construction."""
import sys
from pathlib import Path
import numpy as np
from pycocotools import mask as mask_util

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from inference.probes.run_dense_gate_readout import _labels

def rle(a):
    out = mask_util.encode(np.asfortranarray(np.asarray(a, dtype=np.uint8))); out["counts"] = out["counts"].decode(); return out
def rec(mask): return dict(image_id=1, category_id=1, bbox=[0,0,2,2], score=.9, mask_score=.9, segmentation=rle(mask))
def main():
    current, forced = [rec([[1,0],[0,0]])], [rec([[1,1],[0,0]])]
    gt = [rec([[1,1],[0,0]])]
    matched, labels, oracle = _labels(current, forced, gt, .5)
    assert matched.tolist() == [True] and labels.tolist() == [1.] and oracle == {0}
    print("PASS dense gate readout label contract")
if __name__ == "__main__": main()
