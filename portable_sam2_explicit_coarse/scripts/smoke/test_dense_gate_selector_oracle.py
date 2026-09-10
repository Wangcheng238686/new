#!/usr/bin/env python3
"""CPU regression for dense_gate_selector_oracle's selector invariants."""
import sys
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_util

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from inference.probes.dense_gate_selector_oracle import make_selectors


def rle(mask):
    out = mask_util.encode(np.asfortranarray(np.asarray(mask, dtype=np.uint8)))
    out["counts"] = out["counts"].decode("ascii")
    return out


def rec(image, mask, score=0.9):
    return dict(image_id=image, category_id=1, bbox=[0, 0, 2, 2], score=score,
                mask_score=score, segmentation=rle(mask))


def main():
    gt = [rec(1, [[1, 1], [0, 0]]), rec(2, [[1, 1], [0, 0]])]
    raw = [rec(1, [[1, 0], [0, 0]]), rec(2, [[1, 1], [0, 0]])]
    full = [rec(1, [[1, 1], [0, 0]]), rec(2, [[0, 0], [0, 0]])]
    oracle, randoms, audit = make_selectors(raw, full, gt, .5, 3, 44)
    assert oracle == {0}, oracle
    assert len(randoms) == 3 and all(len(s) == len(oracle) for s in randoms)
    assert audit[0]["choose_forced"] and not audit[1]["choose_forced"]
    bad = [dict(x) for x in full]; bad[0]["score"] = .8
    try:
        make_selectors(raw, bad, gt, .5, 1, 44)
    except ValueError as exc:
        assert "non-mask" in str(exc)
    else:
        raise AssertionError("must reject a non-mask intervention")
    print("PASS dense gate selector oracle")


if __name__ == "__main__":
    main()
