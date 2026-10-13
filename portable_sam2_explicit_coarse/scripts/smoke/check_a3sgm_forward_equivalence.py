#!/usr/bin/env python
"""Forward-equivalence gate for the paper a3sgm checkpoints.

Rebuilds each checkpoint through the real inference path, runs ``model.predict``
on a fixed pair of validation images on CPU, and compares a SHA-256 digest of
the raw prediction arrays (bboxes / scores / labels / masks) against a stored
golden digest. Any behavior drift in the forward chain after refactoring fails
here bit-for-bit.

Usage:
    check_a3sgm_forward_equivalence.py            # compare all cases
    check_a3sgm_forward_equivalence.py --update   # re-record golden digests
    check_a3sgm_forward_equivalence.py --case whu # one case only

CPU is deliberate: GPU kernel/algorithm selection can wobble across library
versions, while CPU execution of an unchanged computation graph is stable, so
a digest mismatch here always means a real code change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

MAINLINE_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_PATH = Path(__file__).resolve().parent / "a3sgm_forward_golden.json"

# Machine-local dataset roots (lthpc); override via environment when porting.
WHU_DATA_ROOT = os.environ.get("WHU1024_DATA_ROOT", "/data/wangcheng/dataset/WHU")
VHR10_DATA_ROOT = os.environ.get(
    "VHR10_DATA_ROOT", "/data/wangcheng/dataset/NWPU VHR-10 dataset"
)
CKPT_ROOT = Path(
    os.environ.get(
        "PORTABLE_SAM2_CHECKPOINT_ROOT",
        "/data/wangcheng/checkpoint/portable_sam2_explicit_coarse/ablations",
    )
)

# Each case = one paper checkpoint on the first two images of its own
# validation split (shuffle=False keeps the order fixed). The NWPU case also
# covers the full_image predict path; the WHU cases cover roi_local plus the
# EMA weight-selection branch.
CASES = {
    "nwpu": {
        "checkpoint": CKPT_ROOT
        / "vhr10_p2v2_matrix300_a3sgm_pbm_sg_udprk64tm_tr1.0_va1.0"
        / "best_composite_model_epoch270.pth",
        "weights": "model",
        "split": "custom",
        "data_root": VHR10_DATA_ROOT,
        "ann_file": "coco_split/NWPU_instances_val.json",
        "image_subdir": "positive image set",
    },
    "whu": {
        "checkpoint": CKPT_ROOT
        / "whu1024_a3sgm_roilocal_full_fast_tr1.0_va1.0"
        / "best_model_epoch118.pth",
        "weights": "model",
        "split": "validation",
        "data_root": WHU_DATA_ROOT,
        "ann_file": None,
        "image_subdir": None,
    },
    "whu_ema": {
        "checkpoint": CKPT_ROOT
        / "whu1024_a3sgm_roilocal_full_fast_tr1.0_va1.0"
        / "best_model_epoch118.pth",
        "weights": "ema",
        "split": "validation",
        "data_root": WHU_DATA_ROOT,
        "ann_file": None,
        "image_subdir": None,
    },
}
NUM_PROBE_IMAGES = 2


def digest_prediction(prediction: dict) -> str:
    """SHA-256 over the raw prediction arrays, in a fixed field order."""
    hasher = hashlib.sha256()
    for key in ("bboxes", "labels", "scores", "mask_scores", "masks"):
        if key not in prediction:
            continue
        value = prediction[key]
        hasher.update(key.encode("ascii"))
        hasher.update(value.tobytes())
    return hasher.hexdigest()


def run_case(case: str) -> dict:
    """Rebuild the case's model in a fresh interpreter and digest its outputs.

    A subprocess per case keeps os.environ mutations from one checkpoint's
    architecture-env restore (done during config resolution) from leaking into
    the next case's resolution.
    """
    spec = CASES[case]
    code = f"""
import sys
sys.path.insert(0, {str(MAINLINE_ROOT / "inference")!r})
sys.path.insert(0, {str(MAINLINE_ROOT)!r})
import argparse
import infer_from_checkpoint as ifc

args = argparse.Namespace(
    checkpoint={str(spec["checkpoint"])!r}, config=None, split={spec["split"]!r},
    data_root={spec["data_root"]!r}, ann_file={spec["ann_file"]!r},
    image_subdir={spec["image_subdir"]!r}, image_size=None, batch_size=1,
    num_workers=0, max_batches={NUM_PROBE_IMAGES}, device="cpu",
    weights={spec["weights"]!r}, sam2_repo=None, sam2_ckpt=None,
    output_dir=None, export_bootstrap_records=False, score_thr=0.0,
    no_eval=True, inspect_only=False, build_only=False, allow_nonstrict=False,
    disable_p2=False, p2_beta=None, disable_decoder_tail=False,
    max_per_img=None, nms_thr=None,
)
checkpoint = ifc._load_checkpoint(args.checkpoint if isinstance(args.checkpoint, __import__("pathlib").Path) else __import__("pathlib").Path(args.checkpoint))
ifc._resolve_sam2_repo(args, ifc._snapshot(checkpoint))
model_config, _source = ifc._resolve_model_config(args, ifc._snapshot(checkpoint))
model = ifc._register_and_build(model_config)
ifc._load_model_state(model, ifc._select_state_dict(checkpoint, args.weights), allow_nonstrict=False)
model.roi_head.mask_head.assert_no_mask_embedding_contract("forward equivalence")
model.to("cpu").eval()
contract = ifc._resolve_dataset_contract(args, ifc._snapshot(checkpoint))
loader = ifc._build_loader(contract, num_workers=0)
import json as _json
digests = []
with __import__("torch").inference_mode():
    for batch_index, batch in enumerate(loader):
        if batch_index >= {NUM_PROBE_IMAGES}:
            break
        imgs, data_samples = ifc._build_data_samples(batch, __import__("torch").device("cpu"))
        processed = model.data_preprocessor({{"inputs": imgs, "data_samples": data_samples}}, training=False)
        outputs = model.predict(processed["inputs"], processed["data_samples"], rescale=False)
        for index, output in enumerate(outputs):
            meta = batch["img_metas"][index]
            pred = output.pred_instances if hasattr(output, "pred_instances") else output
            prediction = ifc._extract_instances_numpy(pred, meta["img_shape"])
            hasher = __import__("hashlib").sha256()
            for key in ("bboxes", "labels", "scores", "mask_scores", "masks"):
                if key in prediction:
                    hasher.update(key.encode("ascii"))
                    hasher.update(prediction[key].tobytes())
            digests.append({{"file": meta.get("filename", ""), "sha256": hasher.hexdigest()}})
print(_json.dumps(digests))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(MAINLINE_ROOT),
    )
    if result.returncode != 0:
        print(result.stdout[-4000:])
        print(result.stderr[-4000:])
        raise SystemExit(f"FAIL case={case}: forward run crashed")
    return json.loads(result.stdout.strip().splitlines()[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=sorted(CASES), default=None)
    parser.add_argument(
        "--update",
        action="store_true",
        help="Re-record golden digests instead of comparing.",
    )
    options = parser.parse_args()

    golden = {}
    if GOLDEN_PATH.is_file():
        golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    cases = [options.case] if options.case else sorted(CASES)

    if options.update:
        for case in cases:
            golden[case] = run_case(case)
            print(f"recorded {case}: {len(golden[case])} images")
        GOLDEN_PATH.write_text(
            json.dumps(golden, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return

    failures = []
    for case in cases:
        expected = golden.get(case)
        if not expected:
            raise SystemExit(f"FAIL case={case}: no golden digest recorded yet")
        actual = run_case(case)
        if actual == expected:
            print(f"PASS {case} ({len(actual)} images)")
        else:
            for index, (want, got) in enumerate(zip(expected, actual)):
                if want != got:
                    print(
                        f"MISMATCH {case} image#{index}: "
                        f"golden={want['sha256'][:16]} actual={got['sha256'][:16]} "
                        f"({got['file']})"
                    )
            failures.append(case)
    if failures:
        raise SystemExit(f"FAIL forward equivalence: {failures}")
    print("forward equivalence holds for all cases")


if __name__ == "__main__":
    main()
