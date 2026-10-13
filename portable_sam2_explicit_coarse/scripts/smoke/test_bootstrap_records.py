#!/usr/bin/env python
"""Unit tests for tools/bootstrap_paired_map.py record handling.

Covers the audit requirements for the bootstrap coordinate fix:
  - xywh validation (positive non-square area, finite, in-bounds) with a
    clear rejection path for xyxy-shaped input;
  - single xywh -> xyxy conversion at group time; COCO-side areas must come
    out as w*h (a double conversion would produce w' = w - x1 < 0);
  - paired image-set equality is enforced and missing images are NOT
    silently treated as zero predictions;
  - resample occurrences with duplicate image ids become independent COCO
    images (fresh unique ids);
  - an image with zero predictions evaluates without error;
  - END-TO-END through the real main() entry (subprocess): execution
    coverage via --baseline-manifest must pass on matching evidence and
    hard-fail when the baseline manifest proves a smaller processed count,
    when gt counts disagree, and when an id-level manifest list disagrees.

Run: python scripts/smoke/test_bootstrap_records.py
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

_SPEC = importlib.util.spec_from_file_location(
    "bootstrap_paired_map",
    PROJECT_ROOT / "tools" / "bootstrap_paired_map.py",
)
BOOT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(BOOT)


def _rect_rec(image_id, x, y, w, h, score=0.9, category_id=1):
    return {
        "image_id": image_id,
        "category_id": category_id,
        "bbox": [float(x), float(y), float(w), float(h)],
        "score": float(score),
        "mask_score": float(score),
        "segmentation": _solid_rle_counts(),
    }


def _solid_rle_counts():
    """Compact column-wise RLE for a fully-foreground 64x64 mask."""
    from pycocotools import mask as mask_util

    import numpy as onp

    solid = onp.ones((64, 64), dtype=np.uint8, order="F")
    rle = mask_util.encode(solid)
    return {"size": list(rle["size"]), "counts": rle["counts"].decode("utf-8")}


def test_validate_bbox():
    sizes = {1: (64, 64)}
    # non-square, in-bounds xywh passes and returns floats
    x, y, w, h = BOOT._validate_bbox_xywh([10, 20, 30, 40], 1, 64, 64, "t")
    assert (x, y, w, h) == (10.0, 20.0, 30.0, 40.0)
    for bad, reason in (
        ([10, 20, -5, 40], "negative width (xyxy feed)"),
        ([10, 20, 0, 40], "zero width"),
        ([10, 20, float("nan"), 40], "non-finite"),
        ([10, 20, 30, 50], "out of bounds (y+h > 64)"),
    ):
        try:
            BOOT._validate_bbox_xywh(bad, 1, 64, 64, "t")
        except SystemExit as err:
            assert reason.split()[0] in str(err) or "bbox" in str(err), (bad, err)
        else:
            raise AssertionError(f"{bad} should have been rejected ({reason})")
    print("PASS bbox xywh validation (non-square, xyxy-reject, bounds)")


def test_group_converts_once():
    sizes = {7: (64, 64)}
    recs = [_rect_rec(7, 10, 20, 30, 40)]
    grouped = BOOT._group(recs, sizes, "test")
    # xywh -> xyxy happens exactly once here: [10,20,30,40] -> [10,20,40,60].
    # build_coco_gt_and_dt converts back to [10,20,30,40] with area 30*40.
    np.testing.assert_allclose(grouped[7]["bboxes"], [[10.0, 20.0, 40.0, 60.0]])

    from utils.coco_eval_utils import build_coco_gt_and_dt

    gt = {
        "bboxes": np.asarray(grouped[7]["bboxes"], dtype=np.float32),
        "labels": np.asarray(grouped[7]["labels"], dtype=np.int64),
        "rles": list(grouped[7]["rles"]),
    }
    coco_gt, _ = build_coco_gt_and_dt(
        [gt],
        [{
            "bboxes": np.zeros((0, 4), dtype=np.float32),
            "labels": np.zeros((0,), dtype=np.int64),
            "rles": [],
            "scores": np.zeros((0,), dtype=np.float32),
            "mask_scores": np.zeros((0,), dtype=np.float32),
        }],
        [{"img_shape": (64, 64)}],
        score_key="scores",
    )
    ann = list(coco_gt.anns.values())[0]
    assert ann["area"] == 30.0 * 40.0, f"area corrupted: {ann['area']}"
    assert ann["bbox"] == [10.0, 20.0, 30.0, 40.0], ann["bbox"]
    print("PASS single conversion; COCO area == w*h on a non-square box")


def test_paired_set_enforcement():
    try:
        BOOT._require_same_images({
            "baseline_dt": [1, 2, 3],
            "treatment_dt": [1, 2],
            "gt_records": [1, 2, 3],
            "images.json": [1, 2, 3],
        })
    except SystemExit as err:
        assert "paired image sets differ" in str(err), err
    else:
        raise AssertionError("mismatched image sets must be rejected")
    BOOT._require_same_images({
        "baseline_dt": [1, 2], "treatment_dt": [2, 1],
        "gt_records": [1, 2], "images.json": [1, 2],
    })  # order-independent equality is fine
    print("PASS paired image-set enforcement (no silent zero-prediction)")


def test_duplicate_and_empty_occurrences():
    from utils.coco_eval_utils import build_coco_gt_and_dt

    # Resample [7, 7, 8]: image 7 occurs twice -> independent COCO images;
    # image 8 has zero predictions -> evaluates without error.
    gt_rows = [_rect_rec(7, 10, 10, 20, 20), _rect_rec(8, 5, 5, 10, 12)]
    sizes = {7: (64, 64), 8: (64, 64)}
    gt_by_img = BOOT._group(gt_rows, sizes, "gt")
    base_by_img = BOOT._group([_rect_rec(7, 10, 10, 20, 20, score=0.9)], sizes, "base")
    treat_by_img = BOOT._group([_rect_rec(7, 10, 10, 20, 20, score=0.95)], sizes, "treat")

    resample = [7, 7, 8]
    all_gt, all_dt, metas = [], [], []
    for new_id, image_id in enumerate(resample, start=1):
        g = gt_by_img.get(image_id, {"bboxes": [], "labels": [], "rles": []})
        all_gt.append({
            "bboxes": np.asarray(g["bboxes"], dtype=np.float32).reshape(-1, 4),
            "labels": np.asarray(g["labels"], dtype=np.int64),
            "rles": list(g["rles"]),
        })
        b = base_by_img.get(image_id)
        all_dt.append({
            "bboxes": np.asarray(b["bboxes"] if b else [], dtype=np.float32).reshape(-1, 4),
            "labels": np.asarray(b["labels"] if b else [], dtype=np.int64),
            "rles": list(b["rles"] if b else []),
            "scores": np.asarray(b["scores"] if b else [], dtype=np.float32),
            "mask_scores": np.asarray(b["mask_scores"] if b else [], dtype=np.float32),
        })
        metas.append({"img_shape": (64, 64)})
    coco_gt, dt = build_coco_gt_and_dt(all_gt, all_dt, metas, score_key="scores")
    assert len(coco_gt.imgs) == 3, f"duplicate occurrence lost fresh id: {len(coco_gt.imgs)}"
    assert len(coco_gt.anns) == 3, "gt of the duplicated image must be duplicated too"
    assert len(dt.anns) == 2, "empty image 8 contributes no predictions"
    from pycocotools.cocoeval import COCOeval

    ev = COCOeval(coco_gt, dt, iouType="segm")
    ev.evaluate(); ev.accumulate()
    assert np.isfinite(ev.stats).all(), "resample eval produced non-finite stats"
    print("PASS duplicate occurrences -> independent images; empty-pred image ok")


def _write_tmp_universe(tmp, processed_images=3, gt_instances=2, manifest_ids=None,
                        processed_ids=None, ann_ids=(1, 2, 3), with_ann_file=True,
                        with_dataset=True):
    """Synthetic Phase-0-style universe: 3 images (ids 1-3), GT on 1&2,
    detections on subsets, plus a baseline manifest recording execution
    coverage.  Identity evidence: production ``processed_image_ids`` (or the
    legacy ``image_ids`` alias), or a COCO-style ann listing whose image-id
    set must match images.json (dataset channel)."""
    tmp.mkdir(parents=True, exist_ok=True)
    treat = tmp / "treatment"
    treat.mkdir(exist_ok=True)
    sizes = {1: (64, 64), 2: (64, 64), 3: (64, 64)}
    (treat / "images.json").write_text(json.dumps(
        [{"image_id": i, "height": h, "width": w} for i, (h, w) in sizes.items()]
    ))
    (treat / "gt_records.json").write_text(json.dumps(
        [_rect_rec(1, 10, 10, 20, 30), _rect_rec(2, 5, 8, 12, 16)]  # non-square
    ))
    (treat / "dt_records.json").write_text(json.dumps(
        [_rect_rec(1, 10, 10, 20, 30, score=0.9)]
    ))
    (tmp / "predictions.json").write_text(json.dumps(
        [_rect_rec(2, 5, 8, 12, 16, score=0.8)]
    ))
    if with_ann_file:
        (tmp / "validation.json").write_text(json.dumps({
            "images": [{"id": i, "height": 64, "width": 64} for i in ann_ids],
            "annotations": [],
            "categories": [{"id": 1, "name": "building"}],
        }))
    manifest = {
        "processed_images": processed_images,
        "ground_truth_instances": gt_instances,
    }
    if processed_ids is not None:
        manifest["processed_image_ids"] = list(processed_ids)  # production field
    if manifest_ids is not None:
        manifest["image_ids"] = manifest_ids  # legacy alias field
    if with_dataset:
        dataset = {"data_root": str(tmp)}
        if with_ann_file:
            dataset["ann_file"] = "validation.json"
        manifest["dataset"] = dataset
    manifest_path = tmp / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return sizes, manifest_path


def _run_tool(tmp, manifest_path, expect_ok):
    out = tmp / "payload.json"
    proc = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "tools" / "bootstrap_paired_map.py"),
         "--baseline-dt", str(tmp / "predictions.json"),
         "--treatment-dir", str(tmp / "treatment"),
         "--baseline-manifest", str(manifest_path),
         "--resamples", "0", "--output", str(out)],
        capture_output=True, text=True,
    )
    if expect_ok:
        assert proc.returncode == 0, f"stderr:\n{proc.stderr[-800:]}"
        payload = json.loads(out.read_text())
        assert np.isfinite(payload["baseline_mAP"])
        assert np.isfinite(payload["treatment_mAP"])
        assert payload["baseline_processed_images"] == 3
        return payload
    assert proc.returncode != 0, "mismatched coverage must hard-fail"
    combined = proc.stderr + proc.stdout
    return combined


def test_main_end_to_end_and_coverage(tmp_path):
    # (a) ann-listing identity matches -> passes through the real main() entry
    sizes, manifest = _write_tmp_universe(tmp_path / "ok")
    payload = _run_tool(tmp_path / "ok", manifest, expect_ok=True)
    assert payload["images"] == 3 and payload["images_with_gt"] == 2
    assert payload["identity_evidence"].startswith("dataset_ann_file"), payload
    # VHR-10 must never silently fall back to the single-class builder.
    vhr_dir = tmp_path / "vhr10"
    _, vhr_manifest = _write_tmp_universe(vhr_dir)
    manifest_json = json.loads(vhr_manifest.read_text())
    manifest_json["dataset"] = {"num_classes": 10, "data_root": str(vhr_dir),
                                "ann_file": "validation.json"}
    vhr_manifest.write_text(json.dumps(manifest_json))
    payload = _run_tool(vhr_dir, vhr_manifest, expect_ok=True)
    assert payload["num_classes"] == 10
    assert payload["category_ids"] == list(range(1, 11)), payload
    # (b) SAME counts but DISJOINT image identities -> must fail on identity
    combined = _run_tool(tmp_path / "disjoint", _write_tmp_universe(
        tmp_path / "disjoint", ann_ids=(101, 102, 103))[1], expect_ok=False)
    assert "paired image sets differ" in combined, combined[-400:]
    # (c) no identity channel at all (no image_ids, no dataset.ann_file)
    #     -> counts alone must NOT be accepted as coverage verified
    combined = _run_tool(tmp_path / "noid", _write_tmp_universe(
        tmp_path / "noid", with_ann_file=False, with_dataset=False)[1],
        expect_ok=False)
    assert "cannot verify baseline image identity" in combined, combined[-400:]
    # (d) id-level manifest list disagreeing with images.json -> fail
    _, manifest = _write_tmp_universe(tmp_path / "ids", manifest_ids=[1, 2])
    combined = _run_tool(tmp_path / "ids", manifest, expect_ok=False)
    assert "paired image sets differ" in combined, combined[-400:]
    # (d2) PRODUCTION format: processed_image_ids matching -> strong channel
    _, manifest = _write_tmp_universe(
        tmp_path / "prod_ok", processed_ids=[3, 1, 2])
    payload = _run_tool(tmp_path / "prod_ok", manifest, expect_ok=True)
    assert payload["identity_evidence"] == "manifest_processed_image_ids", payload
    # (d3) PRODUCTION format, same count but DISJOINT ids, and the dataset
    #      ann listing WOULD match -> must still fail on the id mismatch and
    #      never fall back to the weaker channel
    combined = _run_tool(tmp_path / "prod_bad", _write_tmp_universe(
        tmp_path / "prod_bad", processed_ids=[101, 102, 103],
        ann_ids=(1, 2, 3))[1], expect_ok=False)
    assert "paired image sets differ" in combined, combined[-400:]
    assert "baseline_manifest.processed_image_ids" in combined, combined[-400:]
    # (e) legacy alias id-level list agreeing -> strong channel passes
    _, manifest = _write_tmp_universe(tmp_path / "ids_ok", manifest_ids=[3, 1, 2])
    payload = _run_tool(tmp_path / "ids_ok", manifest, expect_ok=True)
    assert payload["identity_evidence"] == "manifest_image_ids"
    # (f) baseline processed fewer images -> auxiliary count check still bites
    combined = _run_tool(tmp_path / "short", _write_tmp_universe(
        tmp_path / "short", processed_images=2)[1], expect_ok=False)
    assert "execution coverage mismatch" in combined, combined[-400:]
    # (g) gt_records not belonging to the manifest run -> fail
    combined = _run_tool(tmp_path / "gtmm", _write_tmp_universe(
        tmp_path / "gtmm", gt_instances=99)[1], expect_ok=False)
    assert "ground_truth_instances" in combined, combined[-400:]
    print("PASS end-to-end main() entry + image-identity/count enforcement")


def main():
    test_validate_bbox()
    test_group_converts_once()
    test_paired_set_enforcement()
    test_duplicate_and_empty_occurrences()
    test_main_end_to_end_and_coverage(Path(__file__).resolve().parent / "_tmp_bootstrap_test")
    print("ALL bootstrap record tests PASSED")


if __name__ == "__main__":
    main()
