#!/usr/bin/env python3
"""Read-only leave-source-scene identity consensus for R3 isolated components."""
import argparse
import collections
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from build_surfacebev_v4_postsplit import to_world


PACKAGE = Path(__file__).resolve().parents[2]
V3 = PACKAGE / "outputs/SURFBEV-MASK-V3-20261012"
R2 = PACKAGE / "outputs/SURFBEV-V5-PHYSICAL-INSTANCE-R2-20261012"
R3 = PACKAGE / "outputs/SURFBEV-V5-PHYSICAL-INSTANCE-R3-20261012"
OUT = PACKAGE / "outputs/SURFBEV-V5-PHYSICAL-INSTANCE-R4-20261012"
V3_SHA = "8d3da5addd302451337954191dccbf1b76f588a0790f1073415e30439015922f"
R3_ISOLATED_SHA = "a726fef1bcfc9291d4368e56ed30d60c8f3ea767927fa99beda4c55a1e8bd5f3"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v3", type=Path, default=V3)
    parser.add_argument("--r2", type=Path, default=R2)
    parser.add_argument("--r3", type=Path, default=R3)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(args.out)
    if sha(args.v3 / "manifest.jsonl") != V3_SHA:
        raise ValueError("v3 manifest changed")
    if sha(args.r3 / "isolated_world_components.jsonl") != R3_ISOLATED_SHA:
        raise ValueError("R3 isolated list changed")
    r2_summary = json.loads((args.r2 / "summary.json").read_text())
    if r2_summary["probe_sha256"] != sha(args.r2 / "global_probe.jsonl"):
        raise ValueError("R2 manifest changed")
    isolated = [json.loads(x) for x in (args.r3 / "isolated_world_components.jsonl").read_text().splitlines()]
    r2 = {x["global_instance_id"]: x for x in map(json.loads, (args.r2 / "global_probe.jsonl").read_text().splitlines())}
    cases = []
    for item in isolated:
        comp = next(c for c in r2[item["global_instance_id"]]["components"] if c["label"] == item["component_label"])
        cases.append(dict(**item, center_xy_m=comp["center_xy_m"], votes=collections.Counter(),
                          observed_scenes=[], image_in_bounds=0, background=0))
    source = [json.loads(x) for x in (args.v3 / "manifest.jsonl").read_text().splitlines()]
    if len(source) != 1200 or len({x["scene_dir"] for x in source}) != 1200:
        raise ValueError("expected 1200 unique scenes")
    for rec in source:
        if rec["split"] not in ("train", "val"):
            continue
        origin = to_world(rec, (0, 0))
        du = to_world(rec, (1, 0)) - origin
        dv = to_world(rec, (0, 1)) - origin
        inv = np.linalg.inv(np.column_stack((du, dv)))
        inside = []
        for case in cases:
            if rec["scene_dir"] in case["scene_dirs"]:
                continue
            uv = inv @ (np.asarray(case["center_xy_m"]) - origin)
            x, y = int(round(float(uv[0]))), int(round(float(uv[1])))
            if 0 <= x < 512 and 0 <= y < 512:
                inside.append((case, x, y))
        if not inside:
            continue
        if sha(Path(rec["v3_png"])) != rec["v3_sha256"]:
            raise ValueError(f"v3 label changed: {rec['scene_dir']}")
        with Image.open(rec["v3_png"]) as image:
            if image.mode != "L" or image.size != (512, 512):
                raise ValueError(f"invalid label: {rec['scene_dir']}")
            ids = np.asarray(image).copy()
        meta = json.loads((Path(rec["source_png"]).parent / "instances.json").read_text())
        mapping = {int(x["local_instance_id"]): int(x["global_instance_id"]) for x in meta["instances"]}
        for case, x, y in inside:
            case["image_in_bounds"] += 1
            local = int(ids[y, x])
            if local == 0:
                case["background"] += 1
            else:
                if local not in mapping:
                    raise ValueError(f"unmapped label {local}: {rec['scene_dir']}")
                gid = mapping[local]
                case["votes"][gid] += 1
                case["observed_scenes"].append(dict(scene_dir=rec["scene_dir"], observed_global_id=gid,
                                                     pixel_xy=[x, y]))
    rows = []
    for case in cases:
        votes = case.pop("votes")
        n = sum(votes.values())
        winner, winner_votes = votes.most_common(1)[0] if votes else (None, 0)
        target = winner if winner is not None and winner != case["global_instance_id"] and n >= 3 and winner_votes / n >= .90 else None
        case.update(nonbackground_votes=n, vote_counts={str(k): v for k, v in sorted(votes.items())},
                    winner_global_id=winner, winner_fraction=round(winner_votes / n, 6) if n else 0.0,
                    identity_correction_candidate=target,
                    category="IDENTITY_CORRECTION_CANDIDATE" if target is not None else "PHYSICAL_SPLIT_OR_HOLD")
        rows.append(case)
    args.out.mkdir(parents=True)
    manifest = args.out / "identity_consensus.jsonl"
    manifest.write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in rows))
    count = collections.Counter(x["category"] for x in rows)
    positive = [x for x in rows if x["global_instance_id"] == 170002]
    stable = [x for x in rows if x["global_instance_id"] == 1874325323]
    summary = dict(protocol="SURFBEV-V5-PHYSICAL-INSTANCE-R4-20261012", status="DIAGNOSTIC_ONLY",
                   v3_manifest_sha256=V3_SHA, r3_isolated_sha256=R3_ISOLATED_SHA,
                   cases=len(rows), category_counts=dict(count),
                   positive_170002_to_161009=bool(positive and positive[0]["identity_correction_candidate"] == 161009),
                   stable_1874325323_not_identity_correction=all(x["identity_correction_candidate"] is None for x in stable),
                   manifest_sha256=sha(manifest),
                   note="Center point lookup only; no labels changed, no train reader switch.")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
