#!/usr/bin/env python3
"""Deduplicate code-ranked review cases; global IDs used only as an offline cache key."""

import json
from collections import defaultdict
from pathlib import Path

from probe_surfacebev_v6_local_split_scan import ROOT, OUT, jsonl, render


def main():
    m3 = {r["scene_dir"]: r for r in jsonl(ROOT / "outputs/SURFBEV-V5-IDENTITY-M3-20261012/manifest.jsonl")}
    cache = {}

    def review_key(row):
        scene = row["scene_dir"]
        if scene not in cache:
            metadata = json.loads(Path(m3[scene]["v5_metadata"]).read_text())
            cache[scene] = {int(x["local_instance_id"]): int(x["global_instance_id"])
                            for x in metadata["visible_instances"]}
        # This key collapses repeated camera views for efficient review; it is
        # neither a model input nor a criterion for physical-building truth.
        return cache[scene][row["local_instance_id"]]

    disconnected = [r for r in jsonl(OUT/"disconnected.jsonl") if r["strong"]]
    neck = list(jsonl(OUT/"neck.jsonl"))
    selected = {}
    for name, rows, metric, title in (
        ("strong_disconnected", disconnected, lambda r:(r["second_area"],r["gap_px"]), "disconnected"),
        ("neck", neck, lambda r:(r["lobe_areas"][1],r["gradient_ratio"]), "neck"),
    ):
        groups=defaultdict(list)
        for row in rows:
            groups[review_key(row)].append(row)
        picked=[max(group,key=metric) for group in groups.values()]
        picked.sort(key=metric,reverse=True)
        paths=render(picked,title,f"cluster_{name}",limit=len(picked))
        selected[name]=dict(candidate_observations=len(rows),review_groups=len(groups),
                            cases=[dict(scene_dir=r["scene_dir"],split=r["split"],
                                        local_instance_id=r["local_instance_id"],
                                        area_px=r["area_px"],
                                        second_area=r.get("second_area"),gap_px=r.get("gap_px"),
                                        lobe_areas=r.get("lobe_areas"),
                                        gradient_ratio=r.get("gradient_ratio")) for r in picked],
                            sheets=paths)
    (OUT/"cluster_review.json").write_text(json.dumps(selected,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({name:{"observations":v["candidate_observations"],
                            "review_groups":v["review_groups"],"sheets":v["sheets"]}
                      for name,v in selected.items()},indent=2))


if __name__=="__main__":
    main()
