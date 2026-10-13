#!/usr/bin/env python3
"""Read-only V4/M3 partition census and train/val candidate review PNGs."""

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import cv2
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[2]
V4 = ROOT / "outputs/SURFBEV-V4-POSTSPLIT-20261012-R3"
M3 = ROOT / "outputs/SURFBEV-V5-IDENTITY-M3-20261012"
R3 = ROOT / "outputs/SURFBEV-V5-PHYSICAL-INSTANCE-R3-20261012"
OUT = ROOT / "outputs/SURFBEV-V6-INTRAIMAGE-REVIEW-20261012"
BG = (18, 20, 24)
CYAN = np.array([20, 235, 235], dtype=np.float32)
PINK = np.array([255, 75, 170], dtype=np.float32)


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def rgb_and_ids(rgb_path, id_path):
    with Image.open(rgb_path) as image:
        rgb = np.asarray(image.convert("RGB"))
    with Image.open(id_path) as image:
        ids = np.asarray(image)
    if rgb.shape != (512, 512, 3) or ids.shape != (512, 512):
        raise ValueError(f"image contract changed: {rgb_path} {id_path}")
    return rgb, ids


def crop_box(mask, min_side=144, margin=20):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return 0, 0, 512, 512
    side = min(512, max(min_side, int(max(xs.max()-xs.min()+1, ys.max()-ys.min()+1) + 2*margin)))
    cx = (int(xs.min()) + int(xs.max())) // 2
    cy = (int(ys.min()) + int(ys.max())) // 2
    x0 = min(max(0, cx-side//2), 512-side)
    y0 = min(max(0, cy-side//2), 512-side)
    return x0, y0, x0+side, y0+side


def overlaid(rgb, masks_colors, box, size=320):
    frame = rgb.astype(np.float32).copy()
    for mask, color in masks_colors:
        frame[mask] = frame[mask]*0.45 + color*0.55
    frame = np.asarray(np.clip(frame, 0, 255), dtype=np.uint8)
    return Image.fromarray(frame).crop(box).resize((size, size), Image.Resampling.NEAREST)


def panel(left, right, title, subtitle):
    w = left.width + right.width
    canvas = Image.new("RGB", (w, left.height+54), BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 5), title[:96], fill=(255, 255, 255))
    draw.text((8, 27), subtitle[:105], fill=(200, 210, 218))
    canvas.paste(left, (0, 54))
    canvas.paste(right, (left.width, 54))
    return canvas


def partition_delta(a, b):
    common = (a != 0) & (b != 0)
    pairs = np.unique(np.stack((a[common], b[common]), axis=1), axis=0)
    a_to_b = defaultdict(set)
    b_to_a = defaultdict(set)
    for old, new in pairs:
        a_to_b[int(old)].add(int(new))
        b_to_a[int(new)].add(int(old))
    split = {k: sorted(v) for k, v in a_to_b.items() if len(v) > 1}
    merge = {k: sorted(v) for k, v in b_to_a.items() if len(v) > 1}
    return split, merge, int(np.count_nonzero((a != 0) != (b != 0)))


def hash_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_m3(rows):
    records = []
    sheet_tiles = []
    for row in rows:
        if not row["identity_actions"]:
            continue
        if hash_file(row["v4_png"]) != row["v4_sha256"] or hash_file(row["v5_png"]) != row["v5_sha256"]:
            raise ValueError(f"manifest SHA mismatch: {row['scene_dir']}")
        rgb, a = rgb_and_ids(row["satellite_rgb"], row["v4_png"])
        _, b = rgb_and_ids(row["satellite_rgb"], row["v5_png"])
        split, merge, fg_delta = partition_delta(a, b)
        old_ids = sorted({int(x["source_local_id"]) for x in row["identity_actions"]})
        new_ids = sorted({int(x["new_local_id"]) for x in row["identity_actions"] if x.get("new_local_id")})
        old_mask = np.isin(a, old_ids)
        changed = (a != b) & (a != 0)
        # Center the review on the reassigned component; a whole old ID can
        # span the entire tile and make a small corner correction invisible.
        box = crop_box(changed, min_side=192, margin=48)
        before = overlaid(rgb, [(old_mask, PINK)], box)
        after = overlaid(rgb, [(old_mask & ~changed, PINK), (np.isin(b, new_ids), CYAN)], box)
        short = row["scene_dir"].split("_")[1]
        title = f"{short} {row['split']} | pink=old parent, cyan=M3 target"
        subtitle = f"split {len(split)} merge {len(merge)} foreground delta {fg_delta}px; old {old_ids} -> new {new_ids}"
        tile = panel(before, after, title, subtitle)
        tile.save(OUT / f"m3_{short}.png")
        sheet_tiles.append(tile.resize((480, 280)))
        records.append(dict(scene_dir=row["scene_dir"], split=row["split"], old_local_ids=old_ids,
                            new_local_ids=new_ids, split_old_ids=split, merged_new_ids=merge,
                            foreground_delta_px=fg_delta, crop_xyxy=box,
                            image=f"m3_{short}.png"))
    for i in range(0, len(sheet_tiles), 6):
        sheet = Image.new("RGB", (960, 840), BG)
        for j, tile in enumerate(sheet_tiles[i:i+6]):
            sheet.paste(tile, ((j%2)*480, (j//2)*280))
        sheet.save(OUT / f"m3_sheet_{i//6+1:02d}.png")
    return records


def make_connected_risk():
    risk = read_rows(R3 / "connected_risk.jsonl")
    # One large, non-border representative per global. Avoid repeated views of one asset.
    by_global = {}
    for row in risk:
        if row["touches_border"] or row["area_px"] < 512:
            continue
        gid = row["global_instance_id"]
        if gid not in by_global or row["area_px"] > by_global[gid]["area_px"]:
            by_global[gid] = row
    candidates = sorted(by_global.values(), key=lambda r: (-r["area_px"], r["scene_dir"]))
    selected = []
    scenes = set()
    for row in candidates:
        if row["scene_dir"] in scenes:
            continue
        selected.append(row)
        scenes.add(row["scene_dir"])
        if len(selected) == 12:
            break
    if len(selected) < 12:
        raise ValueError("too few non-border connected-risk cases")
    random.Random(20261012).shuffle(selected)
    records = []
    tiles = []
    for index, row in enumerate(selected, 1):
        if hash_file(row["v3_png"]) != row["v3_sha256"]:
            raise ValueError(f"risk SHA mismatch: {row['scene_dir']}")
        rgb, ids = rgb_and_ids(row["satellite_rgb"], row["v3_png"])
        mask = ids == row["local_instance_id"]
        box = crop_box(mask)
        left = Image.fromarray(rgb).crop(box).resize((320, 320), Image.Resampling.NEAREST)
        right = overlaid(rgb, [(mask, CYAN)], box)
        short = row["scene_dir"].split("_")[1]
        title = f"C{index:02d} {short} {row['split']} | one local ID in cyan"
        subtitle = f"area {row['area_px']} bbox-fill {row['bbox_fill']:.2f} hull-fill {row['hull_fill']:.2f}"
        tile = panel(left, right, title, subtitle)
        tile.save(OUT / f"connected_{index:02d}_{short}.png")
        tiles.append(tile.resize((480, 280)))
        records.append(dict(case=f"C{index:02d}", scene_dir=row["scene_dir"], split=row["split"],
                            local_instance_id=row["local_instance_id"], area_px=row["area_px"],
                            bbox_fill=row["bbox_fill"], hull_fill=row["hull_fill"],
                            crop_xyxy=box, image=f"connected_{index:02d}_{short}.png"))
    for i in range(0, len(tiles), 6):
        sheet = Image.new("RGB", (960, 840), BG)
        for j, tile in enumerate(tiles[i:i+6]):
            sheet.paste(tile, ((j%2)*480, (j//2)*280))
        sheet.save(OUT / f"connected_sheet_{i//6+1:02d}.png")
    return records


def make_nonhit_controls(rows):
    """Deterministic train/val sample outside the R3 connected-risk inventory."""
    flagged = {(r["scene_dir"], int(r["local_instance_id"])) for r in read_rows(R3 / "connected_risk.jsonl")}
    shuffled = [r for r in rows if r["split"] in ("train", "val")]
    rng = random.Random(20261012)
    rng.shuffle(shuffled)
    selected = []
    for row in shuffled:
        rgb, ids = rgb_and_ids(row["satellite_rgb"], row["v3_png"])
        counts = np.bincount(ids.ravel().astype(np.int64))
        candidates = []
        for lid in np.flatnonzero(counts >= 128):
            if lid == 0 or (row["scene_dir"], int(lid)) in flagged:
                continue
            mask = (ids == lid).astype(np.uint8)
            n, _ = cv2.connectedComponents(mask, connectivity=8)
            if n == 2:
                candidates.append(int(lid))
        if not candidates:
            continue
        lid = rng.choice(candidates)
        selected.append((row, rgb, ids == lid, lid))
        if len(selected) == 12:
            break
    if len(selected) != 12:
        raise ValueError("too few non-hit controls")
    records, tiles = [], []
    for index, (row, rgb, mask, lid) in enumerate(selected, 1):
        box = crop_box(mask)
        raw = Image.fromarray(rgb).crop(box).resize((320,320), Image.Resampling.NEAREST)
        over = overlaid(rgb, [(mask,CYAN)], box)
        short = row["scene_dir"].split("_")[1]
        title = f"N{index:02d} {short} {row['split']} | not in R3 risk inventory"
        subtitle = f"local {lid}, one component, area {int(mask.sum())} px"
        tile = panel(raw, over, title, subtitle)
        name = f"nonhit_{index:02d}_{short}.png"
        tile.save(OUT / name)
        tiles.append(tile.resize((480,280)))
        records.append(dict(case=f"N{index:02d}", scene_dir=row["scene_dir"], split=row["split"],
                            local_instance_id=lid, area_px=int(mask.sum()), crop_xyxy=box, image=name))
    for i in range(0, len(tiles), 6):
        sheet = Image.new("RGB", (960,840), BG)
        for j, tile in enumerate(tiles[i:i+6]):
            sheet.paste(tile, ((j%2)*480,(j//2)*280))
        sheet.save(OUT / f"nonhit_sheet_{i//6+1:02d}.png")
    return records


def make_targeted_uncertain():
    """Exploratory visual cases, not a prevalence-estimating random sample."""
    risk = read_rows(R3 / "connected_risk.jsonl")
    selected_prior = read_rows(ROOT / "outputs/SURFBEV-V5-PHYSICAL-INSTANCE-CONNECTED-VIS-20261012/selected.jsonl")
    target_gids = (500002, 152002, 511017, 174001)
    by_gid = {int(row["global_instance_id"]): row for row in selected_prior}
    records = []
    for gid in target_gids:
        chosen = by_gid[gid]
        row = next(r for r in risk if r["global_instance_id"] == gid and
                   r["scene_dir"] == chosen["scene_dir"] and
                   r["local_instance_id"] == chosen["local_instance_id"])
        rgb, ids = rgb_and_ids(row["satellite_rgb"], row["v3_png"])
        mask = ids == row["local_instance_id"]
        box = crop_box(mask, min_side=192, margin=48)
        raw = Image.fromarray(rgb).crop(box).resize((400, 400), Image.Resampling.NEAREST)
        overlay = overlaid(rgb, [(mask, CYAN)], box, size=400)
        context = Image.fromarray(rgb).resize((400, 400), Image.Resampling.NEAREST)
        draw = ImageDraw.Draw(context)
        draw.rectangle(tuple(round(v*400/512) for v in box), outline=(255, 65, 190), width=3)
        canvas = Image.new("RGB", (1200, 450), BG)
        short = row["scene_dir"].split("_")[1]
        d = ImageDraw.Draw(canvas)
        d.text((10, 8), f"{gid} / {short} | raw crop / same local ID cyan / scene context", fill=(255,255,255))
        d.text((10, 28), "QUESTION: one connected roof or more than one physical building?", fill=(210,220,230))
        canvas.paste(raw, (0, 50))
        canvas.paste(overlay, (400, 50))
        canvas.paste(context, (800, 50))
        name = f"uncertain_connected_{gid}_{short}.png"
        canvas.save(OUT / name)
        records.append(dict(global_instance_id=gid, scene_dir=row["scene_dir"],
                            local_instance_id=row["local_instance_id"], image=name,
                            selection="exploratory_after_initial_false-positive_review"))
    return records


def make_003085_detail(rows):
    row = next(r for r in rows if r["scene_dir"].startswith("Coord_003085"))
    rgb, a = rgb_and_ids(row["satellite_rgb"], row["v4_png"])
    _, b = rgb_and_ids(row["satellite_rgb"], row["v5_png"])
    changed = b == 61
    box = crop_box(changed, min_side=96, margin=8)
    size = 400
    raw = Image.fromarray(rgb).crop(box).resize((size,size), Image.Resampling.NEAREST)
    source_4 = (a == 4) & changed
    source_44 = (a == 44) & changed
    before = overlaid(rgb, [(source_4, CYAN), (source_44, PINK)], box, size)
    after = overlaid(rgb, [(changed, np.array([255, 220, 35], dtype=np.float32))], box, size)
    canvas = Image.new("RGB", (1200, 450), BG)
    d = ImageDraw.Draw(canvas)
    d.text((10, 8), "003085: two original local IDs -> one M3 local ID", fill=(255,255,255))
    d.text((10, 28), "raw / old ID 4 cyan + old ID 44 pink / M3 merged ID 61 yellow", fill=(210,220,230))
    for index, im in enumerate((raw,before,after)):
        canvas.paste(im, (index*size, 50))
    name = "m3_003085_two_sources_detail.png"
    canvas.save(OUT / name)
    return dict(scene_dir=row["scene_dir"], image=name, crop_xyxy=box,
                source_4_px=int(source_4.sum()), source_44_px=int(source_44.sum()))


def make_003085_overlap(rows):
    """Adjacent train view as visual support, not an identity proof."""
    by_id = {r["scene_dir"].split("_")[1]: r for r in rows}
    src, nbr = by_id["003085"], by_id["003092"]
    rgb_a, ids_a = rgb_and_ids(src["satellite_rgb"], src["v5_png"])
    rgb_b, ids_b = rgb_and_ids(nbr["satellite_rgb"], nbr["v5_png"])
    # RGB template match, deliberately independent of either label raster.
    tpl = rgb_a[416:512, 416:512]
    score = cv2.matchTemplate(rgb_b, tpl, cv2.TM_CCOEFF_NORMED)
    _, best, _, (bx, by) = cv2.minMaxLoc(score)
    if best < 0.80:
        raise ValueError(f"neighbor RGB alignment too weak: {best}")
    dx, dy = bx-416, by-416
    yy, xx = np.where(ids_a == 61)
    mapped = ids_b[yy+dy, xx+dx]
    vals, counts = np.unique(mapped[mapped != 0], return_counts=True)
    local = int(vals[np.argmax(counts)])
    box_a = crop_box(ids_a == 61, min_side=192, margin=24)
    box_b = crop_box(ids_b == local, min_side=192, margin=24)
    size = 400
    raw_a = Image.fromarray(rgb_a).crop(box_a).resize((size,size), Image.Resampling.NEAREST)
    raw_b = Image.fromarray(rgb_b).crop(box_b).resize((size,size), Image.Resampling.NEAREST)
    over_b = overlaid(rgb_b, [(ids_b == local, CYAN)], box_b, size)
    canvas = Image.new("RGB", (1200,450), BG)
    d = ImageDraw.Draw(canvas)
    d.text((10,8), f"003085 edge -> 003092 overlapping train view | RGB match {best:.3f}", fill=(255,255,255))
    d.text((10,28), f"003085 raw / 003092 raw / 003092 local {local} cyan (visual aid, not proof)", fill=(210,220,230))
    for i, im in enumerate((raw_a, raw_b, over_b)):
        canvas.paste(im, (i*size,50))
    name = "m3_003085_overlap_003092.png"
    canvas.save(OUT / name)
    return dict(image=name, source_scene=src["scene_dir"], neighbor_scene=nbr["scene_dir"],
                rgb_match_score=round(float(best),6), shift_xy=[dx,dy],
                neighbor_local_id=local, matched_nonbackground_px=int(counts.max()),
                source_component_px=int(len(xx)))


def main():
    # This renderer only overwrites its own deterministic review PNGs/index.
    OUT.mkdir(parents=True, exist_ok=True)
    rows = read_rows(M3 / "manifest.jsonl")
    m3 = make_m3(rows)
    risk = make_connected_risk()
    nonhit = make_nonhit_controls(rows)
    uncertain = make_targeted_uncertain()
    detail = make_003085_detail(rows)
    overlap = make_003085_overlap(rows)
    result = dict(protocol="SURFBEV-V6-INTRAIMAGE-REVIEW-20261012", status="VISUAL_REVIEW_PENDING",
                  m3_input_sha256=hash_file(M3 / "manifest.jsonl"),
                  r3_input_sha256=hash_file(R3 / "connected_risk.jsonl"),
                  m3_action_scenes=len(m3),
                  m3_partition_change_scenes=sum(bool(x["split_old_ids"] or x["merged_new_ids"]) for x in m3),
                  m3_foreground_change_scenes=sum(x["foreground_delta_px"] > 0 for x in m3),
                  connected_risk_review_scenes=len(risk), nonhit_review_scenes=len(nonhit),
                  m3=m3, connected_risk=risk, nonhit=nonhit,
                  targeted_uncertain=uncertain, m3_003085_detail=detail,
                  m3_003085_overlap=overlap)
    (OUT / "index.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("m3", "connected_risk", "nonhit", "targeted_uncertain")}, indent=2))


if __name__ == "__main__":
    main()
