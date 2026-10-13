#!/usr/bin/env python3
"""Read-only classical-CV inventory of residual per-image instance-merge risks."""

import collections
import csv
import hashlib
import json
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[2]
M3 = ROOT / "outputs/SURFBEV-V5-IDENTITY-M3-20261012"
OLD = ROOT / "outputs/SURFBEV-DISCONNECT-QA-20261012-R3"
OUT = ROOT / "outputs/SURFBEV-V6-LOCAL-SPLIT-SCAN-20261012-R3"
EXPECTED_M3_SHA = "3e36bd8ac37dc1e0fb60bfa2182eac407fb6f8eb3e5353d6adda3ce3916e1261"
EXPECTED_OLD_SHA = "8d3da5addd302451337954191dccbf1b76f588a0790f1073415e30439015922f"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def local_masks(ids):
    ys, xs = np.nonzero(ids)
    local, inv, counts = np.unique(ids[ys,xs], return_inverse=True, return_counts=True)
    order = np.argsort(inv, kind="stable")
    offsets = np.r_[0, np.cumsum(counts)]
    for i, lid in enumerate(local):
        where = order[offsets[i]:offsets[i+1]]
        xx, yy = xs[where], ys[where]
        x0, y0, x1, y1 = int(xx.min()), int(yy.min()), int(xx.max())+1, int(yy.max())+1
        mask = np.zeros((y1-y0, x1-x0), dtype=np.uint8)
        mask[yy-y0,xx-x0] = 1
        yield int(lid), (x0,y0,x1,y1), mask, int(counts[i])


def connected_stats(mask, box, area):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n < 3:
        return None
    order = sorted(range(1,n), key=lambda i: int(stats[i,cv2.CC_STAT_AREA]), reverse=True)
    first, second = order[:2]
    second_area = int(stats[second,cv2.CC_STAT_AREA])
    if second_area < 64 or second_area/area < .10:
        return None
    dist = cv2.distanceTransform((labels!=first).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    gap = float(dist[labels==second].min())
    x0,y0,_,_ = box
    centroids = []
    for idx in (first,second):
        yy,xx = np.where(labels==idx)
        centroids.append([round(float(xx.mean()+x0),2),round(float(yy.mean()+y0),2)])
    return dict(components_8=n-1, largest_area=int(stats[first,cv2.CC_STAT_AREA]),
                second_area=second_area, second_frac=round(second_area/area,6),
                gap_px=round(gap,3), centroids_xy=centroids,
                strong=bool(second_area>=128 and second_area/area>=.20 and gap>=8))


def narrow_neck(mask, box, area, gray, seed):
    if area < 256 or box[0]==0 or box[1]==0 or box[2]==512 or box[3]==512:
        return None
    # A neck is meaningful here only when the original instance is one 8CC.
    # A small pre-existing fragment must not enter this separate candidate arm.
    if cv2.connectedComponents(mask, connectivity=8)[0] != 2:
        return None
    # Pixels removed by opening are only candidate necks; never a proposed GT edit.
    winner = None
    for radius in (2,4,6):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1))
        opened = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(opened,connectivity=8)
        major = sorted((int(stats[i,cv2.CC_STAT_AREA]) for i in range(1,n)),reverse=True)
        if len(major)<2 or major[1]<128 or major[1]/area<.15:
            continue
        yy,xx = np.where((mask!=0)&(opened==0))
        if len(xx)==0:
            continue
        x0,y0,x1,y1 = box
        crop_gray = gray[y0:y1,x0:x1]
        gx=cv2.Sobel(crop_gray,cv2.CV_32F,1,0,ksize=3)
        gy=cv2.Sobel(crop_gray,cv2.CV_32F,0,1,ksize=3)
        gradient = cv2.magnitude(gx,gy)
        # Equal-size internal-pixel sham: gradient is telemetry, not a split gate.
        interior = cv2.erode(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))!=0
        pool = gradient[interior & (opened!=0)]
        if len(pool)==0:
            continue
        rng=np.random.default_rng(seed+radius)
        sham=pool[rng.integers(0,len(pool),size=len(xx))]
        bridge_mean=float(gradient[yy,xx].mean())
        sham_mean=float(sham.mean())
        row=dict(radius_px=radius,lobe_areas=major[:2],removed_px=int(len(xx)),
                 removed_gradient_mean=round(bridge_mean,3),sham_gradient_mean=round(sham_mean,3),
                 gradient_ratio=round(bridge_mean/max(sham_mean,1e-6),3))
        if winner is None or (major[1],-radius)>(winner["lobe_areas"][1],-winner["radius_px"]):
            winner=row
    return winner


def render(rows, title, output, limit=12):
    selected = rows[:limit]
    if not selected:
        return []
    paths=[]
    for batch in range(0,len(selected),6):
        sheet=Image.new("RGB",(1200,900),(16,18,22))
        for j,row in enumerate(selected[batch:batch+6]):
            with Image.open(row["rgb"]) as im:
                rgb=np.asarray(im.convert("RGB"))
            with Image.open(row["label_png"]) as im:
                ids=np.asarray(im)
            mask=(ids==row["local_instance_id"])
            x0,y0,x1,y1=row["bbox_xyxy"]
            side=min(512,max(128,max(x1-x0,y1-y0)+32))
            cx,cy=(x0+x1)//2,(y0+y1)//2
            left=max(0,min(512-side,cx-side//2));top=max(0,min(512-side,cy-side//2))
            crop=(left,top,left+side,top+side)
            raw=Image.fromarray(rgb).crop(crop).resize((200,260),Image.Resampling.NEAREST)
            overlay=rgb.astype(np.float32)
            overlay[mask]=overlay[mask]*.45+np.array([25,230,235])*.55
            over=Image.fromarray(overlay.astype(np.uint8)).crop(crop).resize((200,260),Image.Resampling.NEAREST)
            x=(j%3)*400;y=(j//3)*450
            d=ImageDraw.Draw(sheet)
            d.text((x+4,y+5),f"{row['scene_dir'].split('_')[1]} {row['split']} local {row['local_instance_id']}",fill=(255,255,255))
            detail=(f"2nd {row['second_area']} gap {row['gap_px']}" if title=="disconnected" else
                    f"r {row['radius_px']} lobes {row['lobe_areas'][:2]} RGB {row['gradient_ratio']}" if title=="neck" else
                    f"non-hit area {row['area_px']}")
            d.text((x+4,y+24),detail,fill=(200,210,220))
            sheet.paste(raw,(x,y+50));sheet.paste(over,(x+200,y+50))
        path=OUT/f"{output}_{batch//6+1:02d}.png"
        sheet.save(path)
        paths.append(str(path))
    return paths


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(OUT)
    if sha(M3/"manifest.jsonl") != EXPECTED_M3_SHA:
        raise ValueError("M3 manifest drift")
    if sha(ROOT/"outputs/SURFBEV-MASK-V3-20261012/manifest.jsonl") != EXPECTED_OLD_SHA:
        raise ValueError("V3 manifest drift")
    manifest=list(jsonl(M3/"manifest.jsonl"))
    if len(manifest)!=1200 or len({r["scene_dir"] for r in manifest})!=1200:
        raise ValueError("M3 scene contract changed")
    old={(r["scene_dir"],int(r["local_instance_id"])) for r in csv.DictReader((OLD/"candidates.csv").open())}
    dis=[]; neck=[]; nonhits=[]; rgb_inputs=[]; totals=collections.Counter(); splits=collections.Counter()
    for rec in manifest:
        split=rec["split"]
        if split not in ("train","val"):
            continue
        splits[split]+=1
        rgb_inputs.append(dict(scene_dir=rec["scene_dir"],sha256=sha(rec["satellite_rgb"])))
        label=Path(rec["v5_png"])
        if sha(label)!=rec["v5_sha256"]:
            raise ValueError(f"PNG SHA drift {rec['scene_dir']}")
        with Image.open(label) as im:
            if im.size!=(512,512):
                raise ValueError(f"label shape drift {rec['scene_dir']}")
            ids=np.asarray(im)
        if ids.ndim!=2:
            raise ValueError("IDs must be 2D")
        gray=None
        for lid,box,mask,area in local_masks(ids):
            totals["instances"]+=1
            base=dict(split=split,scene_dir=rec["scene_dir"],local_instance_id=lid,
                      area_px=area,bbox_xyxy=box,touches_border=bool(box[0]==0 or box[1]==0 or box[2]==512 or box[3]==512),
                      rgb=rec["satellite_rgb"],label_png=str(label),v5_sha256=rec["v5_sha256"])
            measured=connected_stats(mask,box,area)
            if measured:
                row=dict(base,**measured,old_v3_m2_same_local=(rec["scene_dir"],lid) in old)
                dis.append(row);totals["m2"]+=1
                if row["strong"]:
                    totals["strong"]+=1
                continue
            if area>=256 and not base["touches_border"]:
                if gray is None:
                    with Image.open(rec["satellite_rgb"]) as im:
                        rgb=np.asarray(im.convert("RGB"))
                    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
                seed=int.from_bytes(hashlib.sha256(f"{rec['scene_dir']}:{lid}".encode()).digest()[:8],"big")
                candidate=narrow_neck(mask,box,area,gray,seed)
                if candidate:
                    neck.append(dict(base,**candidate));totals["neck"]+=1
                    continue
            nonhits.append(base)
    if splits!={"train":694,"val":100}:
        raise ValueError(f"split count drift: {splits}")
    if len({r["scene_dir"] for r in manifest if r["split"] in ("train","val")})!=794:
        raise ValueError("scene count drift")
    dis.sort(key=lambda r:(-int(r["strong"]),-r["second_area"],-r["gap_px"],r["scene_dir"],r["local_instance_id"]))
    neck.sort(key=lambda r:(-r["lobe_areas"][1],-r["gradient_ratio"],r["scene_dir"],r["local_instance_id"]))
    # Hash scenes first, then select a reproducible instance within each scene.
    # Instance-count variation therefore cannot bias scene selection.
    nonhits.sort(key=lambda r: (hashlib.sha256(
        f"20261012:scene:{r['scene_dir']}".encode()).hexdigest(),
        hashlib.sha256(f"20261012:instance:{r['scene_dir']}:{r['local_instance_id']}".encode()).hexdigest()))
    random_nonhits=[]; random_scenes=set()
    for row in nonhits:
        if row["scene_dir"] not in random_scenes:
            random_nonhits.append(row)
            random_scenes.add(row["scene_dir"])
        if len(random_nonhits)>=12:
            break
    OUT.mkdir(parents=True)
    for name,rows in (("disconnected.jsonl",dis),("neck.jsonl",neck),
                      ("nonhit_random.jsonl",random_nonhits),("rgb_inputs.jsonl",rgb_inputs)):
        with (OUT/name).open("w") as stream:
            for row in rows:
                stream.write(json.dumps(row,ensure_ascii=False)+"\n")
    top_dis=[];seen=set()
    for row in dis:
        if row["strong"] and row["scene_dir"] not in seen:
            top_dis.append(row);seen.add(row["scene_dir"])
        if len(top_dis)>=12:break
    top_neck=[];seen=set()
    for row in neck:
        if row["scene_dir"] not in seen:
            top_neck.append(row);seen.add(row["scene_dir"])
        if len(top_neck)>=12:break
    dis_imgs=render(top_dis,"disconnected","disconnected_review")
    neck_imgs=render(top_neck,"neck","neck_review")
    nonhit_imgs=render(random_nonhits,"negative","nonhit_random_review")
    summary=dict(protocol="SURFBEV-V6-LOCAL-SPLIT-SCAN-20261012",status="INVENTORY_ONLY",
                 m3_manifest_sha256=EXPECTED_M3_SHA,v3_manifest_sha256=EXPECTED_OLD_SHA,
                 split_counts=dict(splits),counts=dict(totals),disconnected_scenes=len({r["scene_dir"] for r in dis}),
                 strong_scenes=len({r["scene_dir"] for r in dis if r["strong"]}),
                 neck_scenes=len({r["scene_dir"] for r in neck}),
                 prior_v3_m2=len(old),
                 old_same_local_overlap_diagnostic_only=sum(r["old_v3_m2_same_local"] for r in dis),
                 nonhit_random_count=len(random_nonhits),
                 disconnected_sha256=sha(OUT/"disconnected.jsonl"),neck_sha256=sha(OUT/"neck.jsonl"),
                 nonhit_random_sha256=sha(OUT/"nonhit_random.jsonl"),
                 rgb_inputs_sha256=sha(OUT/"rgb_inputs.jsonl"),
                 review_pngs=dis_imgs+neck_imgs+nonhit_imgs,
                 note="Shape/RGB telemetry only; local ID changes prevent interpreting old/new set difference as fixes; no instance split or AP claim.")
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(summary,indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()
