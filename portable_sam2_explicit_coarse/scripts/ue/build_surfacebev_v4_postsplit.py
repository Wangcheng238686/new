#!/usr/bin/env python3
"""Post-process one user-confirmed merged global ID into two physical instances."""
import argparse
import collections
import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


PACKAGE = Path(__file__).resolve().parents[2]
V3 = PACKAGE / "outputs/SURFBEV-MASK-V3-20261012"
OUT = PACKAGE / "outputs/SURFBEV-V4-POSTSPLIT-20261012-R3"
V3_SHA = "8d3da5addd302451337954191dccbf1b76f588a0790f1073415e30439015922f"
PARENT = 1874325323
KNOWN_SCENE = "Coord_002917_pt_gx_-5_gy_-2_s_002_gx_-5_gy_-2_s_2_X_-90000_Y_-37500"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def child_global(index):
    value = f"SURFBEV-V4:{PARENT}:{index}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big") & ((1<<63)-1)


def load_ids(path):
    with Image.open(path) as image:
        if image.mode != "L" or image.size != (512,512):
            raise ValueError(f"invalid v3 label: {path}")
        return np.asarray(image).copy()


def source_local(rec):
    metadata = json.loads((Path(rec["source_png"]).parent/"instances.json").read_text())
    found = [int(x["local_instance_id"]) for x in metadata["instances"] if int(x["global_instance_id"]) == PARENT]
    if len(found) > 1:
        raise ValueError(f"same global ID mapped to multiple local IDs: {rec['scene_dir']}")
    return found[0] if found else None


def components(ids, local):
    binary = (ids == local).astype(np.uint8)
    n, labels, stats, centers = cv2.connectedComponentsWithStats(binary, connectivity=8)
    order = sorted(range(1,n), key=lambda i:int(stats[i,cv2.CC_STAT_AREA]), reverse=True)
    total = int(binary.sum())
    if len(order) < 2:
        return None
    first, second = order[:2]
    areas = [int(stats[first,cv2.CC_STAT_AREA]), int(stats[second,cv2.CC_STAT_AREA])]
    return dict(labels=labels, stats=stats, centers=centers, order=order, area=total,
                major_ids=(first,second), major_areas=areas, remainder=total-sum(areas))


def to_world(rec, uv):
    sat = Path(rec["source_png"]).parent
    intr = json.loads((sat/"camera_intrinsics.json").read_text())
    ext = json.loads((sat/"camera_extrinsics.json").read_text())
    if ext["extrinsics_contract"]["transform"] != "world_to_camera" or ext["extrinsics_contract"]["world_units"] != "cm":
        raise ValueError(f"unexpected camera contract: {rec['scene_dir']}")
    c2w = np.asarray(ext["camera_to_world_cv"],dtype=np.float64)
    u,v = float(uv[0])+0.5,float(uv[1])+0.5
    ray_camera = np.asarray([(u-intr["cx"])/intr["fx"],(v-intr["cy"])/intr["fy"],1.],dtype=np.float64)
    ray_world = c2w[:3,:3] @ ray_camera
    position = c2w[:3,3]
    if ray_world[2] >= -.1:
        raise ValueError("satellite ray does not point downward")
    hit = position + (-position[2]/ray_world[2])*ray_world
    return hit[:2]/100.0  # metres, approximate Z=0 ground-plane mapping


def eligible(data):
    return data is not None and min(data["major_areas"]) >= 128 and data["remainder"] <= .1*data["area"]


def complete_view(data):
    if not eligible(data) or min(data["major_areas"])/data["area"] < .2:
        return False
    for index in data["major_ids"]:
        x,y,w,h,_ = (int(value) for value in data["stats"][index])
        if x==0 or y==0 or x+w==512 or y+h==512:
            return False
    return True


def world_pair(rec, data):
    return np.stack([to_world(rec,data["centers"][index]) for index in data["major_ids"]])


def fit_centers(pairs, known):
    if len(pairs) < 5 or known not in pairs:
        raise ValueError(f"too few eligible val scenes or known scene missing: {len(pairs)}")
    initial = pairs[known]
    centers = initial[np.lexsort((initial[:,1],initial[:,0]))].copy()
    for _ in range(30):
        assigned = []
        for points in pairs.values():
            cost_same = np.linalg.norm(points[0]-centers[0])+np.linalg.norm(points[1]-centers[1])
            cost_swap = np.linalg.norm(points[0]-centers[1])+np.linalg.norm(points[1]-centers[0])
            assigned.append(points if cost_same <= cost_swap else points[::-1])
        updated = np.mean(np.stack(assigned),axis=0)
        if np.max(np.abs(updated-centers)) < 1e-9:
            centers=updated
            break
        centers=updated
    return centers


def assign_major(points, centers):
    dist = np.linalg.norm(points[:,None,:]-centers[None,:,:],axis=2)
    assignment = np.argmin(dist,axis=1)
    margins = dist[np.arange(2),1-assignment]-dist[np.arange(2),assignment]
    return assignment, margins, dist


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--v3",type=Path,default=V3)
    parser.add_argument("--out",type=Path,default=OUT)
    args=parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"refusing to overwrite {args.out}")
    if sha(args.v3/"manifest.jsonl") != V3_SHA:
        raise ValueError("v3 manifest changed")
    v3summary=json.loads((args.v3/"summary.json").read_text())
    if v3summary["split_manifest_sha256"] != "17f49430b15037c8da25c63a1bf7fe4d510cf5ff153263d2feba62c4393441d5":
        raise ValueError("unexpected split")
    manifest=[json.loads(line) for line in (args.v3/"manifest.jsonl").read_text().splitlines() if line]
    if len(manifest)!=1200 or len({r["scene_dir"] for r in manifest})!=1200:
        raise ValueError("expected 1200 unique scenes")
    parent_rows=[]
    old_globals=set()
    for rec in manifest:
        src=Path(rec["source_png"])
        metadata=json.loads((src.parent/"instances.json").read_text())
        old_globals.update(int(x["global_instance_id"]) for x in metadata["instances"])
        local=source_local(rec)
        if local is not None:
            parent_rows.append((rec,local))
    children=(child_global(0),child_global(1))
    if children[0]==children[1] or any(x in old_globals for x in children):
        raise ValueError("derived global ID collision")
    val_pairs={}
    for rec,local in parent_rows:
        if rec["split"] != "val":
            continue
        ids=load_ids(rec["v3_png"])
        if not np.any(ids==local):
            continue
        data=components(ids,local)
        if complete_view(data):
            val_pairs[rec["scene_dir"]]=world_pair(rec,data)
    if len(val_pairs)<5:
        raise ValueError(f"fewer than 5 complete val scenes: {len(val_pairs)}")
    centers=fit_centers(val_pairs,KNOWN_SCENE)
    separation=float(np.linalg.norm(centers[0]-centers[1]))
    radii=[]; val_min_margin=float("inf")
    if separation < 20:
        raise ValueError(f"cluster separation <20m: {separation:.3f}")
    for scene,points in val_pairs.items():
        assignment,margins,dist=assign_major(points,centers)
        if len(set(assignment.tolist()))!=2 or float(margins.min()) < 5:
            raise ValueError(f"val cluster assignment gate failed: {scene}: {margins}")
        radii.extend(dist[np.arange(2),assignment].tolist())
        val_min_margin=min(val_min_margin,float(margins.min()))
    if max(radii)>15:
        raise ValueError(f"val cluster radius >15m: {max(radii):.3f}")

    (args.out/"labels").mkdir(parents=True)
    rows=[]; split_counts=collections.Counter(); child_splits=collections.defaultdict(set)
    for rec in manifest:
        src=Path(rec["v3_png"])
        if sha(src)!=rec["v3_sha256"]:
            raise ValueError(f"v3 label hash changed: {rec['scene_dir']}")
        dest=args.out/"labels"/(rec["scene_dir"]+".png")
        local=next((lid for item,lid in parent_rows if item["scene_dir"]==rec["scene_dir"]),None)
        action="UNCHANGED"; child_map={}; pixels_changed=0; reason=""
        if local is None:
            shutil.copyfile(src,dest)
        else:
            ids=load_ids(src)
            if not np.any(ids==local):
                action="INVISIBLE"; reason="parent absent in v3 raster"; shutil.copyfile(src,dest)
            else:
                data=components(ids,local)
                if not eligible(data):
                    action="HOLD"; reason="two-major-area-or-remainder-gate"; shutil.copyfile(src,dest)
                else:
                    points=world_pair(rec,data)
                    assignment,margins,dist=assign_major(points,centers)
                    min_margin=5 if complete_view(data) else 15
                    radius_ok=not complete_view(data) or float(dist[np.arange(2),assignment].max())<=15
                    if len(set(assignment.tolist()))!=2 or float(margins.min())<min_margin or not radius_ok:
                        action="HOLD"; reason="cross-scene-assignment-gate"; shutil.copyfile(src,dest)
                    else:
                        metadata=json.loads((Path(rec["source_png"]).parent/"instances.json").read_text())
                        declared={int(item["local_instance_id"]) for item in metadata["instances"]}
                        used=set(int(x) for x in np.unique(ids[ids>0])) | declared
                        free=next((candidate for candidate in range(1,256) if candidate not in used),None)
                        if free is None:
                            raise ValueError(f"no free local ID: {rec['scene_dir']}")
                        component_to_child={int(data["major_ids"][i]):int(assignment[i]) for i in range(2)}
                        for component_id in data["order"][2:]:
                            xy=to_world(rec,data["centers"][component_id])
                            component_to_child[int(component_id)]=int(np.argmin(np.linalg.norm(centers-xy,axis=1)))
                        out=ids.copy()
                        for component_id,child in component_to_child.items():
                            out[data["labels"]==component_id] = local if child==0 else free
                        original=ids==local
                        if not np.array_equal(ids[~original],out[~original]) or not np.array_equal(original,(out==local)|(out==free)):
                            raise ValueError(f"pixel union/parity failed: {rec['scene_dir']}")
                        Image.fromarray(out,"L").save(dest)
                        action="SPLIT"; pixels_changed=int((out!=ids).sum())
                        child_map={str(local):children[0],str(free):children[1]}
                        child_splits[children[0]].add(rec["split"]); child_splits[children[1]].add(rec["split"])
        split_counts[(rec["split"],action)]+=1
        rows.append(dict(scene_dir=rec["scene_dir"],split=rec["split"],v3_png=str(src),v3_sha256=rec["v3_sha256"],
                         v4_png=str(dest.resolve()),v4_sha256=sha(dest),action=action,reason=reason,
                         parent_global_id=PARENT if local is not None else None,parent_local_id=local,
                         child_local_to_global=child_map,pixels_changed=pixels_changed,
                         satellite_rgb=rec["satellite_rgb"]))
    for child,splits in child_splits.items():
        if len(splits & {"train","val","test"})>1:
            raise ValueError(f"derived child crosses evaluation split: {child} {splits}")
    with (args.out/"manifest.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row,sort_keys=True)+"\n")
    summary=dict(protocol="SURFBEV-V4-POSTSPLIT-20261012",status="BUILT_UNREVIEWED",parent_global_id=PARENT,
                 child_global_ids=list(children),v3_manifest_sha256=V3_SHA,output_scenes=len(rows),
                 calibration_val_scenes=len(val_pairs),center_xy_m=centers.tolist(),separation_m=separation,
                 max_val_radius_m=max(radii),min_val_assignment_margin_m=val_min_margin,
                 split_action_counts={split:{action:split_counts[(split,action)] for action in ("SPLIT","HOLD","INVISIBLE","UNCHANGED")} for split in ("train","val","test","excluded")},
                 pixels_changed=sum(x["pixels_changed"] for x in rows),
                 child_splits={str(key):sorted(value) for key,value in child_splits.items()},
                 manifest_sha256=sha(args.out/"manifest.jsonl"),
                 note="Only user-confirmed parent global ID split. No UE rerender, no model-side ignore, no AP claim.")
    (args.out/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(summary,indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()
