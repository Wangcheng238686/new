#!/usr/bin/env python3
"""Render v3/v4 overlays for the confirmed post-processed parent ID."""
import argparse
import html
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


PACKAGE=Path(__file__).resolve().parents[2]
ROOT=PACKAGE/"outputs/SURFBEV-V4-POSTSPLIT-20261012-R3"


def tint(rgb,mask,color):
    image=rgb.copy()
    image[mask]=(image[mask].astype(np.float32)*.55+np.asarray(color)*.45).astype(np.uint8)
    contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image,contours,-1,color,2)
    return Image.fromarray(image)


def render(row,out):
    old=np.asarray(Image.open(row["v3_png"]).convert("L"))
    new=np.asarray(Image.open(row["v4_png"]).convert("L"))
    rgb=np.asarray(Image.open(row["satellite_rgb"]).convert("RGB"))
    if old.shape!=(512,512) or new.shape!=(512,512) or rgb.shape!=(512,512,3):
        raise ValueError("unexpected image shape")
    local=int(row["parent_local_id"])
    old_mask=old==local
    before=tint(rgb,old_mask,(255,45,170))
    after=rgb.copy()
    child_map={int(k):v for k,v in row["child_local_to_global"].items()}
    if row["action"]=="SPLIT":
        if len(child_map)!=2 or local not in child_map:
            raise ValueError("invalid child mapping")
        other=next(k for k in child_map if k!=local)
        if not np.array_equal(old_mask,(new==local)|(new==other)):
            raise ValueError("old/new union mismatch")
        after=np.asarray(tint(after,new==local,(20,220,240)))
        after=np.asarray(tint(after,new==other,(255,45,170)))
    else:
        after=np.asarray(tint(after,new==local,(255,45,170)))
    canvas=Image.new("RGB",(1536,548),(33,36,42))
    for i,panel in enumerate((Image.fromarray(rgb),before,Image.fromarray(after))):
        canvas.paste(panel,(512*i,36))
    draw=ImageDraw.Draw(canvas)
    for i,label in enumerate(("Satellite RGB","v3: one local ID","v4: two IDs" if row["action"]=="SPLIT" else "v4: held unchanged")):
        draw.text((512*i+8,10),label,fill=(245,245,245))
    name=row["split"]+"_"+row["scene_dir"]+".jpg"
    canvas.save(out/"full"/name,quality=90,subsampling=0)
    return name


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--manifest",type=Path,default=ROOT/"manifest.jsonl")
    parser.add_argument("--out",type=Path,default=ROOT/"visual_review")
    args=parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"refusing overwrite {args.out}")
    rows=[json.loads(line) for line in args.manifest.read_text().splitlines() if line]
    rows=[row for row in rows if row["action"] in ("SPLIT","HOLD")]
    if sum(row["action"]=="SPLIT" for row in rows)!=17 or sum(row["action"]=="HOLD" for row in rows)!=6:
        raise ValueError("unexpected split/hold count")
    (args.out/"full").mkdir(parents=True)
    parts=["<!doctype html><html lang=zh-CN><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>",
           "<title>v4 后处理实例对照</title><style>:root{color-scheme:light dark;font-family:system-ui,sans-serif}body{max-width:1600px;margin:auto;padding:20px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(480px,1fr));gap:16px}.case{min-width:0;border:1px solid color-mix(in srgb,currentColor 20%,transparent);padding:8px}.case img{width:100%;height:auto}code{overflow-wrap:anywhere}</style>",
           "<h1>已确认误并 ID：v3 与 v4 标签对照</h1><p>左为卫星 RGB，中为 v3 的同一 ID（粉色），右为 v4 的两个 ID（青色/粉色）。HOLD 图不拆，前后应一致。展示全部 17 个修改场景和 6 个保留场景，人工只需抽样检查。</p><main>"]
    for row in rows:
        name=render(row,args.out)
        parts.append(f'<article class="case"><a href="full/{html.escape(name)}"><img loading="lazy" src="full/{html.escape(name)}" alt="{html.escape(row["scene_dir"])} 后处理前后对照"></a><div>{html.escape(row["split"])} / {html.escape(row["action"])} · 原 local {row["parent_local_id"]} · 变更像素 {row["pixels_changed"]}<br><code>{html.escape(row["scene_dir"])}</code></div></article>')
    parts.append("</main></html>")
    (args.out/"index.html").write_text("\n".join(parts),encoding="utf-8")
    print(f"rendered {len(rows)} views: {args.out/'index.html'}")


if __name__=="__main__":
    main()
