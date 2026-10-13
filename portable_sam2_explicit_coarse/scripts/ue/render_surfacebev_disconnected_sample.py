#!/usr/bin/env python3
"""Render preselected disconnected-mask examples for semantic spot checking."""
import argparse
import csv
import html
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


PACKAGE = Path(__file__).resolve().parents[2]
ROOT = PACKAGE / "outputs/SURFBEV-DISCONNECT-QA-20261012-R3"


def render(row, out):
    with Image.open(row["v3_png"]) as image:
        ids = np.asarray(image).copy()
    rgb = np.asarray(Image.open(row["rgb"]).convert("RGB"))
    local = int(row["local_instance_id"])
    mask = (ids == local).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    order = sorted(range(1,n), key=lambda i:int(stats[i,cv2.CC_STAT_AREA]), reverse=True)
    if len(order) < 2:
        raise ValueError(f"candidate lost components: {row['scene_dir']} / {local}")
    first, second = labels == order[0], labels == order[1]
    other = (mask > 0) & ~first & ~second
    canvas = rgb.copy()
    for selection, color in ((first,(20,220,240)), (second,(255,45,175)), (other,(255,220,40))):
        canvas[selection] = (canvas[selection].astype(np.float32)*.5+np.asarray(color)*.5).astype(np.uint8)
        contours,_ = cv2.findContours(selection.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas,contours,-1,color,2)
    x0,y0,x1,y1 = (int(row[f"bbox_{key}"]) for key in ("x0","y0","x1","y1"))
    cv2.rectangle(canvas,(x0,y0),(x1-1,y1-1),(255,255,255),1)
    side = min(512,max(128,x1-x0+32,y1-y0+32))
    cx,cy = (x0+x1)/2,(y0+y1)/2
    left = max(0,min(512-side,int(round(cx-side/2))))
    top = max(0,min(512-side,int(round(cy-side/2))))
    zoom = Image.fromarray(canvas[top:top+side,left:left+side]).resize((512,512),Image.Resampling.NEAREST)
    view=Image.new("RGB",(1536,548),(33,36,42))
    for index,panel in enumerate((Image.fromarray(rgb),Image.fromarray(canvas),zoom)):
        view.paste(panel,(512*index,36))
    d=ImageDraw.Draw(view)
    for index,label in enumerate(("Satellite RGB","two largest components + bbox","bbox zoom")):
        d.text((512*index+8,10),label,fill=(245,245,245))
    name=f'{row["split"]}_{row["scene_dir"]}_id{local}.jpg'
    view.save(out/"full"/name,quality=90,subsampling=0)
    return name


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--sample",type=Path,default=ROOT/"visual_sample.csv")
    parser.add_argument("--out",type=Path,default=ROOT/"visual_sample")
    args=parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"refusing overwrite: {args.out}")
    rows=list(csv.DictReader(args.sample.open()))
    if any(row["split"] not in ("train","val") for row in rows):
        raise ValueError("test/excluded visualization forbidden")
    (args.out/"full").mkdir(parents=True)
    parts=["<!doctype html><html lang=zh-CN><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>",
           "<title>非连通实例抽样</title><style>:root{color-scheme:light dark;font-family:system-ui,sans-serif}body{max-width:1600px;margin:auto;padding:20px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(480px,1fr));gap:16px}.case{min-width:0;border:1px solid color-mix(in srgb,currentColor 20%,transparent);padding:8px}.case img{width:100%;height:auto}code{overflow-wrap:anywhere}</style>",
           "<h1>非连通实例：train/val 固定抽样</h1><p>青色与粉色是同一 local ID 的两大连通块，黄色为其余块；白框是该 ID 的共同 bbox。是否属两栋独立建筑需按物理对象判断，不能仅凭颜色分开。</p><main>"]
    for row in rows:
        name=render(row,args.out)
        parts.append(f'<article class="case"><a href="full/{html.escape(name)}"><img loading="lazy" src="full/{html.escape(name)}" alt="{html.escape(row["scene_dir"])} 的非连通实例"></a><div>{html.escape(row["split"])} · {"强候选" if row["m2_strong"]=="1" else "边界对照"} · local {row["local_instance_id"]} / global {row["global_instance_id"]} · 第二块 {row["second_area"]} px / {row["second_frac"]}，间距 {row["min_gap_px"]} px<br><code>{html.escape(row["scene_dir"])}</code></div></article>')
    parts.append("</main></html>")
    (args.out/"index.html").write_text("\n".join(parts),encoding="utf-8")
    print(f"rendered {len(rows)} images to {args.out/'index.html'}")


if __name__=="__main__":
    main()
