#!/usr/bin/env python3
"""Render three code-selected ambiguous connected-component cases, no label edits."""

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from probe_surfacebev_v6_local_split_scan import OUT, jsonl


CASES = ("002749", "003252", "002739")


def main():
    all_rows=list(jsonl(OUT/"disconnected.jsonl"))
    result=[]
    for short in CASES:
        row=next(r for r in all_rows if r["scene_dir"].split("_")[1]==short and r["strong"])
        with Image.open(row["rgb"]) as im:
            rgb=np.asarray(im.convert("RGB"))
        with Image.open(row["label_png"]) as im:
            ids=np.asarray(im)
        mask=(ids==row["local_instance_id"]).astype(np.uint8)
        n, labels, stats, _=cv2.connectedComponentsWithStats(mask,connectivity=8)
        ranked=sorted(range(1,n),key=lambda i:int(stats[i,cv2.CC_STAT_AREA]),reverse=True)
        if int(stats[ranked[1],cv2.CC_STAT_AREA])!=row["second_area"]:
            raise ValueError(f"component census drift: {short}")
        first,second=ranked[:2]
        first_mask=labels==first;second_mask=labels==second
        ys,xs=np.nonzero(first_mask|second_mask)
        side=min(512,max(160,int(max(xs.max()-xs.min()+1,ys.max()-ys.min()+1)+48)))
        cx,cy=(int(xs.min())+int(xs.max()))//2,(int(ys.min())+int(ys.max()))//2
        x0=max(0,min(512-side,cx-side//2));y0=max(0,min(512-side,cy-side//2))
        crop=(x0,y0,x0+side,y0+side)
        raw=Image.fromarray(rgb).crop(crop).resize((400,400),Image.Resampling.NEAREST)
        overlay=rgb.astype(np.float32)
        overlay[first_mask]=overlay[first_mask]*.45+np.array([255,75,175])*.55
        overlay[second_mask]=overlay[second_mask]*.45+np.array([25,230,235])*.55
        over=Image.fromarray(overlay.astype(np.uint8)).crop(crop).resize((400,400),Image.Resampling.NEAREST)
        full=Image.fromarray(rgb).resize((400,400),Image.Resampling.NEAREST)
        d=ImageDraw.Draw(full)
        d.rectangle(tuple(round(x*400/512) for x in crop),outline=(255,80,180),width=3)
        canvas=Image.new("RGB",(1200,454),(18,20,24))
        d=ImageDraw.Draw(canvas)
        d.text((10,7),f"{short} {row['split']} local {row['local_instance_id']} | two largest parts",fill=(255,255,255))
        d.text((10,28),f"pink {row['largest_area']} px / cyan {row['second_area']} px / min gap {row['gap_px']} px",fill=(205,215,225))
        for i,im in enumerate((raw,over,full)):
            canvas.paste(im,(400*i,54))
        name=f"key_{short}.png"
        canvas.save(OUT/name)
        result.append(dict(scene_dir=row["scene_dir"],split=row["split"],
                           local_instance_id=row["local_instance_id"],
                           largest_area=row["largest_area"],second_area=row["second_area"],
                           gap_px=row["gap_px"],crop_xyxy=crop,image=str(OUT/name)))
    (OUT/"key_component_review.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    render_neck_key()
    print(json.dumps(result,indent=2,ensure_ascii=False))


def render_neck_key():
    row=next(r for r in jsonl(OUT/"neck.jsonl") if r["scene_dir"].split("_")[1]=="003019")
    with Image.open(row["rgb"]) as im:
        rgb=np.asarray(im.convert("RGB"))
    with Image.open(row["label_png"]) as im:
        ids=np.asarray(im)
    mask=(ids==row["local_instance_id"]).astype(np.uint8)
    radius=row["radius_px"]
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1))
    opened=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
    n,labels,stats,_=cv2.connectedComponentsWithStats(opened,8)
    ordered=sorted(range(1,n),key=lambda i:int(stats[i,cv2.CC_STAT_AREA]),reverse=True)
    if [int(stats[i,cv2.CC_STAT_AREA]) for i in ordered[:2]]!=row["lobe_areas"]:
        raise ValueError("narrow-neck lobes drift")
    x0,y0,x1,y1=row["bbox_xyxy"]
    side=min(512,max(160,max(x1-x0,y1-y0)+48))
    cx,cy=(x0+x1)//2,(y0+y1)//2
    left=max(0,min(512-side,cx-side//2));top=max(0,min(512-side,cy-side//2))
    box=(left,top,left+side,top+side)
    raw=Image.fromarray(rgb).crop(box).resize((400,400),Image.Resampling.NEAREST)
    over=rgb.astype(np.float32)
    over[mask!=0]=over[mask!=0]*.45+np.array([25,230,235])*.55
    mask_img=Image.fromarray(over.astype(np.uint8)).crop(box).resize((400,400),Image.Resampling.NEAREST)
    lobes=rgb.astype(np.float32)
    lobes[labels==ordered[0]]=lobes[labels==ordered[0]]*.45+np.array([255,75,175])*.55
    lobes[labels==ordered[1]]=lobes[labels==ordered[1]]*.45+np.array([255,220,35])*.55
    lobes_img=Image.fromarray(lobes.astype(np.uint8)).crop(box).resize((400,400),Image.Resampling.NEAREST)
    canvas=Image.new("RGB",(1200,454),(18,20,24))
    d=ImageDraw.Draw(canvas)
    d.text((10,7),"003019 connected mask: raw / original one ID / two opening lobes",fill=(255,255,255))
    d.text((10,28),f"r={radius}, lobes {row['lobe_areas']}, gradient ratio {row['gradient_ratio']} (candidate only)",fill=(205,215,225))
    for i,im in enumerate((raw,mask_img,lobes_img)):
        canvas.paste(im,(i*400,54))
    canvas.save(OUT/"neck_key_003019.png")


if __name__=="__main__":
    main()
