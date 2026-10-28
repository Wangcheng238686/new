"""Raw UAV RGB reprojection vs independently rendered SAT RGB, train only.

Uses direct centimeter homogeneous transforms, not model caches or labels.
Actual depth and a declared sampling-origin plane are compared on identical
per-view target pixels. Photometry is evidence, not certified visibility/truth.
"""
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
from pathlib import Path
import hashlib
import json

import cv2
import numpy as np
from PIL import Image, ImageDraw

cv2.setNumThreads(1)
DATA = Path('/data/wangcheng/dataset/CitySample_SatUAV_TrainMinimal_v1_20261021')
OUT = Path('/data/wangcheng/checkpoint/ue_compact_joint_20261024/rgb_sat_image_correspondence')
YAWS = (0,72,144,216,288)


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def matrix(view):
    a=view['intrinsics']
    assert a['width']==a['height']==512
    return np.array([[a['fx'],0,a['cx']],[0,a['fy'],a['cy']],[0,0,1.]])


def project(rgb, zcm, source, satellite):
    yy,xx=np.mgrid[:512,:512]
    k=matrix(source);sk=matrix(satellite)
    t=np.array(source['camera_to_world_cv'],np.float64)
    e=np.array(satellite['world_to_camera_cv'],np.float64)
    valid=np.isfinite(zcm)&(zcm>0)&(zcm<1e6)
    z=zcm[valid]
    ray=np.stack(((xx[valid]-k[0,2])/k[0,0],(yy[valid]-k[1,2])/k[1,1],np.ones(len(z))),1)
    world=(ray*z[:,None])@t[:3,:3].T+t[:3,3]
    target=world@e[:3,:3].T+e[:3,3]
    p=target@sk.T
    with np.errstate(divide='ignore',invalid='ignore'):uv=p[:,:2]/p[:,2:]
    good=(target[:,2]>0)&np.isfinite(uv).all(1)&(uv>=0).all(1)&(uv<512).all(1)
    uv=uv[good];sat_z=target[good,2]
    pixels=uv[:,1].astype(np.int64)*512+uv[:,0].astype(np.int64)
    color=rgb[valid][good]
    order=np.lexsort((sat_z,pixels));sp=pixels[order]
    keep=np.r_[True,sp[1:]!=sp[:-1]] if len(sp) else np.empty(0,bool)
    chosen=order[keep]
    image=np.zeros((512*512,3),np.uint8);depth=np.full(512*512,np.inf)
    image[pixels[chosen]]=color[chosen];depth[pixels[chosen]]=sat_z[chosen]
    return image.reshape(512,512,3),depth.reshape(512,512)


def plane_depth(source,z_world_cm):
    yy,xx=np.mgrid[:512,:512];k=matrix(source);t=np.array(source['camera_to_world_cv'],np.float64)
    ray=np.stack(((xx-k[0,2])/k[0,0],(yy-k[1,2])/k[1,1],np.ones((512,512))),-1)
    world_direction=ray@t[:3,:3].T
    with np.errstate(divide='ignore',invalid='ignore'):
        depth=(z_world_cm-t[2,3])/world_direction[:,:,2]
    return depth


def zncc(a,b):
    a=a.astype(np.float64);b=b.astype(np.float64)
    a-=a.mean(0);b-=b.mean(0)
    den=np.sqrt(np.square(a).sum()*np.square(b).sum())
    return float((a*b).sum()/den) if den>0 else None


def main():
    assert not OUT.exists()
    OUT.mkdir();(OUT/'contacts').mkdir()
    ids=json.loads((DATA/'splits/train.json').read_text());indices=np.linspace(0,len(ids)-1,8,dtype=int).tolist()
    selected=[int(ids[i]) for i in indices]
    protocol=dict(split='train',image_ids=selected,indices=indices,yaw=list(YAWS),
        source_sha256=sha(__file__),split_sha256=sha(DATA/'splits/train.json'),
        geometry='independent raw camera-Z cm, camera_to_world/world_to_camera cm; no model/cache/labels',
        reference='horizontal world-Z=sampling-origin Z, not measured ground',
        photometry='per-view true/plane ZNCC on identical support intersection, central margin40; channels centered separately, norms pooled',
        controls=[[-32,0],[32,0],[0,-32],[0,32]],
        proxy='nearest observed surface per target pixel, not complete satellite visibility',
        limitations='lighting/material/view/LOD/occlusion/aliasing can change RGB; all surfaces and low frequencies included; no subpixel calibration certificate, semantic truth, extra-information or AP proof')
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');(OUT/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    sources={};rows=[]
    for ident in selected:
        folder=DATA/'points'/('%06d'%ident)
        for p in [folder/'cameras.json',folder/'satellite/rgb.png']:sources[str(p)]=sha(p)
        c=json.loads((folder/'cameras.json').read_text());assert c['world_units']=='cm'
        sat=cv2.imread(str(folder/'satellite/rgb.png'))[:,:,::-1].copy()
        true_image=np.zeros_like(sat);plane_image=np.zeros_like(sat)
        true_z=np.full((512,512),np.inf);plane_z=true_z.copy()
        for yaw in YAWS:
            rgb_path=folder/('uav/yaw_%03d/rgb.png'%yaw);dep_path=folder/('uav/yaw_%03d/depth_camera_z.exr'%yaw)
            for p in [rgb_path,dep_path]:sources[str(p)]=sha(p)
            rgb=cv2.imread(str(rgb_path))[:,:,::-1].copy();zcm=cv2.imread(str(dep_path),cv2.IMREAD_UNCHANGED)[:,:,2].astype(np.float64)
            view=c['uav']['yaw_%03d'%yaw]
            measured,mz=project(rgb,zcm,view,c['satellite'])
            planar,pz=project(rgb,plane_depth(view,c['center_ue_cm'][2]),view,c['satellite'])
            replace=mz<true_z;true_image[replace]=measured[replace];true_z[replace]=mz[replace]
            replace=pz<plane_z;plane_image[replace]=planar[replace];plane_z[replace]=pz[replace]
            support=np.isfinite(mz)&np.isfinite(pz)
            support[:40]=False;support[-40:]=False;support[:,:40]=False;support[:,-40:]=False
            yy,xx=np.nonzero(support)
            if len(xx)>=100:
                true_score=zncc(measured[support],sat[support]);plane_score=zncc(planar[support],sat[support])
                control=[dict(dx=dx,dy=dy,ZNCC=zncc(measured[support],sat[yy+dy,xx+dx])) for dx,dy in protocol['controls']]
            else:true_score=plane_score=None;control=[]
            rows.append(dict(image_id=ident,yaw=yaw,common_pixels=len(xx),measured_ZNCC=true_score,
                plane_ZNCC=plane_score,controls=control,measured_supported_pixels=int(np.isfinite(mz).sum()),
                plane_supported_pixels=int(np.isfinite(pz).sum())))
        yy,xx=np.mgrid[:512,:512];check=((yy//8+xx//8)%2).astype(bool)&np.isfinite(true_z)
        checker=sat.copy();checker[check]=true_image[check]
        canvas=Image.new('RGB',(2048,548),'white');draw=ImageDraw.Draw(canvas)
        for j,(im,title) in enumerate([(sat,'actual satellite RGB'),(true_image,'raw true-depth RGB warp'),
                (checker,'8px SAT/true-warp checker'),(plane_image,'reference-plane RGB warp')]):
            canvas.paste(Image.fromarray(im),(j*512,36));draw.text((j*512+4,4),title,fill='black')
        draw.text((4,20),'ID%06d nearest observed surface, holes black; no labels'%ident,fill='black')
        canvas.save(OUT/'contacts'/('%06d.png'%ident))
        (OUT/'progress.json').write_text(json.dumps(dict(images=len(rows)//5,total=8))+'\n')
    assert len(rows)==40 and all(sha(p)==value for p,value in sources.items())
    assert sha(__file__)==protocol['source_sha256']
    usable=[r for r in rows if r['measured_ZNCC'] is not None and r['plane_ZNCC'] is not None]
    summary=dict(views=40,defined_views=len(usable),common_pixels=sum(r['common_pixels'] for r in usable),
        mean_view_measured_ZNCC=float(np.mean([r['measured_ZNCC'] for r in usable])),
        mean_view_plane_ZNCC=float(np.mean([r['plane_ZNCC'] for r in usable])),
        measured_above_plane_views=int(sum(r['measured_ZNCC']>r['plane_ZNCC'] for r in usable)),
        measured_above_mean_shift_control_views=int(sum(r['measured_ZNCC']>np.mean([v['ZNCC'] for v in r['controls']]) for r in usable)))
    (OUT/'rows.json').write_text(json.dumps(rows,indent=2)+'\n')
    (OUT/'complete.json').write_text(json.dumps(dict(complete=True,protocol=protocol,source_sha256=sources,
        summary=summary,visual_review_pending=True),indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
