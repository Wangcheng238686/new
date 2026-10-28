"""Independent image-content proxy for RGB/depth registration, fixed train views.

No matrices are used to create the evidence correspondence. Geometric depth
edges should often coincide with RGB contours, but textures/materials/LOD can
break this relation. Translation estimates are diagnostic, never applied to data.
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
OUT = Path('/data/wangcheng/checkpoint/ue_compact_joint_20261024/rgb_depth_image_alignment')
YAWS = (0, 72, 144, 216, 288)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def contact(rgb, z, valid, depth_edges, rgb_edges, text, path):
    values = z[valid]
    lo, hi = np.quantile(values, [.02, .98])
    grey = np.clip((np.where(valid, z, lo)-lo)/max(hi-lo, 1e-6), 0, 1)
    grey = (grey*255).astype(np.uint8)
    depth = np.repeat(grey[..., None], 3, axis=2)
    depth[~valid] = 0
    overlay = (rgb.astype(float)*.6).astype(np.uint8)
    overlay[rgb_edges] = [255, 0, 255]
    overlay[depth_edges] = [0, 255, 255]
    overlay[rgb_edges & depth_edges] = [255, 255, 255]
    canvas = Image.new('RGB', (1536, 560), 'white')
    d = ImageDraw.Draw(canvas)
    for col, (image, title) in enumerate([(rgb, 'original RGB'), (depth, 'camera-Z grey, per-view p2-p98'),
                                        (overlay, 'depth edges cyan / RGB edges magenta')]):
        canvas.paste(Image.fromarray(image), (col*512, 48))
        d.text((col*512+4, 4), title, fill='black')
    d.text((4, 24), text, fill='black')
    canvas.save(path)


def main():
    assert not OUT.exists(), 'Preserve started audits'
    OUT.mkdir()
    (OUT/'contacts').mkdir()
    ids = json.loads((DATA/'splits/train.json').read_text())
    indices = np.linspace(0, len(ids)-1, 8, dtype=int).tolist()
    selected = [int(ids[i]) for i in indices]
    protocol = dict(split='train', indices=indices, image_ids=selected, yaw=list(YAWS),
        source_sha256=sha(__file__), split_sha256=sha(DATA/'splits/train.json'),
        depth_units='EXR R camera-Z cm -> m; finite 0<Z<10000m',
        depth_edge='valid neighbor |delta Z| > max(2m,.01*min(Zleft,Zright)); mark both endpoints',
        rgb_edge='grey Gaussian3x3 sigma1; Canny80/160 L2',
        score='depth-edge->nearest RGB-edge distance, clipped12px mean; within2/4px fractions',
        crop_margin=40, grid_offsets='dx,dy integers [-8,8]; ties nearest zero then dy/dx',
        fixed_controls=[[-32,0],[32,0],[0,-32],[0,32]],
        same_points_for_all_offsets=True, visual_contacts='all40, no rank-based selection',
        offset_sign='add dx/dy to depth-edge coordinates to query RGB distance map',
        limitations='edge proxy only; texture/shadows/low contrast/material/LOD/sparse edges confound; no semantic truth, calibration certificate, AP evidence, input correction or test use')
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    (OUT/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    rows, score_grids = [], []
    sources = {}
    for ident in selected:
        folder = DATA/'points'/('%06d'%ident)
        camera = folder/'cameras.json'
        sources[str(camera)] = sha(camera)
        c = json.loads(camera.read_text())
        for yaw in YAWS:
            rgb_path = folder/('uav/yaw_%03d/rgb.png'%yaw)
            depth_path = folder/('uav/yaw_%03d/depth_camera_z.exr'%yaw)
            for p in (rgb_path, depth_path):sources[str(p)] = sha(p)
            raw = cv2.imread(str(rgb_path));exr = cv2.imread(str(depth_path),cv2.IMREAD_UNCHANGED)
            assert raw.shape == (512,512,3) and exr.shape[:2] == (512,512)
            rgb = raw[:,:,::-1].copy();z = exr[:,:,2].astype(np.float64)/100
            valid = np.isfinite(z)&(z>0)&(z<10000)
            zsafe = np.where(valid,z,0)
            edge = np.zeros((512,512),bool)
            for axis in (0,1):
                if axis == 0:
                    a,b = zsafe[:-1],zsafe[1:];good = valid[:-1]&valid[1:]
                else:
                    a,b = zsafe[:,:-1],zsafe[:,1:];good = valid[:,:-1]&valid[:,1:]
                jumps = good & (np.abs(a-b)>np.maximum(2,.01*np.minimum(a,b)))
                if axis == 0:
                    edge[:-1] |= jumps;edge[1:] |= jumps
                else:
                    edge[:,:-1] |= jumps;edge[:,1:] |= jumps
            gray = cv2.GaussianBlur(cv2.cvtColor(raw,cv2.COLOR_BGR2GRAY),(3,3),1)
            rgb_edge = cv2.Canny(gray,80,160,L2gradient=True)>0
            distance = cv2.distanceTransform((~rgb_edge).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
            region = edge.copy();region[:40]=False;region[-40:]=False;region[:,:40]=False;region[:,-40:]=False
            yy,xx = np.nonzero(region)
            def score(dx,dy):
                ds = distance[yy+dy,xx+dx]
                return dict(mean_clipped_distance=float(np.minimum(ds,12).mean()),
                    within2=int((ds<=2).sum()),within4=int((ds<=4).sum()),count=len(ds))
            grid = np.full((17,17),np.nan)
            if len(xx):
                zero = score(0,0)
                for dy in range(-8,9):
                    for dx in range(-8,9):grid[dy+8,dx+8] = score(dx,dy)['mean_clipped_distance']
                ranked = sorted((grid[dy+8,dx+8],dx*dx+dy*dy,dy,dx) for dy in range(-8,9) for dx in range(-8,9))
                _,_,dy,dx = ranked[0]
                best = dict(dx=dx,dy=dy,**score(dx,dy))
                controls = [dict(dx=dx,dy=dy,**score(dx,dy)) for dx,dy in protocol['fixed_controls']]
            else:
                zero=best=None;controls=[]
            row = dict(image_id=ident,yaw=yaw,valid_depth_pixels=int(valid.sum()),
                depth_edge_pixels_all=int(edge.sum()),depth_edge_pixels_scored=len(xx),rgb_edge_pixels=int(rgb_edge.sum()),
                at_zero=zero,best_local=best,controls=controls,
                camera_intrinsics=c['uav']['yaw_%03d'%yaw]['intrinsics'])
            rows.append(row);score_grids.append(grid)
            text = 'ID%06d yaw%03d scored=%d zero=%.3f best=(%s,%s)' % (ident,yaw,len(xx),zero['mean_clipped_distance'] if zero else -1,
                best['dx'] if best else 'NA',best['dy'] if best else 'NA')
            contact(rgb,z,valid,edge,rgb_edge,text,OUT/'contacts'/('%06d_yaw%03d.png'%(ident,yaw)))
            (OUT/'progress.json').write_text(json.dumps(dict(views=len(rows),total=40,image_id=ident,yaw=yaw))+'\n')
    assert len(rows)==40 and all(sha(path)==value for path,value in sources.items())
    assert sha(__file__)==protocol['source_sha256']
    np.savez_compressed(OUT/'offset_score_grids.npz',scores=np.stack(score_grids))
    supported = [r for r in rows if r['at_zero']]
    den = sum(r['at_zero']['count'] for r in supported)
    summary = dict(views=len(rows),views_with_edges=len(supported),scored_depth_edge_pixels=den,
        pooled_zero_mean_clipped_distance=sum(r['at_zero']['mean_clipped_distance']*r['at_zero']['count'] for r in supported)/den,
        pooled_within2=sum(r['at_zero']['within2'] for r in supported)/den,
        pooled_within4=sum(r['at_zero']['within4'] for r in supported)/den,
        best_offset_counts={('%d,%d'%(dx,dy)):sum(r['best_local']['dx']==dx and r['best_local']['dy']==dy for r in supported)
            for dx,dy in sorted({(r['best_local']['dx'],r['best_local']['dy']) for r in supported})})
    (OUT/'rows.json').write_text(json.dumps(rows,indent=2)+'\n')
    (OUT/'complete.json').write_text(json.dumps(dict(complete=True,protocol=protocol,summary=summary,
        source_sha256=sources,score_grids_sha256=sha(OUT/'offset_score_grids.npz'),visual_review_pending=True),indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':main()
