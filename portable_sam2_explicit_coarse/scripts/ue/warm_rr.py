#!/usr/bin/env python3
"""WARM-RR: decision-level recall rescue (2026-10-24).
EVI-DECODER v2 evidence on UAV views -> candidate projection to satellite
frame -> connected-component proposals -> inject ONLY where the frozen base
has no overlapping prediction. Base untouched; thr/score fixed conservative
(0.6/0.40) pending train-side calibration; test read + constant control."""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
import cv2
from scipy import ndimage
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path('/home/wangcheng/project/new/.worktrees/dual-stream-dev/portable_sam2_explicit_coarse')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'inference'))
sys.path.insert(0, str(ROOT / 'scripts' / 'ue'))

import ue.dual  # noqa
from ue.dual import _UAV_MEAN, _UAV_STD
from rsprompter.backbone.sam2_vision import RSSAM2VisionEncoder
from probe_evi_decoder_v2 import EviDecoder

DS = Path('/data1/wangcheng/dataset/CitySample_SatUAV_TrainMinimal_v1_20261021')
CACHE = Path('/data2/wangcheng/remote-lthpc/ue_v1_local_adapter_geometry')
DECODER = '/data2/wangcheng/checkpoint/surfbev_b0/probes/EVI-DECODER-20261021/decoder.pt'
OUT = Path('/data2/wangcheng/remote-lthpc/warm_rr')
YAWS = (0, 72, 144, 216, 288)
import os
THR, SCORE = float(os.environ.get('RR_THR', 0.6)), float(os.environ.get('RR_SCORE', 0.40))


def to_rle(e):
    if isinstance(e, dict):
        return {'size': e['size'], 'counts': e['counts'].encode() if isinstance(e['counts'], str) else e['counts']}
    return e


@torch.no_grad()
def sat_evidence(enc, dec, pid, appearance='correct'):
    rgbs = []
    for yaw in YAWS:
        img = cv2.imread(str(DS / 'points' / pid / 'uav' / f'yaw_{yaw:03d}' / 'rgb.png'))
        assert img is not None
        rgbs.append(img[:, :, ::-1].transpose(2, 0, 1).copy())
    rgb = torch.from_numpy(np.stack(rgbs)).float().cuda()
    if appearance == 'constant':
        rgb[:] = torch.tensor([123.675, 116.28, 103.53], device='cuda').view(1, 3, 1, 1)
    mean = rgb.new_tensor(_UAV_MEAN)[:, None, None]
    std = rgb.new_tensor(_UAV_STD)[:, None, None]
    _, fpn = enc((rgb - mean) / std)
    desc = max(tuple(fpn[-4:]), key=lambda f: f.shape[-2] * f.shape[-1]).float()
    ev = torch.sigmoid(dec(desc))[:, 0].cpu().numpy()      # [5,128,128]
    with np.load(CACHE / f'{pid}.npz') as a:
        index = a['index']
        geom = a['geom'].astype(np.float32)                # [16384,8,6]
    tflat = index.reshape(-1)                               # [16384*8]
    valid = tflat >= 0
    view, rem = np.divmod(tflat[valid].astype(np.int64), 128 * 128)
    v, u = np.divmod(rem, 128)
    vals = ev[view, v, u]
    cells = np.repeat(np.arange(16384), index.shape[1])[valid]
    sat = np.zeros(16384, np.float32)
    np.maximum.at(sat, cells, vals)
    # depth channel: per-cell max candidate height (world-Z above scene
    # origin, normalized by /100 in the cache build) - elevated surfaces
    height = np.zeros(16384, np.float32)
    np.maximum.at(height, cells, geom.reshape(-1, 6)[valid, 2])
    # intrinsics/extrinsics channel: distinct UAV views observing the cell
    seen_view = np.zeros((16384, 5), bool)
    seen_view[cells, view] = True
    views = seen_view.sum(1).astype(np.float32)
    return sat.reshape(128, 128), height.reshape(128, 128), views.reshape(128, 128)


def proposals(sat_evi, height=None, views=None, thr=None, hthr=0.0, min_views=1, min_cells=2):
    thr = THR if thr is None else thr
    m = sat_evi >= thr
    if height is not None and hthr > 0:
        m = m & (height >= hthr)
    if views is not None and min_views > 1:
        m = m & (views >= min_views)
    lab, n = ndimage.label(m)
    out = []
    for c in range(1, n + 1):
        cells = lab == c
        if cells.sum() < min_cells:
            continue
        out.append(np.kron(cells, np.ones((4, 4), np.uint8)).astype(bool))
    return out


def evaluate(recs, enc, dec, appearance, hthr=0.0, min_views=1, refine=False):
    all_gt, all_dt = [], []
    added_total = 0
    for rec in recs:
        pid = f"{rec['image_id']:06d}"
        if not (CACHE / f'{pid}.npz').exists():
            continue
        gt_rles = [to_rle(e) for e in rec['gt']['rles']]
        dt_rles = [to_rle(e) for e in rec['dt']['rles']]
        dt_scores = list(rec['dt']['scores'])
        sat, height, views = sat_evidence(enc, dec, pid, appearance)
        for m in proposals(sat, height, views, hthr=hthr, min_views=min_views):
            ys, xs = np.where(m)
            y1, y2, x1, x2 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            ok = True
            for r in dt_rles:
                g = maskUtils.toBbox(r)
                if g[0] < x2 and x1 < g[0] + g[2] and g[1] < y2 and y1 < g[1] + g[3]:
                    inter = (m & maskUtils.decode(r).astype(bool)).sum()
                    if inter > 0.3 * min(m.sum(), maskUtils.area(r)):
                        ok = False
                        break
            if ok:
                dt_rles.append(maskUtils.encode(np.asfortranarray(m.astype(np.uint8))))
                dt_scores.append(SCORE)
                added_total += 1
        all_gt.append([maskUtils.decode(g).astype(bool) for g in gt_rles])
        all_dt.append([maskUtils.decode(d).astype(bool) for d in dt_rles])
    coco = COCO()
    anns, dts = [], []
    for i, (g, d) in enumerate(zip(all_gt, all_dt)):
        iid = i + 1
        for m in g:
            r = maskUtils.encode(np.asfortranarray(m.astype(np.uint8)))
            x, y, w, h = maskUtils.toBbox(r)
            anns.append({'id': len(anns) + 1, 'image_id': iid, 'category_id': 1, 'segmentation': r,
                         'bbox': [float(x), float(y), float(w), float(h)], 'area': float(maskUtils.area(r)), 'iscrowd': 0})
        for m, s in zip(d, [SCORE] * 0 + []):  # placeholder, replaced below
            pass
        # base scores preserved + added at SCORE
        scores = recs[i]['dt']['scores'] if i < len(recs) else [0.5] * len(d)
        for m, s in zip(d, scores + [SCORE] * (len(d) - len(scores))):
            r = maskUtils.encode(np.asfortranarray(m.astype(np.uint8)))
            x, y, w, h = maskUtils.toBbox(r)
            dts.append({'image_id': iid, 'category_id': 1, 'score': float(s), 'segmentation': r,
                        'bbox': [float(x), float(y), float(w), float(h)]})
    coco.dataset = {'images': [{'id': i + 1, 'height': 512, 'width': 512} for i in range(len(all_gt))],
                    'annotations': anns, 'categories': [{'id': 1, 'name': 'b'}]}
    coco.createIndex()
    dt = coco.loadRes(dts)
    ev = COCOeval(coco, dt, 'segm')
    ev.params.maxDets = [1, 10, 250]
    ev.evaluate(); ev.accumulate(); ev.summarize()
    prec = ev.eval['precision']
    v = prec[:, :, 0, 0, 2]
    v = v[v > -1]
    return dict(added=added_total, AP=float(v.mean()) if v.size else None,
                AP50=float(ev.stats[1]), AP75=float(ev.stats[2]),
                AP_s=float(ev.stats[3]), AP_m=float(ev.stats[4]), AP_l=float(ev.stats[5]))


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument('--stage', choices=['test', 'control'], default='test')
    a = ap_.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    enc = RSSAM2VisionEncoder(checkpoint_path='/data1/wangcheng/pretrained/sam2_hiera_base_plus.pt',
                              model_size='base_plus', freeze_backbone=True,
                              use_gradient_checkpointing=False, use_lora=False).cuda().eval()
    dec = EviDecoder().cuda().eval()
    dec.load_state_dict(torch.load(DECODER, map_location='cpu', weights_only=False)['state_dict'])
    recs = json.load(open('/data2/wangcheng/remote-lthpc/evals/base_disabled/disabled/records.json'))
    appearance = 'constant' if a.stage == 'control' else 'correct'
    params = json.load(open(OUT / 'params.json')) if (OUT / 'params.json').exists() else {}
    res = evaluate(recs, enc, dec, appearance,
                   hthr=float(params.get('hthr', 0.0)), min_views=int(params.get('min_views', 1)))
    res['stage'] = a.stage; res['params'] = params
    print(json.dumps(res, indent=2))
    json.dump(res, open(OUT / f'{a.stage}_result.json', 'w'))


if __name__ == '__main__':
    main()
