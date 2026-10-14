#!/usr/bin/env python3
"""DUAL-R0-DECOMP: AP-deficit decomposition + three oracle ceilings on the
frozen EXP-0006 anchor (val 100, test sealed). Runs inside the reconstructed
R1a worktree (anchor loading identical to DUAL-P4). See
docs/DUAL-R0-DECOMP-20261014_preregistration.md.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'inference'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from data.ue_raw_pair_dataset import UERawPairDataset
from probe_surfacebev_p1_h0 import cameras, sha_file, model_input, digest_predictions, devicez_to_camera_z
from probe_dual_p4_harvest_anchor import build, H0, EXPECTED_B0_PATH, EXPECTED_B0_SHA256

SEED = 20261014
VERSION = 'DUAL-R0-DECOMP-v1'
S_HEADLINE = 0.7
S_SENS = (0.9, 0.5)
Z_PLANES = (0.0, 2200.0)  # cm; 0 headline, 22m sensitivity
VIEW_PIX_FRAC = 0.05
DEPTH_SLACK = 0.02
IOU_MATCH = 0.5


def rle(mask: np.ndarray):
    return maskUtils.encode(np.asfortranarray(mask.astype(np.uint8)))


def mask_area(r):
    return float(maskUtils.area(r))


def mask_iou_matrix(rs_pred, rs_gt):
    if not len(rs_pred) or not len(rs_gt):
        return np.zeros((len(rs_pred), len(rs_gt)), np.float64)
    return maskUtils.iou(rs_pred, rs_gt, [0] * len(rs_gt))


def box_iou_matrix(bx, by):
    if not len(bx) or not len(by):
        return np.zeros((len(bx), len(by)), np.float64)
    ax = bx[:, 0][:, None]; ay = bx[:, 1][:, None]
    aw = bx[:, 2][:, None] - ax; ah = bx[:, 3][:, None] - ay
    bx_ = by[:, 0][None, :]; by_ = by[:, 1][None, :]
    bw = by[:, 2][None, :] - bx_; bh = by[:, 3][None, :] - by_
    iw = np.clip(np.minimum(ax + aw, bx_ + bw) - np.maximum(ax, bx_), 0, None)
    ih = np.clip(np.minimum(ay + ah, by_ + bh) - np.maximum(ay, by_), 0, None)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / np.maximum(union, 1e-9)


def greedy_match(iou: np.ndarray, scores: np.ndarray):
    """COCO-style: preds by desc score, each takes best unmatched GT."""
    matched = {}
    for p in np.argsort(-scores):
        if not iou.shape[1]:
            break
        row = iou[p]
        best = int(np.argmax(row))
        if row[best] >= IOU_MATCH:
            matched[p] = best
            iou[:, best] = -1.0
    return matched


def coco_eval(gt_dict, pred_list, iou_type):
    coco = COCO()
    coco.dataset = gt_dict
    coco.createIndex()
    coco_dt = coco.loadRes([dict(p) for p in pred_list])
    e = COCOeval(coco, coco_dt, iou_type)
    e.evaluate(); e.accumulate(); e.summarize()
    out = {'AP': float(e.stats[0]), 'AP50': float(e.stats[1]), 'AP75': float(e.stats[2]),
           'APs': float(e.stats[3]), 'APm': float(e.stats[4]), 'APl': float(e.stats[5])}
    return out, e


@torch.inference_mode()
def queryability(item, gt_masks, dev):
    """Per GT instance, per UAV view: fraction of footprint pixels visible
    when the footprint is lifted to each z plane. Round-trip pixel check
    validates the satellite unprojection once per scene."""
    uk, uc, sk, sw = cameras(item, dev)
    depth = devicez_to_camera_z(item['devicez_rgb'].to(dev))  # 5 x 512 x 512 cam-z
    kinv = torch.inverse(sk)
    rinv = sw[:3, :3].inverse()
    center = -rinv @ sw[:3, 3]
    h, w = gt_masks.shape[1:] if gt_masks.ndim == 3 else (1024, 1024)
    # 1024-space pixel grid -> 512-space for intrinsics
    yy, xx = torch.meshgrid(torch.arange(h, device=dev, dtype=torch.float32),
                            torch.arange(w, device=dev, dtype=torch.float32), indexing='ij')
    u512 = (xx + .5) * 512.0 / w
    v512 = (yy + .5) * 512.0 / h
    ray = torch.stack((u512, v512, torch.ones_like(u512)), -1) @ kinv.T @ rinv.T  # (h,w,3) world dir
    # round-trip check on center pixel
    mid = (h // 2, w // 2)
    p0 = center + ray[mid] * 1000.0
    c0 = sw[:3, :3] @ p0 + sw[:3, 3]
    su = sk[0, 0] * (c0[0] / c0[2]) + sk[0, 2]
    sv = sk[1, 1] * (c0[1] / c0[2]) + sk[1, 2]
    rt_err = float(torch.hypot(su - (mid[1] + .5) * 512.0 / w, sv - (mid[0] + .5) * 512.0 / h))
    if rt_err > 1.0:
        raise RuntimeError(f'satellite round-trip error {rt_err:.3f}px')
    out = np.zeros((len(gt_masks), len(uk), len(Z_PLANES)), np.float32)
    for i, m in enumerate(gt_masks):
        ys, xs = np.nonzero(m)
        if not len(ys):
            continue
        pick = np.random.default_rng(np.random.SeedSequence([SEED, i])).choice(
            len(ys), min(512, len(ys)), replace=False)
        ys_t = torch.from_numpy(ys[pick].astype(np.float32)).to(dev)
        xs_t = torch.from_numpy(xs[pick].astype(np.float32)).to(dev)
        u = (xs_t + .5) * 512.0 / w
        v = (ys_t + .5) * 512.0 / h
        d = torch.stack((u, v, torch.ones_like(u)), -1) @ kinv.T @ rinv.T
        for zi, zp in enumerate(Z_PLANES):
            denom = d[:, 2]
            s = (zp - center[2]) / torch.where(denom.abs() < 1e-6, torch.full_like(denom, 1e-6), denom)
            world = center + d * s[:, None]
            for view in range(len(uk)):
                w2c = _w2c(uc[view])[:3, :]
                cam = world @ w2c[:3, :3].T + w2c[:3, 3]  # (P, 3) camera coords
                zpos = cam[:, 2]
                pu = uk[view, 0, 0] * cam[:, 0] / zpos.clamp_min(1e-6) + uk[view, 0, 2]
                pv = uk[view, 1, 1] * cam[:, 1] / zpos.clamp_min(1e-6) + uk[view, 1, 2]
                ix = (pu / 512.0 * depth.shape[-1]).long()
                iy = (pv / 512.0 * depth.shape[-2]).long()
                inb = (zpos > 0) & (ix >= 0) & (ix < depth.shape[-1]) & (iy >= 0) & (iy < depth.shape[-2])
                if not bool(inb.any()):
                    continue
                rend = depth[view][iy[inb], ix[inb]]
                vis = rend >= zpos[inb] * (1 - DEPTH_SLACK)
                out[i, view, zi] = float(vis.float().mean())
    return out, rt_err


def _w2c(c2w):
    return torch.inverse(c2w)


def run_infer(a, out):
    h = json.loads(H0.read_text())
    if h.get('verdict') != 'H0_PASS':
        raise RuntimeError('H0 not passed')
    dev = torch.device(a.device)
    ns = argparse.Namespace(checkpoint=a.checkpoint, sam2_ckpt=a.sam2_ckpt)
    model, enc, _, _ = build(ns, dev)
    ds = UERawPairDataset(a.root, 'val', True, 'surfbev_b0_split_v2.json', load_targets=True)
    scenes = out / 'scenes'
    scenes.mkdir(parents=True, exist_ok=True)
    n = min(a.max_val, len(ds)) if a.max_val else len(ds)
    (out / 'contract.json').write_text(json.dumps({
        'version': VERSION, 'seed': SEED, 'checkpoint_sha256': EXPECTED_B0_SHA256,
        'split': 'surfbev_b0_split_v2.json', 'split_part': 'val', 'n': n,
        'z_planes_cm': list(Z_PLANES), 'view_pix_frac': VIEW_PIX_FRAC,
        'depth_slack': DEPTH_SLACK, 'iou_match': IOU_MATCH}, indent=2))
    for i in range(n):
        f = scenes / f'{i:04d}.json'
        if f.exists():
            print('val', i + 1, '/', n, 'cached', flush=True)
            continue
        it = ds[i]
        data = model_input(model, it, dev)
        data2 = model_input(model, it, dev)
        o1 = model.predict(data['inputs'], data['data_samples'], rescale=False)
        d1 = digest_predictions(o1)
        o2 = model.predict(data2['inputs'], data2['data_samples'], rescale=False)
        if d1 != digest_predictions(o2):
            raise RuntimeError('passive parity')
        pi = o1[0].pred_instances
        boxes = pi.bboxes.detach().cpu().numpy().astype(np.float32)
        scores = pi.scores.detach().cpu().numpy().astype(np.float32)
        masks = pi.masks
        masks = masks.masks if hasattr(masks, 'masks') else (masks.detach().cpu().numpy() if isinstance(masks, torch.Tensor) else np.asarray(masks))
        masks = np.asarray(masks).astype(bool)
        gt = np.asarray(it['gt_masks']).astype(bool) if len(it['gt_masks']) else np.zeros((0, 1024, 1024), bool)
        vis, rt_err = queryability(it, gt, dev)
        rec = {'scene_id': it['img_metas']['scene_id'],
               'pred': [{'bbox': b.tolist(), 'score': float(s), 'rle': {'size': r['size'], 'counts': r['counts'].decode('latin1')}} for b, s, r in zip(boxes, scores, [rle(m) for m in masks])],
               'gt_rle': [{'size': r['size'], 'counts': r['counts'].decode('latin1')} for r in (rle(m) for m in gt)],
               'gt_vis': vis.tolist(), 'roundtrip_err_px': rt_err}
        tmp = f.with_suffix('.tmp'); tmp.write_text(json.dumps(rec)); os.replace(tmp, f)
        print('val', i + 1, '/', n, flush=True)


def load_scenes(out):
    scenes = sorted((out / 'scenes').glob('*.json'))
    recs = [json.loads(f.read_text()) for f in scenes]
    for r in recs:
        for p in r['pred']:
            p['rle']['counts'] = p['rle']['counts'].encode('latin1')
        for g in r['gt_rle']:
            g['counts'] = g['counts'].encode('latin1')
    return recs


def run_analyze(out):
    recs = load_scenes(out)
    images = [{'id': i, 'height': 1024, 'width': 1024} for i in range(len(recs))]
    ann_id = 0
    gt_anns, pred_all, gt_meta = [], [], []
    for i, r in enumerate(recs):
        for g in r['gt_rle']:
            ann_id += 1
            a_ = float(maskUtils.area(g))
            x, y, w_, h_ = maskUtils.toBbox(g)
            gt_anns.append({'id': ann_id, 'image_id': i, 'category_id': 1,
                            'segmentation': g, 'bbox': [float(x), float(y), float(w_), float(h_)],
                            'area': a_, 'iscrowd': 0})
        for p in r['pred']:
            pred_all.append({'image_id': i, 'category_id': 1, 'bbox': p['bbox'],
                             'score': p['score'], 'segmentation': p['rle']})
    # rebuild gt_meta straightforwardly (gt_vis index per scene)
    gt_meta = []
    for r in recs:
        v = np.asarray(r['gt_vis'])
        for j in range(len(r['gt_rle'])):
            gt_meta.append({'scene': r['scene_id'], 'vis': v[j]})
    gt_dict = {'images': images, 'annotations': gt_anns, 'categories': [{'id': 1, 'name': 'building'}]}
    areas = np.array([a_['area'] for a_ in gt_anns])
    t1, t2 = np.quantile(areas, [1 / 3, 2 / 3])

    results = {}
    base_b, _ = coco_eval(gt_dict, pred_all, 'bbox')
    base_s, _ = coco_eval(gt_dict, pred_all, 'segm')
    results['baseline'] = {'bbox': base_b, 'segm': base_s, 'area_terciles': [float(t1), float(t2)]}

    # per-modality matching at IoU .5
    def matched_pred_idx(iou, scores):
        m = greedy_match(iou.copy(), scores)
        return set(m.keys()), set(m.values())

    by_img_b = {}; by_img_s = {}
    for i, r in enumerate(recs):
        pb = np.array([p['bbox'] for p in r['pred']], np.float64).reshape(-1, 4)
        ps = np.array([p['score'] for p in r['pred']], np.float64)
        gb = np.array([a_['bbox'] for a_ in gt_anns if a_['image_id'] == i], np.float64).reshape(-1, 4)
        prl = [p['rle'] for p in r['pred']]
        grl = [g for g in r['gt_rle']]
        mb, ub = matched_pred_idx(box_iou_matrix(pb, gb), ps)
        ms, us = matched_pred_idx(mask_iou_matrix(prl, grl), ps)
        by_img_b[i] = (mb, ub)
        by_img_s[i] = (ms, us)

    # oracle A: inject perfect predictions for unmatched GT
    def oracle_miss(s):
        preds = [dict(p) for p in pred_all]
        for i, r in enumerate(recs):
            _, ub = by_img_b[i]
            _, us = by_img_s[i]
            for j, g in enumerate(r['gt_rle']):
                x, y, w_, h_ = maskUtils.toBbox(g)
                if j in us:
                    preds.append({'image_id': i, 'category_id': 1, 'score': s,
                                  'segmentation': g, 'bbox': [float(x), float(y), float(w_), float(h_)]})
        b, _ = coco_eval(gt_dict, preds, 'bbox')
        s_, _ = coco_eval(gt_dict, preds, 'segm')
        return {'bbox': b, 'segm': s_}

    results['oracle_miss'] = {str(s): oracle_miss(s) for s in (S_HEADLINE, *S_SENS)}
    # oracle B: rescore matched to 1.0
    preds = []
    for i, r in enumerate(recs):
        mb, _ = by_img_b[i]; ms, _ = by_img_s[i]
        for k, p in enumerate(r['pred']):
            q = {'image_id': i, 'category_id': 1, 'bbox': p['bbox'],
                 'score': 1.0 if k in mb else p['score'], 'segmentation': p['rle']}
            preds.append(q)
    rb, _ = coco_eval(gt_dict, preds, 'bbox')
    rs, _ = coco_eval(gt_dict, preds, 'segm')
    results['oracle_rescore'] = {'bbox': rb, 'segm': rs}
    # oracle C: matched masks -> GT masks (segm only)
    preds = []
    for i, r in enumerate(recs):
        _, us = by_img_s[i]
        ms_map = greedy_match(mask_iou_matrix([p['rle'] for p in r['pred']], r['gt_rle']),
                              np.array([p['score'] for p in r['pred']])) if r['pred'] else {}
        for k, p in enumerate(r['pred']):
            q = {'image_id': i, 'category_id': 1, 'bbox': p['bbox'],
                 'score': p['score'],
                 'segmentation': r['gt_rle'][ms_map[k]] if k in ms_map else p['rle']}
            preds.append(q)
    rc, _ = coco_eval(gt_dict, preds, 'segm')
    results['oracle_mask'] = {'segm': rc}

    # unmatched GT stats + queryability (segm modality)
    unmatched = []
    for i, r in enumerate(recs):
        _, us = by_img_s[i]
        for j in range(len(r['gt_rle'])):
            if j in us:
                unmatched.append((i, j))
    areas_un = np.array([float(maskUtils.area(recs[i]['gt_rle'][j])) for i, j in unmatched]) if unmatched else np.zeros(0)
    qflag = []
    for i, j in unmatched:
        v = np.asarray(recs[i]['gt_vis'])[j]  # (views, planes)
        q0 = int((v[:, 0] >= VIEW_PIX_FRAC).sum())
        q22 = int((v[:, 1] >= VIEW_PIX_FRAC).sum())
        qflag.append((q0 >= 2, q22 >= 2))
    n_un = len(unmatched)
    d_miss = results['oracle_miss'][str(S_HEADLINE)]
    d_bbox = d_miss['bbox']['AP'] - base_b['AP']
    d_segm = d_miss['segm']['AP'] - base_s['AP']
    d_resc = results['oracle_rescore']['bbox']['AP'] - base_b['AP']
    d_mask = results['oracle_mask']['segm']['AP'] - base_s['AP']
    qrate = float(np.mean([q[0] for q in qflag])) if n_un else 0.0
    if d_bbox >= 2.0 and qrate >= 0.5:
        decision = 'R0_ROUTE2'
    elif d_bbox < 1.0 and d_mask >= 2.0:
        decision = 'R0_ROUTE1'
    elif d_bbox < 1.0 and d_resc < 1.0 and d_mask < 1.0:
        decision = 'R0_RETHINK'
    else:
        decision = 'R0_MIXED'
    summary = {'experiment': 'DUAL-R0-DECOMP-20261014', 'status': 'complete', 'decision': decision,
               'n_val': len(recs), 'n_gt': len(gt_anns), 'n_pred': len(pred_all),
               'n_unmatched_gt': n_un,
               'unmatched_area_tercile_frac': [float((areas_un <= t1).mean()) if n_un else 0.0,
                                               float(((areas_un > t1) & (areas_un <= t2)).mean()) if n_un else 0.0,
                                               float((areas_un > t2).mean()) if n_un else 0.0],
               'queryable_rate_2view_z0': qrate,
               'queryable_rate_2view_z22': float(np.mean([q[1] for q in qflag])) if n_un else 0.0,
               'deltas': {'miss_bbox_AP': d_bbox, 'miss_segm_AP': d_segm,
                          'rescore_bbox_AP': d_resc, 'mask_segm_AP': d_mask},
               'results': results,
               'gates': {'miss_bbox_ge_2': d_bbox >= 2.0, 'qrate_ge_05': qrate >= 0.5,
                         'mask_ge_2': d_mask >= 2.0, 'rescore_lt_1': d_resc < 1.0}}
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps({'decision': decision, 'deltas': summary['deltas'],
                      'qrate': qrate, 'n_unmatched': n_un}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', choices=('infer', 'analyze'), required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--checkpoint', type=Path, default=EXPECTED_B0_PATH)
    ap.add_argument('--root', type=Path, default=ROOT / 'data/ue_raw_hero_uav_pairset_20261011')
    ap.add_argument('--sam2-ckpt', default='/data1/wangcheng/pretrained/sam2_hiera_base_plus.pt')
    ap.add_argument('--max-val', type=int, default=0)
    a = ap.parse_args()
    out = a.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if a.part == 'infer':
        run_infer(a, out)
    else:
        run_analyze(out)


if __name__ == '__main__':
    main()
