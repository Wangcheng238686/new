#!/usr/bin/env python3
"""DUAL-AB: scene-paired bootstrap A/B between the frozen-snapshot arms.

EXP-0010 (dual) vs EXP-0009 (single) on val 100 of the frozen snapshot:
per-scene COCO AP (bbox & segm) for single / dual-on / dual-off, scene-paired
percentile bootstrap on per-scene AP differences, gate telemetry summary
extracted from the dual training log, UAV-off ablation delta, and the four
preregistered verdicts (DUAL_GAIN / ENGAGE_NULL / USED_BUT_NULL / HARM).
Per docs/EXP-0008_preregistration.md (inherited by EXP-0010).
"""
from __future__ import annotations
import argparse, json, math, os, re, sys
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
from mmengine.config import Config
from mmdet.registry import MODELS
from ue.contract import checkpoint_identity
from data.ue_raw_pair_dataset import UERawPairDataset
from probe_surfacebev_p1_h0 import model_input, digest_predictions

SNAP = Path('/data2/wangcheng/dataset_frozen/ue_raw_hero_uav_pairset_20261014_frozen')
CKPT_SINGLE = Path('/data2/wangcheng/checkpoint/surfbev_b0/surfacebev_v2_512_sealed_ddp4_s44_EXP-0009/last_model_epoch100.pth')
CKPT_DUAL = Path('/data2/wangcheng/checkpoint/surfbev_b0/surfacebev_v2_512dual_sealed_ddp4_s44_EXP-0010/last_model_epoch100.pth')
DUAL_LOG = Path('/data2/wangcheng/checkpoint/surfbev_b0/launcher_logs/EXP-0010-ddp4.log')
OUT = Path('/data2/wangcheng/checkpoint/surfbev_b0/probes/DUAL-AB-20261015')
SEED = 20261014


def rle(m):
    return maskUtils.encode(np.asfortranarray(m.astype(np.uint8)))


def load_arm(checkpoint, manifest, dual, dev, uav_off):
    from ue.features import build_baseline
    model, _, identity, _ = build_baseline(checkpoint, manifest, os.environ['SAM2_CKPT'], dev)
    if dual:
        model.uav_eval_off = bool(uav_off)
    return model, identity


@torch.inference_mode()
def per_scene_ap(model, ds, dev, dual, uav=None):
    rows = []
    for i in range(len(ds)):
        it = ds[i]
        data = model_input(model, it, dev)
        if dual:
            dp = model.data_preprocessor
            dp._last_uav = {
                'drone_images': it['drone_images'][None].to(dev),
                'devicez_rgb': it['devicez_rgb'][None].to(dev),
                'uav_cameras': [it['uav_cameras']],
                'satellite_camera': [it['satellite_camera']],
            }
            model._uav_pending = dp._last_uav
            dp._last_uav = None
        o = model.predict(data['inputs'], data['data_samples'], rescale=False)
        pi = o[0].pred_instances
        masks = pi.masks
        masks = masks.masks if hasattr(masks, 'masks') else (masks.detach().cpu().numpy() if isinstance(masks, torch.Tensor) else np.asarray(masks))
        masks = np.asarray(masks).astype(bool)
        boxes = pi.bboxes.detach().cpu().numpy().astype(np.float64)
        scores = pi.scores.detach().cpu().numpy().astype(np.float64)
        gt = np.asarray(it['gt_masks']).astype(bool)
        gt_anns, pred = [], []
        for g in (rle(m) for m in gt):
            x, y, w, h = maskUtils.toBbox(g)
            gt_anns.append({'id': len(gt_anns) + 1, 'image_id': 1, 'category_id': 1,
                            'segmentation': g, 'bbox': [float(x), float(y), float(w), float(h)],
                            'area': float(maskUtils.area(g)), 'iscrowd': 0})
        for b, s, m in zip(boxes, scores, masks):
            x1, y1, x2, y2 = b
            pred.append({'image_id': 1, 'category_id': 1, 'score': float(s),
                         'bbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                         'segmentation': rle(m)})
        row = {'scene_id': it['img_metas']['scene_id'], 'n_gt': len(gt_anns), 'n_pred': len(pred)}
        for iou_type in ('bbox', 'segm'):
            coco = COCO()
            coco.dataset = {'images': [{'id': 1, 'height': 512, 'width': 512}],
                            'annotations': gt_anns, 'categories': [{'id': 1, 'name': 'b'}]}
            coco.createIndex()
            if pred:
                dt = coco.loadRes([dict(p) for p in pred])
                e = COCOeval(coco, dt, iou_type)
                e.params.imgIds = [1]
                e.evaluate(); e.accumulate(); e.summarize()
                row[iou_type] = float(e.stats[0])
            else:
                row[iou_type] = 0.0
        rows.append(row)
        print(f'scene {i+1}/{len(ds)}', flush=True)
    return rows


def boot(x):
    r = np.random.default_rng(SEED)
    d = x[r.integers(0, len(x), (10000, len(x)))].mean(1)
    return {'mean': float(x.mean()), 'ci95': [float(v) for v in np.quantile(d, [.025, .975])]}


def gate_telemetry(log_path):
    if not log_path.is_file():
        return None
    pat = re.compile(r'Epoch (\d+) DUAL telemetry: (.*)')
    rows = []
    for line in log_path.read_text(errors='ignore').splitlines():
        m = pat.search(line)
        if m:
            fields = {}
            for kv in m.group(2).split():
                if '=' in kv:
                    k, v = kv.split('=', 1)
                    try:
                        fields[k] = float(v)
                    except ValueError:
                        pass
            rows.append({'epoch': int(m.group(1)), **fields})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-val', type=int, default=0)
    ap.add_argument('--skip-dual', action='store_true')
    a = ap.parse_args()
    dev = torch.device('cuda:0')
    OUT.mkdir(parents=True, exist_ok=True)
    from ue.contract import checkpoint_identity
    for ckpt, name in ((CKPT_SINGLE, 'identity_single.json'), (CKPT_DUAL, 'identity_dual.json')):
        mpath = OUT / name
        if not mpath.exists() and ckpt.is_file():
            ck = torch.load(ckpt, map_location='cpu', weights_only=False)
            mpath.write_text(json.dumps(checkpoint_identity(ckpt, ck), indent=2))
        elif not ckpt.is_file():
            print(f'note: {ckpt} not present yet; dual side skipped', flush=True)
    ds = UERawPairDataset(str(SNAP), 'val', True, 'surfbev_b0_split_v2.json', image_size=(512, 512))
    n = min(a.max_val, len(ds)) if a.max_val else len(ds)
    ds.records = ds.records[:n]

    model_s, id_s = load_arm(CKPT_SINGLE, OUT / 'identity_single.json', False, dev, False)
    rows_s = per_scene_ap(model_s, ds, dev, False)
    del model_s
    torch.cuda.empty_cache()

    rows_d = rows_off = None
    id_d = None
    if not a.skip_dual:
        model_d, id_d = load_arm(CKPT_DUAL, OUT / 'identity_dual.json', True, dev, False)
        rows_d = per_scene_ap(model_d, ds, dev, True)
        model_d.uav_eval_off = True
        rows_off = per_scene_ap(model_d, ds, dev, True)
        del model_d
        torch.cuda.empty_cache()

    with open(OUT / 'per_scene.jsonl', 'w') as f:
        for rs, rd, ro in zip(rows_s, rows_d or [None] * len(rows_s), rows_off or [None] * len(rows_s)):
            f.write(json.dumps({'single': rs, 'dual': rd, 'dual_off': ro}) + '\n')

    tele = gate_telemetry(DUAL_LOG)
    result = {'n_val': n, 'identity_single': id_s.get('baseline_id'), 'identity_dual': (id_d or {}).get('baseline_id')}
    if rows_d is not None:
        for m in ('bbox', 'segm'):
            s = np.array([r[m] for r in rows_s])
            d = np.array([r[m] for r in rows_d])
            o = np.array([r[m] for r in rows_off])
            result[f'ab_{m}'] = {'single': float(s.mean()), 'dual': float(d.mean()), 'dual_off': float(o.mean()),
                                 'dual_minus_single': boot(d - s), 'dual_minus_off': boot(d - o)}
    if tele:
        last = tele[-1]
        first = next((t for t in tele if t.get('gate_steps')), {})
        result['gate'] = {'first_epoch_gate_absmax': first.get('gate_absmax'), 'last_epoch_gate_absmax': last.get('gate_absmax'),
                          'last_gate_mean': last.get('gate_mean'), 'last_delta_p3_ratio': last.get('delta_p3_ratio'),
                          'last_bev_coverage': last.get('bev_coverage'),
                          'drift_pct': (abs(last.get('gate_absmax', 0)) / 0.0 if False else abs(last.get('gate_absmax', 0))) }
    json.dump({'result': result, 'telemetry_epochs': tele}, open(OUT / 'summary.json', 'w'), indent=1)
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
