#!/usr/bin/env python3
"""Gate-0 zero-training snap probe (SNAP-20261026, prereg 2026-10-26).

Question: do frozen-mask boundaries sit near height-gradient evidence in the
UAV native geometry map? For every instance of the frozen EXP-0025 v1best
model on the VAL split (459 scenes; test untouched), outer-boundary pixels
march along the outward normal within +/-5 px and greedily snap to the
position of maximal |grad(height)| where support==1.

Conditions (per prereg):
  nosnap    original masks (baseline)
  real      snap using the real geometry map
  sham      same procedure, but the snap target is drawn uniformly among the
            support==1 candidates along the normal (seed 20261021+scene)
  constant  height replaced by the train-only mean constant under support
            (|grad|==0 everywhere -> provably identical to nosnap; computed
            explicitly for audit)

Verdict inputs: boundary_0.01 AP_l (7px) global + per-scene paired bootstrap
(500 draws, seed 20261021) on real-nosnap, real-sham, real-constant. Gate:
real-nosnap >= +1.0pp, real-sham and real-constant CI lower bounds > 0 /
>= +0.5pp respectively per prereg wording. Reuses eval_citysample_boundary
(model build, mask_to_boundary, COCO machinery) unchanged.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy import ndimage

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
UE = ROOT / 'scripts' / 'ue'
sys.path.insert(0, str(UE))

import eval_citysample_boundary as E  # noqa: E402
from mmengine.config import Config  # noqa: E402
from mmdet.registry import MODELS  # noqa: E402
from mmdet.structures import DetDataSample  # noqa: E402
from pycocotools.coco import COCO  # noqa: E402
from pycocotools.cocoeval import COCOeval  # noqa: E402
from citysample_pair import decode_rle  # noqa: E402

CKPT = Path('/data2/wangcheng/checkpoint/citysample_v1/'
            'citysample_v1_512single_defaug_ddp4_s44_EXP-0025/best_model.pth')
DATA = Path('/data1/wangcheng/dataset/CitySample_SatUAV_TrainMinimal_v1_20261021')
CACHE = Path('/data2/wangcheng/remote-lthpc/cache/ue_v1_native_geometry')
CONSTANTS = Path('/data2/wangcheng/remote-lthpc/checkpoint/'
                 'ue_geometry_mechanism_20261022/common_constants.json')
OUT = Path('/data2/wangcheng/remote-lthpc/checkpoint/ue_ubr_probe_20261026/snap')
TAG = 'SNAP-20261026'
RATIO, DIL = 0.01, 7
SEARCH_PX, STEPS = 5, 21
SEED = 20261021


def height_field(image_id: int, mode: str):
    """Real / constant height channel + support for one scene.

    Constant mode fills the train-only mean height EVERYWHERE (not just
    under support): the zero-training snap consumes gradients, and any
    support-masked constant would develop spurious gradients at hole edges
    after smoothing (v1 bug — constant arm moved). A everywhere-constant
    field has provably zero gradient, hence never snaps and is bit-identical
    to nosnap. The support-matched fill discipline still governs the
    LEARNED UBR port (ubr.build_constant_geometry), not this probe.
    """
    with np.load(CACHE / f'{image_id:06d}.npz') as z:
        h = z['fields'][0].astype(np.float32)
        support = z['support'] > 0
    if mode == 'constant':
        h = np.full_like(h, float(json.loads(CONSTANTS.read_text())['mean_fields'][0]))
    return h, support


def snap_mask(m: np.ndarray, h: np.ndarray, support: np.ndarray,
              rng) -> np.ndarray:
    """Greedy per-pixel normal snap of the outer boundary; interior kept.

    A boundary pixel snaps only to a candidate with strictly positive
    |grad(height)| (real mode) or to a uniformly drawn support==1 candidate
    (sham mode, rng not None); with no admissible candidate it stays put.
    Constant height has zero gradient everywhere, hence never snaps and is
    bit-identical to nosnap. Vectorized over band pixels.
    """
    m = m.astype(bool)  # guard: ~ on uint8 is bitwise (254/255), not logical
    inner = cv2.erode(m.astype(np.uint8), E.KERNEL, iterations=1) > 0
    band = m & ~inner
    if not band.any():
        return m.astype(np.uint8)
    gy, gx = np.gradient(ndimage.gaussian_filter(h, 1.0))
    gmag = np.hypot(gx, gy)
    sdt = ndimage.distance_transform_edt(~m) - ndimage.distance_transform_edt(m)
    ny, nx = np.gradient(sdt)
    ys, xs = np.nonzero(band)
    norm = np.hypot(nx[ys, xs], ny[ys, xs])
    norm = np.where(norm > 1e-6, norm, 1.0)
    ux, uy = nx[ys, xs] / norm, ny[ys, xs] / norm
    ts = np.linspace(-SEARCH_PX, SEARCH_PX, STEPS)
    qy = np.rint(ys[:, None] + ts[None, :] * uy[:, None]).astype(np.int64)
    qx = np.rint(xs[:, None] + ts[None, :] * ux[:, None]).astype(np.int64)
    qyc, qxc = np.clip(qy, 0, 511), np.clip(qx, 0, 511)
    ok = ((qy >= 0) & (qy < 512) & (qx >= 0) & (qx < 512) & support[qyc, qxc])
    rows = np.arange(len(ys))
    if rng is None:
        val = np.where(ok, gmag[qyc, qxc], -1.0)
        k = val.argmax(axis=1)
        fires = ok[rows, k] & (val[rows, k] > 0.0)
    else:
        g = rng.random(ok.shape)
        k = np.where(ok, g, -1.0).argmax(axis=1)
        fires = ok.any(axis=1)
    snapped = np.zeros_like(m, dtype=bool)
    stay = rows[~fires]
    snapped[ys[stay], xs[stay]] = True
    hit = rows[fires]
    snapped[qyc[hit, k[hit]], qxc[hit, k[hit]]] = True
    # Reconstruction: the original 1px frame is vacated and replaced by the
    # snapped positions; 3x3 closing seals sub-2px gaps (corner leaks defeat
    # plain fill_holes), then fill_holes repairs the interior.
    sealed = ndimage.binary_closing(inner | snapped, structure=np.ones((3, 3), bool))
    return ndimage.binary_fill_holes(sealed).astype(np.uint8)


def scene_ap(gt_bnd, dets_bnd) -> float:
    """Per-scene boundary_0.01 AP with the reference evaluator's machinery."""
    coco = COCO()
    coco.dataset = {'images': [{'id': 1, 'width': 512, 'height': 512}],
                    'annotations': gt_bnd, 'categories': [{'id': 1, 'name': 'b'}]}
    coco.createIndex()
    if not dets_bnd:
        return 0.0
    ev = COCOeval(coco, coco.loadRes([dict(d) for d in dets_bnd]), 'segm')
    ev.params.imgIds = [1]
    ev.params.maxDets = [1, 10, 250]
    ev.evaluate()
    ev.accumulate()
    prec = ev.eval['precision'][:, :, 0, 0, 2]
    v = prec[prec > -1]
    return float(v.mean()) if v.size else 0.0


def paired_bootstrap(a: np.ndarray, b: np.ndarray, draws: int = 500) -> dict:
    rng = np.random.default_rng(SEED)
    n = len(a)
    stats = []
    for _ in range(draws):
        idx = rng.integers(0, n, n)
        stats.append(float(a[idx].mean() - b[idx].mean()))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return {'mean': float(a.mean() - b.mean()), 'ci_low': float(lo), 'ci_high': float(hi)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device('cuda:0')
    cfg = Config.fromfile(str(ROOT / 'configs' / 'ue_whu_a3sgm_512.py'))
    model = MODELS.build(cfg.model).to(dev).eval()
    model.test_cfg['rcnn']['max_per_img'] = 250
    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    missing, unexpected = model.load_state_dict(ck['model'], strict=False)
    assert not [k for k in missing if 'lora' not in k] and not unexpected
    assert ck.get('epoch') == 71, f'unexpected epoch {ck.get("epoch")}'

    ann = json.load(open(DATA / 'annotations/val.json'))
    dets = {c: [] for c in ('nosnap', 'real', 'sham', 'constant')}
    scene_aps = {c: [] for c in dets}
    for n, e in enumerate(ann['images']):
        img = cv2.imread(str(DATA / e['file_name']))
        sample = DetDataSample()
        sample.set_metainfo(dict(ori_shape=(512, 512), img_shape=(512, 512),
                                 scale_factor=(1.0, 1.0)))
        data = model.data_preprocessor(
            {'inputs': [torch.from_numpy(img.copy()).permute(2, 0, 1)],
             'data_samples': [sample]}, training=False)
        with torch.no_grad():
            o = model.predict(data['inputs'], data['data_samples'], rescale=False)
        pi = o[0].pred_instances
        masks = pi.masks.masks if hasattr(pi.masks, 'masks') else (
            pi.masks.detach().cpu().numpy() if isinstance(pi.masks, torch.Tensor)
            else np.asarray(pi.masks))
        boxes = pi.bboxes.detach().cpu().numpy().astype(np.float64)
        scores = pi.scores.detach().cpu().numpy().astype(float)
        masks = np.asarray(masks).astype(bool)

        h_real, support = height_field(e['id'], 'real')
        h_const, _ = height_field(e['id'], 'constant')
        rng = np.random.default_rng(SEED + e['id'])
        per_cond_masks = {
            'nosnap': masks,
            'real': [snap_mask(m, h_real, support, None) for m in masks],
            'sham': [snap_mask(m, h_real, support, rng) for m in masks],
            'constant': [snap_mask(m, h_const, support, None) for m in masks],
        }
        gt_bnd = [{'id': a_['id'], 'image_id': 1, 'category_id': 1,
                   'area': a_['area'], 'iscrowd': 0,
                   'segmentation': E.rle(E.mask_to_boundary(decode_rle(a_['segmentation']), DIL))}
                  for a_ in ann['annotations'] if a_['image_id'] == e['id']]
        for cond, cms in per_cond_masks.items():
            det_bnd = []
            for b, s, m in zip(boxes, scores, cms):
                x1, y1, x2, y2 = b
                rle_b = E.rle(E.mask_to_boundary(m.astype(np.uint8), DIL))
                dets[cond].append({'image_id': e['id'], 'category_id': 1,
                                   'score': float(s),
                                   'bbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                                   'segmentation': rle_b})
                det_bnd.append({'image_id': 1, 'category_id': 1, 'score': float(s),
                                'bbox': [0, 0, 1, 1],
                                'segmentation': E.rle(E.mask_to_boundary(m.astype(np.uint8), DIL))})
            scene_aps[cond].append(scene_ap(gt_bnd, det_bnd))
        if (n + 1) % 25 == 0:
            print(f'{n+1}/{len(ann["images"])} scenes', flush=True)

    imgs = [{'id': e_['id'], 'width': 512, 'height': 512} for e_ in ann['images']]
    cats = [{'id': 1, 'name': 'building'}]
    out = {'tag': TAG, 'ckpt': str(CKPT), 'split': 'val', 'ratio': RATIO}

    def global_ap(cond):
        gt_bnd_all = {}
        for a_ in ann['annotations']:
            m = decode_rle(a_['segmentation'])
            gt_bnd_all.setdefault(a_['image_id'], []).append(
                {'id': a_['id'], 'image_id': a_['image_id'], 'category_id': 1,
                 'area': a_['area'], 'iscrowd': 0,
                 'segmentation': E.rle(E.mask_to_boundary(m, DIL))})
        coco = COCO()
        coco.dataset = {'images': imgs,
                        'annotations': [x for v in gt_bnd_all.values() for x in v],
                        'categories': cats}
        coco.createIndex()
        ev = COCOeval(coco, coco.loadRes([dict(d) for d in dets[cond]]), 'segm')
        ev.params.maxDets = [1, 10, 250]
        ev.evaluate()
        ev.accumulate()
        return E.coco_stats(ev)

    for cond in dets:
        out[f'boundary_{RATIO}_{cond}'] = global_ap(cond)
        out[f'scene_ap_{cond}'] = [float(v) for v in scene_aps[cond]]
    out['bootstrap'] = {
        'real_vs_nosnap': paired_bootstrap(np.array(scene_aps['real']),
                                            np.array(scene_aps['nosnap'])),
        'real_vs_sham': paired_bootstrap(np.array(scene_aps['real']),
                                         np.array(scene_aps['sham'])),
        'real_vs_constant': paired_bootstrap(np.array(scene_aps['real']),
                                              np.array(scene_aps['constant'])),
    }
    (OUT / 'snap_result.json').write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if not k.startswith('scene_ap')}, indent=2))


if __name__ == '__main__':
    main()
