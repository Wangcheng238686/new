#!/usr/bin/env python3
"""Frozen-inference feature cache for UBR-ADDPROBE-20261026 (prereg level 2).

One deterministic pass of the frozen EXP-0025 v1best (ep71, sha256 prefix
2fcae2ff) over a split (--split train|val; test untouched). Per scene it
caches everything the additive UBR trainer consumes:

  boxes / scores / full predicted masks   packed bool; source of the
                                          neighbors channel and of the
                                          binarized masks UBR refines
  final mask probability crops            pasted post-UDPR logits z' on the
                                          512 image grid, cropped to
                                          box +/-32 px
  decoder feature crops                   32-ch SAM2 upscaled_embedding on
                                          the full-image stride-4 grid,
                                          cropped to box/4 +/-8 px
  greedy GT matches                       per instance best-IoU GT >= 0.5
                                          else -1; packed GT crop (same
                                          window as the logits crop) for
                                          matched instances

Tap evidence (traced 2026-10-26 on val scene 1, GPU3):
  z' post-UDPR        forward hook on roi_head.mask_head ->
                     MaskHeadOutput.mask_logits [N,1,128,128]
                     (rsprompter/cspm/mask_head.py L402-449; the UDPR tail
                     rewrite is L411-418)
  upscaled_embedding  forward hook on roi_head.mask_head.mask_decoder,
                     output tuple element [3] [N,32,128,128]
                     (mask_head.py L402-412 consumes the wrapper's 4-tuple;
                     __init__ L258 forces capture_upscaled_embedding=True
                     whenever the tail is on; captured inside SAM2 at
                     rsprompter/decoder/sam2_decoder.py L111-126)
  final 512 probs     mmdet _do_paste_mask(sigmoid(z'), boxes, 512, 512) --
                     the exact op FCNMaskHead._predict_by_feat_single runs
                     in roi_local mode; asserted BIT-EXACT (>= mask_thr)
                     against model.pred_instances.masks per scene.
                     upscaled_embedding lives on the FULL-IMAGE stride-4
                     grid (mask_head.py L394-400 repeat-interleaves
                     full-image high-res feats per RoI; cross-RoI content
                     correlation 0.54 outside both boxes == 0.54 full map),
                     hence box/4 cropping; z' is ROI-local (mask targets are
                     per-RoI crops), hence paste-into-512 storage.

Geometry is NOT cached: the trainer reads the native geometry npz by scene
id directly (version-asserted there).

Protocol invariants (fail-closed): ckpt sha256 prefix + epoch header 71,
lora-filtered strict load, per-scene paste parity, mask-within-window
containment.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import sys

import numpy as np
import torch
import cv2

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
UE = ROOT / 'scripts' / 'ue'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(UE))

import eval_citysample_boundary as E  # noqa: E402,F401  (DetDataPreprocessor)
from citysample_pair import decode_rle  # noqa: E402
from mmengine.config import Config  # noqa: E402
from mmdet.registry import MODELS  # noqa: E402
from mmdet.structures import DetDataSample  # noqa: E402
from mmdet.models.roi_heads.mask_heads.fcn_mask_head import (  # noqa: E402
    _do_paste_mask,
)

CKPT = Path('/data2/wangcheng/remote-lthpc/checkpoint/ue_ubr_probe_20261026/'
            'frozen_refs/exp0025_v1best_ep71.pth')
CKPT_SHA_PREFIX = '2fcae2ff'  # frozen-refs/SHA256SUMS, level-1 manifest
EPOCH_HEADER = 71
DATA = Path('/data1/wangcheng/dataset/CitySample_SatUAV_TrainMinimal_v1_20261021')
OUT_ROOT = Path('/data2/wangcheng/remote-lthpc/checkpoint/ue_ubr_probe_20261026/cache')
LOGIT_MARGIN = 32  # px around the box on the 512 grid (task spec)
FEAT_MARGIN = 8    # px around box/4 on the 128 stride-4 grid (task spec)
MAX_PER_IMG = 250
SPLIT_SIZES = {'train': 2142, 'val': 459}  # dataset build 2026-10-21


def sha256_prefix(path: Path, n: int = 8) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()[:n]


def build_model(dev):
    """Frozen reference build, exactly the E.main() recipe, plus taps."""
    assert sha256_prefix(CKPT) == CKPT_SHA_PREFIX, 'frozen ref sha mismatch'
    cfg = Config.fromfile(str(ROOT / 'configs' / 'ue_whu_a3sgm_512.py'))
    model = MODELS.build(cfg.model).to(dev).eval()
    model.test_cfg['rcnn']['max_per_img'] = MAX_PER_IMG
    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    missing, unexpected = model.load_state_dict(ck['model'], strict=False)
    assert not [k for k in missing if 'lora' not in k] and not unexpected
    assert ck.get('epoch') == EPOCH_HEADER, f'unexpected epoch {ck.get("epoch")}'
    mh = model.roi_head.mask_head
    assert mh.decoder_tail_enabled, 'frozen ref must run with UDPR tail on'
    caps = {'z': [], 'up': []}

    def hook_head(_m, _i, o):
        caps['z'].append(o.mask_logits.detach())

    def hook_dec(_m, _i, o):
        caps['up'].append(o[3].detach())

    mh.register_forward_hook(hook_head)
    mh.mask_decoder.register_forward_hook(hook_dec)
    return model, caps


def scene_predict(model, caps, img_path):
    """One predict; returns boxes/scores/masks and the tapped tensors."""
    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError('unreadable image ' + str(img_path))
    sample = DetDataSample()
    sample.set_metainfo(dict(ori_shape=(512, 512), img_shape=(512, 512),
                             scale_factor=(1.0, 1.0)))
    data = model.data_preprocessor(
        {'inputs': [torch.from_numpy(img.copy()).permute(2, 0, 1)],
         'data_samples': [sample]}, training=False)
    caps['z'].clear()
    caps['up'].clear()
    with torch.no_grad():
        o = model.predict(data['inputs'], data['data_samples'], rescale=False)
    pi = o[0].pred_instances
    masks = pi.masks.masks if hasattr(pi.masks, 'masks') else (
        pi.masks.detach().cpu().numpy() if isinstance(pi.masks, torch.Tensor)
        else np.asarray(pi.masks))
    masks = np.asarray(masks).astype(bool)
    boxes = pi.bboxes.detach().cpu().numpy().astype(np.float32)
    scores = pi.scores.detach().cpu().numpy().astype(np.float32)
    z = torch.cat(caps['z'], 0)
    up = torch.cat(caps['up'], 0)
    assert z.shape[0] == up.shape[0] == len(masks) == len(boxes)
    assert z.shape[1:] == (1, 128, 128) and up.shape[1:] == (32, 128, 128)
    return boxes, scores, masks, z, up


def paste_probs(z, boxes, masks, thr, dev):
    """Reproduce FCNMaskHead's roi_local paste; assert bit-exact parity."""
    probs, _ = _do_paste_mask(
        torch.sigmoid(z).float(), torch.from_numpy(boxes).to(dev), 512, 512,
        skip_empty=False)
    recon = (probs >= thr).cpu().numpy()
    assert (recon == masks).all(), 'paste parity broken: final-logits tap wrong'
    return probs.cpu().numpy()  # [N,512,512] float32


def window(box, margin, limit):
    """Clamped int window (y0,x0,h,w) around box on a limit x limit grid."""
    x1, y1, x2, y2 = [float(v) for v in box]
    w0 = max(0, int(np.floor(x1)) - margin)
    w1 = min(limit, int(np.ceil(x2)) + margin)
    h0 = max(0, int(np.floor(y1)) - margin)
    h1 = min(limit, int(np.ceil(y2)) + margin)
    assert 0 <= h0 < h1 <= limit and 0 <= w0 < w1 <= limit
    return h0, w0, h1 - h0, w1 - w0


def match_greedy(masks, gt_masks, dev):
    """Per-pred best-IoU GT; matched iff IoU >= 0.5 (may share a GT)."""
    n, g = len(masks), len(gt_masks)
    if n == 0 or g == 0:
        return np.full(n, -1, np.int32), np.zeros(n, np.float32)
    p = torch.from_numpy(masks.reshape(n, -1)).to(dev).half()
    q = torch.from_numpy(gt_masks.reshape(g, -1).astype(np.uint8)).to(dev).half()
    inter = torch.zeros(n, g, device=dev)
    for s in range(0, n, 64):  # 64 x 262144 fp16 slices keep temps small
        inter[s:s + 64] = p[s:s + 64] @ q.T
    union = p.sum(1, keepdim=True) + q.sum(1) - inter
    iou = (inter / union.clamp(min=1)).float().cpu().numpy()
    best = iou.argmax(1)
    best_iou = iou[np.arange(n), best]
    match = np.where(best_iou >= 0.5, best, -1).astype(np.int32)
    return match, best_iou.astype(np.float32)


def instance_buffers(masks, probs, up_np, win, fwin, match, gt_masks):
    """Ragged per-instance crops as flat buffers + int64 offset tables."""
    n = len(masks)
    feat_buf, log_buf, gt_buf = [], [], []
    feat_off, log_off, gt_off = [0], [0], [0]
    for i in range(n):
        h0, w0, h, w = win[i]
        log_buf.append(probs[i, h0:h0 + h, w0:w0 + w].astype(np.float16).ravel())
        log_off.append(log_off[-1] + log_buf[-1].size)
        fh0, fw0, fh, fw = fwin[i]
        feat_buf.append(
            up_np[i, :, fh0:fh0 + fh, fw0:fw0 + fw].astype(np.float16).ravel())
        feat_off.append(feat_off[-1] + feat_buf[-1].size)
        if match[i] >= 0:
            pk = np.packbits(gt_masks[match[i]][h0:h0 + h, w0:w0 + w])
            gt_buf.append(pk)
            gt_off.append(gt_off[-1] + pk.size)
        else:
            gt_buf.append(np.zeros(0, np.uint8))
            gt_off.append(gt_off[-1])
        rest = masks[i].copy()  # containment: pasted mask inside window
        rest[h0:h0 + h, w0:w0 + w] = False
        assert not rest.any(), f'inst {i} mask outside its window'
    def cat(xs, dt):
        return np.concatenate(xs) if xs else np.zeros(0, dt)

    return dict(
        feat_buf=cat(feat_buf, np.float16), feat_off=np.array(feat_off, np.int64),
        log_buf=cat(log_buf, np.float16), log_off=np.array(log_off, np.int64),
        gt_buf=cat(gt_buf, np.uint8), gt_off=np.array(gt_off, np.int64),
    )


def process_scene(model, caps, entry, gt_list, dev):
    boxes, scores, masks, z, up = scene_predict(
        model, caps, DATA / entry['file_name'])
    thr = float(model.test_cfg['rcnn']['mask_thr_binary'])
    probs = paste_probs(z, boxes, masks, thr, dev)
    gt_masks = (np.array([decode_rle(g['segmentation']) for g in gt_list])
                if gt_list else np.zeros((0, 512, 512), bool))
    match, iou = match_greedy(masks, gt_masks, dev)
    win = np.array([window(b, LOGIT_MARGIN, 512) for b in boxes], np.int32) \
        if len(boxes) else np.zeros((0, 4), np.int32)
    fwin = np.array([window(b / 4.0, FEAT_MARGIN, 128) for b in boxes],
                    np.int32) if len(boxes) else np.zeros((0, 4), np.int32)
    arrays = dict(
        boxes=boxes.astype(np.float32), scores=scores.astype(np.float32),
        win=win, fwin=fwin, match_gt=match, match_iou=iou,
        masks_pack=(np.packbits(masks.reshape(len(masks), -1), axis=1)
                    if len(masks) else np.zeros((0, 512 * 512 // 8), np.uint8)),
    )
    arrays.update(instance_buffers(masks, probs, up.cpu().numpy(), win, fwin,
                                   match, gt_masks))
    n_empty = int((~masks.any(axis=(1, 2))).sum())
    return arrays, n_empty


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', choices=['train', 'val'], required=True)
    ap.add_argument('--limit', type=int, default=0, help='first K scenes only')
    ap.add_argument('--out-root', default=str(OUT_ROOT))
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--overwrite', action='store_true')
    a = ap.parse_args()
    dev = torch.device(a.device)
    out_dir = Path(a.out_root) / a.split
    out_dir.mkdir(parents=True, exist_ok=True)

    ann = json.load(open(DATA / f'annotations/{a.split}.json'))
    gt_by_img = {}
    for g in ann['annotations']:
        gt_by_img.setdefault(g['image_id'], []).append(g)
    images = ann['images'][:a.limit] if a.limit else ann['images']
    model, caps = build_model(dev)

    mf = open(out_dir / 'manifest.jsonl', 'a')
    tot = dict(scenes=0, inst=0, matched=0, empty_masks=0, bytes=0)
    comp = dict(masks=0, feat=0, logits=0, gt=0)
    t0 = time.time()
    for n, e in enumerate(images):
        f_npz = out_dir / f"{e['id']:06d}.npz"
        if f_npz.exists() and not a.overwrite:
            with np.load(f_npz) as zr:
                rec = dict(id=e['id'], n_inst=int(zr['boxes'].shape[0]),
                           n_matched=int((zr['match_gt'] >= 0).sum()),
                           bytes=f_npz.stat().st_size, skip=True)
            mf.write(json.dumps(rec) + '\n')
        else:
            t1 = time.time()
            arrays, n_empty = process_scene(model, caps, e,
                                            gt_by_img.get(e['id'], []), dev)
            np.savez_compressed(f_npz, **arrays)
            rec = dict(id=e['id'], n_inst=int(arrays['boxes'].shape[0]),
                       n_matched=int((arrays['match_gt'] >= 0).sum()),
                       bytes=f_npz.stat().st_size, empty_masks=n_empty,
                       secs=round(time.time() - t1, 2))
            mf.write(json.dumps(rec) + '\n')
            for k, key in (('masks', 'masks_pack'), ('feat', 'feat_buf'),
                           ('logits', 'log_buf'), ('gt', 'gt_buf')):
                comp[k] += int(arrays[key].nbytes)
        tot['scenes'] += 1
        tot['inst'] += rec['n_inst']
        tot['matched'] += rec['n_matched']
        tot['empty_masks'] += rec.get('empty_masks', 0)
        tot['bytes'] += rec['bytes']
        mf.flush()
        if (n + 1) % 25 == 0 or n + 1 == len(images):
            est = tot['bytes'] / (n + 1) * SPLIT_SIZES[a.split]
            print(f'{n + 1}/{len(images)} scenes inst={tot["inst"]} '
                  f'matched={tot["matched"]} empty={tot["empty_masks"]} '
                  f'{tot["bytes"] / 1e9:.2f}G est_full={est / 1e9:.1f}G '
                  f'({time.time() - t0:.0f}s)', flush=True)
    mf.close()
    summary = dict(
        tag='UBR-ADDPROBE-20261026-cache', split=a.split, n_scenes=len(images),
        ckpt=str(CKPT), sha_prefix=CKPT_SHA_PREFIX, epoch=EPOCH_HEADER,
        logit_margin=LOGIT_MARGIN, feat_margin=FEAT_MARGIN,
        totals=dict(tot), raw_component_bytes=dict(comp),
        extrapolated_full_split_bytes=tot['bytes'] / len(images)
        * SPLIT_SIZES[a.split],
        budget_note='train split must stay <=~40GB; shrink margins if exceeded',
    )
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
