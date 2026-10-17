#!/usr/bin/env python3
"""DUAL-AB-LADDER: preregistered cross-arm analysis over the 512-snapshot ladder.

Extends probe_dual_ab_snapshot.py (kept untouched, EXP-0009/0010 lineage) to the
full five-arm ladder per docs/EXP-0008_preregistration.md v2/v2.1/v2.2/v2.3:
  single=EXP-0009, dual_p3=EXP-0010, dual_p2p3=EXP-0011,
  dual_p2p3_lora=EXP-0012, dual_p2=EXP-0013 (last-epoch checkpoints only).

Per arm: per-scene COCO AP (bbox & segm) on val 100 of the frozen snapshot;
dual arms additionally evaluated UAV-off (uav_eval_off). Summarize step adds
scene-paired percentile bootstrap (10k draws, seed 20261014), dual-vs-single
and dual-vs-dual contrasts, suffix-aware last-epoch gate telemetry, the v2.1
drift operator delta_L = mean|gate_L| / mean|proj_L(bev)| (last-epoch weights,
first val batch forward) and the four preregistered verdicts.

Usage:
  probe_dual_ab_ladder.py --arm TAG [--off] [--max-val N]   # eval one arm-run
  probe_dual_ab_ladder.py --summarize [--skip-delta]        # merge + verdicts
"""
from __future__ import annotations
import argparse, json, os, sys
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
from ue.contract import checkpoint_identity
import ue.dual  # noqa: F401 registers RSPrompterAnchorDual + UEDualPreprocessor
from data.ue_raw_pair_dataset import UERawPairDataset
from probe_surfacebev_p1_h0 import model_input

SNAP = Path('/data2/wangcheng/dataset_frozen/ue_raw_hero_uav_pairset_20261014_frozen')
CKPT_DIR = Path('/data2/wangcheng/checkpoint/surfbev_b0')
LOG_DIR = CKPT_DIR / 'launcher_logs'
ARMS = {
    'single': dict(dual=False, ckpt=CKPT_DIR / 'surfacebev_v2_512_sealed_ddp4_s44_EXP-0009/last_model_epoch100.pth',
                   log=LOG_DIR / 'EXP-0009-ddp4.log'),
    'dual_p3': dict(dual=True, ckpt=CKPT_DIR / 'surfacebev_v2_512dual_sealed_ddp4_s44_EXP-0010/last_model_epoch100.pth',
                    log=LOG_DIR / 'EXP-0010-ddp4.log'),
    'dual_p2p3': dict(dual=True, ckpt=CKPT_DIR / 'surfacebev_v2_512dualv2_sealed_ddp4_s44_EXP-0011/last_model_epoch100.pth',
                      log=LOG_DIR / 'EXP-0011-ddp4.log'),
    'dual_p2p3_lora': dict(dual=True, ckpt=CKPT_DIR / 'surfacebev_v2_512duallora_sealed_ddp4_s44_EXP-0012/last_model_epoch100.pth',
                           log=LOG_DIR / 'EXP-0012-ddp4.log'),
    'dual_p2': dict(dual=True, ckpt=CKPT_DIR / 'surfacebev_v2_512dualv2l0_sealed_ddp4_s44_EXP-0013/last_model_epoch100.pth',
                    log=LOG_DIR / 'EXP-0013-ddp4.log'),
    # EXP-0014 (parallel lineage, docs/EXP-0014_conclusion.md): per-channel
    # rescaled BEV, single variable vs dual_p3; rows_* produced by that run
    'dual_norm': dict(dual=True, ckpt=CKPT_DIR / 'surfacebev_v2_512dualnorm_sealed_ddp4_s44_EXP-0014/last_model_epoch100.pth',
                      log=LOG_DIR / 'EXP-0014-ddp4.log'),
    # EXP-0015/0016 (docs/EXP-0015-0016_preregistration.md): fusion wd exemption
    # + L0-LoRA / embed-only injection site
    'dual_l0lora': dict(dual=True, ckpt=CKPT_DIR / 'surfacebev_v2_512dualv2l0lora_sealed_ddp4_s44_EXP-0015/last_model_epoch100.pth',
                        log=LOG_DIR / 'EXP-0015-ddp4.log'),
    'dual_embed': dict(dual=True, ckpt=CKPT_DIR / 'surfacebev_v2_512dualembed_sealed_ddp4_s44_EXP-0016/last_model_epoch100.pth',
                       log=LOG_DIR / 'EXP-0016-ddp4.log'),
}
OUT = CKPT_DIR / 'probes' / 'DUAL-AB-LADDER-20261016'
SEED = 20261014
LEVEL_HW = {0: (128, 128), 1: (64, 64)}  # neck level -> BEV raster size at 512 input


def rle(m):
    return maskUtils.encode(np.asfortranarray(m.astype(np.uint8)))


def manifest_for(tag):
    mpath = OUT / f'identity_{tag}.json'
    if not mpath.exists():
        ck = torch.load(ARMS[tag]['ckpt'], map_location='cpu', weights_only=False)
        mpath.write_text(json.dumps(checkpoint_identity(ARMS[tag]['ckpt'], ck), indent=2))
    return mpath


def load_arm(tag, dev, uav_off):
    from ue.features import build_baseline
    model, _, identity, _ = build_baseline(ARMS[tag]['ckpt'], manifest_for(tag), os.environ['SAM2_CKPT'], dev)
    if ARMS[tag]['dual']:
        model.uav_eval_off = bool(uav_off)
    return model, identity


def uav_payload(it, dev):
    return {'drone_images': it['drone_images'][None].to(dev),
            'devicez_rgb': it['devicez_rgb'][None].to(dev),
            'uav_cameras': [it['uav_cameras']],
            'satellite_camera': [it['satellite_camera']]}


@torch.inference_mode()
def per_scene_ap(model, ds, dev, dual):
    rows = []
    for i in range(len(ds)):
        it = ds[i]
        data = model_input(model, it, dev)
        if dual:
            model._uav_pending = uav_payload(it, dev)
        o = model.predict(data['inputs'], data['data_samples'], rescale=False)
        pi = o[0].pred_instances
        masks = pi.masks
        masks = masks.masks if hasattr(masks, 'masks') else (masks.detach().cpu().numpy() if isinstance(masks, torch.Tensor) else np.asarray(masks))
        masks = np.asarray(masks).astype(bool)
        boxes = pi.bboxes.detach().cpu().numpy().astype(np.float64)
        scores = pi.scores.detach().cpu().numpy().astype(float)
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
                # per-size AP fields (stats[3]/[4]/[5] = s/m/l) for the
                # preregistered mAP_s protection gate; extra keys are
                # backward compatible with existing consumers
                for si, suffix in ((3, '_s'), (4, '_m'), (5, '_l')):
                    row[iou_type + suffix] = float(e.stats[si])
            else:
                row[iou_type] = 0.0
        rows.append(row)
        print(f'[{os.environ.get("ARM_TAG", "?")}{" off" if os.environ.get("ARM_OFF") else ""}] scene {i+1}/{len(ds)}', flush=True)
    return rows


def boot(x):
    r = np.random.default_rng(SEED)
    d = x[r.integers(0, len(x), (10000, len(x)))].mean(1)
    return {'mean': float(x.mean()), 'ci95': [float(v) for v in np.quantile(d, [.025, .975])]}


def gate_telemetry(log_path):
    import re
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


@torch.inference_mode()
def delta_operator(tag, dev):
    """v2.1 drift operator per level from last-epoch weights + first val batch.

    delta_L = mean(|gate_L|) / mean(|proj_L(bev)|); returns per-level dict.
    Registered caveat: proj(bev) mixes cm-scale stat channels, so delta_L is
    scale-conservative; interpretation leans on delta_p3_ratio + UAV-off.
    """
    model, _ = load_arm(tag, dev, uav_off=False)
    ds = UERawPairDataset(str(SNAP), 'val', True, 'surfbev_b0_split_v2.json', image_size=(512, 512))
    it = ds[0]
    uav = uav_payload(it, dev)
    single = len(model.target_levels) == 1
    def site(name, proj_t, gate_t, hw):
        bev, cov = model._uav_bev(uav, hw)
        proj_out = proj_t(bev.unsqueeze(0).float())
        delta = gate_t.view(1, -1, 1, 1) * proj_out
        return {name: {'gate_mean_abs': float(gate_t.abs().mean()),
                       'proj_bev_mean_abs': float(proj_out.abs().mean()),
                       'delta_L': float(gate_t.abs().mean() / proj_out.abs().mean().clamp_min(1e-12)),
                       # v2.3 effective opening (dimensionless; legacy delta_L is
                       # scale-contaminated on unscaled-BEV arms)
                       'open_eff': float(delta.abs().sum() / proj_out.abs().sum().clamp_min(1e-12)),
                       'bev_coverage': cov}}
    out = {}
    for t in model.target_levels:
        out.update(site(f'L{t}', model.proj if single else model.proj[str(t)],
                      model.gate if single else model.gate[str(t)], LEVEL_HW[t]))
    if getattr(model, 'embed_fusion', False):
        out.update(site('E', model.proj_e, model.gate_e, (32, 32)))
    del model
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=sorted(ARMS))
    ap.add_argument('--off', action='store_true')
    ap.add_argument('--max-val', type=int, default=0)
    ap.add_argument('--summarize', action='store_true')
    ap.add_argument('--skip-delta', action='store_true')
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device)

    if a.arm:
        tag = a.arm + ('_off' if a.off else '')
        os.environ['ARM_TAG'], os.environ['ARM_OFF'] = a.arm, ('1' if a.off else '')
        rows_path = OUT / f'rows_{tag}.jsonl'
        if rows_path.exists():
            print(f'{rows_path} exists; skip', flush=True)
            return
        model, identity = load_arm(a.arm, dev, uav_off=a.off)
        ds = UERawPairDataset(str(SNAP), 'val', True, 'surfbev_b0_split_v2.json', image_size=(512, 512))
        if a.max_val:
            ds.records = ds.records[:a.max_val]
        rows = per_scene_ap(model, ds, dev, ARMS[a.arm]['dual'])
        tmp = rows_path.with_suffix('.tmp')
        tmp.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        os.replace(tmp, rows_path)
        print(f'wrote {rows_path} ({identity.get("baseline_id")})', flush=True)
        return

    assert a.summarize
    # ---- merge rows ----
    arm_rows, arm_ids = {}, {}
    for tag in ARMS:
        for suffix in (['', '_off'] if ARMS[tag]['dual'] else ['']):
            p = OUT / f'rows_{tag}{suffix}.jsonl'
            if not p.is_file():
                print(f'note: {p} missing; contrasts involving it are skipped', flush=True)
                continue
            rows = [json.loads(l) for l in p.read_text().splitlines()]
            arm_rows[tag + suffix] = rows
            arm_ids[tag + suffix] = [r['scene_id'] for r in rows]
    ref_ids = arm_ids.get('single')
    for k, ids in arm_ids.items():
        assert ids == ref_ids, f'scene alignment mismatch for {k}'

    def arr(key, metric):
        return np.array([r[metric] for r in arm_rows[key]])

    result = {'n_val': len(ref_ids), 'seed': SEED, 'draws': 10000,
              'identities': {t: json.loads((OUT / f'identity_{t}.json').read_text()).get('baseline_id')
                             for t in ARMS if (OUT / f'identity_{t}.json').is_file()}}
    for key in sorted(arm_rows):
        result.setdefault('means', {})[key] = {m: float(arr(key, m).mean()) for m in ('bbox', 'segm')}
    dual_tags = [t for t in ('dual_p3', 'dual_p2p3', 'dual_p2p3_lora', 'dual_p2', 'dual_norm', 'dual_l0lora', 'dual_embed') if t in arm_rows]
    for t in dual_tags:
        for m in ('bbox', 'segm'):
            s, d, o = arr('single', m), arr(t, m), arr(t + '_off', m)
            result.setdefault('vs_single', {})[f'{t}_{m}'] = {'dual_minus_single': boot(d - s),
                                                              'dual_minus_off': boot(d - o)}
    for x, y in (('dual_p2', 'dual_p2p3'), ('dual_p2p3', 'dual_p3'), ('dual_p2p3_lora', 'dual_p2p3'), ('dual_norm', 'dual_p3'), ('dual_l0lora', 'dual_p2'), ('dual_embed', 'dual_p3')):
        if x in arm_rows and y in arm_rows:
            for m in ('bbox', 'segm'):
                result.setdefault('dual_vs_dual', {})[f'{x}_minus_{y}_{m}'] = boot(arr(x, m) - arr(y, m))

    # ---- telemetry (suffix-aware) ----
    result['gate_last_epoch'] = {}
    for t in dual_tags:
        tele = gate_telemetry(ARMS[t]['log'])
        if tele:
            last = tele[-1]
            levels = {}
            for k, v in last.items():
                if k.endswith(('_L0', '_L1')) and k != 'gate_steps':
                    base, lv = k.rsplit('_L', 1)
                    levels.setdefault('L' + lv, {})[base] = v
                elif not k.endswith(('_L0', '_L1')) and k != 'epoch':
                    levels.setdefault('L?', {})[k] = v
            result['gate_last_epoch'][t] = {'epoch': last['epoch'], 'gate_steps': last.get('gate_steps'), 'levels': levels}
            result['gate_last_epoch'][t]['first_gate_absmax'] = next(
                (r.get('gate_absmax', r.get('gate_absmax_L0')) for r in tele
                 if r.get('gate_steps')), None)

    # ---- v2.1 drift operator + verdicts ----
    if not a.skip_delta:
        result['delta_operator'] = {t: delta_operator(t, dev) for t in dual_tags}
    verdicts = {}
    for t in dual_tags:
        v = {}
        engaged_any = False
        if 'delta_operator' in result and t in result.get('delta_operator', {}):
            engaged_any = any(lv.get('open_eff', 0) >= 0.20 or lv['delta_L'] >= 0.20
                              for lv in result['delta_operator'][t].values())
        for m in ('bbox', 'segm'):
            c = result['vs_single'][f'{t}_{m}']['dual_minus_single']
            v[f'{m}_mean'] = c['mean']
            v[f'{m}_ci95'] = c['ci95']
        gain = all(v[f'{m}_ci95'][0] > 0 and v[f'{m}_mean'] >= 0.005 for m in ('bbox', 'segm'))
        harm = any(v[f'{m}_ci95'][1] < 0 for m in ('bbox', 'segm'))
        off_sig = any(result['vs_single'][f'{t}_{m}']['dual_minus_off']['ci95'][0] > 0 for m in ('bbox', 'segm'))
        v['engaged_any_level'] = engaged_any
        v['uav_off_delta_sig'] = off_sig
        v['verdict'] = ('DUAL_GAIN' if gain else 'DUAL_HARM' if harm else
                        'DUAL_USED_BUT_NULL' if (engaged_any and off_sig) else 'DUAL_ENGAGE_NULL')
        verdicts[t] = v
    result['verdicts'] = verdicts
    tmp = OUT / 'summary.json.tmp'
    tmp.write_text(json.dumps(result, indent=1))
    os.replace(tmp, OUT / 'summary.json')
    print(json.dumps({k: result[k] for k in ('means', 'verdicts') if k in result}, indent=1))


if __name__ == '__main__':
    main()
