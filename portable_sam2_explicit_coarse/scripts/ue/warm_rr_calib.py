#!/usr/bin/env python3
"""WARM-RR calibration on a train subset: cache satellite-frame evidence maps
+ base predictions, sweep (thr, score, mask-refine) maximizing segm AP."""
import json, sys
from pathlib import Path
import numpy as np
import torch
import cv2
from scipy import ndimage
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path('/home/wangcheng/project/new/.worktrees/dual-stream-dev/portable_sam2_explicit_coarse')
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'inference')); sys.path.insert(0, str(ROOT / 'scripts' / 'ue'))
import ue.dual  # noqa
from ue.dual import _UAV_MEAN, _UAV_STD
from rsprompter.backbone.sam2_vision import RSSAM2VisionEncoder
from probe_evi_decoder_v2 import EviDecoder
from warm_rr import sat_evidence, to_rle

DS = Path('/data1/wangcheng/dataset/CitySample_SatUAV_TrainMinimal_v1_20261021')
CACHE = Path('/data2/wangcheng/remote-lthpc/ue_v1_local_adapter_geometry')
OUT = Path('/data2/wangcheng/remote-lthpc/warm_rr')
N = 250

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, '/data2/wangcheng/remote-lthpc/portable_sam2_explicit_coarse')
    from data.ue_minimal_dataset import MinimalSatelliteDataset
    from mmengine.config import Config
    from mmdet.registry import MODELS
    from mmdet.models.data_preprocessors import DetDataPreprocessor
    from mmengine.registry import MODELS as MM
    try: MM.register_module(module=DetDataPreprocessor)
    except Exception: pass
    from mmdet.structures import DetDataSample

    enc = RSSAM2VisionEncoder(checkpoint_path='/data1/wangcheng/pretrained/sam2_hiera_base_plus.pt',
                              model_size='base_plus', freeze_backbone=True,
                              use_gradient_checkpointing=False, use_lora=False).cuda().eval()
    dec = EviDecoder().cuda().eval()
    dec.load_state_dict(torch.load('/data2/wangcheng/checkpoint/surfbev_b0/probes/EVI-DECODER-20261021/decoder.pt',
                                   map_location='cpu', weights_only=False)['state_dict'])
    cfg = Config.fromfile(str(ROOT / 'configs/ue_whu_a3sgm_512.py'))
    model = MODELS.build(cfg.model).cuda().eval()
    model.test_cfg['rcnn']['max_per_img'] = 250
    ck = torch.load('/data2/wangcheng/remote-lthpc/base_ckpt/best_model_epoch73.pth', map_location='cpu', weights_only=False)
    model.load_state_dict(ck['model'], strict=False)

    ds = MinimalSatelliteDataset(DS, 'train')
    scenes = []
    for i in range(min(N, len(ds))):
        s = ds.samples[i]
        imgp = Path(s['img_path']) if 'img_path' in s else Path(s.get('file_name', ''))
        if not imgp.is_absolute(): imgp = DS / imgp
        pid = imgp.parent.parent.name
        if not (CACHE / f'{pid}.npz').exists(): continue
        img = cv2.imread(str(imgp))
        sample = DetDataSample()
        sample.set_metainfo(dict(ori_shape=(512, 512), img_shape=(512, 512), scale_factor=(1.0, 1.0)))
        data = model.data_preprocessor({'inputs': [torch.from_numpy(img.copy()).permute(2, 0, 1)],
                                        'data_samples': [sample]}, training=False)
        with torch.inference_mode():
            o = model.predict(data['inputs'], data['data_samples'], rescale=True)[0].pred_instances
        masks = o.masks.masks if hasattr(o.masks, 'masks') else o.masks.detach().cpu().numpy()
        pr = [to_rle(maskUtils.encode(np.asfortranarray(m.astype(np.uint8)))) for m in np.asarray(masks).astype(bool)]
        evi, height, views = sat_evidence(enc, dec, pid)
        gt_rles = []
        gt = [to_rle(maskUtils.encode(np.asfortranarray(m.astype(np.uint8))))
              for m in np.asarray(ds[i]['gt_masks'] if 'gt_masks' in ds[i] else s.get('masks', [])).astype(bool)] if False else None
        # GT from dataset annotations via decode of stored rles is complex; use records-style: annotations in samples
        gt_rles = []
        for ann in s.get('annotations', []):
            r = ann.get('segmentation')
            if r is None: continue
            if isinstance(r, dict):
                r = {'size': r['size'], 'counts': r['counts'].encode() if isinstance(r['counts'], str) else r['counts']}
                if isinstance(r['counts'], list):
                    r = maskUtils.frPyObjects(r, r['size'][0], r['size'][1])
            gt_rles.append(r)
        scenes.append(dict(pid=pid, dt_rles=pr, scores=o.scores.detach().cpu().numpy().tolist(),
                           gt_rles=gt_rles, evi=evi, height=height, views=views))
        if (i + 1) % 50 == 0: print('done', i + 1, flush=True)
    np.save(OUT / 'calib_scenes.npy', np.array([0]))  # marker
    torch.save(scenes, OUT / 'calib_scenes.pt')
    print('cached', len(scenes), 'scenes')

    def eval_with(thr, score, min_cells, refine, hthr=0.0, min_views=1):
        coco = COCO(); anns = []; dts = []
        added = 0
        for i, sc in enumerate(scenes):
            iid = i + 1
            for g in sc['gt_rles']:
                x, y, w, h = maskUtils.toBbox(g)
                anns.append({'id': len(anns)+1, 'image_id': iid, 'category_id': 1, 'segmentation': g,
                             'bbox': [float(x), float(y), float(w), float(h)], 'area': float(maskUtils.area(g)), 'iscrowd': 0})
            dt_rles = list(sc['dt_rles']); dt_scores = list(sc['scores'])
            evi = sc['evi']
            gate = np.ones_like(evi, bool)
            if hthr > 0: gate = gate & (sc['height'] >= hthr)
            if min_views > 1: gate = gate & (sc['views'] >= min_views)
            lab, n = ndimage.label((evi >= thr) & gate)
            for c in range(1, n+1):
                cells = lab == c
                if cells.sum() < min_cells: continue
                if refine:
                    e512 = cv2.resize(evi, (512, 512), interpolation=cv2.INTER_LINEAR)
                    m = (e512 >= thr + 0.1) & np.kron(cells, np.ones((4,4), np.uint8)).astype(bool)
                    if m.sum() < 16: continue
                else:
                    m = np.kron(cells, np.ones((4,4), np.uint8)).astype(bool)
                ys, xs = np.where(m)
                y1, y2, x1, x2 = ys.min(), ys.max()+1, xs.min(), xs.max()+1
                ok = True
                for r in dt_rles:
                    g = maskUtils.toBbox(r)
                    if g[0] < x2 and x1 < g[0]+g[2] and g[1] < y2 and y1 < g[1]+g[3]:
                        if (m & maskUtils.decode(r).astype(bool)).sum() > 0.3*min(m.sum(), maskUtils.area(r)):
                            ok = False; break
                if ok:
                    dt_rles.append(to_rle(maskUtils.encode(np.asfortranarray(m.astype(np.uint8)))))
                    dt_scores.append(score); added += 1
            for r, s_ in zip(dt_rles, dt_scores):
                x, y, w, h = maskUtils.toBbox(r)
                dts.append({'image_id': iid, 'category_id': 1, 'score': float(s_), 'segmentation': r,
                            'bbox': [float(x), float(y), float(w), float(h)]})
        coco.dataset = {'images': [{'id': i+1, 'height': 512, 'width': 512} for i in range(len(scenes))],
                        'annotations': anns, 'categories': [{'id': 1, 'name': 'b'}]}
        coco.createIndex()
        dt = coco.loadRes(dts)
        ev = COCOeval(coco, dt, 'segm'); ev.params.maxDets = [1, 10, 250]
        ev.evaluate(); ev.accumulate(); ev.summarize()
        v = ev.eval['precision'][:, :, 0, 0, 2]; v = v[v > -1]
        return float(v.mean()), added

    results = {}
    for thr in (0.6, 0.7, 0.8):
        for score in (0.30, 0.40):
            for hthr, mv in ((0.0, 1), (0.03, 2)):
                ap_, added = eval_with(thr, score, 3, False, hthr, mv)
                results[f'{thr}/{score}/{hthr}/{mv}'] = (ap_, added)
                print(f'thr={thr} score={score} hthr={hthr} views={mv}: AP={ap_:.4f} added={added}', flush=True)
    best = max(results.items(), key=lambda kv: kv[1][0])
    print('BEST:', best)
    json.dump({k: list(v) for k, v in results.items()}, open(OUT / 'calib_grid.json', 'w'))
    thr_s, score_s, hthr_s, mv_s = best[0].split('/')
    json.dump({'thr': float(thr_s), 'score': float(score_s), 'hthr': float(hthr_s), 'min_views': int(mv_s)},
              open(OUT / 'params.json', 'w'))

if __name__ == '__main__':
    main()
