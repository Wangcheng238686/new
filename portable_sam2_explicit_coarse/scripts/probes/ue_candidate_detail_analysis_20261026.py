"""Fixed-E10 paired effects and explicit dense/small-object diagnostics."""
import argparse
import contextlib
import io
import json
from pathlib import Path

import numpy as np
from pycocotools import mask as mu
from pycocotools.cocoeval import COCOeval

from ue_candidate_detail_run_20261026 import ROOT, PROTOCOL, hashes, base
from ue_joint_encoding_analysis import evaluator, precision_ap
from ue_joint_encoding_fast_analysis import fast_sample_ap, audits


def metric_mean(value):
    selected = value[value >= 0]
    return float(selected.mean()) if selected.size else None


def difficult(rows):
    """Score-greedy single-category matching at bbox and mask IoU .5.

    Full set counts are not AP's confidence-integrated decomposition. Background
    is a max-IoU<.1 proxy, not human-certified semantic background.
    """
    areas = [np.asarray(mu.area(r['gt']['rles'])) for r in rows]
    reports = {}
    for kind in ('bbox', 'mask'):
        totals = dict(gt=0, small_gt=0, TP=0, small_TP=0, FN=0, small_FN=0,
                      FP=0, duplicate_proxy=0, localization_proxy=0, background_proxy=0)
        detections = []
        image_counts = []
        for row,area in zip(rows, areas):
            gt, dt = row['gt'], row['dt']
            n, m = len(gt['bboxes']), len(dt['bboxes'])
            small = area < 1024
            if kind == 'mask':
                overlap = mu.iou(dt['rles'], gt['rles'], [0]*n) if m and n else np.zeros((m, n))
            else:
                g = np.asarray(gt['bboxes']).reshape(-1, 4)
                d = np.asarray(dt['bboxes']).reshape(-1, 4)
                wh = np.maximum(0, np.minimum(d[:, None, 2:], g[None, :, 2:])-
                                np.maximum(d[:, None, :2], g[None, :, :2]))
                inter = wh.prod(-1)
                overlap = inter/np.maximum(1e-12, (d[:, 2:]-d[:, :2]).prod(-1)[:, None]+
                    (g[:, 2:]-g[:, :2]).prod(-1)[None]-inter)
            matched = np.zeros(n, bool)
            tp = 0
            fp = dict(duplicate_proxy=0, localization_proxy=0, background_proxy=0)
            for i in np.argsort(-np.asarray(dt['scores']), kind='stable'):
                candidate = np.where(~matched, overlap[i], -1)
                j = int(candidate.argmax()) if n else -1
                success = j >= 0 and candidate[j] >= .5
                small_success = False
                if success:
                    matched[j] = True
                    tp += 1
                    small_success = bool(small[j])
                else:
                    maximum = float(overlap[i].max()) if n else 0.
                    label = 'duplicate_proxy' if maximum >= .5 else (
                        'localization_proxy' if maximum >= .1 else 'background_proxy')
                    fp[label] += 1
                detections.append((float(dt['scores'][i]), int(success), int(small_success)))
            small_tp = int((matched & small).sum())
            counts = dict(gt=n, small_gt=int(small.sum()), TP=tp, small_TP=small_tp,
                FN=n-tp, small_FN=int(small.sum())-small_tp, FP=m-tp, **fp)
            for key,value in counts.items():
                totals[key] += value
            image_counts.append(dict(image_id=row['image_id'], **counts))
        ordered = np.asarray(sorted(detections, key=lambda x:-x[0]))
        fp_curve = np.cumsum(1-ordered[:, 1])/len(rows)
        recall = np.cumsum(ordered[:, 2])/max(1, totals['small_gt'])
        froc = {}
        for budget in [.1, .5, 1., 2., 5., 10.]:
            accepted = np.flatnonzero(fp_curve <= budget)
            froc[str(budget)] = float(recall[accepted[-1]]) if len(accepted) else 0.
        reports[kind] = dict(totals=totals, image_counts=image_counts,
            small_recall_at_FP_per_image=froc,
            definition='all-GT greedy matching IoU>=.5; small true-mask area<1024; all scores; shared FP budget includes all sizes')
    return reports


def size_and_density(rows):
    canonical = evaluator(rows)
    all_ap = precision_ap(canonical)
    for collection in (canonical.cocoGt, canonical.cocoDt):
        for ann in collection.anns.values():
            ann['area'] = float(mu.area(ann['segmentation']))
    corrected = COCOeval(canonical.cocoGt, canonical.cocoDt, 'segm')
    corrected.params.maxDets = [1, 10, 250]
    with contextlib.redirect_stdout(io.StringIO()):
        corrected.evaluate()
        corrected.accumulate()
    assert abs(precision_ap(corrected)-all_ap) < 1e-12
    sizes = {}
    for index,name in enumerate(('all', 'small', 'medium', 'large')):
        sizes[name] = dict(MaskAP=metric_mean(corrected.eval['precision'][:, :, :, index, -1]),
            MaskAR250=metric_mean(corrected.eval['recall'][:, :, index, -1]))
    density = {}
    for name,low,high in [('le50', 0, 50), ('51to100', 51, 100), ('gt100', 101, 100000)]:
        subset = [r for r in rows if low <= len(r['gt']['bboxes']) <= high]
        density[name] = dict(images=len(subset), gt=sum(len(r['gt']['bboxes']) for r in subset),
            MaskAP=precision_ap(evaluator(subset)) if subset else None)
    return dict(mask_area_groups=sizes, density_groups=density,
        area_definition='GT and DT area from RLE masks, COCO thresholds32square/96square; all-area AP unchanged')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--split', choices=['val', 'test'], required=True)
    args = parser.parse_args()
    manifest = json.loads((args.root/'manifest.json').read_text())
    assert manifest['source_hashes'] == hashes()
    data, evidence = {}, {}
    for content in ['satellite', 'correct', 'constant', 'plane']:
        key = 'base_'+content
        folder = args.root/'reference'/args.split/content
        identity = json.loads((folder/'complete.json').read_text())
        assert identity['images'] == 459 and identity['split'] == args.split
        assert identity['source_hashes'] == base.sources()
        data[key] = json.loads((folder/'records.json').read_text())
        evidence[str(folder/'records.json')] = base.core.C.file_sha(folder/'records.json')
    for variant,content in PROTOCOL['training_jobs']:
        folder = args.root/'evaluation'/args.split/variant/content/'matched'
        identity = json.loads((folder/'complete.json').read_text())
        assert identity['source_hashes'] == hashes() and identity['images'] == 459
        done = json.loads((args.root/variant/content/'training_complete.json').read_text())
        assert done['updates'] == 2670 and done['epoch'] == 10
        assert identity['checkpoint_sha256'] == done['checkpoint_sha256']
        assert base.core.C.file_sha(args.root/variant/content/'epoch10.pth') == done['checkpoint_sha256']
        data[variant+'_'+content] = json.loads((folder/'records.json').read_text())
        evidence[str(folder/'records.json')] = base.core.C.file_sha(folder/'records.json')
    for mode in ['constant', 'donor']:
        folder = args.root/'evaluation'/args.split/'both'/'correct'/mode
        identity = json.loads((folder/'complete.json').read_text())
        assert identity['checkpoint_sha256'] == json.loads((args.root/'both/correct/training_complete.json').read_text())['checkpoint_sha256']
        assert identity['model_state_unchanged'] and identity['source_hashes'] == hashes()
        data['both_'+mode+'_input'] = json.loads((folder/'records.json').read_text())
        evidence[str(folder/'records.json')] = base.core.C.file_sha(folder/'records.json')
    reference = data['base_correct']
    for rows in data.values():
        assert len(rows) == 459
        assert [r['image_id'] for r in rows] == [r['image_id'] for r in reference]
        assert all(r['gt'] == b['gt'] for r,b in zip(rows, reference))
    contrasts = {'baseline_content':{'base_correct':1, 'base_constant':-1}}
    for v in PROTOCOL['variants']:
        contrasts[v+'_content'] = {v+'_correct':1, v+'_constant':-1}
        contrasts[v+'_vs_satellite'] = {v+'_correct':1, 'base_satellite':-1}
        contrasts[v+'_vs_baseline'] = {v+'_correct':1, 'base_correct':-1}
        contrasts[v+'_content_interaction'] = {v+'_correct':1, v+'_constant':-1, 'base_correct':-1, 'base_constant':1}
    contrasts.update(both_depth={'both_correct':1, 'both_plane':-1},
        both_donor_dependency={'both_correct':1, 'both_donor_input':-1},
        both_constant_dependency={'both_correct':1, 'both_constant_input':-1},
        query_effect_without_detail={'query_correct':1, 'base_correct':-1},
        query_effect_with_detail={'both_correct':1, 'detail_correct':-1},
        detail_effect_without_query={'detail_correct':1, 'base_correct':-1},
        detail_effect_with_query={'both_correct':1, 'query_correct':-1},
        read_detail_interaction={'both_correct':1, 'query_correct':-1, 'detail_correct':-1, 'base_correct':1})
    samples = np.random.default_rng(20261021).integers(0, 459, (500, 459))
    out = args.root/'analysis'/args.split
    out.mkdir(parents=True, exist_ok=True)
    result = dict(split=args.split, selection='fixed E10 prefix', paths={}, evidence_sha256=evidence,
        bootstrap=500, seed=20261021, source_hashes=hashes(),
        limits='single training seed, spatial correlation, multiple exploratory contrasts, development-exposed v1 test; no test selection')
    for path in ('combined', 'original_path'):
        mapped = {key:([dict(r, dt=r['original_path']) for r in rows] if path == 'original_path' else rows)
                  for key,rows in data.items()}
        result['paths'][path] = {}
        for name,kind,ratio in [('MaskAP', 'segm', None), ('BBoxAP', 'bbox', None),
                                ('BoundaryAP2', 'segm', .02), ('BoundaryAP1', 'segm', .01)]:
            points, boot = {}, {}
            for key,rows in mapped.items():
                ev = evaluator(rows, kind=kind, ratio=ratio)
                points[key] = precision_ap(ev)
                assert abs(fast_sample_ap(ev, np.arange(459))-points[key]) < 1e-12
                boot[key] = np.asarray([fast_sample_ap(ev, ix) for ix in samples])
                print(path, name, key, points[key], flush=True)
            effects = {}
            for key,weights in contrasts.items():
                point = sum(points[k]*v for k,v in weights.items())
                draws = sum(boot[k]*v for k,v in weights.items())
                effects[key] = dict(coefficients=weights, delta=point,
                    paired_image_bootstrap95=np.quantile(draws, [.025, .975]).tolist())
            result['paths'][path][name] = dict(points=points, comparisons=effects)
            base.core.C.dump(out/'summary.partial.json', result)
        diagnostics = {}
        for key,rows in mapped.items():
            diagnostics[key] = dict(**size_and_density(rows), errors=difficult(rows))
            print('difficult', path, key, flush=True)
        base.core.C.dump(out/(path+'_difficult.json'), diagnostics)
    result['calibration'] = audits
    base.core.C.dump(out/'summary.json', result)


if __name__ == '__main__':
    main()
