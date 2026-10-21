#!/usr/bin/env python
"""RSIISN test 评估: result.pkl -> 标准 COCO dt -> pycocotools 12 指标

官方补丁版 results2json 的 image_id 与 GT 不符(loadRes IndexError 被吞成
"empty")，故绕开 mmdet2 的 json 路线，直接用数据集 img_infos 真实 id。
协议: maxDets=[1,10,100](统计口径=100)，与其他基线一致。

用法: python eval_rsiisn_pkl.py RESULT_PKL --config CFG --gt GT_JSON --out-dir DIR
"""
import argparse
import json
import os
import pickle

import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('result_pkl')
    ap.add_argument('--config',
                    default='/home/wangcheng/project/RSIISN/configs/whu1024/rsiisn_cascade_swinT_whu1024.py')
    ap.add_argument('--gt', default='/data1/wangcheng/dataset/WHU/2.4 annotation/annotation/test.json')
    ap.add_argument('--out-dir', default=None)
    args = ap.parse_args()

    from mmcv.utils import Config
    from mmdet.datasets import build_dataset
    cfg = Config.fromfile(args.config)
    cfg.data.test.test_mode = True
    ds = build_dataset(cfg.data.test)
    res = pickle.load(open(args.result_pkl, 'rb'))
    assert len(res) == len(ds), f'{len(res)} != {len(ds)}'

    preds = []
    for idx, r in enumerate(res):
        img_id = ds.data_infos[idx]['id']
        det, seg = r if isinstance(r, tuple) else (r, None)
        # mmdet2 结构: det/seg 均为"每类一个数组/列表"(单类 -> 一层嵌套)
        boxes = np.concatenate(
            [np.asarray(d).reshape(-1, 5) for d in det if len(d)], 0
        ) if any(len(d) for d in det) else np.zeros((0, 5))
        seg_flat = []
        if seg is not None:
            for per_cls in seg:
                seg_flat.extend(per_cls)
        for i in range(len(boxes)):
            x1, y1, x2, y2, s = boxes[i]
            item = dict(image_id=img_id, category_id=1, score=float(s),
                        bbox=[float(x1), float(y1), float(x2 - x1), float(y2 - y1)])
            if i < len(seg_flat):
                rles = []
                # 每实例 seg 可能是 [rle] / [[rle,...]] / polygon, 统一展开
                def _flat(x):
                    if isinstance(x, list):
                        for e in x:
                            yield from _flat(e)
                    else:
                        yield x
                for s_el in _flat(seg_flat[i]):
                    if isinstance(s_el, dict) and 'counts' in s_el:
                        c = s_el['counts']
                        # COCO json 惯例: counts 为 utf-8 str(pycocotools 内部再编码)
                        rles.append(dict(size=list(s_el['size']),
                                         counts=c.decode('utf-8') if isinstance(c, bytes) else c))
                    elif isinstance(s_el, dict):
                        rles.append(s_el)
                    else:
                        rles.append([float(v) for v in np.asarray(s_el).flatten()])
                item['segmentation'] = rles
            preds.append(item)

    print(f'instances: {len(preds)}')
    out_dir = args.out_dir or os.path.dirname(args.result_pkl)
    os.makedirs(out_dir, exist_ok=True)
    json.dump(preds, open(os.path.join(out_dir, 'pred_coco.json'), 'w'))

    coco_gt = COCO(args.gt)
    metrics = {}
    for task in ('bbox', 'segm'):
        coco_dt = coco_gt.loadRes(json.loads(json.dumps(preds)))
        if task == 'segm':
            for p in coco_dt.anns.values():
                img = coco_gt.imgs[p['image_id']]
                h, w = img['height'], img['width']
                rles = []
                for el in p['segmentation']:
                    if isinstance(el, dict) and 'counts' in el:
                        c = el['counts']
                        if isinstance(c, (bytes, str)):
                            rles.append(el)
                        else:
                            rles.append(mask_utils.frPyObjects(el, h, w))
                    else:
                        poly = [float(v) for v in np.asarray(el).flatten()]
                        rles.append(mask_utils.frPyObjects([poly], h, w)[0])
                p['segmentation'] = rles[0] if len(rles) == 1 else mask_utils.merge(
                    [r for r in rles])
        E = COCOeval(coco_gt, coco_dt, task)
        E.evaluate()
        E.accumulate()
        E.summarize()
        names = ['mAP', 'mAP50', 'mAP75', 'mAPs', 'mAPm', 'mAPl']
        for i, n in enumerate(names):
            metrics[f'{task}_{n}'] = round(float(E.stats[i]), 4)
    out = os.path.join(out_dir, 'test_metrics.json')
    json.dump(metrics, open(out, 'w'), indent=2)
    print('saved', out)


if __name__ == '__main__':
    main()
