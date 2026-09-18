#!/usr/bin/env python
"""从 Ours (PromptMiner-SAM2) 管线导出 proposal 框 + coarse mask, 供 teaser panel(a) 使用.

原理: monkeypatch SAM2MaskHead._forward_coarse_p2, 捕获每张图 predict 时 mask head
收到的 boxes(全图坐标) 与 coarse_outputs(raw/refined logits, ROI-local).
导出: 每条 proposal 的框/分数 + coarse mask(整图 RLE) + 匹配的最终检测分数.
用法: python dump_coarse_teaser.py --stems 300838,310401 --out coarse_teaser/
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

MAINLINE = Path(__file__).resolve().parents[2]
REPO = MAINLINE.parent
sys.path.insert(0, str(MAINLINE))
sys.path.insert(0, str(MAINLINE / 'inference'))
sys.path.insert(0, str(Path(__file__).parent))

import infer_from_checkpoint as ifc  # noqa: E402

TEST_JSON = '/data1/wangcheng/dataset/WHU/2.4 annotation/annotation/test.json'
OUR_CKPT = ('/data1/wangcheng/checkpoint/portable_sam2_explicit_coarse/ablations/'
            'paper_promptminer_rd_p2_whu_full_tr1.0_va1.0/best_model_epoch98.pth')


def make_mini_ann(stems, out_path):
    data = json.load(open(TEST_JSON))
    keep = {im['id'] for im in data['images'] if Path(im['file_name']).stem in stems}
    mini = {
        'images': [im for im in data['images'] if im['id'] in keep],
        'annotations': [a for a in data['annotations'] if a['image_id'] in keep],
        'categories': data.get('categories', []),
    }
    json.dump(mini, open(out_path, 'w'))
    return {Path(im['file_name']).stem: im['id'] for im in mini['images']}


def paste_roi(mask_small, box, size=1024):
    """ROI-local 掩码按框贴回整图(与 roi_local_bbox_paste 同约定)."""
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    w, h = max(1, x2 - x1), max(1, y2 - y1)
    m = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((size, size), np.uint8)
    xs1, ys1 = max(0, x1), max(0, y1)
    xs2, ys2 = min(size, x1 + w), min(size, y1 + h)
    if xs2 > xs1 and ys2 > ys1:
        canvas[ys1:ys2, xs1:xs2] = m[ys1 - y1:ys2 - y1, xs1 - x1:xs2 - x1]
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stems', required=True, help='逗号分隔的影像 stem')
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cuda:3')
    ap.add_argument('--iou-match', type=float, default=0.5)
    args = ap.parse_args()

    stems = [s.strip() for s in args.stems.split(',')]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    mini_ann = out_dir / 'mini_ann.json'
    make_mini_ann(stems, mini_ann)

    from pycocotools import mask as mm

    ckpt_path = Path(OUR_CKPT).resolve()
    checkpoint = ifc._load_checkpoint(ckpt_path)
    snapshot = ifc._snapshot(checkpoint)
    args_ns = argparse.Namespace(
        checkpoint=str(ckpt_path), config=None, split='custom',
        data_root='/data1/wangcheng/dataset/WHU',
        ann_file=str(mini_ann), image_subdir='2.2 test/test',
        image_size=None, batch_size=1, num_workers=0,
        device=args.device, weights='model',
        sam2_repo=None, sam2_ckpt=None,
        allow_nonstrict=False, disable_p2=False, p2_beta=None,
        max_per_img=None, nms_thr=None,
    )
    sam2_repo = ifc._resolve_sam2_repo(args_ns, snapshot)
    model_config, _ = ifc._resolve_model_config(args_ns, snapshot)
    device = ifc._resolve_device(args_ns.device)
    model = ifc._register_and_build(model_config)
    ifc._load_model_state(model, ifc._select_state_dict(checkpoint, 'model'), False)
    model.to(device).eval()

    # --- 捕获 coarse 输出 + 挖掘点 ---
    import rsprompter.sam2_mask_head as smh
    from rsprompter.shape_point_miner import ShapePointMiner
    orig_p2 = smh.RSPrompterAnchorMaskHeadSAM2._forward_coarse_p2
    orig_mine = ShapePointMiner.forward
    captures = []
    point_captures = []

    def patched(self, x, image_embeddings, boxes, roi_img_ids, p2f, pr, rf, dbg):
        out = orig_p2(self, x, image_embeddings, boxes, roi_img_ids, p2f, pr, rf, dbg)
        _, coarse_outputs, _, _ = out
        if boxes is not None and coarse_outputs is not None:
            captures.append({
                'boxes': boxes.detach().float().cpu().numpy(),
                'raw': coarse_outputs['raw_logits'].detach().sigmoid().cpu().numpy(),
                'refined': coarse_outputs['refined_logits'].detach().sigmoid().cpu().numpy(),
            })
        return out

    def patched_mine(self, mask_logits, local_boxes, pe_size, *a, **kw):
        coords, labels, stats = orig_mine(self, mask_logits, local_boxes, pe_size, *a, **kw)
        point_captures.append((coords.detach().float().cpu().numpy(),
                               labels.detach().cpu().numpy(),
                               float(pe_size) if not torch.is_tensor(pe_size) else float(pe_size)))
        return coords, labels, stats

    smh.RSPrompterAnchorMaskHeadSAM2._forward_coarse_p2 = patched
    ShapePointMiner.forward = patched_mine
    roi_sam = bool(getattr(model.roi_head.mask_head, 'roi_sam_enabled', False))

    # --- 只跑目标图 ---
    contract = dict(ifc._resolve_dataset_contract(args_ns, snapshot))
    contract['batch_size'] = 1
    loader = ifc._build_loader(contract, 0)
    pre = model.data_preprocessor
    records = {}
    with torch.inference_mode():
        for batch in loader:
            stem = Path(batch['img_metas'][0]['filename']).stem
            imgs, samples = ifc._build_data_samples(batch, device)
            processed = pre({'inputs': imgs, 'data_samples': samples}, training=False)
            captures.clear()
            point_captures.clear()
            outputs = model.predict(processed['inputs'], processed['data_samples'], rescale=False)
            pred = outputs[0].pred_instances
            pb = pred.bboxes.cpu().numpy()
            ps = pred.scores.cpu().numpy()
            prop = []
            for cap in captures:  # mask head 可能分块处理 ROI, 合并全部块
                for i in range(len(cap['boxes'])):
                    box = cap['boxes'][i]
                    refined = (cap['refined'][i, 0] > 0.5).astype(np.uint8)
                    raw = (cap['raw'][i, 0] > 0.5).astype(np.uint8)
                    full_r = paste_roi(refined, box)
                    full_w = paste_roi(raw, box)
                    # 匹配最终检测(同框 IoU 最大者)取分数
                    ious = []
                    for j in range(len(pb)):
                        xx1 = max(box[0], pb[j][0]); yy1 = max(box[1], pb[j][1])
                        xx2 = min(box[2], pb[j][2]); yy2 = min(box[3], pb[j][3])
                        inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
                        a1 = (box[2] - box[0]) * (box[3] - box[1])
                        a2 = (pb[j][2] - pb[j][0]) * (pb[j][3] - pb[j][1])
                        ious.append(inter / max(1e-6, a1 + a2 - inter))
                    jbest = int(np.argmax(ious)) if ious else -1
                    score = float(ps[jbest]) if jbest >= 0 and ious[jbest] > 0.5 else None
                    # 挖掘点: miner 与 coarse 同一次 forward, ROI 行序一致.
                    # roi_sam_enabled=False 时 miner 收到全图框, coords 已是全图坐标;
                    # =True 时 coords 为 ROI-local PE 画布坐标, 需线性映射回全图.
                    pts = []
                    if len(point_captures) == len(captures):
                        ci = next(k2 for k2, c2 in enumerate(captures) if c2 is cap)
                        if ci < len(point_captures):
                            pc, pl, pe_size = point_captures[ci]
                            if i < len(pc):
                                if roi_sam:
                                    bw = max(1e-6, box[2] - box[0]); bh = max(1e-6, box[3] - box[1])
                                    tr = lambda pt_: (box[0] + pt_[0] * bw / pe_size,
                                                      box[1] + pt_[1] * bh / pe_size)
                                else:
                                    tr = lambda pt_: (pt_[0], pt_[1])
                                for pt_, lb_ in zip(pc[i], pl[i]):
                                    x_, y_ = tr(pt_)
                                    pts.append({'x': float(x_), 'y': float(y_),
                                                'label': int(lb_)})
                    prop.append({
                        'box': [float(v) for v in box],
                        'det_score': score,
                        'points': pts,
                        'coarse_rle': mm.encode(np.asfortranarray(full_r)).__class__(
                            {'size': list(full_r.shape), 'counts': mm.encode(np.asfortranarray(full_r))['counts'].decode('ascii')},
                        ) if full_r.sum() else None,
                        'raw_rle': (
                            {'size': list(full_w.shape),
                             'counts': mm.encode(np.asfortranarray(full_w))['counts'].decode('ascii')}
                            if full_w.sum() else None
                        ),
                    })
            records[stem] = prop
            print(f'{stem}: proposals={len(prop)} final_dets={len(pb)}')
    smh.RSPrompterAnchorMaskHeadSAM2._forward_coarse_p2 = orig_p2

    json.dump(records, open(out_dir / 'coarse_dump.json', 'w'))
    print(f'saved -> {out_dir / "coarse_dump.json"}')


if __name__ == '__main__':
    main()
