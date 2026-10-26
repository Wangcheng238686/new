"""Frozen inference audit: original-anchor targets versus decoded candidate value.

Fixed 32 train / 16 test density quantiles; no optimization or structure selection.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '2'
import json
from pathlib import Path

import numpy as np
import torch
from mmdet.models.task_modules.prior_generators.utils import anchor_inside_flags
from mmdet.structures.bbox import bbox_overlaps
from mmengine.structures import InstanceData
import ue_detection_attrition_probe_20261023 as d

OUT = Path('/data/wangcheng/checkpoint/ue_rpn_supervision_audit_20261024')
CACHE = Path('/data/wangcheng/cache/ue_v1_native_geometry')


def main():
    OUT.mkdir(exist_ok=True)
    torch.cuda.set_per_process_memory_fraction(.23)
    d.common.seed(44)
    model, _ = d.build()
    rows = []
    with torch.inference_mode():
        for split, count in [('train', 32), ('test', 16)]:
            ds = d.common.MinimalSatelliteDataset(d.DATA, split, augment=False)
            order = np.argsort([len(s['annotations']) for s in ds.samples], kind='stable')
            indices = [int(order[round((j+.5)/count*(len(ds)-1))]) for j in range(count)]
            assert len(set(indices)) == count
            for index in indices:
                d.common.seed(20261024+index)
                p = d.common.processed(model, d.common.rtmdet_collate_fn([ds[index]]))
                gt = p['data_samples'][0].gt_instances
                meta = p['data_samples'][0].metainfo
                f = model.extract_feat(p['inputs'])
                rpn = model.rpn_head
                cls, reg = rpn(f[0])
                shapes = [z.shape[-2:] for z in cls]
                anchor_list, flag_list = rpn.get_anchors(shapes, [meta], device='cuda')
                priors = anchor_list[0]
                flat = torch.cat(priors)
                flags = torch.cat(flag_list[0])
                inside = anchor_inside_flags(flat, flags, meta['img_shape'][:2], rpn.train_cfg['allowed_border'])
                ar = rpn.assigner.assign(InstanceData(priors=flat[inside]), gt, None)
                assignment = torch.full((len(flat),), -1, device='cuda', dtype=torch.long)
                assignment[inside] = ar.gt_inds  # -1 ignore, 0 background, >0 GT index+1
                targets = rpn.get_targets(anchor_list, flag_list, [gt], [meta])
                labels = torch.cat([z.reshape(-1) for z in targets[0]])
                weights = torch.cat([z.reshape(-1) for z in targets[1]])
                folder = d.common.point_folder(ds, index, d.DATA)
                with np.load(CACHE/f'{folder.name}.npz') as z:
                    support = torch.from_numpy(z['support'].copy()).cuda().bool()
                best_all = torch.zeros(len(gt), device='cuda')
                best_1000 = torch.zeros_like(best_all)
                best_2000 = torch.zeros_like(best_all)
                gt_stats = [dict(good=0, good_supported=0, good_assigned_positive=0,
                    good_assigned_negative=0, good_assigned_ignore=0, good_sampled_positive=0,
                    good_sampled_negative=0) for _ in range(len(gt))]
                stat_keys = list(gt_stats[0])
                stat_values = torch.zeros((len(gt),len(stat_keys)),device='cuda',dtype=torch.long)
                counts = dict(anchors=0, sampled_positive=0, sampled_negative=0,
                    decoded_good_sampled_positive=0, decoded_good_sampled_negative=0,
                    saturated_sampled_positive=0, saturated_sampled_negative=0)
                sampled_losses = []
                offset = 0
                for c, b, prior in zip(cls, reg, priors):
                    logits = c[0].permute(1,2,0).reshape(-1)
                    decoded = rpn.bbox_coder.decode(prior, b[0].permute(1,2,0).reshape(-1,4), max_shape=meta['img_shape'])
                    rank = logits.sigmoid().sort(descending=True)[1]
                    n = len(logits)
                    a = assignment[offset:offset+n]
                    lab = labels[offset:offset+n]
                    w = weights[offset:offset+n]
                    pos, neg = (w>0)&(lab==0), (w>0)&(lab==1)
                    centers = ((prior[:,:2]+prior[:,2:])/2).round().long()
                    x, y = centers[:,0], centers[:,1]
                    valid = (x>=0)&(x<512)&(y>=0)&(y<512)
                    supported = valid.clone(); supported[valid] = support[y[valid],x[valid]]
                    good_anchor = torch.zeros(n, device='cuda', dtype=torch.bool)
                    for start in range(0,n,2048):
                        ov = bbox_overlaps(gt.bboxes, decoded[start:start+2048])
                        best_all = torch.maximum(best_all, ov.max(1).values)
                        good = ov >= .5
                        good_anchor[start:start+2048] = good.any(0)
                        aa = a[start:start+2048]
                        indicators = torch.stack((torch.ones_like(aa,dtype=torch.bool),
                            supported[start:start+2048],aa>0,aa==0,aa<0,
                            pos[start:start+2048],neg[start:start+2048]),1)
                        stat_values += (good.float() @ indicators.float()).long()
                    for k, dest in [(1000,best_1000),(2000,best_2000)]:
                        ov = bbox_overlaps(gt.bboxes, decoded[rank[:k]])
                        dest.copy_(torch.maximum(dest, ov.max(1).values))
                    counts['anchors'] += n
                    counts['sampled_positive'] += int(pos.sum())
                    counts['sampled_negative'] += int(neg.sum())
                    counts['decoded_good_sampled_positive'] += int((pos&good_anchor).sum())
                    counts['decoded_good_sampled_negative'] += int((neg&good_anchor).sum())
                    counts['saturated_sampled_positive'] += int((pos&(logits>10)).sum())
                    counts['saturated_sampled_negative'] += int((neg&(logits< -10)).sum())
                    sampled = w>0
                    losses = torch.nn.functional.binary_cross_entropy_with_logits(logits[sampled], (lab[sampled]==0).float(), reduction='none')
                    sampled_losses.extend(losses.cpu().tolist())
                    offset += n
                stat_cpu = stat_values.cpu().tolist()
                for j,s in enumerate(gt_stats):
                    s.update(zip(stat_keys,stat_cpu[j]))
                    s.update(gt_index=j, best_all=float(best_all[j]), best_top1000=float(best_1000[j]), best_top2000=float(best_2000[j]))
                rows.append(dict(split=split, image_id=ds.samples[index]['image_id'], dataset_index=index,
                    gt=len(gt), counts=counts, mean_sampled_bce=float(np.mean(sampled_losses)), instances=gt_stats))
                d.common.dump(OUT/'progress.json', dict(images=len(rows), split=split, last_id=ds.samples[index]['image_id'], pid=os.getpid()))
                print('IMAGE', split, index, counts, flush=True)
    assert d.common.state_digest(model) == d.FROZEN_SHA
    summary = {}
    for split in ['train','test']:
        rr = [r for r in rows if r['split']==split]
        instances = [g for r in rr for g in r['instances']]
        lost = [g for g in instances if g['best_all']>=.5>g['best_top1000']]
        recovered = [g for g in lost if g['best_top2000']>=.5]
        summary[split] = dict(images=len(rr), gt=len(instances),
            counts={key:sum(r['counts'][key] for r in rr) for key in rr[0]['counts']},
            mean_image_sampled_bce=float(np.mean([r['mean_sampled_bce'] for r in rr])))
        for name, group in [('topk_lost',lost),('budget_recovered',recovered)]:
            summary[split][name] = dict(gt=len(group),
                with_any_supported_good=sum(g['good_supported']>0 for g in group),
                with_any_assigned_positive_good=sum(g['good_assigned_positive']>0 for g in group),
                with_any_sampled_positive_good=sum(g['good_sampled_positive']>0 for g in group),
                with_any_sampled_negative_good=sum(g['good_sampled_negative']>0 for g in group),
                counts={key:sum(g[key] for g in group) for key in ('good','good_supported','good_assigned_positive','good_assigned_negative','good_assigned_ignore','good_sampled_positive','good_sampled_negative')})
    d.common.dump(OUT/'records.json',rows)
    d.common.dump(OUT/'summary.json',dict(complete=True, summary=summary,
        satellite_state_unchanged=True, optimizer_steps=0, model_state_sha256=d.FROZEN_SHA,
        source_sha256=d.common.file_sha(__file__), base_sha256=d.common.file_sha(d.BASE),
        selection='32 train /16 test equally spaced mid-density-quantiles, stable sort by GT count; no augmentation; seed20261024+index',
        scope='Diagnostic samples, not full-split estimates. Good means decoded IoU>=.5 with any GT. '
              'Assigned-positive may correspond to a different GT; sampled is a single deterministic anchor draw. '
              'Negative good candidates show original-anchor vs decoded-quality target mismatch, not annotation error.'))
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':
    main()
