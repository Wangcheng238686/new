"""Authoritative FROC supplement: thresholds accept complete equal-score groups.

Independent postprocessing; do not edit the frozen training/analysis sources.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from pycocotools import mask as mu


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def attainable_curve(detections, small_gt, images, budgets=(.1,.5,1.,2.,5.,10.)):
    """Rows are score,TP,smallTP after fixed per-image greedy matching."""
    if not len(detections):
        return {str(b):0. if small_gt else None for b in budgets}
    values = np.asarray(detections, dtype=np.float64)
    values = values[np.argsort(-values[:, 0], kind='stable')]
    endpoints = np.flatnonzero(np.r_[values[:-1, 0] != values[1:, 0], True])
    fp = np.cumsum(1-values[:, 1])[endpoints]
    tp_small = np.cumsum(values[:, 2])[endpoints]
    result = {}
    for budget in budgets:
        accepted = np.flatnonzero(fp <= budget*images)
        result[str(budget)] = (float(tp_small[accepted[-1]]/small_gt) if len(accepted) else 0.) if small_gt else None
    return result


def froc(rows, kind):
    detections, denominator = [], 0
    for row in rows:
        gt,dt = row['gt'],row['dt']
        n,m = len(gt['bboxes']),len(dt['bboxes'])
        small = np.asarray(mu.area(gt['rles'])) < 1024
        denominator += int(small.sum())
        if kind == 'mask':
            overlap = mu.iou(dt['rles'],gt['rles'],[0]*n) if n and m else np.zeros((m,n))
        else:
            g = np.asarray(gt['bboxes']).reshape(-1,4)
            d = np.asarray(dt['bboxes']).reshape(-1,4)
            wh = np.maximum(0,np.minimum(d[:,None,2:],g[None,:,2:])-np.maximum(d[:,None,:2],g[None,:,:2]))
            inter = wh.prod(-1)
            overlap = inter/np.maximum(1e-12,(d[:,2:]-d[:,:2]).prod(-1)[:,None]+(g[:,2:]-g[:,:2]).prod(-1)[None]-inter)
        matched = np.zeros(n,bool)
        for i in np.argsort(-np.asarray(dt['scores']),kind='stable'):
            candidate = np.where(~matched,overlap[i],-1)
            j = int(candidate.argmax()) if n else -1
            success = j >= 0 and candidate[j] >= .5
            if success:
                matched[j] = True
            detections.append((float(dt['scores'][i]),int(success),int(success and small[j])))
    return dict(small_gt=denominator,images=len(rows),detections=len(detections),
        small_recall_at_FP_per_image=attainable_curve(detections,denominator,len(rows)),
        matching='same per-image score-stable greedy one-to-one IoU>=.5; all sizes contribute FP',
        thresholds='score>=threshold; equal scores always accepted together; empty set allowed')


def compute(folder, output):
    meta = json.loads((folder/'complete.json').read_text())
    assert meta['complete'] and meta['images'] == 459 and meta['world_size'] == 4
    rows = json.loads((folder/'records.json').read_text())
    assert len(rows) == len({r['image_id'] for r in rows}) == 459
    result = dict(authoritative_for='fixed-FP-per-image small recall only; supersedes frozen analysis prefix-based FROC',
        records_sha256=digest(folder/'records.json'),identity_sha256=digest(folder/'complete.json'),
        source_sha256=digest(__file__),checkpoint_sha256=meta['checkpoint_sha256'],
        split=meta['split'],paths={})
    for path in ['combined','original_path']:
        mapped = rows if path == 'combined' else [dict(r,dt=r['original_path']) for r in rows]
        result['paths'][path] = {kind:froc(mapped,kind) for kind in ['bbox','mask']}
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary = output.with_suffix('.tmp')
    temporary.write_text(json.dumps(result,indent=2)+'\n')
    temporary.replace(output)


def watch(root, owner):
    record = Path('/proc',str(owner),'stat')
    assert record.exists()
    ticks = record.read_text().rsplit(') ',1)[1].split()[19]
    source = digest(__file__)
    protocol = json.loads((root/'manifest.json').read_text())['protocol']
    tasks = []
    for split in ['val','test']:
        for variant,content in protocol['training_jobs']:
            modes = ['matched','constant','donor'] if (variant,content) == ('both','correct') else ['matched']
            for mode in modes:
                folder = root/'evaluation'/split/variant/content/mode
                output = root/'analysis_corrected_froc'/split/(variant+'_'+content+'_'+mode+'.json')
                tasks.append((folder,output))
    status = root/'analysis_corrected_froc/state.json'
    status.parent.mkdir(parents=True,exist_ok=True)
    while True:
        assert digest(__file__) == source
        done = []
        for folder,output in tasks:
            if output.exists():
                d = json.loads(output.read_text())
                assert d['source_sha256'] == source and d['records_sha256'] == digest(folder/'records.json')
                assert d['identity_sha256'] == digest(folder/'complete.json')
                done.append(str(output))
            elif (folder/'complete.json').exists():
                compute(folder,output)
                done.append(str(output))
        status.write_text(json.dumps(dict(pid=os.getpid(),done=done,total=len(tasks),
            source_sha256=source,producer_owner_pid=owner,producer_start_ticks=ticks,observed=time.time()),indent=2)+'\n')
        if len(done) == len(tasks):
            (status.parent/'complete.json').write_text(json.dumps(dict(complete=True,outputs={p:digest(p) for p in done},source_sha256=source),indent=2)+'\n')
            return
        assert record.exists(), 'Producer exited before all fixed evaluations; audit rather than restart'
        current = record.read_text().rsplit(') ',1)[1].split()
        assert current[0] not in ['Z','X'] and current[19] == ticks
        time.sleep(30)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--watch-root',type=Path)
    p.add_argument('--owner-pid',type=int)
    p.add_argument('--input',type=Path)
    p.add_argument('--output',type=Path)
    args = p.parse_args()
    if args.watch_root:
        assert args.owner_pid
        watch(args.watch_root,args.owner_pid)
    else:
        assert args.input and args.output
        compute(args.input,args.output)


if __name__ == '__main__':
    main()
