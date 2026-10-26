"""Paired AP/Boundary AP assessment from fixed complete v1 adapter predictions."""
import argparse
import contextlib
import copy
import io
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from pycocotools import mask as mu
from pycocotools.cocoeval import COCOeval
from utils.coco_eval_utils import build_coco_gt_and_dt

cv2.setNumThreads(1)  # Bound CPU use when independent comparisons run concurrently.


def boundary(rle, ratio):
    mask=mu.decode(rle).astype(np.uint8)
    d=max(1,int(round(ratio*np.hypot(*mask.shape))))
    padded=np.pad(mask,((1,1),(1,1)))
    eroded=cv2.erode(padded,np.ones((3,3),np.uint8),iterations=d)[1:-1,1:-1]
    return mu.encode(np.asfortranarray(mask-eroded))


class BoundaryEval(COCOeval):
    def __init__(self,gt,dt,ratio):
        super().__init__(gt,dt,'segm')
        self.ratio=ratio
        self.bcache={}

    def computeIoU(self,imgId,catId):
        mask_iou=super().computeIoU(imgId,catId)
        if not len(mask_iou):return mask_iou
        gt=self._gts[imgId,catId];dt=self._dts[imgId,catId]
        inds=np.argsort([-d['score'] for d in dt],kind='mergesort')
        dt=[dt[i] for i in inds[:self.params.maxDets[-1]]]
        def rle(a,kind):
            key=(kind,a['id'])
            if key not in self.bcache:self.bcache[key]=boundary(a['segmentation'],self.ratio)
            return self.bcache[key]
        biou=mu.iou([rle(a,'dt') for a in dt],[rle(a,'gt') for a in gt],[a['iscrowd'] for a in gt])
        return np.minimum(mask_iou,biou)


def precision_ap(ev):
    a=ev.eval['precision'][:,:,:,0,-1]
    return float(a[a>=0].mean())


def evaluator(rows,kind='segm',ratio=None):
    def arrays(a):
        b=dict(a)
        for k in ('bboxes','labels','scores','mask_scores'):
            if k in b:b[k]=np.array(b[k])
        return b
    g=[arrays(r['gt']) for r in rows];d=[arrays(r['dt']) for r in rows]
    metas=[dict(img_shape=(512,512)) for _ in rows]
    with contextlib.redirect_stdout(io.StringIO()):
        gt,dt=build_coco_gt_and_dt(g,d,metas,score_key='scores')
        ev=BoundaryEval(gt,dt,ratio) if ratio else COCOeval(gt,dt,kind)
        ev.params.maxDets=[1,10,250];ev.evaluate();ev.accumulate()
    return ev


def sample_ap(ev,indices):
    n=len(indices)
    original=ev.evalImgs
    # Reuse per-image matches. Entries carry independent one-to-one assignments;
    # accumulate only reads their match/ignore/score arrays, not original image IDs.
    ev.evalImgs=[original[a*n+int(i)] for a in range(len(ev.params.areaRng)) for i in indices]
    primary=copy.deepcopy(ev.params)
    primary.areaRng=[ev.params.areaRng[0]];primary.areaRngLbl=['all'];primary.maxDets=[250]
    with contextlib.redirect_stdout(io.StringIO()):ev.accumulate(primary)
    value=precision_ap(ev)
    ev.evalImgs=original
    return value


def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--bootstrap',type=int,default=500)
    p.add_argument('--split',choices=['val','test'],default='val');args=p.parse_args()
    base=json.loads(args.baseline.read_text());cand=json.loads(args.candidate.read_text())
    assert len(base)==len(cand)==459
    assert [r['image_id'] for r in base]==[r['image_id'] for r in cand]
    for b,c in zip(base,cand):
        assert b['gt']==c['gt'],'GT changed'
        for k in ('bboxes','labels','scores'):
            assert b['dt'][k]==c['dt'][k],f'Detector output changed {k}'
    result=dict(images=459,evaluation_split=args.split,detector_and_gt_exact=True,selection='last',metrics={},
        analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        bootstrap_scope='all-area AP at maxDets250; full point metrics evaluated before resampling')
    rng=np.random.default_rng(20261021)
    samples=rng.integers(0,len(base),(args.bootstrap,len(base)))
    for name,ratio in [('MaskAP',None),('BoundaryAP_2pct',.02),('BoundaryAP_1pct',.01)]:
        eb,ec=evaluator(base,ratio=ratio),evaluator(cand,ratio=ratio)
        vb,vc=precision_ap(eb),precision_ap(ec)
        # Identity re-accumulation is a required calibration for the bootstrap path.
        assert abs(sample_ap(eb,np.arange(459))-vb)<1e-12
        assert abs(sample_ap(ec,np.arange(459))-vc)<1e-12
        ds=[]
        for i,ix in enumerate(samples):
            ds.append(sample_ap(ec,ix)-sample_ap(eb,ix))
            if (i+1)%100==0:print(name,'bootstrap',i+1,flush=True)
        ci=np.quantile(ds,[.025,.975]).tolist()
        result['metrics'][name]=dict(baseline=vb,candidate=vc,delta=vc-vb,
            paired_image_bootstrap95=ci,bootstrap_samples=len(ds),seed=20261021)
        print(name,result['metrics'][name],flush=True)
    m=result['metrics']['MaskAP']
    result['gate']=dict(direction_positive=m['delta']>0,
        clear_v1_checkpoint_gain=m['delta']>=.002 and m['paired_image_bootstrap95'][0]>0,
        limitations='single training seed; image bootstrap does not cover training variance or spatial dependence')
    args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
