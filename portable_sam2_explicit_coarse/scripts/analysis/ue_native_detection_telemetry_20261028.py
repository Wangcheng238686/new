"""CPU-only descriptive validation detection telemetry, never a loss or selector.

Fixed score thresholds (.05,.25,.5), IoU thresholds (.5,.75), stable score greedy
one-to-one bbox matching. All GT are building instances, no crowd/ignore records.
These thresholded counts are NOT COCO AP. No test-set paths are read.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from pycocotools import mask as mask_utils

SCORES=(.05,.25,.5)
IOUS=(.5,.75)
SIZES=('small','medium','large')
ERRORS=('center_dx_px','center_dy_px','center_dx_over_gt_width','center_dy_over_gt_height',
        'width_error_px','height_error_px','log_width_ratio','log_height_ratio')
BINS=('no_candidates','lt_0.1','0.1_to_lt_0.5','0.5_to_lt_0.75','ge_0.75')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def size_group(area):return 'small' if area<32**2 else 'medium' if area<96**2 else 'large'


def overlaps(boxes,gt):
    boxes=np.asarray(boxes,dtype=np.float64).reshape(-1,4);gt=np.asarray(gt,dtype=np.float64).reshape(-1,4)
    inter=np.maximum(0,np.minimum(boxes[:,None,2:],gt[None,:,2:])-np.maximum(boxes[:,None,:2],gt[None,:,:2])).prod(-1)
    ba=np.maximum(0,boxes[:,2:]-boxes[:,:2]).prod(-1);ga=np.maximum(0,gt[:,2:]-gt[:,:2]).prod(-1)
    return inter/np.maximum(ba[:,None]+ga[None,:]-inter,1e-12)


def overlap_bin(value):
    if value is None:return 'no_candidates'
    return 'lt_0.1' if value<.1 else '0.1_to_lt_0.5' if value<.5 else '0.5_to_lt_0.75' if value<.75 else 'ge_0.75'


def match_image(gt_boxes,pred_boxes,scores,score_threshold,iou_threshold):
    """Return original prediction indices, GT indices; stable ties preserve input."""
    gt=np.asarray(gt_boxes,np.float64).reshape(-1,4);pred=np.asarray(pred_boxes,np.float64).reshape(-1,4)
    scores=np.asarray(scores,np.float64);assert len(pred)==len(scores)
    assert np.isfinite(gt).all() and np.isfinite(pred).all() and np.isfinite(scores).all()
    assert (gt[:,2:]>gt[:,:2]).all() and (pred[:,2:]>pred[:,:2]).all(),'Positive-area boxes required'
    matrix=overlaps(pred,gt);selected=np.flatnonzero(scores>=score_threshold)
    order=selected[np.argsort(-scores[selected],kind='stable')];used=set();tp=[];fp=[]
    for index in order:
        eligible=matrix[index].copy()
        if used:eligible[list(used)]=-1
        best=int(np.argmax(eligible)) if len(eligible) else -1
        if best>=0 and eligible[best]>=iou_threshold:used.add(best);tp.append((int(index),best))
        else:
            best_iou=float(matrix[index].max()) if len(gt) else 0.
            kind='duplicate' if best_iou>=iou_threshold else 'localization' if best_iou>=.1 else 'background'
            fp.append((int(index),kind))
    return tp,fp,[index for index in range(len(gt)) if index not in used],selected,matrix


def summarize(values):
    if not values:return dict(count=0,mean_signed=None,mean_abs=None,median_abs=None,p90_abs=None)
    x=np.asarray(values,np.float64)
    return dict(count=len(x),mean_signed=float(x.mean()),mean_abs=float(np.abs(x).mean()),median_abs=float(np.median(np.abs(x))),p90_abs=float(np.quantile(np.abs(x),.9)))


def calculate(records,view,score_threshold,iou_threshold):
    count=dict(images=len(records),GT=0,predictions_before_score=0,predictions=0,TP=0,FP=0,FN=0)
    fp_types={name:0 for name in ('duplicate','localization','background')}
    gt_sizes={name:dict(GT=0,TP=0,FN=0) for name in SIZES}
    fp_sizes={name:dict(FP=0,duplicate=0,localization=0,background=0) for name in SIZES}
    fn_bins={name:{key:0 for key in BINS} for name in ('retained_after_score','all_before_score')}
    errors={key:[] for key in ERRORS}
    for row in records:
        gt=row['gt'];dt=row[view];g=np.asarray(gt['bboxes'],np.float64).reshape(-1,4);p=np.asarray(dt['bboxes'],np.float64).reshape(-1,4)
        assert len(gt['rles'])==len(g) and len(dt['scores'])==len(p)
        assert all(int(x)==0 for x in gt['labels']+dt['labels']),'Single-building-class contract'
        areas=np.array([float(mask_utils.area(rle)) for rle in gt['rles']]);assert (areas>0).all()
        tp,fp,fn,selected,matrix=match_image(g,p,dt['scores'],score_threshold,iou_threshold)
        count['GT']+=len(g);count['predictions_before_score']+=len(p);count['predictions']+=len(selected)
        count['TP']+=len(tp);count['FP']+=len(fp);count['FN']+=len(fn)
        for area in areas:gt_sizes[size_group(area)]['GT']+=1
        for pi,gi in tp:
            gt_sizes[size_group(areas[gi])]['TP']+=1
            gw,gh=g[gi,2:]-g[gi,:2];pw,ph=p[pi,2:]-p[pi,:2]
            dx,dy=(p[pi,:2]+p[pi,2:]-g[gi,:2]-g[gi,2:])/2
            values=(dx,dy,dx/gw,dy/gh,pw-gw,ph-gh,np.log(pw/gw),np.log(ph/gh))
            for key,value in zip(ERRORS,values):errors[key].append(float(value))
        for pi,kind in fp:
            fp_types[kind]+=1;area=float(np.prod(p[pi,2:]-p[pi,:2]));item=fp_sizes[size_group(area)];item['FP']+=1;item[kind]+=1
        for gi in fn:
            gt_sizes[size_group(areas[gi])]['FN']+=1
            fn_bins['retained_after_score'][overlap_bin(float(matrix[selected,gi].max()) if len(selected) else None)]+=1
            fn_bins['all_before_score'][overlap_bin(float(matrix[:,gi].max()) if len(p) else None)]+=1
    assert count['TP']+count['FP']==count['predictions'] and count['TP']+count['FN']==count['GT']
    assert sum(fp_types.values())==count['FP']
    for item in gt_sizes.values():
        assert item['GT']==item['TP']+item['FN'];item['recall']=item['TP']/item['GT'] if item['GT'] else None
    for bins in fn_bins.values():assert sum(bins.values())==count['FN']
    return dict(score_threshold=score_threshold,iou_threshold=iou_threshold,counts=count,FP_types=fp_types,
        precision=count['TP']/count['predictions'] if count['predictions'] else None,
        recall=count['TP']/count['GT'] if count['GT'] else None,
        GT_mask_area_recall=gt_sizes,FP_predicted_box_area=fp_sizes,FN_best_candidate_IoU=fn_bins,
        TP_bbox_errors={key:summarize(values) for key,values in errors.items()})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/data/wangcheng/checkpoint/ue_g_ubr_native_full100_20261028'))
    parser.add_argument('--arm',required=True,choices=['A_nohead','B_zero','C_correct']);parser.add_argument('--epoch',required=True,type=int,choices=range(1,101))
    parser.add_argument('--output',type=Path);args=parser.parse_args()
    folder=args.root/args.arm/('validation_epoch%d'%args.epoch);path=folder/'records.json';completion=folder/'complete.json'
    output=args.output or args.root/'telemetry/detection'/args.arm/('epoch%d.json'%args.epoch)
    if output.exists():raise RuntimeError('Preserve existing telemetry output: '+str(output))
    evidence={str(p.resolve()):sha(p) for p in (path,completion,Path(__file__))}
    contract=args.root/args.arm/'contract.json'
    if contract.exists():evidence[str(contract.resolve())]=sha(contract)
    meta=json.loads(completion.read_text());records=json.loads(path.read_text())
    assert meta['complete'] and meta['split']=='val' and meta['arm']==args.arm and meta['images']==len(records)==459
    assert len({r['image_id'] for r in records})==459
    report=dict(complete=True,arm=args.arm,epoch=args.epoch,split='val',evidence_sha256=evidence,
        purpose='descriptive only; never loss/gating/checkpoint selection; no test inputs',
        definitions=dict(matching='bbox IoU; score>=fixed threshold; descending stable score greedy one-to-one; ties GT first input index',
            FP='unmatched: duplicate if maxGTIoU>=match threshold; localization if .1<=maxGTIoU<threshold; background if maxGTIoU<.1 (empty GT=0)',
            FN='unmatched GT; max IoU bins use ALL candidates before score or retained candidates after score, including predictions matched to other GT; not proposal recall',
            size='small area<1024; medium1024<=area<9216; large>=9216. Recall uses GT mask RLE area; FP uses predicted bbox area, separate denominators',
            bbox='TP only; signed pred-minus-GT; dx/dy pixel centers; normalized dx/GTwidth,dy/GTheight; log width/height ratios; summaries count/mean signed/mean abs/median abs/P90 abs; not COCO AP'),
        views={view:[calculate(records,key,score,iou) for score in SCORES for iou in IOUS] for view,key in [('combined','dt'),('original_path','original_path')]})
    assert all(sha(p)==digest for p,digest in evidence.items()),'Input/source changed during analysis'
    output.parent.mkdir(parents=True,exist_ok=True)
    # Exclusive create guarantees no overwriting even when independent watchers race.
    with output.open('x') as stream:json.dump(report,stream,indent=2,sort_keys=True);stream.write('\n')
    print(json.dumps(dict(complete=True,arm=args.arm,epoch=args.epoch,output=str(output))))

if __name__=='__main__':main()
