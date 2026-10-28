"""Offline fixed-record AP/bootstrap, factorial contrasts and error accounting."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import cv2
from pycocotools import mask as mu
from scipy.optimize import linear_sum_assignment
from ue_v1_adapter_analysis import evaluator,precision_ap,sample_ap
from ue_v1_adapter import dump,file_sha
cv2.setNumThreads(1)


def point_and_bootstrap(records,out,bootstrap=500):
    rows=json.loads(Path(records).read_text());assert len(rows)==459
    samples=np.random.default_rng(20261021).integers(0,459,(bootstrap,459))
    report=dict(images=459,records_sha256=file_sha(records),bootstrap_samples=bootstrap,seed=20261021,metrics={})
    vectors={}
    for name,ratio in [('MaskAP',None),('BoundaryAP_2pct',.02),('BoundaryAP_1pct',.01)]:
        ev=evaluator(rows,ratio=ratio);point=precision_ap(ev)
        precision=ev.eval['precision'];recall=ev.eval['recall']
        def mean_valid(x):return float(x[x>=0].mean()) if (x>=0).any() else None
        report['metrics'][name]=dict(AP=point,AP75=mean_valid(precision[5,:,:,0,-1]),
            AP_small=mean_valid(precision[:,:,:,1,-1]),AP_medium=mean_valid(precision[:,:,:,2,-1]),
            AP_large=mean_valid(precision[:,:,:,3,-1]),AR250=mean_valid(recall[:,:,0,-1]))
        assert abs(sample_ap(ev,np.arange(459))-point)<1e-12
        values=[]
        for i,ix in enumerate(samples):
            values.append(sample_ap(ev,ix))
            if (i+1)%100==0:print(name,i+1,flush=True)
        vectors[name]=np.array(values)
    bbox=evaluator(rows,kind='bbox')
    report['bbox_AP']=precision_ap(bbox)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'bootstrap.npz',**vectors)
    dump(out/'metrics.json',report)


def assert_paired(base,candidate):
    assert len(base)==len(candidate)==459
    for b,c in zip(base,candidate):
        assert b['image_id']==c['image_id'] and b['gt']==c['gt']
        assert all(b['dt'][k]==c['dt'][k] for k in ('bboxes','labels','scores'))


def contrasts(root,manifest):
    root=Path(root);outputs={}
    names=['baseline']+[a['name'] for a in manifest['arms']]
    metrics={n:json.loads((root/'analysis'/n/'metrics.json').read_text()) for n in names}
    vectors={n:dict(np.load(root/'analysis'/n/'bootstrap.npz')) for n in names}
    specs={f'{n}-baseline':{n:1,'baseline':-1} for n in names if n!='baseline'}
    specs.update({'read_B-A':{'B':1,'A':-1},'read_D-C':{'D':1,'C':-1},
        'fusion_C-A':{'C':1,'A':-1},'fusion_D-B':{'D':1,'B':-1},
        'interaction':{'D':1,'C':-1,'B':-1,'A':1}})
    for n in 'ABCD':specs[n+'-constant_train']={n:1,n+'_constant':-1}
    for n in 'CD':specs[n+'-zero_train']={n:1,n+'_zero':-1}
    for content,suffix in [('correct',''),('constant','_constant'),('zero','_zero')]:
        specs['aux_'+content]={'D_aux'+suffix:1,'D'+suffix:-1}
    for title,coeff in specs.items():
        result={}
        for metric in ('MaskAP','BoundaryAP_2pct','BoundaryAP_1pct'):
            delta=sum(c*metrics[n]['metrics'][metric]['AP'] for n,c in coeff.items())
            values=sum(c*vectors[n][metric] for n,c in coeff.items())
            result[metric]=dict(delta=delta,paired_image_bootstrap95=np.quantile(values,[.025,.975]).tolist(),
                bootstrap_samples=500,seed=20261021)
        ap=result['MaskAP'];outputs[title]=dict(coefficients=coeff,metrics=result,
            clear_gain_reference=ap['delta']>=.002 and ap['paired_image_bootstrap95'][0]>0)
    dump(root/'contrasts.json',dict(primary_split='test',selection='E10 last',contrasts=outputs,
        limitations=['single training seed','image bootstrap excludes training variance and spatial dependence',
                    'multiple exploratory comparisons; nominal intervals','same v1 test previously exposed']))
    return outputs


def bbox_matching(g,d):
    g=np.array(g,float).reshape(-1,4);d=np.array(d,float).reshape(-1,4)
    if not len(g) or not len(d):return [],len(g),len(d)
    # Serialized gt_arrays boxes are xyxy; no mask-derived boxes.
    lo=np.maximum(g[:,None,:2],d[None,:,:2]);hi=np.minimum(g[:,None,2:],d[None,:,2:])
    inter=np.maximum(hi-lo,0).prod(-1)
    area_g=np.maximum(g[:,2:]-g[:,:2],0).prod(-1);area_d=np.maximum(d[:,2:]-d[:,:2],0).prod(-1)
    iou=inter/(area_g[:,None]+area_d[None,:]-inter).clip(1e-9)
    gi,di=linear_sum_assignment(-iou);pairs=[(int(x),int(y)) for x,y in zip(gi,di) if iou[x,y]>=.5]
    return pairs,len(g)-len(pairs),len(d)-len(pairs)


def mechanisms(base_path,candidate_path,cache,out):
    base=json.loads(Path(base_path).read_text());candidate=json.loads(Path(candidate_path).read_text());assert_paired(base,candidate)
    totals={zone:dict(pixels=0,corrected=0,introduced=0,baseline_errors=0) for zone in ('boundary','interior','background')}
    unmatched_gt=unmatched_dt=0;changed=0;images=[];offset=[]
    for b,c in zip(base,candidate):
        pairs,ug,ud=bbox_matching(b['gt']['bboxes'],b['dt']['bboxes']);unmatched_gt+=ug;unmatched_dt+=ud
        gt=mu.decode(b['gt']['rles']) if len(b['gt']['rles']) else np.zeros((512,512,0),np.uint8)
        bm=mu.decode(b['dt']['rles']) if len(b['dt']['rles']) else np.zeros((512,512,0),np.uint8)
        cm=mu.decode(c['dt']['rles']) if len(c['dt']['rles']) else np.zeros((512,512,0),np.uint8)
        changed+=int(np.count_nonzero(bm!=cm))
        for gi,di in pairs:
            g=gt[:,:,gi].astype(bool);old=bm[:,:,di].astype(bool);new=cm[:,:,di].astype(bool)
            inner=cv2.erode(g.astype(np.uint8),np.ones((3,3),np.uint8),iterations=2,borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)
            outer=cv2.dilate(g.astype(np.uint8),np.ones((3,3),np.uint8),iterations=2,borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)
            box=np.array([b['gt']['bboxes'][gi],b['dt']['bboxes'][di]])
            lo=np.maximum(np.floor(box[:,:2].min(0)).astype(int)-8,0);hi=np.minimum(np.ceil(box[:,2:].max(0)).astype(int)+8,512)
            region=np.zeros((512,512),bool);region[lo[1]:hi[1],lo[0]:hi[0]]=True
            zones=dict(boundary=(outer&~inner)&region,interior=inner&region,background=(~outer)&region)
            olderr=old!=g;newerr=new!=g
            for key,z in zones.items():
                totals[key]['pixels']+=int(z.sum());totals[key]['corrected']+=int((z&olderr&~newerr).sum())
                totals[key]['introduced']+=int((z&~olderr&newerr).sum());totals[key]['baseline_errors']+=int((z&olderr).sum())
        union=gt.any(-1);path=Path(cache)/f"{b['image_id']:06d}.npz"
        # Archive IDs are the source point ID. Fall back to report mapping outside
        # this routine rather than silently substituting a different image.
        if not path.exists():raise FileNotFoundError(path)
        from rsprompter.uav.s2db_geometry import observed_support
        with np.load(path) as a:support=observed_support(a['index'],a['anchor'])
        coverage=float((union&support).sum()/max(int(union.sum()),1))
        count=gt.shape[-1];group='low' if coverage<.5 else ('mid' if coverage<.8 else 'high')
        density='le50' if count<=50 else ('51to100' if count<=100 else 'gt100')
        images.append(dict(image_id=b['image_id'],coverage=coverage,coverage_group=group,density_group=density,
            matched=len(pairs),unmatched_gt=ug,unmatched_dt=ud,stats=c.get('stats',{})))
        if c.get('read_observation'):offset.append(c['read_observation'])
    group_metrics={}
    for column,labels in [('coverage_group',['low','mid','high']),('density_group',['le50','51to100','gt100'])]:
        for label in labels:
            inds=[i for i,row in enumerate(images) if row[column]==label]
            if not inds:group_metrics[column+'/'+label]=dict(images=0,MaskAP_delta=None);continue
            eb=evaluator([base[i] for i in inds]);ec=evaluator([candidate[i] for i in inds])
            group_metrics[column+'/'+label]=dict(images=len(inds),MaskAP_baseline=precision_ap(eb),MaskAP_candidate=precision_ap(ec),
                MaskAP_delta=precision_ap(ec)-precision_ap(eb))
    dump(out,dict(zones=totals,zone_denominator='matched instance-pixel counts; overlapping instances may repeat pixels',
        unmatched_gt=unmatched_gt,unmatched_dt=unmatched_dt,bbox_match_iou=.5,matching='Hungarian bbox IoU, same frozen assignments',
        changed_instance_pixels=changed,changed_denominator='all detection instance pixels, including unmatched detections',
        images=images,groups=group_metrics,read_observations=offset,
        baseline_sha256=file_sha(base_path),candidate_sha256=file_sha(candidate_path)))



def content_changes(first,second,out):
    a=json.loads(Path(first).read_text());b=json.loads(Path(second).read_text());assert_paired(a,b)
    instances=changed_instances=changed_pixels=0
    rows=[]
    for x,y in zip(a,b):
        xr,yr=x['dt']['rles'],y['dt']['rles'];assert len(xr)==len(yr)
        count=pixels=0
        for old,new in zip(xr,yr):
            if old!=new:
                n=int(np.count_nonzero(mu.decode(old)!=mu.decode(new)))
                count+=int(n>0);pixels+=n
        instances+=len(xr);changed_instances+=count;changed_pixels+=pixels
        rows.append(dict(image_id=x['image_id'],instances=len(xr),changed_instances=count,changed_pixels=pixels))
    dump(out,dict(first_sha256=file_sha(first),second_sha256=file_sha(second),images=459,
        instances=instances,changed_instances=changed_instances,changed_pixels=changed_pixels,
        pixel_denominator=instances*512*512,scope='binary instance predictions; not soft logits',rows=rows))

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='action',required=True)
    q=sub.add_parser('metrics');q.add_argument('--records',type=Path,required=True);q.add_argument('--out',type=Path,required=True);q.add_argument('--bootstrap',type=int,default=500)
    q=sub.add_parser('contrasts');q.add_argument('--root',type=Path,required=True)
    q=sub.add_parser('changes');q.add_argument('--first',type=Path,required=True);q.add_argument('--second',type=Path,required=True);q.add_argument('--out',type=Path,required=True)
    q=sub.add_parser('mechanisms');q.add_argument('--baseline',type=Path,required=True);q.add_argument('--candidate',type=Path,required=True)
    q.add_argument('--cache',type=Path,required=True);q.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.action=='metrics':point_and_bootstrap(a.records,a.out,a.bootstrap)
    elif a.action=='contrasts':contrasts(a.root,json.loads((a.root/'manifest.json').read_text()))
    elif a.action=='changes':content_changes(a.first,a.second,a.out)
    else:mechanisms(a.baseline,a.candidate,a.cache,a.out)

if __name__=='__main__':main()
