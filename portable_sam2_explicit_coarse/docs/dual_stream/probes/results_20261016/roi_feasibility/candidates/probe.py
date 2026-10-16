"""Partial cached-candidate feasibility; CPU, no model imports."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment

OUT = Path(__file__).parent
SRC = Path('/tmp/ue_uav_experiments/20261014_ad_seed42/final_analysis')
PAIR = Path('/tmp/ue_adjacent_pair_20261016/outputs')
RISK = Path('/data/wangcheng/ue/CitySample/CitySample_5.5/data/Content/Python/visualizations/satellite_connectivity_audit_20261016/instance_issues.csv')
identity = {}
def load(path):
    identity[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())
def group(rows):
    d=defaultdict(list)
    for r in rows: d[r['image_id']].append(r)
    return d
def iou(a,b):
    a=np.array(a,dtype=float).reshape(-1,4); b=np.array(b,dtype=float).reshape(-1,4)
    inter=np.maximum(0,np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2])).prod(-1)
    return inter/np.maximum((a[:,2:]-a[:,:2]).prod(-1)[:,None]+(b[:,2:]-b[:,:2]).prod(-1)[None,:]-inter,1e-12)
def unique(boxes):
    out=[]
    for b in boxes:
        if b is None: continue
        if not out or not np.any(np.max(np.abs(np.array(out)-b),axis=1)<=1e-4): out.append(b)
    return out
def match(m,t):
    n,p=m.shape
    w=np.full((n,p+n),-1e6);w[:,p:]=0;w[:,:p]=np.where(m>=t,n+1+m,-1e6)
    a,b=linear_sum_assignment(w,maximize=True)
    return {int(x) for x,y in zip(a,b) if y<p and m[x,y]>=t}
def feasible(m,bb,slots,t):
    a,b=[np.flatnonzero(m[s]>=t) for s in slots]
    if not len(a) or not len(b):return False,False,None
    distinct=a[:,None]!=b[None,:]
    any_distinct=bool(distinct.any())
    okay=distinct & (bb[np.ix_(a,b)]<=.5)
    choices=np.argwhere(okay)
    witness=None if not len(choices) else [int(a[choices[0,0]]),int(b[choices[0,1]])]
    return any_distinct,bool(okay.any()),witness

def main():
    pairs=load(PAIR/'pairs.json'); instances=load(PAIR/'instances.json')
    inst={(x['image_id'],x['gt_slot']):x for x in instances}
    identity[str(RISK)]=hashlib.sha256(RISK.read_bytes()).hexdigest()
    risks={(int(r['index']),int(r['local_id'])) for r in csv.DictReader(RISK.open()) if int(r['components_8'])>1 or int(r['large_secondary_count'])>0 or int(r['foreign_hole_pixels'])>0}
    all_rows=[];counts={};gains=[]
    perimage=group(pairs)
    for mode in ['single_best','dual_correct']:
        stage=load(SRC/'full_stage_audit'/f'{mode}.json');sg=group(stage['instances']); dt=group(load(SRC/mode/'dt_records.json'))
        counts[mode]={'eligible_saved_rows':0,'eligible_unique':0,'union_unique':0,'final_rows':0,'final_dedup':0,'gt_best_iou_max_difference':0.}
        for im,truth in sg.items():
            truth.sort(key=lambda x:x['gt_slot']);assert [x['gt_slot'] for x in truth]==list(range(len(truth)))
            gt=[x['bbox_xyxy'] for x in truth]
            final=[[r['bbox'][0],r['bbox'][1],r['bbox'][0]+r['bbox'][2],r['bbox'][1]+r['bbox'][3]] for r in dt[im]]
            elig=[r.get('eligible_best_box') for r in truth]; e=unique(elig);u=unique(e+final);f=unique(final)
            c=counts[mode];c['eligible_saved_rows']+=len(elig);c['eligible_unique']+=len(e);c['union_unique']+=len(u);c['final_rows']+=len(final);c['final_dedup']+=len(f)
            mats={k:iou(gt,v) for k,v in [('E',e),('U',u),('F',f)]};bbs={k:iou(v,v) for k,v in [('E',e),('U',u),('F',f)]}
            c['gt_best_iou_max_difference']=max(c['gt_best_iou_max_difference'],float(np.max(np.abs(mats['E'].max(axis=1)-[r['eligible_best_iou'] for r in truth]))))
            for t in [.5,.75]:
                matching={k:match(v,t) for k,v in mats.items()}
                for p in perimage[im]:
                    slots=p['slots'];risk=any((im,truth[s]['local_id']) in risks for s in slots)
                    interior=all(inst[im,s]['area']>=1024 and all([truth[s]['bbox_xyxy'][0]>0,truth[s]['bbox_xyxy'][1]>0,truth[s]['bbox_xyxy'][2]<512,truth[s]['bbox_xyxy'][3]<512]) for s in slots)
                    row={'mode':mode,'image_id':im,'slots':slots,'physical_pair':p['physical_pair'],'threshold':t,'risk':risk,'interior_nonsmall':interior}
                    for k in mats:
                        feasible2,nms_ok,w=feasible(mats[k],bbs[k],slots,t)
                        row[k]={'global_pair_success':all(s in matching[k] for s in slots),'pair_local_feasible':feasible2,'pair_local_nms_compatible':nms_ok}
                        if k=='U' and w is not None:row['witness']={'indices':w,'boxes':[u[x] for x in w],'gt_ious':[float(mats[k][s,x]) for s,x in zip(slots,w)],'mutual_iou':float(bbs[k][w[0],w[1]])}
                    # Validate final-only reproduces previous image-wide assignment.
                    old=p['modes'][mode][str(t)]['box_pair_success']
                    assert old==row['F']['global_pair_success'],(mode,im,slots,t,old,row['F'])
                    row['final_box_failure']=not row['F']['global_pair_success']
                    row['partial_union_rerank_opportunity']=row['final_box_failure'] and row['U']['pair_local_nms_compatible']
                    row['partial_union_fixed_nms_blocked']=row['final_box_failure'] and row['U']['pair_local_feasible'] and not row['U']['pair_local_nms_compatible']
                    row['partial_union_no_distinct_geometry']=row['final_box_failure'] and not row['U']['pair_local_feasible']
                    row['already_final_pair_local_possible']=row['final_box_failure'] and row['F']['pair_local_feasible']
                    row['cached_addition_enables_pair']=row['partial_union_rerank_opportunity'] and not row['F']['pair_local_feasible']
                    row['mask_failure']=not p['modes'][mode][str(t)]['mask_pair_success']
                    row['mask_failure_with_rerank_opportunity']=row['mask_failure'] and row['partial_union_rerank_opportunity']
                    all_rows.append(row)
    strata={'all':lambda r:True,'interior_nonsmall':lambda r:r['interior_nonsmall'],'risk_flagged':lambda r:r['risk'],'risk_unflagged':lambda r:not r['risk'],'interior_nonsmall_unflagged':lambda r:r['interior_nonsmall'] and not r['risk']}
    summary={'scope':'Partial GT-selected saved candidate geometry, NOT UAV learnability or AP','candidate_counts':counts,'strata':{}}
    indicators=['final_box_failure','partial_union_rerank_opportunity','partial_union_fixed_nms_blocked','partial_union_no_distinct_geometry','already_final_pair_local_possible','cached_addition_enables_pair','mask_failure','mask_failure_with_rerank_opportunity']
    for name,fn in strata.items():
        summary['strata'][name]={}
        for mode in counts:
            summary['strata'][name][mode]={}
            for t in [.5,.75]:
                rows=[r for r in all_rows if r['mode']==mode and r['threshold']==t and fn(r)];phys=defaultdict(list)
                for r in rows:phys[tuple(r['physical_pair'])].append(r)
                result={'appearances':len(rows),'physical_pairs':len(phys),'counts':{k:sum(r[k] for r in rows) for k in indicators},'physical_pair_equal_weight_rates':{k:float(np.mean([np.mean([r[k] for r in rs]) for rs in phys.values()])) for k in indicators}}
                result['candidate_sets']={k:{v:sum(r[k][v] for r in rows) for v in ['global_pair_success','pair_local_feasible','pair_local_nms_compatible']} for k in ['F','E','U']}
                summary['strata'][name][mode][str(t)]=result
    (OUT/'pair_records.json').write_text(json.dumps(all_rows,indent=2))
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    (OUT/'source_sha256.json').write_text(json.dumps(identity,indent=2))
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
