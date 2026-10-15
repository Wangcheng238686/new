"""Independent CPU validation/aggregation of the fixed GT prompt intervention.
No model execution. One preregistered comparison; no AP or training claims.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path('/tmp/ue_uav_experiments/20261014_ad_seed42/final_analysis')

def read(p): return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()
def require(condition,message):
    if not condition: raise AssertionError(message)
def identity(r): return tuple(r[k] for k in ('image_id','gt_slot','global_id','bbox_area','roi_level'))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-dir',type=Path,default=ROOT/'gt_prompt_oracle/full');ap.add_argument('--output',type=Path);args=ap.parse_args()
    d=args.run_dir;m=read(d/'run_manifest.json');checks=read(d/'invariance_checks.json')
    require(m['variants']==['baseline','joint_gt_prompt'],'Fixed two cells only')
    require(m['training'] is False and m['optimizer_created'] is False and m['parameter_versions_unchanged'] is True,'Frozen run')
    require(m['strict'] and not m['load_report']['missing_keys'] and not m['load_report']['unexpected_keys'],'Strict EMA load')
    require(m['weights']=='ema','EMA weights')
    require(m['oracle_logit_amplitude']==4,'Fixed preregistered oracle amplitude')
    require(m['manifest_sha256']=='495841b2174242c87661e6b40dba2ddcca13d64b13798116efc8c933c679ca59','Frozen dataset version')
    ids=m['image_ids'];require(len(ids)==len(set(ids))==m['processed_images'],'Unique processed image IDs')
    complete=m['complete'];require(complete==(len(ids)==153),'Completion matches scope')
    require(m['mask_thr_binary']==.4 and abs(m['mask_logit_threshold']-np.log(.4/.6))<1e-12,'Deployment threshold')
    require(m['dense_shape_alpha']==.32814547419548035 and m['dense_detach_input'] is True,'Original shape gate/detach')
    require(m['explicit_prompt_mode']=='points_box_dense' and m['shape_fusion_type']=='roi_only','Original prompt contract')
    for key,path in [('checkpoint_sha256',Path(m['checkpoint'])),('script_sha256',Path('/tmp/ue_gt_prompt_oracle_probe.py')),('helper_script_sha256',Path('/tmp/ue_rgb_content_probe.py')),('trace_helper_sha256',Path('/tmp/ue_p2_prompt_trace.py')),('preregistration_sha256',d.parent/'PREREGISTRATION.md'),('manifest_sha256',ROOT.parent/'dataset/manifest.json')]:
        require(sha(path)==m[key],f'SHA mismatch {key}')
    rows={k:read(d/(k+'.json')) for k in m['variants']};b=rows['baseline'];v=rows['joint_gt_prompt'];n=len(b)
    require(n==len(v)==m['ground_truth_instances'],'Row counts')
    require([identity(x) for x in b]==[identity(x) for x in v],'Paired GT order')
    require(len(set((x['image_id'],x['gt_slot']) for x in b))==n,'Unique GT appearances')
    require([x['image_id'] for x in checks]==ids,'Check/image order')
    prior=read(ROOT/'p2_prompt_trace/full/baseline.json');old={(x['image_id'],x['gt_slot']):x for x in prior}
    for x,y in zip(b,v):
        z=old[(x['image_id'],x['gt_slot'])]
        require(identity(x)==identity(z) and x['gt_iou']==z['gt_iou'],'Exact old baseline IoU/identity')
        for t in (x,y):
            require(np.isfinite(t['gt_iou']) and 0<=t['gt_iou']<=1,'IoU range')
            require(len(t['point_valid'])==len(t['point_correct'])==len(t['point_labels'])==len(t['point_coords'])==len(t['point_local_yx'])==4,'Four point slots')
            require(0<=t['coarse_iou']<=1 and np.isfinite(t['coarse_bce']) and t['coarse_bce']>=0,'Coarse quality legal')
            require(t['sparse_token_count'] in (2,6),'Actual sparse token count')
            require(all(a in (-1,0,1) and bool(q)==(a>=0) for a,q in zip(t['point_labels'],t['point_valid'])),'Point label validity')
            require(all(len(p)==2 and all(np.isfinite(z) for z in p) for p in t['point_coords']),'Finite image point coordinates')
            require(all(len(p)==2 and all(isinstance(z,int) and 0<=z<64 for z in p) for p in t['point_local_yx']),'Local point coordinates')
            require(t['dense_support_cells']==x['dense_support_cells'],'Support unchanged')
            require(all((not c) or a for c,a in zip(t['point_correct'],t['point_valid'])),'Correct only valid slots')
            if t['dense_support_cells']==0: require(t['dense_no_mask_delta_norm']==0,'No-support dense equals pretrained base')
            for stat in t['stages'].values():
                require(all(np.isfinite(stat[k]) for k in ('rms','mae','relative_l2')),'Finite change statistics')
        require(y['oracle_input_exact'] and x['oracle_input_exact'] is False,'Exact intervention tensor; baseline not applicable')
        fg=x['coarse_target_fg_pixels'];require(isinstance(fg,int) and 0<=fg<=4096 and y['coarse_target_fg_pixels']==fg,'GT target area')
        require(y['coarse_iou']==(1 if fg else 0),'Oracle binary shape exact (empty union convention)')
        require(y['predicted_coarse_iou']==x['predicted_coarse_iou']==x['coarse_iou'],'Predicted quality stays baseline')
        for t in (x,y):
            q=t['original_coarse_logit_quantiles'];require(len(q)==5 and all(np.isfinite(z) for z in q) and q==sorted(q),'Ordered finite original logit quantiles')
            require(np.isfinite(t['original_coarse_logit_mean']) and np.isfinite(t['original_coarse_logit_std']) and t['original_coarse_logit_std']>=0,'Original logit moments')
        for key in ('original_coarse_logit_quantiles','original_coarse_logit_mean','original_coarse_logit_std'): require(x[key]==y[key],'Same original logits diagnostics')
        require(not y['stages']['roi']['strict_changed'] and y['stages']['roi']['rms']==0,'Fixed RoI features')
        require(x['paste_xor_pixels']==0 and x['point_changed_count']==0,'Baseline self comparison')
        for stat in x['stages'].values(): require(not stat['strict_changed'] and stat['rms']==stat['mae']==stat['relative_l2']==0,'Baseline zero deltas')
        for key in ('dense_support_cells','coarse_iou','coarse_bce','point_valid','point_correct','point_labels','point_coords','point_local_yx','sparse_token_count'):
            require(x[key]==z[key],f'Exact historical baseline {key}')
    require(set(x['image_id'] for x in b)==set(ids),'All image records')
    counts=np.array([sum(x['image_id']==i for x in b) for i in ids]);require((counts>0).all(),'Nonempty images')
    require([x['image_id'] for x in b]==[i for i,c in zip(ids,counts) for _ in range(c)],'Contiguous image rows')
    for iid,c in zip(ids,counts): require([x['gt_slot'] for x in b if x['image_id']==iid]==list(range(c)),'GT slot order')
    # Runner check keys are verified explicitly below after implementation review.
    for c,count in zip(checks,counts):
        chunk=m['mask_roi_chunk_size'] or int(count)
        expected_chunks=[min(chunk,int(count)-i) for i in range(0,int(count),chunk)]
        require(c['chunk_sizes']==expected_chunks,'Exact production chunk slices')
        for key in ('features_unchanged','geometry_rgb_restored','hooks_removed','baseline_endpoint_iou_exact','original_coarse_unchanged','oracle_input_exact','roi_unchanged'): require(c[key] is True,f'Explicit invariant {key}')
        require(c['gt_slots_in_order']==list(range(int(count))),'Exact declared GT order')
        bounds=np.r_[0,np.cumsum(expected_chunks)].tolist();require(c['oracle_target_ranges']==[[a,b] for a,b in zip(bounds[:-1],bounds[1:])],'Exact target chunk ranges')
        for key in ('observer_smoke_pixel_exact','sham_smoke_pixel_exact'): require(c[key] is (None if complete else True),f'Scope-dependent smoke check {key}')
    if complete: require(n==4496 and len(set(x['global_id'] for x in b))==237,'Full scope')
    area=np.array([x['bbox_area'] for x in b]);level=np.array([x['roi_level'] for x in b]);support=np.array([x['dense_support_cells'] for x in b]);groups={'all':np.ones(n,dtype=bool),'small':area<1024,'medium':(area>=1024)&(area<9216),'large':area>=9216,'P2':level==0,'nonP2':level!=0,'dense_support_zero':support==0,'dense_support_nonzero':support>0}
    def aggregate(rr,sel):
        a=[r for r,s in zip(rr,sel) if s];k=len(a)
        if not k:return {'n':0}
        slots={}
        for j,slot in enumerate(('p1','p2','n1','n2')):
            valid=sum(r['point_valid'][j] for r in a);correct=sum(r['point_correct'][j] for r in a)
            slots[slot]={'valid':valid,'correct':correct,'accuracy':correct/valid if valid else None}
        return dict(n=k,final_iou=float(np.mean([r['gt_iou'] for r in a])),coarse_iou_diagnostic=float(np.mean([r['coarse_iou'] for r in a])),point_slots=slots,point_changed_rois=sum(r['point_changed_count']>0 for r in a),point_changed_slots=sum(r['point_changed_count'] for r in a),paste_changed_rois=sum(r['paste_xor_pixels']>0 for r in a),mean_paste_xor_pixels=float(np.mean([r['paste_xor_pixels'] for r in a])),dense_support_zero_count=sum(r['dense_support_cells']==0 for r in a),stages={s:dict(strict_changed_rois=sum(r['stages'][s]['strict_changed'] for r in a),rms_gt1e6_rois=sum(r['stages'][s]['changed_gt1e6'] for r in a),mean_rms=float(np.mean([r['stages'][s]['rms'] for r in a]))) for s in a[0]['stages']})
    delta=np.array([y['gt_iou']-x['gt_iou'] for x,y in zip(b,v)]);starts=np.r_[0,np.cumsum(counts)[:-1]];sums=np.add.reduceat(delta,starts);require(abs(sums.sum()/n-delta.mean())<1e-12,'Image grouped sums')
    rng=np.random.default_rng(1760);draw=rng.integers(0,len(ids),(1000,len(ids)));boot=sums[draw].sum(1)/counts[draw].sum(1);ci=np.quantile(boot,[.025,.975]).tolist()
    buildings=sorted(set(x['global_id'] for x in b));building=[dict(global_id=g,n=sum(x['global_id']==g for x in b),delta=float(delta[np.array([x['global_id']==g for x in b])].mean())) for g in buildings]
    bygroup={g:dict(n=int(s.sum()),mean=float(delta[s].mean()) if s.any() else None,improved_gt002=int((delta[s]>.002).sum()),worse_gt002=int((delta[s]<-.002).sum())) for g,s in groups.items()}
    result=dict(scope=dict(images=len(ids),appearances=n,unique_buildings=len(buildings),complete=complete,current_development_subset=True),source_files_sha256={str(p):sha(p) for p in [d/'baseline.json',d/'joint_gt_prompt.json',d/'run_manifest.json',d/'invariance_checks.json',Path(__file__)]},validation='PASS',invariant_counts=dict(images_checked=len(checks),multi_chunk_images=sum(len(c['chunk_sizes'])>1 for c in checks),total_chunks=sum(len(c['chunk_sizes']) for c in checks),support_zero_rois=int((support==0).sum())),target_shape_diagnostics=dict(empty_count=sum(x['coarse_target_fg_pixels']==0 for x in b),full_count=sum(x['coarse_target_fg_pixels']==4096 for x in b)),original_logit_diagnostics=dict(mean_of_per_roi_quantiles=np.mean([x['original_coarse_logit_quantiles'] for x in b],axis=0).tolist(),mean_of_per_roi_logit_std=float(np.mean([x['original_coarse_logit_std'] for x in b])),note='Descriptive original-logit distribution only; fixed oracle amplitude +/-4 was preregistered, no amplitude tuning.'),main_comparison=dict(name='joint_gt_prompt_minus_baseline',baseline_mean=float(np.mean([x['gt_iou'] for x in b])),oracle_mean=float(np.mean([x['gt_iou'] for x in v])),delta_mean=float(delta.mean()),image_paired_percentile95=ci,gate=dict(mean_minimum=.005,ci_lower_strictly_positive=True,status=('PASS' if delta.mean()>=.005 and ci[0]>0 else 'FAIL') if complete else 'NOT_EVALUATED_SMOKE',passed=bool(delta.mean()>=.005 and ci[0]>0) if complete else None),building_equal_weight_delta=float(np.mean([x['delta'] for x in building])),by_group=bygroup),groups={name:{g:aggregate(rr,s) for g,s in groups.items()} for name,rr in rows.items()},building_effects=building,bootstrap=dict(seed=1760,resamples=1000,method='Image paired sum/count. Exploratory; does not cover spatial, selection or seed uncertainty.'),interpretation_limits='GT logits +/-4 are a frozen conditional diagnostic, not a mathematical upper bound. Joint point+dense replacement does not isolate either modality. Finite logits, point mining, frozen PE, distribution shift, GT boxes, existing alpha and support may limit response. No AP, UAV-attainable gain or detach-training benefit claim.')
    output=args.output or d.parent/('summary.json' if complete else 'smoke_summary.json');output.write_text(json.dumps(result,indent=2));print(json.dumps(result['main_comparison'],indent=2));print('SAVED',output)
if __name__=='__main__':main()
