"""Read-only stage recall audit of the sealed satellite detector; no fitting."""
import argparse
import gc
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from mmcv.ops import batched_nms
from mmdet.structures.bbox import bbox_overlaps
import ue_v1_adapter as common
from ue_s2db_probe import BASE,DATA,FROZEN_SHA

ROOT=common.ROOT
OUT=Path('/data/wangcheng/checkpoint/ue_detection_attrition_20261023')
REFERENCE=Path('/data/wangcheng/checkpoint/ue_v1_adapter_probe_20261021/test/baseline/disabled/records.json')
STAGES=['rpn_anchors','rpn_decoded_all','rpn_topk','rpn_size_filter','rpn_nms','rpn_cap',
        'roi_decoded','roi_score_filter','roi_nms','final_cap']
THRESHOLDS=(.5,.75)


def cpu(t):
    return t.detach().cpu()


def build():
    cp=torch.load(BASE,map_location='cpu',weights_only=False)
    snapshot=cp['config_snapshot']
    # port shim (lthpc->local, 2026-10-24): remap snapshot-pinned absolute
    # paths; file bytes and archive identity checks unchanged
    import json as _json, os as _os
    _s=_json.dumps(snapshot)
    _s=_s.replace('/data/wangcheng/pretrained-models/sam2/sam2_hiera_base_plus.pt',
                  _os.environ.get('SAM2_CKPT','/data2/wangcheng/remote-lthpc/sam2_hiera_base_plus.pt'))
    _s=_s.replace('/data/wangcheng/dataset/CitySample_SatUAV_TrainMinimal_v1_20261021',str(DATA))
    snapshot=_json.loads(_s)
    assert common.archive_identity(DATA)==snapshot['data_config']['archive_hashes']
    model=common._register_and_build(snapshot['model_config'])
    report=common._load_model_state(model,cp['model'],allow_nonstrict=False)
    assert not report['missing_keys'] and not report['unexpected_keys']
    for p in model.parameters():p.requires_grad_(False)
    model.eval().cuda()
    del cp
    assert common.state_digest(model)==FROZEN_SHA
    return model,snapshot


def coverage(gt,boxes):
    """Any-candidate recall plus maximum-cardinality bipartite capacity."""
    ng,nb=len(gt),len(boxes)
    best=torch.zeros(ng,device=gt.device)
    edges={t:[[],[]] for t in THRESHOLDS}
    for start in range(0,nb,2048):
        iou=bbox_overlaps(gt,boxes[start:start+2048])
        if iou.numel():best=torch.maximum(best,iou.max(1).values)
        for t,(rows,cols) in edges.items():
            rr,cc=torch.where(iou>=t)
            rows.extend(cpu(rr).tolist());cols.extend((cpu(cc)+start).tolist())
    matches={}
    for t,(rr,cc) in edges.items():
        if not ng or not nb:
            count=0
        else:
            graph=csr_matrix((np.ones(len(rr),dtype=np.int8),(rr,cc)),shape=(ng,nb))
            count=int((maximum_bipartite_matching(graph,perm_type='column')>=0).sum())
        matches[str(t)]=count
    return dict(candidates=nb,best_iou=cpu(best).tolist(),one_to_one_capacity=matches)


def trace(model,features,p):
    x=features[0];meta=p['data_samples'][0].metainfo
    assert np.allclose(meta.get('scale_factor',(1,1)),1)
    rpn=model.rpn_head
    cls,reg=rpn(x)
    priors=rpn.prior_generator.grid_priors([z.shape[-2:] for z in cls],dtype=cls[0].dtype,device=cls[0].device)
    cfg=rpn.test_cfg
    boxes0=[];scores0=[];levels0=[];topids=[];offset=0
    for lev,(c,b,pr) in enumerate(zip(cls,reg,priors)):
        logits=c[0].permute(1,2,0).reshape(-1,rpn.cls_out_channels)
        scores=logits.sigmoid() if rpn.use_sigmoid_cls else logits.softmax(-1)[:,:-1]
        assert scores.shape[1]==1
        scores=scores[:,0]
        delta=b[0].permute(1,2,0).reshape(-1,rpn.bbox_coder.encode_size)
        boxes0.append(rpn.bbox_coder.decode(pr,delta,max_shape=meta['img_shape']))
        scores0.append(scores);levels0.append(torch.full_like(scores,lev,dtype=torch.long))
        ids=torch.arange(len(scores),device=scores.device)
        n=cfg.get('nms_pre',-1)
        if 0<n<len(scores):ids=scores.sort(descending=True)[1][:n]
        topids.append(ids+offset);offset+=len(scores)
    anchors=torch.cat(priors);raw=torch.cat(boxes0);rs=torch.cat(scores0);levels=torch.cat(levels0)
    ids=torch.cat(topids);top=raw[ids];ss=rs[ids];ll=levels[ids]
    size=cfg.get('min_bbox_size',-1)
    valid=((top[:,2]-top[:,0])>size)&((top[:,3]-top[:,1])>size) if size>=0 else torch.ones(len(top),device=top.device,dtype=torch.bool)
    sized=top[valid];sized_scores=ss[valid];sized_levels=ll[valid]
    dets,keep=batched_nms(sized,sized_scores,sized_levels,cfg.nms)
    nms_boxes=dets[:,:4];nms_scores=dets[:,4]
    capped=nms_boxes[:cfg.max_per_img];capped_scores=nms_scores[:cfg.max_per_img]
    official_props=rpn.predict_by_feat(cls,reg,batch_img_metas=[meta],rescale=False)
    props=official_props[0]
    assert torch.equal(capped,props.bboxes) and torch.equal(capped_scores,props.scores),'RPN replay differs'
    head=model.roi_head
    xx=head._inject_extra_pe(x)
    official=head.predict_bbox(xx,[meta],official_props,head.test_cfg,rescale=False)[0]
    rois=torch.cat((torch.zeros((len(capped),1),device=capped.device),capped),1)
    br=head._bbox_forward(xx,rois)
    bh=head.bbox_head
    assert bh.num_classes==1 and not bh.custom_cls_channels
    raw_roi=bh._predict_by_feat_single(rois,br['cls_score'],br['bbox_pred'],meta,False,None)
    decoded=raw_roi.bboxes
    assert decoded.shape[1]==4
    fg=raw_roi.scores[:,0];bg=raw_roi.scores[:,1]
    rcfg=head.test_cfg
    passed=fg>rcfg.score_thr
    rboxes=decoded[passed];rscore=fg[passed]
    rdets,rkeep=batched_nms(rboxes,rscore,torch.zeros(len(rscore),device=rscore.device,dtype=torch.long),rcfg.nms)
    final=rdets[:rcfg.max_per_img]
    assert torch.equal(final[:,:4],official.bboxes) and torch.equal(final[:,4],official.scores),'RoI replay differs'
    assert bool((official.labels==0).all())
    stages=dict(zip(STAGES,[anchors,raw,top,sized,nms_boxes,capped,decoded,rboxes,rdets[:,:4],final[:,:4]]))
    return stages,dict(roi_foreground=fg,roi_background=bg,roi_decoded=decoded,
        rpn_pre_nms_scores=sized_scores,roi_threshold=rcfg.score_thr),official


def analyze(model,ds,index,ref,gate=False):
    batch=common.rtmdet_collate_fn([ds[index]])
    p=common.processed(model,batch)
    gt=p['data_samples'][0].gt_instances.bboxes
    assert torch.equal(cpu(gt),torch.tensor(ref['gt']['bboxes'],dtype=gt.dtype))
    features=model.extract_feat(p['inputs'])
    stages,extras,official=trace(model,features,p)
    for key in ('bboxes','scores','labels'):
        a=cpu(getattr(official,key)); b=torch.tensor(ref['dt'][key],dtype=a.dtype)
        assert torch.equal(a,b),(index,key,'baseline differs')
    if gate:
        with common.cached_extract(model,features):
            pred=model.predict(p['inputs'],p['data_samples'],rescale=True)[0].pred_instances
        assert common.serialized(common.gt_arrays(pred))==ref['dt'],'Full original mask prediction differs'
    stage={name:coverage(gt,b) for name,b in stages.items()}
    masks=p['data_samples'][0].gt_instances.masks.to_ndarray()
    area=masks.reshape(len(masks),-1).sum(1)
    bboxes=cpu(gt).numpy();bbox_area=(bboxes[:,2]-bboxes[:,0])*(bboxes[:,3]-bboxes[:,1])
    # Score support is evaluated on decoded boxes before the actual score filter.
    maxfg={}
    for t in THRESHOLDS:
        overlaps=bbox_overlaps(gt,extras['roi_decoded'])
        score=extras['roi_foreground'][None].expand(len(gt),-1)
        vals=score.masked_fill(overlaps<t,-1).max(1).values if score.shape[1] else gt.new_full((len(gt),),-1)
        maxfg[str(t)]=cpu(vals).tolist()
    final_iou=bbox_overlaps(stages['final_cap'],gt)
    dt_best=final_iou.max(1).values if len(gt) else gt.new_zeros(len(stages['final_cap']))
    row=dict(image_id=ds.samples[index]['image_id'],dataset_index=index,gt_count=len(gt),
        mask_area=area.tolist(),bbox_area=bbox_area.tolist(),stages=stage,
        max_foreground_score_among_good_decoded_boxes=maxfg,
        final_boxes=cpu(official.bboxes).tolist(),final_scores=cpu(official.scores).tolist(),
        final_dt_best_gt_iou=cpu(dt_best).tolist(),final_exact_against_baseline=True)
    # RPN and filtering subsets lose coverage monotonically. Regression may recover.
    monotone=[('rpn_decoded_all','rpn_topk'),('rpn_topk','rpn_size_filter'),('rpn_size_filter','rpn_nms'),
              ('rpn_nms','rpn_cap'),('roi_decoded','roi_score_filter'),('roi_score_filter','roi_nms'),('roi_nms','final_cap')]
    for left,right in monotone:
        for t in THRESHOLDS:
            l=np.array(stage[left]['best_iou'])>=t;r=np.array(stage[right]['best_iou'])>=t
            assert not (r&~l).any(),(left,right)
            assert stage[right]['one_to_one_capacity'][str(t)]<=stage[left]['one_to_one_capacity'][str(t)]
    return row


def summarize(rows,out):
    groups={'all':lambda r,i:True,'mask_small':lambda r,i:r['mask_area'][i]<1024,
        'mask_medium':lambda r,i:1024<=r['mask_area'][i]<9216,'mask_large':lambda r,i:r['mask_area'][i]>=9216,
        'density_le50':lambda r,i:r['gt_count']<=50,'density_51to100':lambda r,i:50<r['gt_count']<=100,
        'density_gt100':lambda r,i:r['gt_count']>100}
    result={};transitions={}
    for group,select in groups.items():
        result[group]={}
        for t in THRESHOLDS:
            totals={s:0 for s in STAGES};ng=0;edge={f'{a}->{b}':dict(lost=0,recovered=0) for a,b in zip(STAGES,STAGES[1:])}
            final_miss_last_loss={s:0 for s in STAGES};final_miss_last_loss['never_covered']=0
            for r in rows:
                ix=np.array([select(r,i) for i in range(r['gt_count'])]);ng+=int(ix.sum())
                flags={s:np.array(r['stages'][s]['best_iou'])>=t for s in STAGES}
                for s in STAGES:totals[s]+=int((flags[s]&ix).sum())
                for a,b in zip(STAGES,STAGES[1:]):
                    edge[f'{a}->{b}']['lost']+=int((flags[a]&~flags[b]&ix).sum())
                    edge[f'{a}->{b}']['recovered']+=int((~flags[a]&flags[b]&ix).sum())
                for i in np.flatnonzero(ix&~flags['final_cap']):
                    previous=[j for j,s in enumerate(STAGES) if flags[s][i]]
                    key=STAGES[max(previous)+1] if previous else 'never_covered'
                    final_miss_last_loss[key]+=1
            assert sum(final_miss_last_loss.values())==ng-totals['final_cap']
            result[group][str(t)]=dict(gt=ng,any_candidate_covered=totals,
                recall={s:v/ng if ng else None for s,v in totals.items()},
                final_uncovered_last_transition=final_miss_last_loss)
            transitions[group+'/'+str(t)]=edge
    capacity={str(t):{s:sum(r['stages'][s]['one_to_one_capacity'][str(t)] for r in rows) for s in STAGES} for t in THRESHOLDS}
    dt=sum(r['stages']['final_cap']['candidates'] for r in rows)
    final_proxy={str(t):dict(dt=dt,no_gt_overlap=sum(sum(x<t for x in r['final_dt_best_gt_iou']) for r in rows),
        unmatched_capacity=dt-capacity[str(t)]['final_cap']) for t in THRESHOLDS}
    common.dump(out/'summary.json',dict(images=len(rows),groups=result,transitions=transitions,
        one_to_one_capacity=capacity,final_dt_proxies=final_proxy))
    lines=['# 检测阶段漏失定位','',f'固定EXP-0017 E73，{len(rows)}张开发test图；FP32，无更新。',
        '所有最终框/类别/分数逐值复现原459图基座记录。按阶段统计候选存在与一对一最大匹配容量，均不是COCO AP。','',
        '|阶段|候选总数|IoU≥.5任意覆盖GT|.5一对一容量|IoU≥.75任意覆盖GT|',
        '|---|---:|---:|---:|---:|']
    for s in STAGES:
        lines.append(f"|{s}|{sum(r['stages'][s]['candidates'] for r in rows)}|{result['all']['0.5']['any_candidate_covered'][s]}|{capacity['0.5'][s]}|{result['all']['0.75']['any_candidate_covered'][s]}|")
    lines.extend(['','## 解释范围','',
        'rpn_anchors为未经回归的全部锚框；rpn_decoded_all为全部锚框回归结果、未做每层top-k。其后为实际top-k、尺寸过滤、按层NMS、RPN上限。',
        'roi_decoded为真实RPN候选经RoI回归后的框、未过滤分数；roi_score_filter严格使用原foreground score>0.05规则，不引入argmax分类过滤。',
        '相邻阶段丢失/恢复均记录。回归可以恢复覆盖；final_uncovered_last_transition按最后一次从有覆盖到无覆盖归类，不等于干预该步骤就能恢复最终AP。',
        '一对一容量使用稀疏图最大基数匹配，仅汇总数量；不报告不唯一匹配身份的尺寸归因。分组召回基于任意候选覆盖，公开分母。',
        '尺寸组按真实mask面积，区别于旧canonical evaluator的bbox面积；密度组预先固定<=50、51–100、>100。',
        '最终DT无GT IoU≥阈值及最大匹配容量剩余是误检/重复/定位代理，不是按分数排序的COCO FP分类。',
        '未改变阈值、NMS、候选预算，不据本开发test结果选模型；没有新增训练，不能据诊断直接宣布UAV检测收益。'])
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,default=OUT);ap.add_argument('--worker',action='store_true');ap.add_argument('--gate',action='store_true')
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    if not args.worker:
        # Sequential measured admission; other experiment workers remain live.
        for gate in (True,False):
            need=5000 if gate else json.loads((args.out/'gate/resource.json').read_text())['peak_reserved_MiB']+2048
            while True:
                raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
                available=sorted([(int(b),int(a)) for a,b in (l.split(',') for l in raw.splitlines())],reverse=True)
                if available[0][0]>=need:break
                common.dump(args.out/'state.json',dict(stage='waiting_shared_capacity',required_MiB=need,free=available,pid=os.getpid()))
                time.sleep(10)
            dest=args.out/'gate' if gate else args.out
            cmd=[sys.executable,__file__,'--worker','--out',str(dest)]+(['--gate'] if gate else [])
            common.dump(args.out/'state.json',dict(stage='gate' if gate else 'running',gpu=available[0][1],owner_pid=os.getpid()))
            with (args.out/('gate.log' if gate else 'worker.log')).open('a') as log:
                rc=subprocess.call(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(available[0][1])),stdout=log,stderr=subprocess.STDOUT)
            if rc:
                common.dump(args.out/'state.json',dict(stage='failed',returncode=rc,gate=gate));raise RuntimeError(cmd)
        common.dump(args.out/'state.json',dict(stage='complete',owner_pid=os.getpid()))
        return
    common.seed(44)
    ds=common.MinimalSatelliteDataset(DATA,'test',augment=False)
    ref=json.loads(REFERENCE.read_text());assert len(ds)==len(ref)==459
    indices=[int(np.argmax([len(s['annotations']) for s in ds.samples]))] if args.gate else list(range(len(ds)))
    model,snapshot=build()
    sources={str(Path(__file__).resolve()):common.file_sha(__file__)}
    frozen=json.loads(Path('/data/wangcheng/checkpoint/ue_s2db_probe_20261022/manifest.json').read_text())['source_hashes']
    for rel,h in frozen.items():
        path=ROOT/rel;assert common.file_sha(path)==h
        sources[str(path)]=h
    for method in (model.rpn_head._predict_by_feat_single,model.roi_head.predict_bbox,model.roi_head._bbox_forward,model.roi_head.bbox_head._predict_by_feat_single,model.rpn_head.bbox_coder.decode,model.rpn_head.prior_generator.grid_priors):
        path=Path(inspect.getfile(method));sources[str(path)]=common.file_sha(path)
    synthetic=torch.tensor([[0.,0.,10.,10.],[0.,0.,10.,10.]],device='cuda')
    sanity=coverage(synthetic,synthetic[:1])
    assert sum(x>=.5 for x in sanity['best_iou'])==2 and sanity['one_to_one_capacity']['0.5']==1
    assert coverage(synthetic,synthetic[:0])['one_to_one_capacity']['0.5']==0
    for obj in (type(model),type(model.rpn_head),type(model.roi_head),type(model.roi_head.bbox_head),batched_nms):
        path=Path(inspect.getfile(obj));sources[str(path)]=common.file_sha(path)
    common.dump(args.out/'plan.json',dict(checkpoint=str(BASE),checkpoint_sha256=common.file_sha(BASE),
        archive_hashes=common.archive_identity(DATA),source_hashes=sources,reference=str(REFERENCE),reference_sha256=common.file_sha(REFERENCE),
        split='test',test_role='development',indices=indices,thresholds=list(THRESHOLDS),stages=STAGES,
        test_cfg=snapshot['model_config']['test_cfg'],precision='FP32',optimizer_steps=0,
        max_matching='Maximum-cardinality sparse bipartite matching; any-candidate support separately; not COCO AP'))
    torch.cuda.reset_peak_memory_stats();rows=[]
    with torch.inference_mode():
        for i in indices:
            assert ds.samples[i]['image_id']==ref[i]['image_id']
            rows.append(analyze(model,ds,i,ref[i],args.gate))
            common.dump(args.out/'progress.json',dict(images=len(rows),last_index=i))
            if len(rows)%25==0 or args.gate:print('TRACED',len(rows),i,flush=True)
    assert common.state_digest(model)==FROZEN_SHA
    assert all(p.grad is None for p in model.parameters())
    assert all(common.file_sha(Path(p))==h for p,h in sources.items())
    common.dump(args.out/'records.json',rows);summarize(rows,args.out)
    common.dump(args.out/'resource.json',dict(peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20))
    common.dump(args.out/'complete.json',dict(complete=True,gate_only=args.gate,images=len(rows),
        detector_replay_exact=True,baseline_final_boxes_scores_labels_exact=True,gate_full_masks_exact=args.gate,
        satellite_state_unchanged=True,satellite_state_sha256=FROZEN_SHA,source_unchanged=True,optimizer_steps=0,
        records_sha256=common.file_sha(args.out/'records.json')))

if __name__=='__main__':
    main()
