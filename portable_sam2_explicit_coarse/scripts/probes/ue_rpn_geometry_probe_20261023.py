"""Geometry-to-RPN objectness probe; sealed satellite weights never update."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
import numpy as np
import torch
from torch.utils.data import DataLoader
import ue_geometry_mechanism_probe as g
from ue_geometry_mechanism_probe import (ROOT,seed,dump,file_sha,MinimalSatelliteDataset,archive_identity,
    processed,point_folder,state_digest,gt_arrays,serialized,rtmdet_collate_fn,_register_and_build,_load_model_state,
    build_coco_gt_and_dt,run_coco_eval,tensor_digest,atomic_save,rng_state,restore_rng,payload,constants,bind)
from rsprompter.uav.rpn_geometry_probe import RPNGeometryAdapter
from ue_detection_attrition_probe_20261023 import coverage,trace
BASE=g.BASE;DATA=g.DATA;FROZEN_SHA=g.FROZEN_SHA
STUDY=Path('/data/wangcheng/checkpoint/ue_rpn_geometry_20261023')
SOURCES=['scripts/probes/ue_rpn_geometry_probe_20261023.py','rsprompter/uav/rpn_geometry_probe.py',
    'scripts/probes/ue_geometry_mechanism_probe.py','rsprompter/uav/native_geometry_probe.py',
    'scripts/probes/ue_detection_attrition_probe_20261023.py','scripts/probes/ue_v1_adapter.py',
    'data/ue_minimal_dataset.py','data/whu_instance_dataset.py']


def make_model(args):
    cp=torch.load(args.base,map_location='cpu',weights_only=False);snapshot=cp['config_snapshot']
    assert archive_identity(args.data_root)==snapshot['data_config']['archive_hashes']
    model=_register_and_build(snapshot['model_config'])
    report=_load_model_state(model,cp['model'],allow_nonstrict=False);del cp
    assert not report['missing_keys'] and not report['unexpected_keys']
    for p in model.parameters():p.requires_grad_(False)
    model.eval().cuda();assert state_digest(model)==FROZEN_SHA
    seed(45)
    adapter=RPNGeometryAdapter(model.rpn_head.rpn_cls.in_channels,model.rpn_head.rpn_cls.out_channels).cuda()
    if args.init:
        cp=torch.load(args.init,map_location='cpu',weights_only=False)
        adapter.load_state_dict(cp['adapter'],strict=True);assert tensor_digest(adapter.state_dict())==cp['fingerprint']
    adapter.attach(model);seed(44)
    return model,adapter,snapshot


def contract(args,snapshot,adapter):
    return dict(study='native_geometry_rpn_objectness_v1',content=args.content,
        base_sha256=file_sha(args.base),satellite_state_sha256=FROZEN_SHA,archive_hashes=archive_identity(args.data_root),
        source_hashes={p:file_sha(ROOT/p) for p in SOURCES},initial_fingerprint=tensor_digest(adapter.state_dict()),
        init_file_sha256=file_sha(args.init),constants_sha256=file_sha(args.constants),constant_fields=constants(args).tolist(),
        base_config_snapshot=snapshot,loss='original RPN anchor classification only; original assigner/sampler',
        residual='2*tanh(alpha)*tanh(head)*support, alpha=0 initially',sampling='nearest native floor-bin at actual integer anchor center',
        frozen='complete satellite/RPN/box regression/RoI/mask/UDPR',optimizer='AdamW',weight_decay=.01,clip=.5,
        epochs=10,updates=2670,batch=1,accum=8,lr=1e-4,warmup=50,cosine_min_factor=.1,
        initialization_seed=45,data_seed=44,epoch_seed_formula='1044000+epoch',amp=True,amp_init_scale=512,
        ema=False,selection='E10 last',eval_precision='fp32',max_dets=[1,10,250],
        score_path='Residual enters original RPN forward outputs before sigmoid/topk/NMS/postcap; no score restoration',
        args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})


def train(args, model, adapter, snapshot):
    ds = MinimalSatelliteDataset(args.data_root, 'train', augment=True)
    assert len(ds) == 2142 and not (args.out/'last.pth').exists()
    c = contract(args, snapshot, adapter)
    initial = {k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()}
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-4, weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda', init_scale=512)
    gen = torch.Generator().manual_seed(44)
    start_epoch = step = 0
    history = args.out/'train.jsonl'
    if args.resume_from:
        parent = torch.load(args.resume_from, map_location='cpu', weights_only=False)
        # Compare the complete mathematical/data/source contract, excluding only runtime paths.
        for key in c:
            if key != 'args': assert parent['contract'][key] == c[key], key
        adapter.load_state_dict(parent['adapter'], strict=True)
        optimizer.load_state_dict(parent['optimizer']); scaler.load_state_dict(parent['scaler'])
        start_epoch, step = int(parent['epoch']), int(parent['step'])
        assert step == start_epoch*267
        restore_rng(parent['rng_state'], gen)
        rows = [json.loads(s) for s in history.read_text().splitlines()]
        assert len(rows) == step and [r['update'] for r in rows] == list(range(1, step+1))
        dump(args.out/'resume_identity.json', dict(parent=str(args.resume_from),
             sha256=file_sha(args.resume_from), epoch=start_epoch, update=step, all_rng_restored=True))
        del parent
    else:
        assert not history.exists(), 'Refuse overwriting training history'
    dump(args.out/'frozen_training_contract.json', c)
    for path in SOURCES:
        dest = args.out/'source'/path; dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((ROOT/path).read_bytes())
    model.eval(); adapter.train()
    dump(args.out/'progress.json', dict(stage='train', epoch=start_epoch, update=step, pid=os.getpid()))
    with history.open('a' if args.resume_from else 'w') as log:
        for epoch in range(start_epoch+1, 11):
            epoch_seed = 1044000+epoch
            seed(epoch_seed); gen.manual_seed(epoch_seed)
            loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=2,
                                collate_fn=rtmdet_collate_fn, generator=gen)
            optimizer.zero_grad(set_to_none=True)
            started = time.time(); ids = []; total_loss = 0.
            for i, batch in enumerate(loader):
                if i >= 2136: break
                folder = Path(batch['img_metas'][0]['filename']).parent.parent
                ids.append(folder.name)
                sample = processed(model, batch, True)
                data = payload(args, folder, 'correct' if args.content == 'geometry' else args.content)
                with torch.autocast('cuda', dtype=torch.float16):
                    features = bind(model, adapter, sample, data)
                    outputs = model.rpn_head(features[0])
                    losses = model.rpn_head.loss_by_feat(*outputs, batch_gt_instances=[s.gt_instances for s in sample['data_samples']], batch_img_metas=[s.metainfo for s in sample['data_samples']])
                    loss = sum(losses['loss_rpn_cls'])
                assert torch.isfinite(loss)
                scaler.scale(loss/8).backward()
                stats = {'levels': list(adapter.stats)}; adapter.clear()
                total_loss += float(loss.detach())
                if (i+1) % 8 == 0:
                    step += 1
                    lr = 1e-4*min(1., step/50)*(.1+.9*.5*(1+np.cos(np.pi*step/2670)))
                    for group in optimizer.param_groups: group['lr'] = lr
                    scaler.unscale_(optimizer)
                    norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), .5)
                    assert torch.isfinite(norm)
                    scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
                    row = dict(epoch=epoch, update=step, images=i+1, loss=float(loss.detach()), lr=lr,
                        grad_norm=float(norm), amp_scale=scaler.get_scale(), latest_forward=stats, epoch_seed=epoch_seed)
                    log.write(json.dumps(row)+'\n'); log.flush()
                    if step % 10 == 0: print('TRAIN', json.dumps(row), flush=True)
            assert len(ids) == 2136 and step == epoch*267
            assert all(p.grad is None for p in model.parameters()) and state_digest(model) == FROZEN_SHA
            dump(args.out/f'epoch{epoch}_summary.json', dict(epoch=epoch, update=step,
                seconds=time.time()-started, microbatches=len(ids), loss_microbatch_mean=total_loss/len(ids),
                sample_order=ids, sample_order_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                satellite_state_sha256=FROZEN_SHA))
            cp = dict(adapter=adapter.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                      epoch=epoch, step=step, contract=c, rng_state=rng_state(gen))
            atomic_save(cp, args.out/f'epoch{epoch}.pth')
            dump(args.out/'progress.json', dict(stage='train', epoch=epoch, update=step, pid=os.getpid()))
            print('EPOCH', epoch, time.time()-started, flush=True)
    assert step == 2670
    dump(args.out/'parameter_changes.json', dict(satellite_unchanged=True, satellite_state_sha256=FROZEN_SHA,
        adapter_changes={k: float((v.detach().cpu()-initial[k]).float().norm()) for k, v in adapter.state_dict().items()}))
    # Last includes optimizer/scaler/RNG too, so it is also a complete recovery artifact.
    atomic_save(cp, args.out/'last.pth')


def topk_boxes(rpn,outputs,meta):
    cls,reg=outputs
    priors=rpn.prior_generator.grid_priors([x.shape[-2:] for x in cls],dtype=cls[0].dtype,device=cls[0].device)
    pp=[];dd=[]
    for c,b,prior in zip(cls,reg,priors):
        score=c[0].permute(1,2,0).reshape(-1).sigmoid()
        delta=b[0].permute(1,2,0).reshape(-1,rpn.bbox_coder.encode_size)
        n=rpn.test_cfg.nms_pre
        ids=score.sort(descending=True)[1][:n] if 0<n<len(score) else torch.arange(len(score),device=score.device)
        pp.append(prior[ids]);dd.append(delta[ids])
    return rpn.bbox_coder.decode(torch.cat(pp),torch.cat(dd),max_shape=meta['img_shape'])


def evaluate(args,model,adapter):
    adapter.eval();model.eval();adapter.enabled=args.mode!='disabled'
    if args.nms_pre!=1000:
        assert args.mode=='disabled','Budget diagnostic is satellite only'
        model.rpn_head.test_cfg.nms_pre=args.nms_pre
    ds=MinimalSatelliteDataset(args.data_root,args.eval_split,augment=False)
    loader=DataLoader(ds,batch_size=1,shuffle=False,num_workers=2,collate_fn=rtmdet_collate_fn)
    gt=[];dt=[];metas=[];rows=[]
    before=state_digest(model)
    with torch.inference_mode():
        for i,batch in enumerate(loader):
            if args.max_images and i>=args.max_images:break
            sample=processed(model,batch)
            if adapter.enabled:
                features=bind(model,adapter,sample,payload(args,point_folder(ds,i,args.data_root),args.mode))
            else:features=model.extract_feat(sample['inputs'])
            meta=sample['data_samples'][0].metainfo
            boxes_by_stage,_,bbox_result=trace(model,features,sample)
            result=model.roi_head.predict_mask(model.roi_head._inject_extra_pe(features[0]),[meta],[bbox_result],rescale=True,
                image_embeddings=features[1],image_positional_embeddings=features[2],high_res_features=features[3])[0]
            ga,da=gt_arrays(sample['data_samples'][0].gt_instances),gt_arrays(result)
            boxes=sample['data_samples'][0].gt_instances.bboxes
            stages={name:coverage(boxes,b) for name,b in boxes_by_stage.items()}
            gt.append(ga);dt.append(da);metas.append(batch['img_metas'][0])
            rows.append(dict(image_id=ds.samples[i]['image_id'],gt=serialized(ga),dt=serialized(da),stages=stages))
            adapter.clear()
            if (i+1)%25==0:print('EVAL',args.mode,i+1,flush=True)
    assert state_digest(model)==before==FROZEN_SHA
    cg,cd=build_coco_gt_and_dt(gt,dt,metas,score_key='scores');metrics={}
    for kind in ('segm','bbox'):metrics.update(run_coco_eval(cg,cd,kind,max_dets=[1,10,250]))
    dump(args.out/'records.json',rows);dump(args.out/'metrics.json',metrics)
    dump(args.out/'evaluation_identity.json',dict(images=len(rows),split=args.eval_split,mode=args.mode,
        checkpoint_sha256=file_sha(args.adapter) if args.adapter else None,base_sha256=file_sha(args.base),
        nms_pre=args.nms_pre,rpn_postcap=model.rpn_head.test_cfg.max_per_img,max_dets=[1,10,250],
        inference_precision='fp32',satellite_state_unchanged=True,source_hashes={p:file_sha(ROOT/p) for p in SOURCES},
        constants_sha256=file_sha(args.constants),detector_outputs_may_change=True))


def resume_proof(args,model,adapter,initial):
    from torch.utils.data import Subset
    ds=MinimalSatelliteDataset(args.data_root,'train',augment=True)
    small=Subset(ds,list(range(8)))
    optimizer=torch.optim.AdamW(adapter.parameters(),lr=1e-4,weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda',init_scale=512)
    seed(1044001);gen=torch.Generator().manual_seed(44)
    def mini_epoch():
        optimizer.zero_grad(set_to_none=True);rows=[]
        loader=DataLoader(small,batch_size=1,shuffle=True,num_workers=2,collate_fn=rtmdet_collate_fn,generator=gen)
        for batch in loader:
            sample=processed(model,batch,True)
            folder=Path(batch['img_metas'][0]['filename']).parent.parent
            with torch.autocast('cuda',dtype=torch.float16):
                features=bind(model,adapter,sample,payload(args,folder,'correct'))
                outputs=model.rpn_head(features[0])
                losses=model.rpn_head.loss_by_feat(*outputs,batch_gt_instances=[z.gt_instances for z in sample['data_samples']],batch_img_metas=[z.metainfo for z in sample['data_samples']])
                loss=sum(losses['loss_rpn_cls'])
            scaler.scale(loss/8).backward()
            rows.append((folder.name,hashlib.sha256(sample['inputs'].detach().cpu().numpy().tobytes()).hexdigest(),float(loss.detach())))
            adapter.clear()
        scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(adapter.parameters(),.5)
        scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
        return rows
    mini_epoch()
    saved=dict(adapter=adapter.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),rng=rng_state(gen))
    atomic_save(saved,args.out/'engineering_mini_epoch1.pth')
    expected=mini_epoch();weights={k:v.detach().cpu().clone() for k,v in adapter.state_dict().items()}
    expected_gen=gen.get_state().clone()
    cp=torch.load(args.out/'engineering_mini_epoch1.pth',map_location='cpu',weights_only=False)
    adapter.load_state_dict(cp['adapter']);optimizer.load_state_dict(cp['optimizer']);scaler.load_state_dict(cp['scaler']);restore_rng(cp['rng'],gen)
    actual=mini_epoch()
    assert expected==actual,'Resumed augmented sample order/input/loss differs'
    assert all(torch.equal(v.detach().cpu(),weights[k]) for k,v in adapter.state_dict().items())
    assert torch.equal(expected_gen,gen.get_state())
    adapter.load_state_dict(initial);optimizer.zero_grad(set_to_none=True)
    assert state_digest(model)==FROZEN_SHA
    dump(args.out/'resume_gate.json',dict(passed=True,scope='two 8-image engineering mini epochs, accum8; no formal training budget consumed',
        samples_augmented=True,workers=2,order_inputs_losses_exact=True,adapter_weights_exact=True,
        python_numpy_torch_cuda_loader_generator_restored=True,loader_generator_end_exact=True))


def gate(args,model,adapter):
    from ue_v1_adapter import cached_extract
    ds=MinimalSatelliteDataset(args.data_root,'train',augment=False)
    i=int(np.argmax([len(s['annotations']) for s in ds.samples]))
    sample=processed(model,rtmdet_collate_fn([ds[i]]),True)
    data=payload(args,point_folder(ds,i,args.data_root),'correct')
    before=state_digest(model);initial={k:v.detach().cpu().clone() for k,v in adapter.state_dict().items()}
    torch.cuda.reset_peak_memory_stats();identities={}
    for amp in (False,True):
        with torch.no_grad(),torch.autocast('cuda',enabled=amp,dtype=torch.float16):
            features=model.extract_feat(sample['inputs'])
            adapter.enabled=False
            base=model.rpn_head(features[0])
            with cached_extract(model,features):ref=model.predict(sample['inputs'],sample['data_samples'],rescale=True)[0].pred_instances
            adapter.encode(*data,features[3][0]);adapter.enabled=True
            out=model.rpn_head(features[0])
            assert all(torch.equal(a,b) for aa,bb in zip(base,out) for a,b in zip(aa,bb))
            with cached_extract(model,features):pred=model.predict(sample['inputs'],sample['data_samples'],rescale=True)[0].pred_instances
            assert all(torch.equal(getattr(ref,k),getattr(pred,k)) for k in ('bboxes','scores','labels','masks'))
            # Explicit geometry ramp and a hole at the actual first anchor.
            support=torch.ones_like(data[1]);field=torch.zeros_like(data[2])
            field[:,0]=torch.arange(512,device='cuda')[None,None,:]/512
            adapter.context=(support,field)
            for prior,f in zip(adapter.priors,features[0]):
                sampled,valid=adapter.sample(prior,f.shape[-2:])
                uv=((prior[:,:2]+prior[:,2:])/2).reshape(*f.shape[-2:],adapter.anchors,2)[:,:,0]
                assert torch.allclose(sampled[0,0],uv[:,:,0]/512,atol=1e-6,rtol=0)
            support.zero_();adapter.alpha.fill_(.5)
            adapter.context=(support,field)
            unsupported=model.rpn_head(features[0])
            assert all(torch.equal(a,b) for aa,bb in zip(base,unsupported) for a,b in zip(aa,bb))
            adapter.alpha.zero_();adapter.clear()
            identities[str(amp)]=dict(zero_logit_and_bbox_reg_exact=True,zero_final_predictions_exact=True,
                no_support_nonzero_alpha_exact=True,anchor_center_ramp_exact=True)
    def targets():
        seed(20261023)
        rpn=model.rpn_head
        anchors,flags=rpn.get_anchors([x.shape[-2:] for x in base[0]],[z.metainfo for z in sample['data_samples']],device='cuda')
        return rpn.get_targets(anchors,flags,[z.gt_instances for z in sample['data_samples']],[z.metainfo for z in sample['data_samples']])
    adapter.enabled=False;labels0=targets();adapter.enabled=True;labels1=targets()
    def same(a,b):
        if torch.is_tensor(a):return torch.equal(a,b)
        if isinstance(a,(list,tuple)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
        return a==b
    assert same(labels0,labels1),'Anchor labels/weights/samples changed'
    optimizer=torch.optim.AdamW(adapter.parameters(),lr=1e-3)
    grads=[]
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):
            features=bind(model,adapter,sample,data)
            out=model.rpn_head(features[0]);losses=model.rpn_head.loss_by_feat(*out,
                batch_gt_instances=[s.gt_instances for s in sample['data_samples']],batch_img_metas=[s.metainfo for s in sample['data_samples']])
            loss=sum(losses['loss_rpn_cls'])
        loss.backward()
        grad={name:float(p.grad.float().norm()) if p.grad is not None else None for name,p in adapter.named_parameters()}
        assert all(v is None or np.isfinite(v) for v in grad.values())
        grads.append(grad);optimizer.step();adapter.clear()
    assert grads[0]['alpha'] and any((r.get('query.weight') or 0)>0 for r in grads[1:])
    assert any((r.get('mixer.2.weight') or 0)>0 for r in grads[1:])
    assert state_digest(model)==before==FROZEN_SHA and all(p.grad is None for p in model.parameters())
    with torch.no_grad():
        features=model.extract_feat(sample['inputs']);adapter.encode(*data,features[3][0]);adapter.alpha.fill_(.8)
        changed=model.rpn_head(features[0]);adapter.enabled=False;base=model.rpn_head(features[0]);adapter.enabled=True
        assert all(torch.equal(a,b) for a,b in zip(changed[1],base[1])), 'Box deltas changed'
        assert any(not torch.equal(a,b) for a,b in zip(changed[0],base[0])), 'Objectness did not change'
        ca=model.rpn_head.predict_by_feat(*changed,batch_img_metas=[z.metainfo for z in sample['data_samples']])[0]
        cb=model.rpn_head.predict_by_feat(*base,batch_img_metas=[z.metainfo for z in sample['data_samples']])[0]
        assert not torch.equal(ca.scores,cb.scores), 'New score path did not reach original proposal NMS'
        adapter.clear()
    # 250 proposals through the frozen downstream mask path, not just RPN loss.
    from mmengine.structures import InstanceData
    with torch.inference_mode():
        adapter.clear();features=model.extract_feat(sample['inputs'])
        boxes=sample['data_samples'][0].gt_instances.bboxes
        gt_count=len(boxes)
        boxes=boxes.repeat((250+len(boxes)-1)//len(boxes),1)[:250]
        props=InstanceData(bboxes=boxes,scores=torch.ones(250,device='cuda'),labels=torch.zeros(250,device='cuda',dtype=torch.long))
        pred=model.roi_head.predict_mask(model.roi_head._inject_extra_pe(features[0]),[z.metainfo for z in sample['data_samples']],[props],rescale=True,
            image_embeddings=features[1],image_positional_embeddings=features[2],high_res_features=features[3])[0]
        assert len(pred)==250 and torch.isfinite(pred.scores).all()
    adapter.load_state_dict(initial);optimizer.zero_grad(set_to_none=True)
    resume_proof(args,model,adapter,initial)
    dump(args.out/'gpu_gate.json',dict(passed=True,train_index=i,gt=gt_count,identities=identities,
        gradients=grads,satellite_state_unchanged=True,resource_rois=250,
        peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
        source_hashes={p:file_sha(ROOT/p) for p in SOURCES},formal_initialization_restored=True,modified_scores_reach_original_proposals=True,bbox_deltas_unchanged=True,anchor_targets_and_sampling_exact=True,resume_mini_epochs_exact=True))


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['init','gate','train','eval'])
    p.add_argument('--out',type=Path,required=True);p.add_argument('--data-root',type=Path,default=DATA)
    p.add_argument('--base',type=Path,default=BASE);p.add_argument('--init',type=Path,default=STUDY/'common_init.pth')
    p.add_argument('--cache',type=Path,default=Path('/data/wangcheng/cache/ue_v1_native_geometry'))
    p.add_argument('--constants',type=Path,default=Path('/data/wangcheng/checkpoint/ue_geometry_mechanism_20261022/common_constants.json'))
    p.add_argument('--content',choices=['geometry','constant','zero'],default='geometry')
    p.add_argument('--mode',choices=['correct','constant','zero','shuffle','disabled'],default='correct')
    p.add_argument('--adapter',type=Path);p.add_argument('--resume-from',type=Path)
    p.add_argument('--eval-split',choices=['val','test'],default='test')
    p.add_argument('--nms-pre',type=int,choices=[1000,2000],default=1000);p.add_argument('--max-images',type=int,default=0)
    return p


def main():
    args=parser().parse_args();args.out.mkdir(parents=True,exist_ok=True)
    if args.action=='init':args.init=None
    model,adapter,snapshot=make_model(args)
    if args.action=='init':
        assert not (args.out/'common_init.pth').exists()
        atomic_save(dict(adapter=adapter.state_dict(),fingerprint=tensor_digest(adapter.state_dict())),args.out/'common_init.pth')
        dump(args.out/'initialization.json',dict(fingerprint=tensor_digest(adapter.state_dict()),source_hashes={p:file_sha(ROOT/p) for p in SOURCES}))
    elif args.action=='gate':gate(args,model,adapter)
    elif args.action=='train':train(args,model,adapter,snapshot)
    else:
        if args.adapter:
            cp=torch.load(args.adapter,map_location='cpu',weights_only=False)
            assert cp['epoch']==10 and cp['step']==2670
            for p,h in cp['contract']['source_hashes'].items():assert file_sha(ROOT/p)==h
            adapter.load_state_dict(cp['adapter'],strict=True)
        evaluate(args,model,adapter)

if __name__=='__main__':main()
