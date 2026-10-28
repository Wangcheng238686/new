"""Fixed train-only RGB direction audit; no optimizer or source modifications."""
import argparse
import gc
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
import ue_s2db_probe as run

STUDY=Path('/data/wangcheng/checkpoint/ue_s2db_probe_20261022')
OUT=Path('/data/wangcheng/checkpoint/ue_rgb_content_direction_audit_20261023')


def cpu(t):return t.detach().float().cpu().clone()
def energy(xs):return sum(float(x.double().square().sum()) for x in xs)
def rms(xs):
    n=sum(x.numel() for x in xs)
    return (energy(xs)/n)**.5 if n else None


def fingerprint_sources():
    manifest=json.loads((STUDY/'manifest.json').read_text())
    hashes=manifest['source_hashes']
    actual={k:run.file_sha(run.ROOT/k) for k in hashes}
    assert hashes==actual,'Frozen source changed'
    return actual


def run_arm(arm,cli,indices):
    checkpoint=STUDY/arm/'last.pth'
    cp=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert cp['epoch']==10 and cp['step']==2670
    a=argparse.Namespace(**cp['contract']['args'])
    for k in ('data_root','base','cache','out','init'):
        if getattr(a,k,None) is not None:setattr(a,k,Path(getattr(a,k)))
    a.init=None
    model,adapter,_=run.make_model(a)
    adapter.load_state_dict(cp['adapter'],strict=True);del cp
    model.eval();adapter.eval();adapter.aux_weight=0
    for p in adapter.parameters():p.requires_grad_(False)
    before=run.state_digest(model);abefore=run.tensor_digest(adapter.state_dict())
    assert before==run.FROZEN_SHA
    ds=run.MinimalSatelliteDataset(a.data_root,'train',augment=False)
    decoder=model.roi_head.mask_head.mask_decoder.decoder
    tail=model.roi_head.mask_head.decoder_tail_refiner
    raw_base=[];leaves=[];deltas=[];boxes=[];raw=[];final=[];selected=[]
    capture_leaf=[True]
    def pre_capture(m,args,o):raw_base.append(cpu(o))
    def receiver(m,args,o):
        deltas.append(cpu(o)-raw_base[-1]);boxes.append(cpu(adapter.boxes))
        if not capture_leaf[0]:return o
        leaf=o.detach().requires_grad_(True);leaves.append(leaf)
        return leaf
    def tail_pre(m,args):raw.append(cpu(args[0]))
    def tail_post(m,args,o):
        final.append(cpu(o[0]));selected.append(o[1]['indices'].detach().cpu().clone())
    # prepend observes the actual frozen conv output before adapter's hook.
    handles=[decoder.conv_s0.register_forward_hook(pre_capture,prepend=True),
             decoder.conv_s0.register_forward_hook(receiver),
             tail.register_forward_pre_hook(tail_pre),tail.register_forward_hook(tail_post)]
    records=[]
    for i in indices:
        paired={}
        for mode in ('correct','constant','donor'):
            replay=None
            if cli.gate and mode=='correct':
                run.seed(44);capture_leaf[0]=False
                sample=run.processed(model,run.rtmdet_collate_fn([ds[i]]),True)
                data=run.payload(run.point_folder(ds,i,a.data_root),a.cache,'correct')
                with torch.no_grad():
                    features=run.bind(model,adapter,sample,data)
                    with run.cached_extract(model,features):ref=model.loss(sample['inputs'],sample['data_samples'])
                replay=dict(raw=list(raw),final=list(final),boxes=list(boxes),
                    losses={k:float(v) for k,v in ref.items() if torch.is_tensor(v)})
                del ref,features,data,sample;adapter.clear();capture_leaf[0]=True
            run.seed(44)
            for xs in (raw_base,leaves,deltas,boxes,raw,final,selected):xs.clear()
            donor_i=(i+1)%len(ds)
            donor=run.point_folder(ds,donor_i,a.data_root) if mode=='donor' else None
            sample=run.processed(model,run.rtmdet_collate_fn([ds[i]]),True)
            data=run.payload(run.point_folder(ds,i,a.data_root),a.cache,
                             'constant' if mode=='constant' else 'correct',donor)
            with torch.no_grad():features=run.bind(model,adapter,sample,data)
            with run.cached_extract(model,features):losses=model.loss(sample['inputs'],sample['data_samples'])
            assert leaves and len(leaves)==len(raw)==len(final)==len(deltas)
            vals={k:float(v.detach()) for k,v in losses.items() if torch.is_tensor(v)}
            if replay is not None:
                assert all(torch.equal(x,y) for x,y in zip(replay['raw'],raw)) and len(replay['raw'])==len(raw)
                assert all(torch.equal(x,y) for x,y in zip(replay['final'],final)) and len(replay['final'])==len(final)
                assert all(torch.equal(x,y) for x,y in zip(replay['boxes'],boxes)) and replay['losses']==vals
                run.dump(cli.out/'leaf_replay_gate.json',dict(passed=True,raw_logits_exact=True,
                    final_logits_exact=True,proposal_boxes_exact=True,losses_exact=True))
            gradients={}
            for k,v in losses.items():
                if not torch.is_tensor(v) or not v.requires_grad:continue
                grads=torch.autograd.grad(v,leaves,allow_unused=True,retain_graph=True)
                gradients[k]=[torch.zeros_like(d) if g is None else cpu(g) for d,g in zip(deltas,grads)]
            paired[mode]=dict(losses=vals,gradients=gradients,deltas=list(deltas),
                              boxes=list(boxes),raw=list(raw),final=list(final),base=list(raw_base),selected=list(selected))
            del losses,features,data,sample,grads
            adapter.clear();leaves.clear();gc.collect();torch.cuda.empty_cache()
            print('AUDIT',arm,i,mode,'peak_MiB',torch.cuda.max_memory_reserved()/2**20,flush=True)
        c=paired['correct'];contrasts={}
        for mode in ('constant','donor'):
            m=paired[mode]
            assert len(c['boxes'])==len(m['boxes']) and all(torch.equal(x,y) for x,y in zip(c['boxes'],m['boxes']))
            direction=[x-y for x,y in zip(c['deltas'],m['deltas'])]
            rd=[x-y for x,y in zip(c['raw'],m['raw'])]
            fd=[x-y for x,y in zip(c['final'],m['final'])]
            derivatives={}
            for k,gc0 in c['gradients'].items():
                gm=m['gradients'][k]
                def directional(gs):
                    dot=sum(float((x.double()*d.double()).sum()) for x,d in zip(gs,direction))
                    den=(energy(gs)*energy(direction))**.5
                    return dict(dot=dot,cosine=dot/den if den else None,gradient_rms=rms(gs))
                derivatives[k]=dict(at_correct=directional(gc0),at_intervention=directional(gm),
                    actual_loss_correct_minus_intervention=c['losses'][k]-m['losses'][k])
            same_rank=sum(int((x==y).sum()) for x,y in zip(c['selected'],m['selected']))
            selected_count=sum(x.numel() for x in c['selected'])
            shared_set=sum(sum(len(set(xr.tolist())&set(yr.tolist())) for xr,yr in zip(x,y))
                           for x,y in zip(c['selected'],m['selected']))
            contrasts[mode]=dict(content_direction='correct injected residual minus intervention injected residual',
                content_difference_rms=rms(direction),correct_total_injection_rms=rms(c['deltas']),
                intervention_total_injection_rms=rms(m['deltas']),
                content_difference_over_correct_total_rms=rms(direction)/rms(c['deltas']) if energy(c['deltas']) else None,
                receiver_base_rms=rms(c['base']),raw_sam_logit_difference_rms=rms(rd),
                final_udpr_logit_difference_rms=rms(fd),
                final_over_raw_logit_difference_rms=rms(fd)/rms(rd) if energy(rd) else None,
                udpr_selected_points=selected_count,
                udpr_selected_rank_change_fraction=1-same_rank/selected_count if selected_count else None,
                udpr_selected_set_change_fraction=1-shared_set/selected_count if selected_count else None,
                losses=derivatives)
        records.append(dict(arm=arm,train_index=i,image_id=ds.samples[i]['image_id'],
            gt_instances=len(ds.samples[i]['annotations']),donor_train_index=(i+1)%len(ds),
            calls=len(c['boxes']),rois=sum(len(b) for b in c['boxes']),
            all_losses={k:v['losses'] for k,v in paired.items()},contrasts=contrasts))
        run.dump(cli.out/f'{arm}_partial.json',dict(complete=False,records=records))
        del paired,c,m,direction,rd,fd;gc.collect()
    for h in handles:h.remove()
    assert run.state_digest(model)==before and run.tensor_digest(adapter.state_dict())==abefore
    assert all(p.grad is None for p in model.parameters())
    result=dict(complete=True,arm=arm,checkpoint=str(checkpoint),checkpoint_sha256=run.file_sha(checkpoint),
                satellite_unchanged=True,adapter_unchanged=True,peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20,
                records=records)
    run.dump(cli.out/f'{arm}.json',result)
    del model,adapter;gc.collect();torch.cuda.empty_cache()
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=OUT)
    p.add_argument('--worker',action='store_true');p.add_argument('--gate',action='store_true')
    cli=p.parse_args();cli.out.mkdir(parents=True,exist_ok=True)
    if not cli.worker:
        # Durable admission owner; no exclusive idle-GPU requirement.
        while True:
            lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
            free=sorted([(int(y),int(x)) for x,y in (l.split(',') for l in lines.splitlines())],reverse=True)
            if free[0][0]>=6000:
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(free[0][1]))
                cmd=[sys.executable,__file__,'--worker','--gate','--out',str(cli.out/'gate')]
                rc=subprocess.call(cmd,env=env)
                if rc:raise RuntimeError('Capacity/engineering gate failed; preserve logs and investigate')
                gate=json.loads((cli.out/'gate'/'A.json').read_text())
                reserve=gate['peak_reserved_MiB']
                run.dump(cli.out/'resource_gate.json',dict(gpu=free[0][1],free_before_MiB=free[0][0],peak_reserved_MiB=reserve,
                    admitted=reserve+2048<free[0][0],gate_source_unchanged=True))
                assert reserve+2048<free[0][0]
                while True:
                    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
                    now=sorted([(int(y),int(x)) for x,y in (l.split(',') for l in rows.splitlines())],reverse=True)
                    if now[0][0]>=reserve+2048:
                        env['CUDA_VISIBLE_DEVICES']=str(now[0][1]);break
                    run.dump(cli.out/'owner_state.json',dict(stage='waiting_postgate_shared_capacity',free=now,pid=os.getpid()))
                    time.sleep(15)
                run.dump(cli.out/'worker_admission.json',dict(gpu=now[0][1],free_before_MiB=now[0][0],reserve_MiB=reserve))
                rc=subprocess.call([sys.executable,__file__,'--worker','--out',str(cli.out)],env=env)
                if rc:raise RuntimeError('Audit worker failed')
                return
            run.dump(cli.out/'owner_state.json',dict(stage='waiting_shared_capacity',free=free,pid=os.getpid()))
            time.sleep(15)
    torch.set_num_threads(2);run.seed(44)
    before=fingerprint_sources()
    ds=run.MinimalSatelliteDataset(run.DATA,'train',augment=False)
    order=np.argsort([len(s['annotations']) for s in ds.samples],kind='stable')
    indices=[int(order[round(q*(len(ds)-1))]) for q in (.125,.375,.625,.875)]
    run.dump(cli.out/'manifest.json',dict(script_sha256=run.file_sha(__file__),source_hashes=before,
        train_indices=indices,quantiles=[.125,.375,.625,.875],modes=['correct','constant','donor'],
        donor='next sample in unchanged full train order, wrapping; RGB only, target geometry retained',
        precision='FP32',dropout=False,augmentation=False,optimizer_steps=0,
        randomness='Independent worker; seed44 reset before each matched mode, no optimizer; does not touch RNG of any other process',
        losses='Actual already-weighted tensor entries returned by model.loss, separately by original loss key; mask/tail are not rescaled by this audit',
        capture='all actual conv_s0 calls; enhanced tensor detached as leaf; downstream gradients only',
        interpretation='Positive inner product means increasing intervention toward correct content locally raises loss; not AP or causal success proof. Loss differences and derivatives both reported.'))
    arms=['A'] if cli.gate else ['A','D']
    results=[run_arm(arm,cli,indices[:1] if cli.gate else indices) for arm in arms]
    assert fingerprint_sources()==before
    run.dump(cli.out/'complete.json',dict(complete=True,engineering_gate_only=cli.gate,source_unchanged=True,
        results=[dict(arm=r['arm'],samples=len(r['records'])) for r in results],optimizer_steps=0))


if __name__=='__main__':main()
