"""Matched ten-epoch prefixes; independent of all live compact source files."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time

WORK = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(WORK), str(WORK / 'scripts/probes')]
import ue_compact_joint_run_20261024 as base
from rsprompter.uav.rgb_mechanism_variants import RGBMechanismVariant, install
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT = Path('/data/wangcheng/checkpoint/ue_rgb_mechanism_20261025')
VARIANTS = ['view_preserve', 'spatial_preserve', 'no_sat_copy']
NEW = ['scripts/probes/ue_rgb_mechanism_run_20261025.py', 'rsprompter/uav/rgb_mechanism_variants.py']


def hashes():
    return {**base.sources(), **{s: base.core.C.file_sha(WORK / s) for s in NEW}}


def cpu_gate(root):
    base.core.C.seed(2055)
    selected = torch.randn(30, 8, 32)
    index = torch.randint(0, 81920, (30, 8)); index[0] = -1; index[1, 2:] = -1
    index[2] = 100  # One view; unequal view support must remain equivalent.
    valid = index >= 0
    original = (selected * valid[..., None]).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
    view = RGBMechanismVariant(); view.configure('view_preserve')
    error = float((view.view_reduce(selected, index) - original).abs().max())
    assert error < 1e-6, error
    spatial = RGBMechanismVariant(); spatial.configure('spatial_preserve')
    x = torch.randn(1, 32, 128, 128)
    spatial_error = float((spatial.pool_spatial(x) - F.avg_pool2d(x, 2, 2)).abs().max())
    assert spatial_error < 1e-6
    # Full copied interaction, including the head, matches the old function.
    spatial.output.weight.data.normal_(std=.02)
    from rsprompter.uav.compact_joint import CompactInteraction
    reference = CompactInteraction(); reference.load_state_dict(spatial.state_dict(), strict=False)
    support = torch.rand(1, 1, 128, 128) > .2
    spatial.context = reference.context = (x, support)
    sat = torch.randn(1, 64, 64, 224)
    with torch.no_grad():
        a, b = spatial.interact(sat), reference.interact(sat)
    full_error = float((a - b).abs().max())
    head_error = max(float((a-b).abs().max()) for a,b in zip(spatial.outputs, reference.outputs))
    assert full_error < 1e-5 and head_error < 1e-5, (full_error, head_error)
    reverse = RGBMechanismVariant(); reverse.configure('no_sat_copy')
    reverse.output.weight.data.normal_(std=.02)
    reverse.context = (x, torch.zeros_like(support))
    with torch.no_grad():
        assert torch.equal(reverse.interact(sat), sat)
    reverse.context = (x, support)
    reverse.interact(sat).square().mean().backward()
    assert reverse.uout.weight.grad is not None and reverse.uout.weight.grad.count_nonzero() == 0
    assert reverse.sq.weight.grad is not None and reverse.sq.weight.grad.abs().sum() > 0
    base.core.C.dump(root / 'cpu_gate.json', dict(passed=True,
        candidate_weighted_view_initial_error=error, spatial_initial_error=spatial_error,
        copied_interaction_error=full_error, copied_head_error=head_error,
        no_support_identity=True, reverse_zero_gradient_forward_live=True, source_hashes=hashes()))


def runtime(variant, content):
    rt = base.Runtime(base.ROOT, 'rgbd' if content == 'correct' else 'constant_rgb')
    if variant != 'baseline':
        install(rt.joint.uav, variant)
        # Register added weights before constructing the matched optimizer.
        rt.optimizer = torch.optim.AdamW([p for p in rt.network.parameters() if p.requires_grad],
            lr=base.PROTOCOL['lr'], weight_decay=base.PROTOCOL['weight_decay'])
    base.core.C.seed(base.PROTOCOL['seed'])
    return rt


def make_contract(rt, args):
    return dict(variant=args.variant, content=args.content, source_hashes=hashes(),
        baseline_initialization=json.loads((base.ROOT / 'initialization.json').read_text()),
        original_protocol=base.PROTOCOL, epochs=10, successful_updates=2670,
        learning_rate='unchanged first2670 updates of original100epoch schedule',
        selection='fixed E10 val only; no test selection',
        initialization='same generic foundation/common new heads; not task checkpoint',
        extra_trainable_parameters={n:p.numel() for n,p in rt.network.named_parameters()
            if 'view_mix' in n or 'spatial_mix' in n},
        data_identity=base.core.C.archive_identity(base.core.previous.DATA))


def gate(rt, args):
    out = args.root / 'gates' / args.variant
    ds = base.core.C.MinimalSatelliteDataset(base.core.previous.DATA, 'train')
    i = max(range(len(ds)), key=lambda j:len(ds.samples[j]['annotations']))
    batch = base.core.C.rtmdet_collate_fn([ds[i]])
    ident = ds.samples[i]['image_id']
    rt.network.train(); rt.set_epoch(1)
    changed_before = {n:p.detach().cpu().clone() for n,p in rt.network.named_parameters()
        if 'view_mix' in n or 'spatial_mix' in n or ('rgb_encoder' in n and 'lora_B' in n)}
    values = []
    for step in range(1, 4):
        values.append(base.accumulation(rt, [(batch, ident)] * 8, step))
    changed = {n:not torch.equal(before, dict(rt.network.named_parameters())[n].detach().cpu())
               for n,before in changed_before.items()}
    assert any(v for n,v in changed.items() if 'lora_B' in n)
    if args.variant != 'no_sat_copy':
        assert any(v for n,v in changed.items() if 'mix' in n)
    # FP32 full proposal/combined inference on max-GT train image, no optimizer.
    rt.network.eval()
    with torch.inference_mode():
        sample = base.core.C.processed(rt.satellite, batch)
        payload = rt.payload(ident); rt.joint.prepare_uav(**payload)
        feat = rt.satellite.extract_feat(sample['inputs'])
        prefix, combined, count = base.core.inference_outputs(rt.satellite, rt.joint.uav,
            sample, feat, payload['geometry'])
        assert len(combined['scores']) <= 250 and base.np.isfinite(combined['scores']).all()
        rt.clear()
    torch.cuda.synchronize()
    base.core.C.dump(out / 'gpu_gate.json', dict(passed=True, source_hashes=hashes(),
        image_id=ident, gt_instances=len(ds.samples[i]['annotations']), successful_updates=3,
        update_reports=values, changed_parameters=changed, FP32_predictions=len(combined['scores']),
        peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20,
        peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
        allocator_ceiling_MiB=7800, original_single_parameters_jointly_trainable=True))


def train(rt, args):
    out = args.root / args.variant / args.content
    frozen = make_contract(rt, args)
    g = json.loads((args.root/'gates'/args.variant/'gpu_gate.json').read_text())
    assert g['passed'] and g['source_hashes'] == hashes()
    assert not (out/'last.pth').exists(), 'Refuse implicit overwrite/resume'
    base.core.C.dump(out/'contract.json', frozen)
    ds = base.core.C.MinimalSatelliteDataset(base.core.previous.DATA, 'train', augment=True)
    step = 0
    for epoch in range(1, 11):
        rt.network.train(); rt.set_epoch(epoch)
        loader = DataLoader(ds,batch_size=1,shuffle=True,num_workers=2,
            generator=rt.generator,collate_fn=base.core.C.rtmdet_collate_fn)
        order, prepared, loss_sum, retries = [], [], 0., 0
        began = time.time()
        for microbatch,batch in enumerate(loader):
            if microbatch >= 2136:break
            ident = int(Path(batch['img_metas'][0]['filename']).parent.parent.name)
            order.append(ident); prepared.append((batch,ident))
            if len(prepared) == 8:
                row = base.accumulation(rt, prepared, step+1); step+=1; prepared=[]
                loss_sum+=row['loss_sum']; retries+=row['retries']
                if step % 25 == 0:
                    base.core.C.dump(out/'progress.json',dict(epoch=epoch,update=step,
                        last_loss=row['last_loss'],preclip_norm=row['preclip_norm'],
                        AMP_retries_epoch=retries,pid=os.getpid(),variant=args.variant,content=args.content))
                    print('UPDATE',args.variant,args.content,epoch,step,flush=True)
        assert not prepared and len(set(order))==len(order)==2136 and step==epoch*267
        ref = json.loads((base.ROOT/rt.arm/f'epoch{epoch}.json').read_text())
        assert order == ref['sample_order'], 'Original prefix loader order changed'
        metrics = base.evaluate(rt,'val',out/'validation_current')
        state = rt.state(epoch,step,frozen,dict(score=metrics['segm/mAP'],epoch=epoch))
        base.core.previous.atomic_save(state,out/'last.pth')
        if epoch in (1,10):
            base.core.previous.atomic_save(state,out/f'epoch{epoch}.pth')
        base.core.C.dump(out/f'epoch{epoch}.json',dict(epoch=epoch,step=step,sample_order=order,
            original_prefix_order_exact=True,validation=metrics,mean_microbatch_loss=loss_sum/2136,
            AMP_retries_epoch=retries,seconds=time.time()-began,
            peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20))
        assert hashes()==frozen['source_hashes']
    base.core.C.dump(out/'training_complete.json',dict(complete=True,epoch=10,updates=2670,
        checkpoint_sha256=base.core.C.file_sha(out/'epoch10.pth'),contract=frozen))


def evaluate(rt,args):
    if args.variant=='baseline':
        path=base.ROOT/rt.arm/'epoch10.pth'
        cp=torch.load(path,map_location='cpu',weights_only=False)
        assert cp['epoch']==10 and cp['step']==2670
        rt.network.load_state_dict(cp['model'],strict=True)
    else:
        path=args.root/args.variant/args.content/'epoch10.pth'
        cp=torch.load(path,map_location='cpu',weights_only=False)
        assert cp['contract']==make_contract(rt,args) and cp['epoch']==10 and cp['step']==2670
        rt.network.load_state_dict(cp['model'],strict=True)
    rt.set_epoch(10)
    if args.mode!='matched':
        original=rt.payload
        def intervened(ident):
            value=original(ident)
            if args.mode=='constant':value['rgb'][:]=rt.joint.uav.rgb_mean
            else:
                ds=base.core.C.MinimalSatelliteDataset(base.core.previous.DATA,'val')
                ids=[s['image_id'] for s in ds.samples]
                donor=ids[(ids.index(ident)+len(ids)//2)%len(ids)]
                value['rgb']=rt.uav_inputs.load(donor,geometry_mode='measured',rgb_mode='correct')['rgb']
            return value
        rt.payload=intervened
    out=args.root/'evaluation'/args.variant/args.content/args.mode
    assert not (out/'complete.json').exists(), 'Refuse duplicate evaluation'
    base.evaluate(rt,'val',out,checkpoint_sha256=base.core.C.file_sha(path))
    identity=json.loads((out/'complete.json').read_text())
    identity.update(mechanism_variant=args.variant,training_content=args.content,
                    input_intervention=args.mode,source_hashes=hashes())
    base.core.C.dump(out/'complete.json',identity)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['cpu_gate','gate','train','eval'])
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--variant',choices=['baseline']+VARIANTS,default='view_preserve')
    p.add_argument('--content',choices=['correct','constant'],default='correct')
    p.add_argument('--mode',choices=['matched','constant','donor'],default='matched')
    a=p.parse_args();a.root.mkdir(parents=True,exist_ok=True)
    if a.action=='cpu_gate':cpu_gate(a.root);return
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(7800/(torch.cuda.get_device_properties(0).total_memory/2**20))
    rt=runtime(a.variant,a.content)
    if a.action=='gate':gate(rt,a)
    elif a.action=='train':train(rt,a)
    else:evaluate(rt,a)


if __name__=='__main__':main()
