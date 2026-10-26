"""Offline per-loss gradients on fixed training samples; never updates weights."""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import cv2
import numpy as np
from torch.nn import functional as F
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runner', choices=['ladder','post'], required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--init', type=Path, default=Path('/data/wangcheng/checkpoint/ue_attention_ladder_20261022/common_init.pth'))
    cli = p.parse_args()
    run = importlib.import_module('ue_attention_ladder_probe' if cli.runner == 'ladder' else 'ue_post_activation_probe')
    cp = torch.load(cli.checkpoint, map_location='cpu', weights_only=False)
    c = cp['contract']
    a = argparse.Namespace(**c['args'])
    for k in ('data_root','base','cache','out','init'):
        setattr(a,k,Path(getattr(a,k)))
    a.init = cli.init
    model, adapter, _ = run.make_model(a)
    adapter.load_state_dict(cp['adapter'], strict=True)
    epoch = cp['epoch']; del cp
    model.eval(); adapter.eval()
    before = run.state_digest(model)
    adapter_before = run.tensor_digest(adapter.state_dict())
    ds = run.MinimalSatelliteDataset(a.data_root,'train',augment=False)
    order = np.argsort([len(s['annotations']) for s in ds.samples],kind='stable')
    indices = [int(order[round(q*(len(ds)-1))]) for q in (.125,.375,.625,.875)]
    named = [(n,v) for n,v in adapter.named_parameters() if v.requires_grad]
    params = [v for n,v in named]
    records = []; captures = []; posts = []; capture_boxes = []; post_boxes = []
    decoder = model.roi_head.mask_head.mask_decoder.decoder
    receiver = decoder.output_upscaling[-1] if a.direction == 'N2' else decoder.conv_s0
    def capture(module, inputs, output):
        if output.requires_grad:
            captures.append(output); capture_boxes.append(adapter.boxes.detach().clone())
    def post_capture(module, inputs, output):
        if output.requires_grad:
            posts.append(output); post_boxes.append(adapter.boxes.detach().clone())
    handles = [receiver.register_forward_hook(capture),decoder.output_upscaling[-1].register_forward_hook(post_capture)]
    cli.out.mkdir(parents=True,exist_ok=True)
    for i in indices:
        for mode in ('correct','constant'):
            run.seed(44); captures.clear(); posts.clear(); capture_boxes.clear(); post_boxes.clear()
            sample = run.processed(model,run.rtmdet_collate_fn([ds[i]]),True)
            data = run.payload(a,run.point_folder(ds,i,a.data_root),mode)
            features = run.bind(model,adapter,sample,data)
            with run.cached_extract(model,features): losses = model.loss(sample['inputs'],sample['data_samples'])
            masks = sample['data_samples'][0].gt_instances.masks.to_ndarray().astype(np.uint8)
            union = masks.any(0)
            boundary = np.zeros_like(union)
            for mask in masks:
                boundary |= mask.astype(bool) & ~cv2.erode(mask,np.ones((3,3),np.uint8)).astype(bool)
            union_boundary = union & ~cv2.erode(union.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
            zones = torch.as_tensor(np.stack([boundary,union_boundary,union & ~boundary,~union]),device='cuda',dtype=torch.float32)[None]
            terms = {k:v for k,v in losses.items() if torch.is_tensor(v) and v.requires_grad}
            vectors = {}; details = {}
            targets = params + captures + posts
            assert captures and posts, 'Receiver hook did not capture actual mask path'
            for key, loss in terms.items():
                grad = torch.autograd.grad(loss,targets,allow_unused=True,retain_graph=True)
                per = {}; groups = {}
                vectors[key] = []
                for (name,par),g in zip(named,grad):
                    cpu = None if g is None else g.detach().float().cpu()
                    vectors[key].append(cpu)
                    per[name] = dict(status='none' if cpu is None else 'zero' if not bool(cpu.any()) else 'nonzero',
                        norm=0. if cpu is None else float(cpu.double().norm()),numel=par.numel())
                    group = 'encoder_lora' if name.startswith('encoder.') else name.split('.')[0]
                    agg = groups.setdefault(group,dict(squared_norm=0.,numel=0,connected_numel=0))
                    agg['squared_norm'] += per[name]['norm']**2; agg['numel'] += par.numel()
                    if cpu is not None: agg['connected_numel'] += par.numel()
                for val in groups.values(): val['rms_all_parameters'] = (val['squared_norm']/val['numel'])**.5
                def spatial(gs, boxes_list):
                    result = []
                    for g, boxes in zip(gs, boxes_list):
                        if g is None: result.append(None); continue
                        energy = g.detach().float().square().mean(1)
                        h,w = g.shape[-2:]
                        yy,xx = torch.meshgrid((torch.arange(h,device='cuda')+.5)/h,(torch.arange(w,device='cuda')+.5)/w,indexing='ij')
                        uv = boxes[:,None,None,:2]+torch.stack([xx,yy],-1)[None]*(boxes[:,None,None,2:]-boxes[:,None,None,:2])
                        region = F.grid_sample(zones.expand(len(boxes),-1,-1,-1),uv/256-1,mode='nearest',align_corners=False)>0
                        stats = {}
                        for label, mask in zip(['instance_inner_boundary_1px','union_inner_boundary_1px','foreground_inner','background'],region.unbind(1)):
                            stats[label] = dict(grid_points=int(mask.sum()),rms=float(energy[mask].mean().sqrt()) if bool(mask.any()) else None)
                        result.append(dict(shape=list(g.shape),rms=float(energy.mean().sqrt()),zones=stats,
                            nonzero_fraction=float((energy>0).float().mean())))
                    return result
                details[key] = dict(loss=float(loss.detach()),parameters=per,groups=groups,
                    receiver_gradients=spatial(grad[len(params):len(params)+len(captures)],capture_boxes),
                    post_activation_gradients=spatial(grad[len(params)+len(captures):],post_boxes))
            cosine = []
            keys = list(vectors)
            for ix,k in enumerate(keys):
                for other in keys[ix+1:]:
                    bygroup = {}
                    for (name,_),x,y in zip(named,vectors[k],vectors[other]):
                        group = 'encoder_lora' if name.startswith('encoder.') else name.split('.')[0]
                        z = bygroup.setdefault(group,[0.,0.,0.])
                        if x is not None:z[0] += float(x.double().square().sum())
                        if y is not None:z[1] += float(y.double().square().sum())
                        if x is not None and y is not None:z[2] += float((x.double()*y.double()).sum())
                    cosine.append(dict(left=k,right=other,by_group={g: v[2]/(v[0]*v[1])**.5 if v[0]>0 and v[1]>0 else None for g,v in bygroup.items()}))
            records.append(dict(image_id=ds.samples[i]['image_id'],mode=mode,alpha=float(adapter.alpha.detach()),losses=details,cosines=cosine))
            run.dump(cli.out/'partial.json',dict(samples=records,complete=False))
            del losses,terms,features,targets,vectors,details,grad,captures[:],posts[:]
            adapter.context = None
            print('AUDIT',i,mode,flush=True)
    for h in handles:h.remove()
    assert run.state_digest(model)==before==run.FROZEN_SHA
    assert run.tensor_digest(adapter.state_dict())==adapter_before
    assert all(v.grad is None for v in model.parameters()) and all(v.grad is None for v in params)
    run.dump(cli.out/'gradients.json',dict(complete=True,checkpoint=str(cli.checkpoint),checkpoint_sha256=run.file_sha(cli.checkpoint),
        epoch=epoch,direction=a.direction,split='train',density_quantiles=[.125,.375,.625,.875],precision='fp32',
        dropout=False,augmentation=False,optimizer_steps=0,scaled=False,clipped=False,
        spatial_scope='training GT inner one-pixel instance/union boundaries and foreground/background sampled nearest at RoI grid centers; not Boundary AP tolerance',
        satellite_state_unchanged=True,adapter_state_unchanged=True,samples=records))


if __name__ == '__main__':main()
