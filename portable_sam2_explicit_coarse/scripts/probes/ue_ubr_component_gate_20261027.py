#!/usr/bin/env python3
"""Four-GPU *component* gate, not a model training/evaluation runner.

Example (after source review):
  OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone \
    --nproc_per_node=4 scripts/probes/ue_ubr_component_gate_20261027.py \
    --output /data/wangcheng/checkpoint/NEW_COMPONENT_GATE_DIRECTORY

Runs four AMP AdamW updates and 250 sequential 512² UBR-only inference
instances PER RANK. This does not measure a whole detector, real decoder,
image batch, masks/AP, or data/geometry alignment. Synthetic leaf decoder
features test gradient reachability; they are not trainable decoder parameters.
"""
import os
os.environ['OMP_NUM_THREADS'] = '2'

import argparse
from datetime import timedelta
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import traceback

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import torch.nn.functional as F


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parameter_sha(model):
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        digest.update(name.encode())
        digest.update(parameter.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temp.replace(path)


def all_reports(report):
    reports = [None] * dist.get_world_size()
    dist.all_gather_object(reports, report)
    return reports


def fixture(ubr, rank, count, device):
    """Distinct synthetic input per rank, exact native 512x512 grid."""
    axis = torch.arange(512, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(axis, axis, indexing='ij')
    base = 3 * torch.sin(xx / (23 + rank)) + 2 * torch.cos(yy / 31)
    z = base.reshape(1, 1, 512, 512)[:count].clone().requires_grad_(True)
    features = torch.randn(count, 32, 512, 512, device=device, requires_grad=True)
    neighbors = torch.zeros(count, 1, 512, 512, device=device)
    neighbors[:, :, 50:140, 280:420] = 1
    geometry = torch.randn(count, 5, 512, 512, device=device)
    geometry[:, 4] = (xx > 100).float()
    geometry[:, :4] *= geometry[:, 4:5]
    sdt, band = ubr.prediction_boundary_state(z, threshold=.4, width=8)
    target = (((xx - 250 - rank).square() + (yy - 250).square()) < 170 ** 2).float()
    target = target.reshape(1, 1, 512, 512)[:count].clone()
    return features, z, sdt, neighbors, geometry, band, target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if int(os.environ.get('WORLD_SIZE', '0')) != 4:
        raise RuntimeError('This gate requires torchrun WORLD_SIZE=4')
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    if local_rank not in range(4) or not torch.cuda.is_available() or torch.cuda.device_count() < 4:
        raise RuntimeError('Four visible GPUs with LOCAL_RANK 0..3 are required')
    torch.cuda.set_device(local_rank)
    torch.set_num_threads(2)
    device = torch.device('cuda', local_rank)
    dist.init_process_group('nccl', timeout=timedelta(minutes=15))
    setup = [None]
    if rank == 0:
        try:
            args.output.mkdir(parents=True, exist_ok=False)
            setup[0] = {'ok': True}
        except Exception:
            setup[0] = {'ok': False, 'error': traceback.format_exc()}
    dist.broadcast_object_list(setup, src=0)
    if not setup[0]['ok']:
        raise RuntimeError(setup[0]['error'])

    source_path = Path(__file__).resolve().parents[2] / 'rsprompter/uav/unified_boundary_refiner.py'
    source_sha = {'module': sha_file(source_path), 'gate': sha_file(__file__)}
    spec = importlib.util.spec_from_file_location('standalone_ubr_component', source_path)
    ubr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ubr)
    torch.manual_seed(44 + rank)
    torch.cuda.manual_seed(44 + rank)
    model = ubr.UnifiedBoundaryRefiner(checkpoint_blocks=True).to(device)
    model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank,
                                   broadcast_buffers=False, find_unused_parameters=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01)
    scaler = torch.cuda.amp.GradScaler(init_scale=512.)
    report = {'rank': rank, 'local_rank': local_rank, 'pid': os.getpid(),
              'device_name': torch.cuda.get_device_name(device), 'source_sha': source_sha,
              'world_size': 4, 'synthetic': True, 'scope': 'UBR component only; not whole model',
              'num_parameters': sum(p.numel() for p in model.parameters()), 'updates': []}
    write_json(args.output / f'rank{rank}_started.json', report)
    failed = None
    try:
        # Exact identity at zero output layer on the same native grid.
        data = fixture(ubr, rank, 1, device)
        feature, z, sdt, neighbors, geometry, band, target = data
        with torch.no_grad(), torch.cuda.amp.autocast():
            initial = model.module(feature, z, sdt, neighbors, geometry)
        report['zero_init_exact_identity'] = bool(torch.equal(initial, z))
        if not report['zero_init_exact_identity']:
            raise AssertionError('zero-init logits are not exactly identical')
        del data, feature, z, sdt, neighbors, geometry, band, target, initial
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        train_start = time.perf_counter()
        for step in range(4):
            # Rank3 is deliberately empty once; all parameters remain in graph.
            count = 0 if rank == 3 and step == 0 else 1
            feature, z, sdt, neighbors, geometry, band, target = fixture(ubr, rank, count, device)
            optimizer.zero_grad(set_to_none=True)
            old_scale = scaler.get_scale()
            with torch.cuda.amp.autocast():
                final = model(feature, z, sdt, neighbors, geometry)
                mask_loss = F.binary_cross_entropy_with_logits(final.float(), target) if count else final.sum() * 0
                raw = ubr.boundary_loss(final, target, band,
                                        torch.zeros(count, dtype=torch.long, device=device), num_images=1)
                loss = mask_loss + .1 * raw
            if not torch.isfinite(loss):
                raise AssertionError('nonfinite loss')
            if not torch.equal(final.detach()[~band], z.detach()[~band]):
                raise AssertionError('training modified outside-band logits')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), .5)
            if not torch.isfinite(norm):
                raise AssertionError('nonfinite parameter gradient norm')
            # GradScaler.unscale_ only touches optimizer parameters; the
            # synthetic feature leaf is not in that optimizer.
            decoder_feature_grad = None if feature.grad is None else float(feature.grad.float().norm()) / old_scale
            if step == 1 and (decoder_feature_grad is None or decoder_feature_grad <= 0 or
                              not torch.isfinite(feature.grad).all()):
                raise AssertionError('second-update synthetic decoder-feature gradient unreachable')
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < old_scale:
                raise AssertionError('AMP skipped an optimizer update')
            hashes = all_reports(parameter_sha(model.module))
            if len(set(hashes)) != 1:
                raise AssertionError('DDP replicas diverged after update')
            report['updates'].append({'step': step + 1, 'instances': count,
                                      'local_image_count': 1,
                                      'mask_loss': float(mask_loss.detach()),
                                      'raw_ubr_loss': float(raw.detach()),
                                      'weighted_ubr_loss': .1 * float(raw.detach()),
                                      'loss': float(loss.detach()), 'grad_norm_before_clip': float(norm),
                                      'clip_limit': .5, 'amp_scale': scaler.get_scale(),
                                      'decoder_feature_leaf_grad_norm': decoder_feature_grad,
                                      'replica_sha': hashes[0]})
            del feature, z, sdt, neighbors, geometry, band, target, final, raw, loss, mask_loss
        torch.cuda.synchronize(device)
        report['training_seconds'] = time.perf_counter() - train_start
        report['training_peak_allocated_mib'] = torch.cuda.max_memory_allocated(device) / 2**20
        report['training_peak_reserved_mib'] = torch.cuda.max_memory_reserved(device) / 2**20
        model.eval()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        feature, z, sdt, neighbors, geometry, band, _ = fixture(ubr, rank, 1, device)
        torch.cuda.synchronize(device)
        infer_start = time.perf_counter()
        with torch.inference_mode(), torch.cuda.amp.autocast():
            for index in range(250):
                # Sequential instances, one tensor resident; not a 250-instance batch.
                current_z = z + index * .0001
                # SDT/band are recomputed because logit thresholds can cross.
                current_sdt, current_band = ubr.prediction_boundary_state(current_z, .4, 8)
                final = model.module(feature, current_z, current_sdt, neighbors, geometry)
                if not torch.isfinite(final).all():
                    raise AssertionError('nonfinite inference result')
                if not torch.equal(final[~current_band], current_z[~current_band]):
                    raise AssertionError('inference changed outside-band logits')
                if not torch.equal((final.sigmoid() >= .4)[~current_band],
                                   (current_z.sigmoid() >= .4)[~current_band]):
                    raise AssertionError('inference changed outside-band binary mask')
        torch.cuda.synchronize(device)
        report['inference'] = {'num_instances': 250, 'native_hw': [512, 512],
                               'execution': 'sequential one instance at a time per rank',
                               'includes_scipy_sdt_cpu_transfer': True,
                               'seconds': time.perf_counter() - infer_start,
                               'peak_allocated_mib': torch.cuda.max_memory_allocated(device) / 2**20,
                               'peak_reserved_mib': torch.cuda.max_memory_reserved(device) / 2**20,
                               'finite': True, 'outside_logits_and_binary_exact': True,
                               'amp': True, 'scope': 'UBR only, optimizer states still resident'}
        if source_sha != {'module': sha_file(source_path), 'gate': sha_file(__file__)}:
            raise AssertionError('source changed during component gate')
        report['success'] = True
    except Exception:
        failed = traceback.format_exc()
        report['success'] = False
        report['error'] = failed
        write_json(args.output / f'rank{rank}_result.json', report)
        write_json(args.output / f'failed_rank{rank}.json', {'rank': rank, 'error': failed})
        # Fail immediately: torchrun terminates peer ranks. Do not enter an
        # unmatched object collective while peers may still be in backward.
        raise RuntimeError(failed)
    write_json(args.output / f'rank{rank}_result.json', report)
    # Exceptions during a rank's collectives cause torchrun failure/timeout; no
    # completion marker is written unless all four reports positively succeed.
    reports = all_reports(report)
    if rank == 0:
        complete = {'world_size': 4, 'ranks': reports, 'source_sha': source_sha,
                    'scope': 'synthetic UBR component engineering gate, NOT end-to-end model',
                    'success': all(row['success'] for row in reports)}
        write_json(args.output / ('complete.json' if complete['success'] else 'failed.json'), complete)
    if failed or not all(row['success'] for row in reports):
        raise RuntimeError(failed or 'another rank failed')
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
