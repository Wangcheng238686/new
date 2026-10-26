"""Independent common-foundation joint runner; frozen probes stay untouched."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

WORK = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(WORK), str(WORK / 'scripts/probes')]
import ue_instance_interaction_20261024 as core
from data.ue_joint_inputs import JointUAVInputs
from data.ue_instance_targets import center_targets
from rsprompter.uav.compact_joint import CompactJoint, build_foundation_satellite
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path('/data/wangcheng/checkpoint/ue_compact_joint_20261024')
PLANE = Path('/data/wangcheng/cache/ue_v1_reference_plane_20261024')
FOUNDATION = Path('/data/wangcheng/pretrained-models/sam2/sam2_hiera_base_plus.pt')
PROTOCOL_FILE = WORK / 'configs/ablations/ue_compact_joint_protocol.json'
PROTOCOL = json.loads(PROTOCOL_FILE.read_text())
NEW_SOURCES = ['scripts/probes/ue_compact_joint_run_20261024.py',
    'rsprompter/uav/compact_joint.py', 'rsprompter/uav/rgb_p0_encoder.py',
    'rsprompter/uav/geometry_inputs.py', 'data/ue_joint_inputs.py', 'data/ue_instance_targets.py',
    'configs/ablations/ue_compact_joint_protocol.json', 'configs/ablations/ue_minimal_single.py',
    'configs/ablations/ue_rgbd_single.py', 'configs/a3sgm_whu1024.py']


class AmpOverflow(RuntimeError):
    def __init__(self, old_scale, new_scale):
        super().__init__(f'AMP backoff {old_scale} -> {new_scale}; no optimizer step')
        self.old_scale, self.new_scale = old_scale, new_scale


def sources():
    return {**core.source_hashes(), **{name: core.C.file_sha(WORK / name) for name in NEW_SOURCES}}


def init(root):
    root.mkdir(parents=True, exist_ok=True)
    assert not (root / 'common_init.pth').exists()
    source = sources()
    loads = []
    actual = torch.load
    def generic_only(path, *args, **kwargs):
        name = path if isinstance(path, (str, Path)) else getattr(path, 'name', None)
        assert name and Path(name).resolve() == FOUNDATION.resolve(), name
        loads.append(str(Path(name).resolve()))
        return actual(path, *args, **kwargs)
    core.C.seed(PROTOCOL['seed'])
    with patch('torch.load', side_effect=generic_only):
        satellite, config = build_foundation_satellite(core.C._register_and_build,
            WORK / 'configs/ablations/ue_minimal_single.py', FOUNDATION)
        satellite_sha = core.C.state_digest(satellite)
        joint = CompactJoint(satellite, FOUNDATION, source_seed=2045)
    assert core.C.state_digest(joint.satellite) == satellite_sha
    assert satellite_sha != core.previous.FROZEN_SHA
    state = dict(satellite=joint.satellite.state_dict(), uav=joint.uav.state_dict(), model_config=config,
        satellite_initial_sha=satellite_sha, uav_initial_sha=core.previous.tensor_digest(joint.uav.state_dict()),
        foundation_sha256=core.C.file_sha(FOUNDATION), source_hashes=source, protocol=PROTOCOL,
        actual_torch_load_paths=loads, task_weights_loaded=False)
    core.previous.atomic_save(state, root / 'common_init.pth')
    core.C.dump(root / 'initialization.json', dict(common_init_sha256=core.C.file_sha(root / 'common_init.pth'),
        satellite_initial_sha=satellite_sha, uav_initial_sha=state['uav_initial_sha'],
        foundation_sha256=state['foundation_sha256'], source_hashes=source, protocol=PROTOCOL,
        actual_torch_load_paths=loads, task_weights_loaded=False))
    core.write_sources(root, source)


class Runtime:
    def __init__(self, root, arm):
        assert arm in PROTOCOL['arms']
        self.root, self.arm = Path(root), arm
        identity = json.loads((self.root / 'initialization.json').read_text())
        assert identity['source_hashes'] == sources() and identity['protocol'] == PROTOCOL
        assert identity['common_init_sha256'] == core.C.file_sha(self.root / 'common_init.pth')
        cp = torch.load(self.root / 'common_init.pth', map_location='cpu', weights_only=False)
        assert cp['task_weights_loaded'] is False and cp['foundation_sha256'] == core.C.file_sha(FOUNDATION)
        core.C.seed(PROTOCOL['seed'])
        satellite, cfg = build_foundation_satellite(core.C._register_and_build,
            WORK / 'configs/ablations/ue_minimal_single.py', FOUNDATION)
        assert cfg == cp['model_config']
        satellite.load_state_dict(cp['satellite'], strict=True)
        assert core.C.state_digest(satellite) == cp['satellite_initial_sha']
        self.joint = None
        if arm != 'satellite':
            self.joint = CompactJoint(satellite, FOUNDATION, source_seed=2045)
            self.joint.uav.load_state_dict(cp['uav'], strict=True)
            assert core.previous.tensor_digest(self.joint.uav.state_dict()) == cp['uav_initial_sha']
            self.network = self.joint.cuda()
        else:
            self.network = satellite.cuda()
        self.satellite = satellite
        self.satellite.rpn_head.test_cfg.nms_pre = PROTOCOL['RPN_pre']
        self.uav_inputs = JointUAVInputs(core.previous.DATA, '/data/wangcheng/cache/ue_v1_native_geometry',
            '/data/wangcheng/cache/ue_v1_s2db_geometry_v2', PLANE)
        self.generator = torch.Generator().manual_seed(PROTOCOL['seed'])
        self.optimizer = torch.optim.AdamW([p for p in self.network.parameters() if p.requires_grad],
            lr=PROTOCOL['lr'], weight_decay=PROTOCOL['weight_decay'])
        self.scaler = torch.amp.GradScaler('cuda', init_scale=PROTOCOL['amp_init_scale'])
        core.C.seed(PROTOCOL['seed'])

    def payload(self, ident):
        if self.joint is None:
            return None
        value = self.uav_inputs.load(ident,
            geometry_mode='reference_plane' if self.arm == 'reference_plane' else 'measured',
            rgb_mode='constant' if self.arm == 'constant_rgb' else 'correct')
        provenance = value.pop('provenance')
        assert provenance['measured_depth'] == (self.arm != 'reference_plane')
        return value

    def objective(self, sample, ident):
        if self.joint is None:
            losses = self.satellite.loss(sample['inputs'], sample['data_samples'])
        else:
            boxes = sample['data_samples'][0].gt_instances.bboxes.detach().cpu().numpy()
            target = center_targets(boxes)
            losses = self.joint(sample, **self.payload(ident), instance_target=target,
                                center_loss_fn=core.pilot.loss_fn)
        # Shared objective collector has no dependence on frozen task weights.
        terms = []
        for name, value in losses.items():
            if 'loss' not in name:
                continue
            for tensor in value if isinstance(value, (list, tuple)) else [value]:
                if torch.is_tensor(tensor) and tensor.requires_grad:
                    terms.append(tensor)
        assert terms
        return sum(terms)

    def clear(self):
        if self.joint is not None:
            self.joint.clear()

    def update(self, step, allow_retry=False):
        total = PROTOCOL['epochs'] * PROTOCOL['optimizer_updates_per_epoch']
        lr = PROTOCOL['lr'] * min(1., step / PROTOCOL['warmup_updates']) * (
            PROTOCOL['cosine_min_factor'] + (1 - PROTOCOL['cosine_min_factor']) * .5 *
            (1 + math.cos(math.pi * step / total)))
        for group in self.optimizer.param_groups:
            group['lr'] = lr
        self.scaler.unscale_(self.optimizer)
        try:
            norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), PROTOCOL['clip'], error_if_nonfinite=True)
        except RuntimeError:
            nonfinite = any(p.grad is not None and not torch.isfinite(p.grad).all()
                            for p in self.network.parameters())
            if not allow_retry or not nonfinite:
                raise
            old_scale = self.scaler.get_scale()
            # unscale_ already found the nonfinite gradient, so this call MUST
            # skip AdamW, including weight decay and moment updates.
            self.scaler.step(self.optimizer)
            self.scaler.update()
            new_scale = self.scaler.get_scale()
            self.optimizer.zero_grad(set_to_none=True)
            assert new_scale < old_scale
            raise AmpOverflow(old_scale, new_scale)
        old = self.scaler.get_scale()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        assert self.scaler.get_scale() >= old, 'AMP skipped update; preserve real update budget'
        self.optimizer.zero_grad(set_to_none=True)
        return float(norm)

    def set_epoch(self, epoch):
        for module in self.network.modules():
            setter = getattr(module, 'set_current_epoch', None)
            if callable(setter):
                setter(epoch)

    def state(self, epoch, step, contract, best):
        return dict(model=self.network.state_dict(), optimizer=self.optimizer.state_dict(),
            scaler=self.scaler.state_dict(), rng_state=core.previous.rng_state(self.generator),
            epoch=epoch, step=step, contract=contract, best=best, checkpoint_phase='after_validation')

    def restore(self, path, contract, training=True):
        cp = torch.load(path, map_location='cpu', weights_only=False)
        assert cp['contract'] == contract
        assert cp['checkpoint_phase'] == 'after_validation'
        self.network.load_state_dict(cp['model'], strict=True)
        self.set_epoch(max(1, cp['epoch']))
        if training:
            self.optimizer.load_state_dict(cp['optimizer'])
            self.scaler.load_state_dict(cp['scaler'])
            core.previous.restore_rng(cp['rng_state'], self.generator)
        return cp


def accumulation(runtime, batches, step, before_update=None):
    """One real update; any AMP retry reuses prepared CPU batches and loss RNG.

    Capture after collecting the eight batches: the sampler has already queued
    its next indices. Restoring this generator state does not rewind workers.
    """
    assert len(batches) == PROTOCOL['accumulation']
    rng = core.previous.rng_state(runtime.generator)
    buffers = {name: value.detach().clone() for name, value in runtime.network.named_buffers()}
    history = []
    for attempt in range(5):
        if attempt:
            core.previous.restore_rng(rng, runtime.generator)
            with torch.no_grad():
                for name, value in runtime.network.named_buffers():
                    value.copy_(buffers[name])
        runtime.optimizer.zero_grad(set_to_none=True)
        values = []
        for batch, ident in batches:
            sample = core.C.processed(runtime.satellite, batch, True)
            with torch.autocast('cuda', dtype=torch.float16):
                loss = runtime.objective(sample, ident)
            assert torch.isfinite(loss), 'Nonfinite forward loss is not recoverable by AMP backoff'
            values.append(float(loss.detach()))
            runtime.scaler.scale(loss / PROTOCOL['accumulation']).backward()
            runtime.clear()
        history.append(values)
        if before_update is not None:
            before_update(attempt)
        try:
            norm = runtime.update(step, allow_retry=True)
            return dict(loss_sum=sum(values), last_loss=values[-1], preclip_norm=norm,
                        retries=attempt, attempt_losses=history)
        except AmpOverflow as error:
            print('AMP RETRY', runtime.arm, step, attempt + 1, error.old_scale, error.new_scale, flush=True)
            if attempt == 4:
                raise RuntimeError('AMP failed after four retries; keep last valid epoch checkpoint') from error
    raise AssertionError('unreachable')


def contract(runtime, require_plane_complete=True):
    if require_plane_complete:
        complete = json.loads((PLANE / 'BUILD_COMPLETE.json').read_text())
        assert complete['complete'] and complete['images'] == 3060
    return dict(protocol=PROTOCOL, arm=runtime.arm, source_hashes=sources(),
        initialization=json.loads((runtime.root / 'initialization.json').read_text()),
        data_identity=core.C.archive_identity(core.previous.DATA),
        plane_manifest_sha256=core.C.file_sha(PLANE / 'manifest.json'),
        measured_cache_manifest_sha256=core.C.file_sha(core.OUT / 'input_cache_sha256.json'))


def evaluate(runtime, split, out, checkpoint_sha256=None, limit=None):
    assert split in ('val', 'test') and not torch.is_autocast_enabled()
    runtime.network.eval()
    dataset = core.C.MinimalSatelliteDataset(core.previous.DATA, split)
    records, metas = [], []
    count = min(limit, len(dataset)) if limit else len(dataset)
    with torch.inference_mode():
        for i in range(count):
            ident = dataset.samples[i]['image_id']
            batch = core.C.rtmdet_collate_fn([dataset[i]])
            sample = core.C.processed(runtime.satellite, batch)
            if runtime.joint is not None:
                payload = runtime.payload(ident)
                runtime.joint.prepare_uav(**payload)
                adapter = runtime.joint.uav
                geometry = payload['geometry']
            else:
                adapter = SimpleNamespace(enabled=False)
                geometry = None
            features = runtime.satellite.extract_feat(sample['inputs'])
            prefix, combined, accepted = core.inference_outputs(runtime.satellite, adapter,
                sample, features, geometry)
            records.append(dict(image_id=ident, gt=core.C.serialized(core.C.gt_arrays(
                sample['data_samples'][0].gt_instances)), dt=core.C.serialized(combined),
                geometry_accepted=accepted, original_path=core.C.serialized(prefix)))
            metas.append(batch['img_metas'][0])
            runtime.clear()
    gt = [{**row['gt'], 'bboxes': np.asarray(row['gt']['bboxes'], np.float32),
           'labels': np.asarray(row['gt']['labels'], np.int64)} for row in records]
    dt = [core.consumer.arrays(row['dt']) for row in records]
    cg, cd = core.C.build_coco_gt_and_dt(gt, dt, metas, score_key='scores')
    metrics = {}
    for kind in ('segm', 'bbox'):
        metrics.update(core.C.run_coco_eval(cg, cd, kind, max_dets=[1, 10, 250]))
    # The shared legacy utility names its third recall slot AR@100 even when
    # maxDets[2] is 250. Keep its frozen numeric behavior, label this run truthfully.
    metrics = {name.replace('AR@100', 'AR@250'): value for name, value in metrics.items()}
    core.C.dump(out / 'records.json', records)
    core.C.dump(out / 'metrics.json', metrics)
    core.C.dump(out / 'complete.json', dict(complete=True, split=split, images=len(records),
        precision='FP32', max_dets=250, checkpoint_sha256=checkpoint_sha256, arm=runtime.arm,
        source_hashes=sources(), purpose='engineering_subset' if limit else 'full_protocol'))
    return metrics


def train(runtime, out, frozen, resume=None):
    ready = json.loads((runtime.root / 'engineering_complete.json').read_text())
    assert ready['passed'] and ready['source_hashes'] == sources()
    dataset = core.C.MinimalSatelliteDataset(core.previous.DATA, 'train', augment=True)
    assert len(dataset) == 2142
    out.mkdir(parents=True, exist_ok=True)
    core.C.dump(out / 'contract.json', frozen)
    epoch = step = 0
    best = dict(score=-1., epoch=None)
    if resume:
        cp = runtime.restore(resume, frozen)
        epoch, step, best = cp['epoch'], cp['step'], cp['best']
        assert step == epoch * PROTOCOL['optimizer_updates_per_epoch']
        del cp
    else:
        assert not (out / 'last.pth').exists(), 'Use explicit resume; do not overwrite'
    for epoch in range(epoch + 1, PROTOCOL['epochs'] + 1):
        runtime.network.train()
        runtime.set_epoch(epoch)
        loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2,
                            generator=runtime.generator, collate_fn=core.C.rtmdet_collate_fn)
        runtime.optimizer.zero_grad(set_to_none=True)
        order, loss_sum, retry_count, prepared = [], 0., 0, []
        began = time.time()
        for microbatch, batch in enumerate(loader):
            if microbatch >= PROTOCOL['used_images_per_epoch']:
                break
            ident = int(Path(batch['img_metas'][0]['filename']).parent.parent.name)
            order.append(ident)
            prepared.append((batch, ident))
            if len(prepared) == PROTOCOL['accumulation']:
                value = accumulation(runtime, prepared, step + 1)
                step += 1
                prepared = []
                loss_sum += value['loss_sum']
                retry_count += value['retries']
                if step % 25 == 0:
                    core.C.dump(out / 'progress.json', dict(stage='train', epoch=epoch, update=step,
                        last_microbatch=microbatch + 1, last_loss=value['last_loss'],
                        preclip_norm=value['preclip_norm'], AMP_retries_epoch=retry_count, pid=os.getpid()))
                    print('JOINT UPDATE', runtime.arm, epoch, step, flush=True)
        assert not prepared and len(order) == len(set(order)) == 2136 and step == epoch * 267
        metrics = evaluate(runtime, 'val', out / 'validation_current')
        improved = metrics['segm/mAP'] > best['score']
        if improved:
            best = dict(score=metrics['segm/mAP'], epoch=epoch)
        state = runtime.state(epoch, step, frozen, best)
        core.previous.atomic_save(state, out / 'last.pth')
        if improved:
            shutil.copyfile(out / 'last.pth', out / 'best_model.pth')
            core.C.dump(out / 'best_validation_metrics.json', dict(epoch=epoch, metrics=metrics))
        if epoch in (1, 3, 10, 25, 50, 100):
            shutil.copyfile(out / 'last.pth', out / f'epoch{epoch}.pth')
        core.C.dump(out / f'epoch{epoch}.json', dict(epoch=epoch, step=step, sample_order=order,
            mean_microbatch_loss=loss_sum / len(order), validation=metrics, best=best,
            seconds=time.time() - began, AMP_retries_epoch=retry_count,
            peak_reserved_MiB=torch.cuda.max_memory_reserved() / 2**20))
        assert frozen['source_hashes'] == sources()
    assert step == 26700
    core.C.dump(out / 'training_complete.json', dict(complete=True, epoch=epoch, updates=step,
        best=best, best_checkpoint_sha256=core.C.file_sha(out / 'best_model.pth'),
        last_checkpoint_sha256=core.C.file_sha(out / 'last.pth'), contract=frozen))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['init', 'train', 'test'])
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--arm', choices=PROTOCOL['arms'], default='rgbd')
    p.add_argument('--resume', type=Path)
    args = p.parse_args()
    if args.action == 'init':
        init(args.root)
        return
    runtime = Runtime(args.root, args.arm)
    frozen = contract(runtime)
    out = args.root / args.arm
    if args.action == 'train':
        train(runtime, out, frozen, args.resume)
    else:
        done = json.loads((out / 'training_complete.json').read_text())
        assert done['complete'] and done['contract'] == frozen
        selected = out / 'best_model.pth'
        assert core.C.file_sha(selected) == done['best_checkpoint_sha256']
        assert not (out / 'test/complete.json').exists(), 'Test already completed'
        runtime.restore(selected, frozen, training=False)
        evaluate(runtime, 'test', out / 'test', checkpoint_sha256=done['best_checkpoint_sha256'])


if __name__ == '__main__':
    main()
