"""Measured whole inference path at fixed old weights; no AP/model selection.

Compare strong SAT, old full-source RGB-D, and the exact compact-source
replacement. This is not latency of a trained common-foundation joint model.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

WORK = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(WORK), str(WORK / 'scripts/probes')]
import ue_instance_interaction_20261024 as core
from rsprompter.uav.compact_joint import CompactInteraction
from rsprompter.uav.rgb_p0_encoder import RGBP0Encoder
from data.ue_joint_inputs import JointUAVInputs
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    assert not (args.out / 'complete.json').exists()
    cfg = argparse.Namespace(root=core.OUT, seed=44, family='rgb', rgb='correct', route='internal', content='geometry')
    model, full, _, _ = core.build(cfg)
    checkpoint = core.OUT / 'seed44_rgb_internal_geometry_correct/last.pth'
    cp = torch.load(checkpoint, map_location='cpu', weights_only=False)
    full.load_state_dict(cp['adapter'], strict=True)
    del cp
    full.eval(); model.eval()
    compact = CompactInteraction()
    compact.add_rgb('/data/wangcheng/pretrained-models/sam2/sam2_hiera_base_plus.pt', 2045)
    state = compact.state_dict()
    reference = full.state_dict()
    for name in state:
        if not name.startswith('rgb_encoder.'):
            state[name] = reference[name]
    p0 = RGBP0Encoder(full.rgb_encoder)
    for name, value in p0.state_dict().items():
        state['rgb_encoder.p0.' + name] = value
    compact.load_state_dict(state, strict=True)
    compact.cuda().eval()
    del p0, state, reference
    source = {**core.source_hashes(), **{name: core.C.file_sha(WORK / name) for name in (
        'rsprompter/uav/compact_joint.py', 'rsprompter/uav/rgb_p0_encoder.py',
        'data/ue_joint_inputs.py', 'scripts/analysis/ue_compact_cost_audit_20261024.py')}}
    checkpoint_sha = core.C.file_sha(checkpoint)
    inputs = JointUAVInputs(core.previous.DATA, '/data/wangcheng/cache/ue_v1_native_geometry',
        '/data/wangcheng/cache/ue_v1_s2db_geometry_v2', '/data/wangcheng/cache/ue_v1_reference_plane_20261024')
    ds = core.C.MinimalSatelliteDataset(core.previous.DATA, 'val')
    indices = [0, len(ds) // 2, len(ds) - 1]
    cases = ('satellite', 'full_source_rgbd', 'compact_source_rgbd')
    protocol = dict(source_hashes=source, checkpoint_sha256=checkpoint_sha,
        satellite_state_sha=core.previous.FROZEN_SHA, fixed_val_indices=indices,
        image_ids=[ds.samples[i]['image_id'] for i in indices], warmup_per_case=1,
        repetitions=3, precision='FP32', max_dets=250,
        scope='Old E73 satellite and fixed RGB-D E10; identical full/compact weights; no joint model/AP claim',
        timing='Wall includes eval image/GT loading, UAV cache and RGB reads, preprocessing, inference and RLE output. CUDA event intervals may include host-induced gaps.',
        memory='All three variants share resident old weights; report additional peak above resident, not standalone deployment memory')
    core.C.dump(args.out / 'protocol.json', protocol)
    def once(case, index):
        for adapter in (full, compact):
            for handle in adapter.handles:
                handle.remove()
            adapter.handles = []
            adapter.clear()
            adapter.rgb_context = None
        adapter = full if case == 'full_source_rgbd' else compact
        adapter.enabled = case != 'satellite'
        adapter.attach(model)
        core.C.seed(44)
        torch.cuda.synchronize()
        start_memory = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        began = time.perf_counter()
        ident = ds.samples[index]['image_id']
        batch = core.C.rtmdet_collate_fn([ds[index]])
        sample = core.C.processed(model, batch)
        payload = inputs.load(ident) if adapter.enabled else None
        input_ready = time.perf_counter()
        events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        events[0].record()
        if adapter.enabled:
            adapter.rgb_context = (payload['rgb'], payload['index'], payload['anchor'])
            adapter.encode(payload['geometry'])
        events[1].record()
        features = model.extract_feat(sample['inputs'])
        events[2].record()
        prefix, combined, accepted = core.inference_outputs(model, adapter, sample, features,
            payload['geometry'] if payload else None)
        events[3].record()
        torch.cuda.synchronize()
        wall = (time.perf_counter() - began) * 1000
        serialized = core.C.serialized(combined)
        import hashlib
        digest = hashlib.sha256(json.dumps(serialized, sort_keys=True).encode()).hexdigest()
        value = dict(case=case, image_id=ident, wall_ms=wall,
            eval_input_wall_ms=(input_ready - began) * 1000,
            uav_prepare_stream_ms=events[0].elapsed_time(events[1]),
            satellite_with_interaction_stream_ms=events[1].elapsed_time(events[2]),
            detection_mask_RLE_stream_ms=events[2].elapsed_time(events[3]),
            additional_peak_allocated_MiB=(torch.cuda.max_memory_allocated() - start_memory) / 2**20,
            original_instances=len(prefix['scores']), combined_instances=len(combined['scores']),
            uav_instances=accepted, prediction_sha256=digest)
        adapter.clear(); adapter.rgb_context = None
        return value
    rows = []
    with torch.inference_mode():
        for case in cases:
            once(case, indices[0])
        for repetition in range(3):
            for index in indices:
                # Rotate execution order to reduce a fixed ordering effect.
                order = cases[repetition:] + cases[:repetition]
                paired = {}
                for case in order:
                    value = once(case, index)
                    value['repetition'] = repetition
                    rows.append(value)
                    paired[case] = value
                    print('COST', case, value['image_id'], value['wall_ms'], flush=True)
                assert paired['full_source_rgbd']['prediction_sha256'] == paired['compact_source_rgbd']['prediction_sha256']
    summary = {}
    for case in cases:
        selected = [row for row in rows if row['case'] == case]
        keys = ('wall_ms', 'eval_input_wall_ms', 'uav_prepare_stream_ms',
                'satellite_with_interaction_stream_ms', 'detection_mask_RLE_stream_ms',
                'additional_peak_allocated_MiB', 'combined_instances')
        summary[case] = {key: statistics.median(row[key] for row in selected) for key in keys}
    assert core.C.state_digest(model) == core.previous.FROZEN_SHA
    assert core.C.file_sha(checkpoint) == checkpoint_sha
    assert all(core.C.file_sha(WORK / name) == sha for name, sha in source.items())
    core.C.dump(args.out / 'rows.json', rows)
    core.C.dump(args.out / 'complete.json', dict(complete=True, samples_per_case=9,
        full_compact_prediction_exact=True, summary=summary, source_hashes=source,
        checkpoint_sha256=checkpoint_sha, frozen_satellite_exact=True, limitations=protocol['scope'],
        baseline_comparison='SAT and RGB-D may predict different counts; full versus compact outputs exact. Eval-label/cache loading and warm page cache are included, not deployment latency.'))
    print('COMPACT COST AUDIT COMPLETE', summary, flush=True)


if __name__ == '__main__':
    main()
