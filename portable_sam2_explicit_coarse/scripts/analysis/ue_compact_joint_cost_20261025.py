"""Final selected-weight standalone evaluation-path cost on fixed val images.

Each case runs in a separate otherwise idle GPU process. No AP/model selection.
RGBD original-only keeps feature interaction and computes the instance head, but
skips decoding/masking its appended candidates. Cache production is excluded.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

WORK = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(WORK), str(WORK / 'scripts/probes')]
import ue_compact_joint_run_20261024 as run
import torch

CASES = ('satellite', 'rgbd_combined', 'rgbd_original')


def isolated(physical_gpu):
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid',
        '--format=csv,noheader'], text=True)
    uuids = {int(s.split(',')[0]): s.split(',')[1].strip() for s in raw.splitlines()}
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid',
        '--format=csv,noheader'], text=True)
    other = [int(s.split(',')[0]) for s in raw.splitlines()
             if s.split(',')[1].strip() == uuids[physical_gpu] and int(s.split(',')[0]) != os.getpid()]
    assert not other, ('Cost requires an otherwise idle GPU', other)
    return uuids[physical_gpu]


def pred_sha(row, count=None):
    value = run.core.C.serialized(row)
    result = {k: value[k][:count] if count is not None else value[k]
              for k in ('bboxes', 'labels', 'scores', 'rles')}
    return hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=run.ROOT)
    p.add_argument('--case', choices=CASES, required=True)
    p.add_argument('--physical-gpu', type=int, required=True)
    args = p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == str(args.physical_gpu)
    gpu_uuid = isolated(args.physical_gpu)
    arm = 'satellite' if args.case == 'satellite' else 'rgbd'
    out = args.root / 'joint_cost' / args.case
    assert not out.exists(), 'Preserve started cost runs'
    done = json.loads((args.root / arm / 'training_complete.json').read_text())
    assert done['complete'] and done['epoch'] == 100 and done['updates'] == 26700
    selected = args.root / arm / 'best_model.pth'
    assert run.core.C.file_sha(selected) == done['best_checkpoint_sha256']
    out.mkdir(parents=True)
    runtime = run.Runtime(args.root, arm)
    contract = run.contract(runtime)
    assert contract == done['contract']
    runtime.restore(selected, contract, training=False)
    runtime.network.eval()
    initial = run.core.C.state_digest(runtime.network)
    dataset = run.core.C.MinimalSatelliteDataset(run.core.previous.DATA, 'val')
    indices = [0, len(dataset)//2, len(dataset)-1]
    protocol = dict(case=args.case, arm=arm, physical_gpu=args.physical_gpu, gpu_uuid=gpu_uuid,
        gpu_name=torch.cuda.get_device_name(0), checkpoint_sha256=done['best_checkpoint_sha256'],
        checkpoint_epoch=done['best']['epoch'], fixed_val_indices=indices,
        image_ids=[dataset.samples[i]['image_id'] for i in indices],
        precision='FP32', max_dets=250, warmup_each_image=1, repetitions=3,
        source_hashes=run.sources(), driver_sha256=run.core.C.file_sha(Path(__file__)),
        total_parameters=sum(t.numel() for t in runtime.network.parameters()),
        uav_parameters=sum(t.numel() for t in runtime.joint.uav.parameters()) if runtime.joint else 0,
        timing='wall includes eval image+GT/input loading and cached geometry/PNG IO; prepared inference wall excludes that input stage; both include RLE, exclude prediction hashing, model construction, source auditing and cache production',
        memory='standalone process allocator peaks include model and transient inputs; CUDA context/library allocations are outside allocated/reserved statistics',
        original_only='same RGB-D model/interaction and instance-head forward; skip appended candidate decode/masks only')
    run.core.C.dump(out / 'protocol.json', protocol)
    (out / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]

    def once(index):
        isolated(args.physical_gpu)
        runtime.clear()
        run.core.C.seed(44)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        began = time.perf_counter()
        ident = dataset.samples[index]['image_id']
        batch = run.core.C.rtmdet_collate_fn([dataset[index]])
        sample = run.core.C.processed(runtime.satellite, batch)
        payload = runtime.payload(ident) if runtime.joint else None
        input_ready = time.perf_counter()
        events[0].record()
        if runtime.joint:
            runtime.joint.prepare_uav(**payload)
            adapter = runtime.joint.uav
        else:
            adapter = SimpleNamespace(enabled=False)
        events[1].record()
        features = runtime.satellite.extract_feat(sample['inputs'])
        events[2].record()
        # Interaction has already happened. This switch affects only the
        # optional appended-candidate part of the formal inference function.
        if args.case == 'rgbd_original':
            assert adapter.enabled
            adapter.enabled = False
        try:
            prefix, combined, accepted = run.core.inference_outputs(runtime.satellite, adapter,
                sample, features, payload['geometry'] if payload else None)
        finally:
            if args.case == 'rgbd_original':
                adapter.enabled = True
        events[3].record()
        torch.cuda.synchronize()
        ended = time.perf_counter()
        result = dict(image_id=ident, wall_ms=(ended-began)*1000,
            eval_input_wall_ms=(input_ready-began)*1000,
            prepared_inference_wall_ms=(ended-input_ready)*1000,
            uav_prepare_stream_ms=events[0].elapsed_time(events[1]),
            satellite_with_interaction_stream_ms=events[1].elapsed_time(events[2]),
            detection_mask_RLE_stream_ms=events[2].elapsed_time(events[3]),
            peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
            peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20,
            original_instances=len(prefix['scores']), output_instances=len(combined['scores']),
            appended_instances=accepted, prediction_sha256=pred_sha(combined),
            retained_prefix_sha256=pred_sha(combined, len(prefix['scores'])))
        runtime.clear()
        isolated(args.physical_gpu)
        return result

    rows = []
    with torch.inference_mode():
        for index in indices:
            once(index)
        for repetition in range(3):
            for index in indices:
                value = once(index)
                value['repetition'] = repetition
                rows.append(value)
                run.core.C.dump(out / 'rows.json', rows)
                print('JOINT COST', args.case, value['image_id'], value['wall_ms'], flush=True)
    for index in indices:
        ident = dataset.samples[index]['image_id']
        selected_rows = [r for r in rows if r['image_id'] == ident]
        assert len({r['prediction_sha256'] for r in selected_rows}) == 1
    assert run.core.C.state_digest(runtime.network) == initial and not runtime.optimizer.state
    assert run.core.C.file_sha(selected) == done['best_checkpoint_sha256']
    assert run.sources() == protocol['source_hashes']
    keys = ['wall_ms', 'eval_input_wall_ms', 'prepared_inference_wall_ms', 'uav_prepare_stream_ms',
            'satellite_with_interaction_stream_ms', 'detection_mask_RLE_stream_ms',
            'peak_allocated_MiB', 'peak_reserved_MiB', 'original_instances', 'output_instances', 'appended_instances']
    run.core.C.dump(out / 'complete.json', dict(complete=True, protocol=protocol, samples=9,
        summary={k: statistics.median(r[k] for r in rows) for k in keys},
        model_parameters_and_persistent_buffers_unchanged=True,
        peak_scope='each-case separate process; median per-forward peaks, not a confidence interval',
        limitations='3 fixed val images and warm page/allocator cache; eval-input timing includes GT; events may include host gaps; not throughput, cold start or universal deployment latency'))


if __name__ == '__main__':
    main()
