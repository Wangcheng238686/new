"""Exact source-P0 reduction gate; no probe training or model selection."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys

WORK = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(WORK), str(WORK / 'scripts/probes')]
import ue_instance_interaction_20261024 as probe
from rsprompter.uav.rgb_p0_encoder import RGBP0Encoder
import torch
from torch.nn import functional as F


def measure(function, images, repeats=5):
    for _ in range(2):
        for image in images.split(1):
            function(image)
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    start_memory = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for image in images.split(1):
            function(image)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return dict(five_views_cuda_ms=times, median_ms=sorted(times)[len(times) // 2],
        additional_peak_allocated_MiB=(torch.cuda.max_memory_allocated() - start_memory) / 2**20,
        resident_start_MiB=start_memory / 2**20,
        scope='source encoding only, FP32 inference, shared-device observation; not end-to-end latency')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    assert not (args.out / 'complete.json').exists()
    checkpoint = probe.OUT / 'seed44_rgb_internal_geometry_correct/last.pth'
    cp_sha = probe.C.file_sha(checkpoint)
    frozen = probe.source_hashes()
    sources = {name: probe.C.file_sha(WORK / name) for name in (
        'rsprompter/uav/rgb_p0_encoder.py',
        'scripts/analysis/ue_rgb_p0_efficiency_gate_20261024.py')}
    cp = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert cp['epoch'] == 10 and cp['step'] == 2670
    assert cp['contract']['source_hashes'] == frozen
    adapter = probe.InstanceInteraction()
    adapter.add_rgb('/data/wangcheng/pretrained-models/sam2/sam2_hiera_base_plus.pt', 2045)
    adapter.load_state_dict(cp['adapter'], strict=True)
    del cp
    adapter.cuda().eval()
    full = adapter.rgb_encoder
    slim = RGBP0Encoder(full).cuda().eval()
    tensors = slim.audit_source_state(full)
    ds = probe.C.MinimalSatelliteDataset(probe.previous.DATA, 'val')
    folder = probe.C.point_folder(ds, 0, probe.previous.DATA)
    payload_args = argparse.Namespace(cache=Path('/data/wangcheng/cache/ue_v1_s2db_geometry_v2'))
    rgb, indices, _, anchor = probe.previous.payload(payload_args, folder, 'correct', None)
    images = (rgb - adapter.rgb_mean) / adapter.rgb_std
    full_forward = lambda x: full(x)[1][0]
    checks = []
    with torch.inference_mode():
        for amp in (False, True):
            for view, image in enumerate(images.split(1)):
                with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
                    expected, actual = full_forward(image), slim(image)
                assert torch.equal(expected, actual), (amp, view, (expected - actual).abs().max())
                checks.append(dict(amp=amp, view=view, shape=list(actual.shape), exact=True))
        # Check exactly the downstream normalized continuous reads for all views.
        def read(function):
            maps = torch.cat([adapter.rgb_project(function(x)) for x in images.split(1)])
            features = F.layer_norm(maps.permute(0, 2, 3, 1).reshape(-1, 32), (32,))
            from rsprompter.uav.attention_ladder import sample_source_continuous
            return sample_source_continuous(features, indices, anchor)
        assert torch.equal(read(full_forward), read(slim))
        timing_full = measure(full_forward, images)
        timing_slim = measure(slim, images)
    # Per-call train-mode equivalence from the same CUDA RNG state. Removing
    # unused late-stage dropout changes the END RNG, so whole-run bit replay is
    # deliberately not claimed. New training needs its own fixed RNG contract.
    for name, module in full.named_modules():
        if 'lora_dropout' in name:
            module.train(True)
    slim.train(True)
    generator = torch.Generator(device='cuda').manual_seed(2046)
    weight = torch.randn((1, 256, 16, 16), device='cuda', generator=generator)
    rng = torch.cuda.get_rng_state()
    full.zero_grad(set_to_none=True)
    expected = full_forward(images[:1])
    (expected[:, :, ::8, ::8] * weight).sum().backward()
    full_rng_end = torch.cuda.get_rng_state().clone()
    base = full.encoder.get_base_model()
    reference = {name: p.grad.detach().clone() for name, p in base.named_parameters() if p.grad is not None}
    expected = expected.detach()
    calibration = {name: 0. for name in reference}
    for _ in range(3):
        torch.cuda.set_rng_state(rng)
        full.zero_grad(set_to_none=True)
        replay = full_forward(images[:1])
        (replay[:, :, ::8, ::8] * weight).sum().backward()
        assert torch.equal(expected, replay)
        for name, p in base.named_parameters():
            if name in reference:
                calibration[name] = max(calibration[name], float((p.grad - reference[name]).abs().max()))
        del replay
    torch.cuda.set_rng_state(rng)
    slim.zero_grad(set_to_none=True)
    actual = slim(images[:1])
    (actual[:, :, ::8, ::8] * weight).sum().backward()
    assert torch.equal(expected, actual)
    gradients = []
    for name, p in slim.named_parameters():
        if p.grad is None:
            continue
        key = slim.source_key(name)
        assert key in reference and torch.isfinite(p.grad).all(), name
        error = float((p.grad - reference[key]).abs().max())
        tolerance = max(1e-7, 2 * calibration[key])
        assert error <= tolerance, (name, error, tolerance, calibration[key])
        gradients.append(dict(name=name, numel=p.numel(), norm=float(p.grad.float().norm()),
                              max_abs_error=error, replay_noise=calibration[key], tolerance=tolerance))
    assert len(gradients) == len(reference) == 10
    assert sum(x['numel'] for x in gradients) == 25648
    full.eval()
    slim.eval()
    assert probe.C.file_sha(checkpoint) == cp_sha and probe.source_hashes() == frozen
    assert all(probe.C.file_sha(WORK / name) == sh for name, sh in sources.items())
    slim.audit_source_state(full)
    counts = dict(full_source_parameters=sum(p.numel() for p in full.parameters()),
                  p0_source_parameters=sum(p.numel() for p in slim.parameters()),
                  p0_trainable_parameters=sum(p.numel() for p in slim.parameters() if p.requires_grad))
    torch.save(dict(state_dict=slim.state_dict(), source_checkpoint_sha256=cp_sha,
                    source_hashes=sources, role='exact current source P0 only; not a jointly trained model'),
               args.out / 'p0_source.pth')
    result = dict(complete=True, checkpoint_sha256=cp_sha, source_hashes=sources,
        frozen_source_hashes=frozen, state_tensors_exact=tensors, checks=checks,
        normalized_candidate_reads_exact=True, train_forward_exact=True,
        active_gradients_within_three_replay_noise=True,
        gradients=gradients, parameter_counts=counts, full_timing=timing_full, p0_timing=timing_slim,
        end_rng_equal=torch.equal(full_rng_end, torch.cuda.get_rng_state()),
        rng_scope='Per-call same-start RNG equality only; whole-run dropout stream changes when unused stages are removed.',
        scope='Five real UAV views of fixed first val sample; not all-dataset or final detector-output equivalence, no new AP claim',
        pid=os.getpid())
    probe.C.dump(args.out / 'complete.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
