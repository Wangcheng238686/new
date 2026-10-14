"""Check missing/nonfinite DDP telemetry and unchanged AMP fusion gradients."""

import argparse
import copy
import importlib.util
import json
import logging
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "train"))
from ue_telemetry import UEEpochTelemetry


class TelemetryModel:
    def __init__(self, values):
        self.values = values

    def get_telemetry(self):
        return self.values


def check_epoch():
    distributed = "RANK" in os.environ
    if distributed:
        dist.init_process_group("gloo")
    rank = dist.get_rank() if distributed else 0
    monitor = UEEpochTelemetry("cpu")
    monitor.accumulate(
        TelemetryModel({"UAV/common": rank + 1.0, "UAV/nonfinite": float("nan")})
    )
    monitor.accumulate(
        TelemetryModel({"UAV/common": rank + 3.0, f"UAV/only_rank{rank}": 8.0})
    )
    monitor.record_step(512, 256)
    monitor.record_step(256, 256)
    report = monitor.finish(
        logging.getLogger("test"),
        1,
        distributed,
        rank == 0,
        torch.amp.GradScaler("cpu", enabled=False),
    )
    world = dist.get_world_size() if distributed else 1
    assert report["UAV/common"] == {
        "mean": (world - 1) / 2 + 2,
        "finite_forwards": 2 * world,
    }
    assert "UAV/nonfinite" not in report
    for r in range(world):
        assert report[f"UAV/only_rank{r}"] == {"mean": 8.0, "finite_forwards": 1}
    assert report["runtime"]["rank_step_attempts"] == 2 * world
    assert report["runtime"]["rank_amp_skips"] == world
    assert report["runtime"]["amp_skip_fraction"] == 0.5
    if distributed:
        dist.destroy_process_group()
    return dict(
        missing_and_nonfinite=True,
        cross_rank_denominators=True,
        amp_attempt_denominators=True,
    )


def check_fusion():
    spec = importlib.util.spec_from_file_location(
        "fusion", ROOT / "rsprompter/uav/fusion.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(42)
    model = module.ConfidenceFusion(256).cuda()
    reference = copy.deepcopy(model)
    satellite = torch.randn(1, 256, 8, 8, device="cuda")
    uav = torch.randn_like(satellite)
    coverage = torch.zeros(1, 1, 8, 8, device="cuda")
    coverage[:, :, :4] = 0.5
    confidence = (coverage > 0).float() * 0.8
    with torch.autocast("cuda"):
        result, stats = model(satellite, uav, confidence, coverage)
        residual = reference.project(uav)
        gate = reference.gate(
            torch.cat(
                (
                    satellite,
                    residual,
                    confidence.to(satellite.dtype),
                    coverage.to(satellite.dtype),
                ),
                dim=1,
            )
        ).sigmoid()
        expected = satellite + (
            reference.alpha * (coverage > 0) * confidence * gate * residual
        ).to(satellite.dtype)
    assert torch.equal(result, expected)
    result.float().square().mean().backward()
    expected.float().square().mean().backward()
    for (name, parameter), (other_name, other) in zip(
        model.named_parameters(), reference.named_parameters()
    ):
        assert name == other_name and torch.equal(parameter.grad, other.grad), name
    assert float(stats["supported_fraction"]) == 0.5
    assert abs(float(stats["confidence_supported"]) - 0.8) < 1e-6
    assert abs(float(stats["gate_supported"]) - float(gate[:, :, :4].detach().float().mean())) < 1e-6
    _, empty = model(satellite, uav, confidence * 0, coverage * 0)
    for key in (
        "supported_fraction",
        "gate_supported",
        "confidence_supported",
        "effective_gain_supported",
        "residual_ratio",
    ):
        assert float(empty[key]) == 0, key
    return dict(
        amp_output_identical=True,
        amp_gradients_identical=True,
        supported_denominators=True,
        empty_support=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    print(json.dumps(check_fusion() if args.gpu else check_epoch(), sort_keys=True))
