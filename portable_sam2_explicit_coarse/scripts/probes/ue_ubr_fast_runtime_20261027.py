"""Fresh generic runtime, with an independent efficient P1 controller.

The original runtime/controller files are imported unchanged. Parameters,
initialization and the source capture hook are shared exactly; only the
controller executing the fixed UBR operation is replaced in this process.
"""
import ue_ubr_runtime_20261027 as original
from rsprompter.uav.ubr_detector_fast import FastUBRDetectorIntegration


def build_runtime(content='correct', variant='detail', enabled=True):
    rt = original.build_runtime(content=content, variant=variant, enabled=enabled)
    before = {n: id(p) for n, p in rt.network.named_parameters()}
    fast = FastUBRDetectorIntegration.from_controller(rt.controller, active_microbatch=1, inference_microbatch=4)
    rt.controller = fast
    rt.network.controller = fast
    assert {n: id(p) for n, p in rt.network.named_parameters()} == before
    rt.initialization_audit = dict(rt.initialization_audit,
        controller='FastUBRDetectorIntegration', active_microbatch=1, inference_microbatch=4,
        old_refiner_parameter_objects_preserved=True,
        efficiency_gate_required=True)
    return rt
