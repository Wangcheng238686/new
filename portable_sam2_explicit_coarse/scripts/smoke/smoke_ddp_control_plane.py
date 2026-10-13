#!/usr/bin/env python

"""Exercise delayed rank-0 epoch control on a CPU/Gloo process group."""

import os
import sys
import time
from datetime import timedelta
from pathlib import Path

import torch.distributed as dist

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train.train_rsprompter_fusion import _broadcast_stop_training


def main() -> None:
    dist.init_process_group(
        backend="gloo", init_method="env://", timeout=timedelta(seconds=15)
    )
    rank = int(os.environ["RANK"])
    if rank == 0:
        time.sleep(2.0)
    resolved_stop = _broadcast_stop_training(rank == 0, dist.group.WORLD)
    if not resolved_stop:
        raise AssertionError("rank-0 stop flag was not received on every rank")
    if rank == 0:
        print("DDP CPU/Gloo delayed-control smoke: OK", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
