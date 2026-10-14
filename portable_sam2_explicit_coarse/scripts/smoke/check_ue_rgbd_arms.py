#!/usr/bin/env python
"""Build every UE arm under its runner environment and attack owned env knobs."""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "train")]
os.environ["RSPROMPTER_LIGHT_IMPORT"] = "0"
os.environ.setdefault("SAM2_REPO", str(ROOT.parent / "sam2"))
from mmengine import Config
from mmengine.registry import init_default_scope
from mmdet.registry import MODELS
import rsprompter
from rsprompter.uav.contract import model_contract
from optimizer import _build_optimizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    init_default_scope("mmdet")
    clean_env = {"PATH": os.environ["PATH"], "PYTHON_BIN": sys.executable}
    entry = ["bash", str(ROOT / "scripts/ue_rgbd.sh")]
    records = {}
    for arm in ("single", "dual", "fpn", "sam"):
        owned = json.loads(
            subprocess.check_output([*entry, arm, "dump-env"], env=clean_env, text=True)
        )
        attacked = {
            **clean_env,
            **{key: "poison" for key in owned if key not in ("SAM2_REPO", "SAM2_CKPT")},
        }
        assert owned == json.loads(
            subprocess.check_output([*entry, arm, "dump-env"], env=attacked, text=True)
        )
        os.environ.update(
            {key: value for key, value in owned.items() if value is not None}
        )
        for key, value in owned.items():
            if value is None:
                os.environ.pop(key, None)
        config = Config.fromfile(str(ROOT / f"configs/ablations/ue_rgbd_{arm}.py"))
        contract = model_contract(config.model.to_dict())
        model = MODELS.build(copy.deepcopy(config.model))
        dual = arm != "single"
        assert hasattr(model, "uav_backbone") == dual
        if dual:
            assert (model.fpn_fusion is not None) == (arm in ("dual", "fpn"))
            assert (model.sam_fusion is not None) == (arm in ("dual", "sam"))
            assert not {id(p) for p in model.backbone.parameters()} & {
                id(p) for p in model.uav_backbone.parameters()
            }
        optimizer = _build_optimizer(
            model,
            lr=2e-4,
            backbone_lr_mult=0.1,
            other_lr_mult=1.0,
            bbox_head_lr_mult=1.0,
        )
        ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
        assert (
            len(ids)
            == len(set(ids))
            == sum(p.requires_grad for p in model.parameters())
        )
        records[arm] = dict(
            contract=contract,
            trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
            groups={
                g["name"]: sum(p.numel() for p in g["params"])
                for g in optimizer.param_groups
            },
            pollution_attack=True,
        )
        del optimizer, model
        gc.collect()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
