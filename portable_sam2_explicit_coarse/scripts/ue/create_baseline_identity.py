#!/usr/bin/env python3
"""Create an explicit identity manifest only for a new sealed UE checkpoint."""
import argparse
import json
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ue.contract import checkpoint_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    identity = checkpoint_identity(args.checkpoint, checkpoint)
    with args.output.open("x") as handle:
        json.dump(identity, handle, indent=2)


if __name__ == "__main__":
    main()
