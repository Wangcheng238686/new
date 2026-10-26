"""Local training-reproduction wrapper for the geometry proposal pilot.
Rebinds the pinned roots to the local mirror WITHOUT touching the pilot
source (its sha256 is protocol-locked); argv is forwarded to pilot.main().
Usage: repro_pilot_local.py {gate|train} [--content geometry|constant]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import ue_geometry_proposal_pilot_20261024 as pilot
import ue_geometry_mechanism_probe as g
_L = Path('/data2/wangcheng/remote-lthpc')
pilot.OUT = _L / 'checkpoint/ue_geometry_proposal_pilot_local_20261024'
pilot.CACHE = _L / 'cache/ue_v1_native_geometry'
pilot.CONSTANTS = _L / 'checkpoint/ue_geometry_mechanism_20261022/common_constants.json'
g.DATA = _L / 'dataset/CitySample_SatUAV_TrainMinimal_v1_20261021'
pilot.main()
