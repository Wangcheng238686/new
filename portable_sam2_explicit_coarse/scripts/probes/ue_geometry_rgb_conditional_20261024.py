"""Conditional-eval launcher for the RGB-augmented geometry arms.

Reuses the (already port-adapted) conditional online runner verbatim via
module substitution: NetRGB / rgb cache / rgb content control / rgb pilot
root. The runner's own asserts (checkpoint epoch/step/content, strong-SAT
reference, prefix construction, mask consumption) run unchanged.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import ue_geometry_conditional_online_20261024 as cond
import ue_geometry_proposal_pilot_20261024 as pilot
import ue_geometry_mechanism_probe as g
import ue_geometry_rgb_pilot_20261024 as rgbp

_L = Path('/data2/wangcheng/remote-lthpc')
# conditional's import-time redirect bound pilot.* to the remote pilot root;
# rebind to the RGB pilot root + rgb cache and swap the module hooks
pilot.OUT = rgbp.OUT
pilot.CACHE = rgbp.pilot.CACHE
pilot.Net = rgbp.Net
pilot.content = rgbp.content
g.DATA = _L / 'dataset/CitySample_SatUAV_TrainMinimal_v1_20261021'
# runner OUT: separate tree so the reproduced artifacts stay pristine
cond.OUT = _L / 'checkpoint/ue_geometry_rgb_conditional_20261024'
# GEO_LOCAL_WEIGHTS semantics apply (weights differ from any cached records)
os.environ['GEO_LOCAL_WEIGHTS'] = '1'

if __name__ == '__main__':
    cond.main()
