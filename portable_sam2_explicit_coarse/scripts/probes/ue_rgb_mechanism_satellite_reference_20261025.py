"""Evaluate the already completed common-foundation SAT E10, validation only."""
import torch
from ue_rgb_mechanism_run_20261025 import base, ROOT

if __name__=='__main__':
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(7800/(torch.cuda.get_device_properties(0).total_memory/2**20))
    rt=base.Runtime(base.ROOT,'satellite')
    path=base.ROOT/'satellite'/'epoch10.pth'
    cp=torch.load(path,map_location='cpu',weights_only=False)
    assert cp['epoch']==10 and cp['step']==2670
    rt.network.load_state_dict(cp['model'],strict=True);rt.set_epoch(10)
    out=ROOT/'evaluation'/'satellite'/'correct'/'matched'
    assert not (out/'complete.json').exists()
    base.evaluate(rt,'val',out,checkpoint_sha256=base.core.C.file_sha(path))
