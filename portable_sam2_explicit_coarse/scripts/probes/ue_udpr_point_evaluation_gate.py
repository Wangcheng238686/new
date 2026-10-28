"""Validate paired point-correction telemetry on validation images only."""
import argparse
from pathlib import Path
import torch
import ue_udpr_point_probe as run


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);r=p.parse_args().root
    a=argparse.Namespace(base=run.BASE,data_root=run.DATA,direction='S1',content='correct',
        init=r/'common_init.pth',cache=Path('/data/wangcheng/cache/ue_v1_s2db_geometry_v2'))
    model,adapter,_=run.make_model(a);model.eval();adapter.eval();adapter.observe=True
    ds=run.MinimalSatelliteDataset(a.data_root,'val')
    sample=run.processed(model,run.rtmdet_collate_fn([ds[0]]))
    rows=[]
    for nonzero in [False,True]:
        if nonzero:
            with torch.no_grad():adapter.output.bias.fill_(.02)
        with torch.inference_mode():
            features=run.bind(model,adapter,sample,run.payload(a,run.point_folder(ds,0,a.data_root),'correct'))
            adapter.audit_gt=(sample['data_samples'][0].gt_instances.bboxes.detach(),
                torch.as_tensor(sample['data_samples'][0].gt_instances.masks.to_ndarray(),device='cuda'))
            with run.cached_extract(model,features):model.predict(sample['inputs'],sample['data_samples'],rescale=True)
        diag=adapter.observation['point_diagnostics']
        assert diag['matched_points']>0 and diag['supported_matched_points']>0
        assert diag['new_correct']-diag['old_correct']==diag['corrected']-diag['damaged']
        if not nonzero:assert diag['changed']==0
        rows.append(dict(nonzero_projection=nonzero,diagnostics=diag))
    assert run.state_digest(model)==run.FROZEN_SHA
    run.dump(r/'evaluation_gate.json',dict(passed=True,split='val',images=1,rows=rows,satellite_unchanged=True,
        note='Temporary projection mutations discarded; not AP evidence or formal training state'))


if __name__=='__main__':main()
