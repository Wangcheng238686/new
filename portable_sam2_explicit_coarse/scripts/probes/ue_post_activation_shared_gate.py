"""Same-weight N2 vs shared N2 consumer identity, validation only."""
import argparse
from pathlib import Path
import torch
import ue_post_activation_shared_probe as run


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);r=p.parse_args().root
    a=argparse.Namespace(base=run.BASE,data_root=run.DATA,direction='N2',content='correct',
        init=r/'common_init.pth',cache=Path('/data/wangcheng/cache/ue_v1_s2db_geometry_v2'))
    model,adapter,_=run.make_model(a);model.eval();adapter.eval();adapter.alpha.data.fill_(.02)
    ds=run.MinimalSatelliteDataset(a.data_root,'val');sample=run.processed(model,run.rtmdet_collate_fn([ds[0]]))
    decoder=model.roi_head.mask_head.mask_decoder.decoder;tail=model.roi_head.mask_head.decoder_tail_refiner
    assert model.roi_head.mask_head.final_mask_coordinate_mode=='roi_local'
    enhanced=[];raw=[];tail_features=[]
    h1=decoder.output_upscaling[-1].register_forward_hook(lambda m,inp,out: enhanced.append(out))
    def capture(m,inp):
        raw.append(inp[0].detach().clone());tail_features.append(inp[1].detach().clone())
        if adapter.sync_capture:assert inp[1] is enhanced[-1]
    h2=tail.register_forward_pre_hook(capture)
    snapshots=[]
    for sync in [False,True]:
        enhanced.clear();raw.clear();tail_features.clear();adapter.sync_capture=sync
        with torch.inference_mode():
            features=run.bind(model,adapter,sample,run.payload(a,run.point_folder(ds,0,a.data_root),'correct'))
            with run.cached_extract(model,features):model.predict(sample['inputs'],sample['data_samples'],rescale=True)
        assert raw and len(enhanced)==len(raw)
        snapshots.append((list(raw),list(tail_features)))
    assert len(snapshots[0][0])==len(snapshots[1][0])
    assert all(torch.equal(x,y) for x,y in zip(snapshots[0][0],snapshots[1][0]))
    assert any(not torch.equal(x,y) for x,y in zip(snapshots[0][1],snapshots[1][1]))
    assert run.state_digest(model)==run.FROZEN_SHA
    h1.remove();h2.remove()
    run.dump(r/'consumer_gate.json',dict(passed=True,split='val',images=1,nonzero_alpha=.02,
        same_weights_raw_SAM_logits_exact=True,shared_tensor_identity=True,UDPR_features_changed=True,
        satellite_state_unchanged=True,roi_local_contract=True,temporary_state_discarded=True))


if __name__=='__main__':main()
