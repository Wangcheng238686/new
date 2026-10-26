"""Verify stage4 replacement reaches stage5 and expected FPN consumers."""
import argparse
from pathlib import Path
import torch
import ue_joint_encoding_probe as run


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);r=p.parse_args().root
    a=argparse.Namespace(base=run.BASE,data_root=run.DATA,direction='J1',content='correct',
        init=r/'common_init.pth',cache=Path('/data/wangcheng/cache/ue_v1_s2db_geometry_v2'))
    model,adapter,_=run.make_model(a);model.eval();adapter.eval()
    trunks=[m for m in model.backbone.modules() if hasattr(m,'stage_ends') and hasattr(m,'blocks')];trunk=trunks[0]
    assert tuple(trunk.blocks[2].pool.kernel_size)==(2,2) and tuple(trunk.blocks[2].pool.stride)==(2,2)
    ds=run.MinimalSatelliteDataset(a.data_root,'train');sample=run.processed(model,run.rtmdet_collate_fn([ds[0]]))
    holder=[];received=[]
    h1=trunk.blocks[4].register_forward_hook(lambda m,i,o: holder.append(o))
    h2=trunk.blocks[5].register_forward_pre_hook(lambda m,i: received.append(i[0] is holder[-1]))
    with torch.inference_mode():
        adapter.enabled=False;base=model.extract_feat(sample['inputs'])
        adapter.enabled=True;adapter.output.bias.fill_(.02)
        data=run.payload(a,run.point_folder(ds,0,a.data_root),'correct')
        updated=run.bind(model,adapter,sample,data)
    assert len(holder)==2 and all(received)
    assert holder[-1].shape==(1,64,64,224)
    assert torch.equal(base[3][0],updated[3][0]),'stride4 lateral unexpectedly changed'
    assert not torch.equal(base[3][1],updated[3][1]),'stride8 consumer unchanged'
    assert not torch.equal(base[1],updated[1]),'SAM embedding consumer unchanged'
    assert any(not torch.equal(x,y) for x,y in zip(base[0],updated[0])),'detection neck unchanged'
    assert run.state_digest(model)==run.FROZEN_SHA
    h1.remove();h2.remove()
    run.dump(r/'wiring_gate.json',dict(passed=True,block4_shape=[1,64,64,224],block5_receives_updated_tensor=True,
        stride4_lateral_exact=True,stride8_and_embedding_and_detection_changed=True,
        q_pool_footprint=[2,2],satellite_parameters_buffers_exact=True,split='train',images=1,
        temporary_projection_state_discarded=True))


if __name__=='__main__':main()
