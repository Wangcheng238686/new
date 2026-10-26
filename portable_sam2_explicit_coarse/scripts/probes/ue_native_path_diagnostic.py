"""Read-only decomposition of the native adapter's satellite and geometry paths."""
import argparse,json,sys,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts/probes'),str(ROOT)]
import ue_v1_adapter as c
import torch
import torch.nn.functional as F
import numpy as np
from rsprompter.uav.native_geometry_probe import NativeGeometryAdapter,prepare_native


def rms(x):return float(x.float().square().mean().sqrt())

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 a.base=Path('/data/wangcheng/checkpoint/ue_minimal_single_20261021_EXP-0017/checkpoints/best_model_epoch73.pth')
 a.data_root=Path('/data/wangcheng/dataset/CitySample_SatUAV_TrainMinimal_v1_20261021');a.source_mode='fine';a.route='highres'
 c.seed(44);model,old,snap=c.make_model(a)
 for h in old.handles:h.remove()
 del old;torch.cuda.empty_cache()
 adapter=NativeGeometryAdapter().cuda().eval();path=Path('/data/wangcheng/checkpoint/ue_v1_adapter_probe_20261021/native_geometry/last.pth')
 cp=torch.load(path,map_location='cpu',weights_only=False);adapter.load_state_dict(cp['adapter'],strict=True);del cp
 before=c.state_digest(model);conv=adapter.mixer[0];rows=[]
 with torch.inference_mode():
  for split in ['train','val']:
   ds=c.MinimalSatelliteDataset(a.data_root,split)
   indices=np.linspace(0,len(ds)-1,8,dtype=int).tolist()
   for i in indices:
    folder=c.point_folder(ds,i,a.data_root);sample=c.processed(model,c.rtmdet_collate_fn([ds[i]]))
    high=model.extract_feat(sample['inputs'])[3][0];qimage=adapter.query(high)
    with np.load(prepare_native(folder,Path('/data/wangcheng/cache/ue_v1_native_geometry'))) as d:
     fields=torch.from_numpy(d['fields'].astype(np.float32))[None].cuda();support=torch.from_numpy(d['support'].astype(np.float32))[None,None].cuda()
    # Fixed full-image box isolates branches on the actual decoder's 128x128 sampling grid.
    boxes=torch.tensor([[0.,0.,512.,512.]],device='cuda');size=128
    line=(torch.arange(size,device='cuda',dtype=torch.float32)+.5)/size
    yy,xx=torch.meshgrid(line,line,indexing='ij');grid=torch.stack((xx,yy),-1)[None]*2-1
    q=F.grid_sample(qimage,grid,align_corners=False);cov=F.grid_sample(support,grid,align_corners=False)
    geo=F.grid_sample(fields,grid,align_corners=False)/cov.clamp_min(1e-6)
    valid=F.grid_sample(support,grid,mode='nearest',align_corners=False)>0
    sat=F.conv2d(q,conv.weight[:,:32],None,padding=1)
    geoterm=F.conv2d(geo,conv.weight[:,32:],None,padding=1)
    bias=conv.bias.view(1,-1,1,1)
    assert torch.allclose(conv(torch.cat((q,geo),1)),sat+geoterm+bias,atol=1e-5,rtol=1e-4)
    finish=lambda x:adapter.mixer[2](F.gelu(x))*valid
    outputs={
     'correct':finish(sat+geoterm+bias),
     'zero_fields':finish(sat+bias),
     'zero_query_channels':finish(geoterm+bias),
     'bias_only':finish(torch.zeros_like(sat)+bias)}
    total=outputs['correct'];den=float(total.square().sum())
    zero=outputs['zero_fields'];cos=float((total*zero).sum()/(total.norm()*zero.norm()).clamp_min(1e-12))
    row=dict(split=split,index=i,point=folder.name,support=float(support.mean()),scope='fixed whole-image box at128, not actual detection RoIs',
     q_channel_rms=[rms(q[:,j]) for j in range(32)],geo_channel_rms=[rms(geo[:,j]) for j in range(4)],
     first_conv_satellite_rms=rms(sat),first_conv_geometry_rms=rms(geoterm),first_conv_bias_rms=rms(bias),
     output_rms={k:rms(v) for k,v in outputs.items()},zero_fields_output_cosine=cos,
     removing_geometry_change_rms=rms(total-zero),removing_query_change_rms=rms(total-outputs['zero_query_channels']),
     geometry_term_per_field_rms=[rms(F.conv2d(geo[:,j:j+1],conv.weight[:,32+j:33+j],None,padding=1)) for j in range(4)])
    rows.append(row);print(split,i,'sat',row['first_conv_satellite_rms'],'geo',row['first_conv_geometry_rms'],flush=True)
 assert c.state_digest(model)==before
 weight={k:dict(norm=float(v.float().norm()),rms=rms(v)) for k,v in adapter.state_dict().items()}
 # Matched trained zero model exists; compare learned adapter movement, not capacity claims.
 cp=torch.load(path.parent.parent/'native_zero/last.pth',map_location='cpu',weights_only=False)
 changes={k:float((v.cpu()-cp['adapter'][k]).float().norm()) for k,v in adapter.state_dict().items()}
 c.dump(a.out/'path_decomposition.json',dict(rows=rows,weights=weight,zero_trained_parameter_l2_distance=changes,
  satellite_state_unchanged=True,adapter_sha256=c.file_sha(path),runner_sha256=c.file_sha(__file__),
  selection='8 evenly spaced dataset indices per train/val, no AP selection',limitations='activation diagnostic on fixed full-image box, not AP or causal proof'))
 print('COMPLETE',flush=True)
if __name__=='__main__':main()
