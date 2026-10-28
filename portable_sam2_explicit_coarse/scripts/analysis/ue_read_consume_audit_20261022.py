"""CPU coordinate-contract audit. No model mutation, training, or GPU inference."""
import os
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
import json,hashlib
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[2]
OUT=Path('/data/wangcheng/checkpoint/ue_read_consume_audit_20261022');OUT.mkdir(parents=True,exist_ok=True)
DATA=Path('/data/wangcheng/dataset/CitySample_SatUAV_TrainMinimal_v1_20261021')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
torch.set_num_threads(1)
# Identity grid and subcell floor selection are distinct operations.
y,x=torch.meshgrid(torch.arange(128),torch.arange(128),indexing='ij')
ramp=torch.stack((x,y)).float()[None]
grid=torch.stack(((x+.5)/64-1,(y+.5)/64-1),-1).float()[None]
assert torch.equal(F.grid_sample(ramp,grid,align_corners=False),ramp)
# Same RoI query / evidence grid uses identical affine transform, but the
# receiving base map is a full-image repeat, not this crop.
boxes=torch.tensor([[32.,64.,96.,128.],[256.,320.,288.,352.]])
line=(torch.arange(128)+.5)/128;yy,xx=torch.meshgrid(line,line,indexing='ij')
uv=boxes[:,None,None,:2]+torch.stack((xx,yy),-1)[None]*(boxes[:,None,None,2:]-boxes[:,None,None,:2])
q=F.grid_sample(ramp.expand(2,-1,-1,-1),uv/256-1,align_corners=False)
assert not torch.equal(q[0],q[1])
base=ramp.expand(2,-1,-1,-1)
assert torch.equal(base[0],base[1])
displacement=(q-base).square().sum(1).sqrt()
# Source feature token center vs measured source anchor are separate origins.
# Map token-grid index using the adapter's own edge-domain convention only;
# do not call this the learned feature's physical receptive-field center.
anchors=[];cachehash={}
train=json.loads((DATA/'splits/train.json').read_text())
for i in np.linspace(0,len(train)-1,32,dtype=int):
 p=Path('/data/wangcheng/cache/ue_v1_s2db_geometry_v2')/f'{train[i]:06d}.npz'
 with np.load(p) as a:
  ix=a['index'];an=a['anchor'];valid=ix>=0
  token=ix[valid]%16384
  center=np.stack((token%128*4+2,token//128*4+2),1)
  anchors.append(center-an[valid,:2])
 cachehash[p.name]=sha(p)
a=np.concatenate(anchors)
# Attention may extract geometry without RGB; normalization removes magnitudes
# but not directions, and output bias survives at any supported location.
geom=torch.randn(16,8,6);gv=torch.randn(6,32)
pooled=(geom@gv).mean(1)
assert pooled.abs().sum()>0
bias=torch.arange(32).float()[None].expand(4,-1)
norm=F.layer_norm(bias,(32,))
assert norm.abs().sum()>0
masked=norm*torch.zeros(4,1)
assert torch.count_nonzero(masked)==0
# Theoretical source granularity for median real small GT boxes.
ann=json.loads((DATA/'annotations/test.json').read_text());small=[v for v in ann['annotations'] if v['area']<1024]
widths=np.array([v['bbox'][2:4] for v in small])
files=['rsprompter/uav/local_adapter_probe.py','rsprompter/uav/local_adapter_variants.py','rsprompter/uav/native_geometry_probe.py','rsprompter/uav/s2db_adapter.py','rsprompter/uav/s2db_geometry.py','rsprompter/cspm/mask_head.py','rsprompter/cspm/mask_head_targets.py','rsprompter/decoder/sam2_decoder.py','scripts/probes/ue_s2db_probe.py']
report=dict(scope='CPU synthetic coordinate checks plus fixed32 train cache metadata; test labels for size description only',
 source_hashes={p:sha(ROOT/p) for p in files},script_sha256=sha(__file__),
 zero_offset_token_grid_identity_exact=True,
 same_grid_local_query_evidence_coordinates=True,
 global_receiver_repeat_is_not_RoI_query_crop=True,
 synthetic_receiver_query_displacement_source_feature_pixels=dict(median=float(displacement.median()),max=float(displacement.max())),
 fixed32_train_anchors=len(a),token_edgecenter_minus_measured_source_uv_px=dict(unique_per_axis=[np.unique(a[:,i]).tolist() for i in range(2)],
  mean_abs=np.abs(a).mean(0).tolist(),max_abs=np.abs(a).max(0).tolist()),
 feature_vs_depth_offset_origin_note='xy=token+0.5+offset for feature; uv=measured_anchor+4*offset for depth. Different origins. Not exact shifted 3D validation.',
 cache_sha256=cachehash,constant_rgb_can_keep_geometry_evidence=True,
 output_bias_can_make_supported_residual_without_source_content=True,
 after_normalization_support_mask_blocks_unsupported_residual=True,
 small_test_instances=len(small),small_bbox_width_height_median= np.median(widths,axis=0).tolist(),
 small_bbox_width_height_median_in_stride4_cells=np.median(widths/4,axis=0).tolist(),
 caveats=['Synthetic ramps prove coordinate conventions, not real learned activation misalignment magnitude',
  'Adapter token center convention is not an independently certified physical feature center',
  'No new performance ablation; these checks cannot assign lost AP to one mechanism'])
(OUT/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ('cache_sha256','source_hashes')},indent=2))
