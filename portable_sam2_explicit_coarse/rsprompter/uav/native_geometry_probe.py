"""Native target-raster surface evidence adapter; an orthogonal depth-only probe."""
import json
import os
from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .local_adapter_probe import camera_arrays, file_sha, YAWS

VERSION='native512_height_dx_dy_viewcount_v1'


def prepare_native(folder,cache):
    path=Path(cache)/f'{folder.name}.npz'
    if path.is_file():
        with np.load(path) as a:assert str(a['version'])==VERSION
        return path
    c=json.loads((folder/'cameras.json').read_text())
    assert c['world_units']=='cm' and c['camera_axes']=='x_right_y_down_z_forward'
    sk,se,_=camera_arrays(c['satellite'])
    yy,xx=np.mgrid[:512,:512];u,v=xx.ravel(),yy.ravel()
    pixels=[];depths=[];heights=[];views=[];hashes={'cameras.json':file_sha(folder/'cameras.json')}
    for view,yaw in enumerate(YAWS):
        name=f'uav/yaw_{yaw:03d}/depth_camera_z.exr';hashes[name]=file_sha(folder/name)
        z=cv2.imread(str(folder/name),cv2.IMREAD_UNCHANGED)[:,:,2].ravel().astype(np.float64)/100
        good=np.isfinite(z)&(z>0)&(z<10000)
        k,e,t=camera_arrays(c['uav'][f'yaw_{yaw:03d}'])
        x=np.stack(((u[good]-k[0,2])*z[good]/k[0,0],(v[good]-k[1,2])*z[good]/k[1,1],z[good]),1)
        world=x@t[:3,:3].T+t[:3,3]
        target=world@se[:3,:3].T+se[:3,3];proj=target@sk.T
        uv=proj[:,:2]/proj[:,2:]
        valid=(target[:,2]>0)&np.isfinite(uv).all(1)&(uv>=0).all(1)&(uv<512).all(1)
        # Same continuous edge-bin convention as the existing 4px candidate table.
        uv=uv[valid];pix=(uv[:,1].astype(np.int64)*512+uv[:,0].astype(np.int64))
        pixels.append(pix);depths.append(target[valid,2]);heights.append(world[valid,2]-c['center_ue_cm'][2]/100)
        views.append(np.full(len(pix),view,np.int8))
    pix,z,h,view=map(np.concatenate,(pixels,depths,heights,views))
    order=np.lexsort((z,pix));sorted_pix=pix[order]
    first=np.r_[True,sorted_pix[1:]!=sorted_pix[:-1]];selected=order[first]
    zbuf=np.full(512*512,np.inf);height=np.zeros(512*512,np.float32)
    zbuf[pix[selected]]=z[selected];height[pix[selected]]=h[selected]
    support=np.isfinite(zbuf).reshape(512,512)
    count=np.zeros(512*512,np.float32)
    for vi in range(5):
        near=(view==vi)&(z<=zbuf[pix]+1)
        count[np.unique(pix[near])]+=1
    height=height.reshape(512,512)
    dx=np.zeros_like(height);dy=np.zeros_like(height)
    validx=support[:,:-1]&support[:,1:];validy=support[:-1]&support[1:]
    dx[:,:-1]=np.where(validx,np.diff(height,axis=1),0)
    dy[:-1]=np.where(validy,np.diff(height,axis=0),0)
    fields=np.stack((height/100,np.clip(dx/10,-5,5),np.clip(dy/10,-5,5),count.reshape(512,512)/5))
    fields[:,~support]=0
    assert np.isfinite(fields).all()
    report=dict(version=VERSION,source_hashes=hashes,support_pixels=int(support.sum()),
                nearest_observed_surface_only=True,fields=['height_m/100','dx_m/10_clipped5','dy_m/10_clipped5','views_nearest1m/5'])
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,fields=fields.astype(np.float16),support=support,version=VERSION,report=json.dumps(report))
    tmp.replace(path)
    return path


class NativeGeometryAdapter(nn.Module):
    """One local feature residual, no new mask head, no RGB encoder."""
    def __init__(self):
        super().__init__()
        self.query=nn.Conv2d(256,32,1)
        self.mixer=nn.Sequential(nn.Conv2d(36,32,3,padding=1),nn.GELU(),nn.Conv2d(32,32,1))
        self.alpha=nn.Parameter(torch.zeros(()))
        self.context=None;self.boxes=None;self.enabled=True;self.stats={};self.handles=[]

    def attach(self,model):
        head=model.roi_head.mask_head
        self.handles.append(head.register_forward_pre_hook(self._boxes,with_kwargs=True))
        self.handles.append(head.mask_decoder.decoder.conv_s0.register_forward_hook(self._inject))

    def _boxes(self,module,args,kwargs):self.boxes=kwargs['boxes']

    def encode(self,unused_rgb,support,fields,satellite_high):
        self.context=(support,fields,satellite_high);self.stats={}

    def _chunk(self,qimage,fields,support,boxes,size):
        line=(torch.arange(size,device=boxes.device,dtype=torch.float32)+.5)/size
        yy,xx=torch.meshgrid(line,line,indexing='ij')
        uv=boxes[:,None,None,:2]+torch.stack((xx,yy),-1)[None]*(boxes[:,None,None,2:]-boxes[:,None,None,:2])
        grid=uv/256-1
        q=F.grid_sample(qimage.expand(len(boxes),-1,-1,-1),grid.to(qimage.dtype),align_corners=False)
        cov=F.grid_sample(support.expand(len(boxes),-1,-1,-1),grid,align_corners=False)
        geo=F.grid_sample(fields.expand(len(boxes),-1,-1,-1),grid.to(fields.dtype),align_corners=False)/cov.clamp_min(1e-6)
        valid=F.grid_sample(support.expand(len(boxes),-1,-1,-1),grid,mode='nearest',align_corners=False)>0
        return self.mixer(torch.cat((q,geo.to(q.dtype)),1))*valid

    def _inject(self,module,args,base):
        if not self.enabled or self.context is None:return base
        support,fields,high=self.context;qimage=self.query(high);chunks=[]
        for boxes in self.boxes.split(4):
            fn=lambda q,g,s,b:self._chunk(q,g,s,b,base.shape[-1])
            delta=checkpoint(fn,qimage,fields,support,boxes,use_reentrant=False) if self.training and torch.is_grad_enabled() else fn(qimage,fields,support,boxes)
            chunks.append(delta)
        delta=torch.cat(chunks);applied=self.alpha*delta
        with torch.no_grad():self.stats=dict(alpha=float(self.alpha),support_fraction=float(support.mean()),
            source_delta_rms=float(delta.float().square().mean().sqrt()),applied_delta_rms=float(applied.float().square().mean().sqrt()),
            base_rms=float(base.float().square().mean().sqrt()),rois=len(base),support_grid=512)
        return base+applied
