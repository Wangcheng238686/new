"""CPU-only native selected-ray provenance audit. Never mutates shared caches."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[k]='1'
os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'
os.environ['RSPROMPTER_LIGHT_IMPORT']='1'
import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
import torch
from PIL import Image,ImageDraw
from pycocotools import mask as maskutil
from rsprompter.uav.native_geometry_probe import prepare_native,VERSION
from rsprompter.uav.local_adapter_probe import camera_arrays,YAWS,file_sha
cv2.setNumThreads(1);torch.set_num_threads(2)
DATA=Path('/data/wangcheng/dataset/CitySample_SatUAV_TrainMinimal_v1_20261021')
CACHE=Path('/data/wangcheng/cache/ue_v1_native_geometry')
OUT=Path('/data/wangcheng/checkpoint/ue_native_source_audit_20261022')
SEED=20261022

def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def decode(a):
    s=a['segmentation'];r=maskutil.frPyObjects(s,512,512) if isinstance(s['counts'],list) else s
    return maskutil.decode(r).astype(bool)
def reconstruct(folder):
    c=json.loads((folder/'cameras.json').read_text());sk,se,_=camera_arrays(c['satellite'])
    yy,xx=np.mgrid[:512,:512];u,v=xx.ravel(),yy.ravel()
    parts=[];hashes={'cameras.json':file_sha(folder/'cameras.json')};roundtrip=0.
    for vi,yaw in enumerate(YAWS):
        rel=f'uav/yaw_{yaw:03d}/depth_camera_z.exr';hashes[rel]=file_sha(folder/rel)
        hashes[f'uav/yaw_{yaw:03d}/rgb.png']=file_sha(folder/f'uav/yaw_{yaw:03d}/rgb.png')
        z=cv2.imread(str(folder/rel),cv2.IMREAD_UNCHANGED)[:,:,2].ravel().astype(np.float64)/100
        good=np.isfinite(z)&(z>0)&(z<10000);k,e,t=camera_arrays(c['uav'][f'yaw_{yaw:03d}'])
        x=np.stack(((u[good]-k[0,2])*z[good]/k[0,0],(v[good]-k[1,2])*z[good]/k[1,1],z[good]),1)
        world=x@t[:3,:3].T+t[:3,3]
        replay=world@e[:3,:3].T+e[:3,3];rp=replay@k.T
        roundtrip=max(roundtrip,float(np.abs(rp[:,:2]/rp[:,2:]-np.stack((u[good],v[good]),1)).max()))
        target=world@se[:3,:3].T+se[:3,3];proj=target@sk.T;uv=proj[:,:2]/proj[:,2:]
        valid=(target[:,2]>0)&np.isfinite(uv).all(1)&(uv>=0).all(1)&(uv<512).all(1)
        parts.append(dict(pix=uv[valid,1].astype(np.int64)*512+uv[valid,0].astype(np.int64),sat_z=target[valid,2],
            height=world[valid,2]-c['center_ue_cm'][2]/100,view=np.full(valid.sum(),vi,np.int8),
            source_uv=np.stack((u[good],v[good]),1)[valid].astype(np.int16),source_z=z[good][valid],world=world[valid],sat_uv=uv[valid]))
    arrays={key:np.concatenate([p[key] for p in parts]) for key in parts[0]}
    pix,z,h,view=(arrays[k] for k in ('pix','sat_z','height','view'))
    order=np.lexsort((z,pix));sp=pix[order];selected=order[np.r_[True,sp[1:]!=sp[:-1]]]
    zbuf=np.full(512*512,np.inf);height=np.zeros(512*512,np.float32);selmap=np.full(512*512,-1,np.int64)
    zbuf[pix[selected]]=z[selected];height[pix[selected]]=h[selected];selmap[pix[selected]]=selected
    support=np.isfinite(zbuf).reshape(512,512);count=np.zeros(512*512,np.float32)
    near=np.zeros(len(pix),bool)
    for vi in range(5):
        n=(view==vi)&(z<=zbuf[pix]+1);near|=n;count[np.unique(pix[n])]+=1
    height=height.reshape(512,512);dx=np.zeros_like(height);dy=np.zeros_like(height)
    dx[:,:-1]=np.where(support[:,:-1]&support[:,1:],np.diff(height,axis=1),0)
    dy[:-1]=np.where(support[:-1]&support[1:],np.diff(height,axis=0),0)
    fields=np.stack((height/100,np.clip(dx/10,-5,5),np.clip(dy/10,-5,5),count.reshape(512,512)/5));fields[:,~support]=0
    arrays.update(selected=selected,selected_index_map=selmap.reshape(512,512),near_1m=near,support=support,fields=fields.astype(np.float16))
    hashes['satellite/rgb.png']=file_sha(folder/'satellite/rgb.png')
    return arrays,hashes,roundtrip

def ray(a,i):
    return {k:(int(a[k][i]) if k=='view' else a[k][i].tolist()) for k in ('view','source_uv','source_z','world','sat_uv','sat_z','height')}|{'concat_selection_index':int(i),'yaw_deg':int(YAWS[a['view'][i]])}
def panel(im,title,point=None,zoom=False):
    im=im.convert('RGB');d=ImageDraw.Draw(im)
    if point is not None:
        x,y=point;d.ellipse((x-4,y-4,x+4,y+4),outline='red',width=2);d.line((x-9,y,x+9,y),fill='red');d.line((x,y-9,x,y+9),fill='red')
        if zoom:im=im.crop((int(x)-32,int(y)-32,int(x)+32,int(y)+32))
    im=im.resize((256,256),Image.Resampling.NEAREST if zoom else Image.Resampling.BILINEAR)
    out=Image.new('RGB',(256,288),'white');out.paste(im,(0,32));ImageDraw.Draw(out).text((4,4),title,fill='black');return out

def contact(folder,a,points,masks,path):
    sat=Image.open(folder/'satellite/rgb.png').convert('RGB');satnp=np.array(sat);outline=np.zeros((512,512),bool)
    for m in masks:outline|=m&~cv2.erode(m.astype('uint8'),np.ones((3,3),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)
    satnp[outline]=[0,255,0];sat=Image.fromarray(satnp)
    canvas=Image.new('RGB',(1536,864),'white')
    for row,p in enumerate(points):
        r=p['height_ray'];vi=r['view'];yaw=YAWS[vi];uv=r['source_uv'];suv=r['sat_uv']
        rgb=Image.open(folder/f'uav/yaw_{yaw:03d}/rgb.png').convert('RGB')
        dep=cv2.imread(str(folder/f'uav/yaw_{yaw:03d}/depth_camera_z.exr'),cv2.IMREAD_UNCHANGED)[:,:,2].astype('float64')/100
        good=np.isfinite(dep)&(dep>0)&(dep<10000);lo,hi=np.percentile(dep[good],[2,98]);scaled=np.clip((np.nan_to_num(dep)-lo)/max(hi-lo,1e-9),0,1)
        dc=cv2.applyColorMap((scaled*255).astype('uint8'),cv2.COLORMAP_TURBO)[:,:,::-1];dc[~good]=0;di=Image.fromarray(dc)
        titles=[f"{p['group']} satellite + GT",f"sat zoom ({suv[0]:.2f},{suv[1]:.2f})",f"source yaw{yaw} ({uv[0]},{uv[1]})",'source zoom',f"Z(m) [{lo:.1f},{hi:.1f}]",f"Z center {r['source_z']:.3f}m"]
        ims=[panel(sat,titles[0],suv),panel(sat,titles[1],suv,True),panel(rgb,titles[2],uv),panel(rgb,titles[3],uv,True),panel(di,titles[4],uv),panel(di,titles[5],uv,True)]
        for col,im in enumerate(ims):canvas.paste(im,(256*col,288*row))
    canvas.save(path)

def main():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'sources').mkdir(exist_ok=True);(OUT/'contacts').mkdir(exist_ok=True)
    gen=np.random.Generator(np.random.PCG64(SEED));samples=[];anns={}
    for split in ('train','val'):
        coco=json.loads((DATA/'annotations'/f'{split}.json').read_text());ids=sorted(i['id'] for i in coco['images'])
        picks=gen.choice(ids,4,replace=False).tolist();samples +=[{'split':split,'image_id':i} for i in picks]
        anns[split]=coco
    manifest=dict(seed=SEED,algorithm='single PCG64 generator, choice sorted train IDs then sorted val IDs, replace=False',samples=samples,
        source_sha256={str(p.relative_to(ROOT)):file_sha(p) for p in (Path(__file__),ROOT/'rsprompter/uav/native_geometry_probe.py',ROOT/'rsprompter/uav/local_adapter_probe.py')},
        annotation_sha256={s:file_sha(DATA/'annotations'/f'{s}.json') for s in ('train','val')},
        units={'depth_EXR_R':'camera-Z cm converted /100 to m','world':'m; translation cm /100','height':'world Z minus sample-center Z in m','source_uv':'integer raw source pixel','sat_uv':'continuous raw projection in px; target bin floor','index':'concatenated all valid projected rays in yaw order, source row-major'},version=VERSION)
    mp=OUT/'manifest.json'
    if mp.exists():assert json.loads(mp.read_text())['samples']==samples
    dump(mp,manifest) # Always before reconstruction or point inspection.
    results=[]
    for sample in samples:
        iid=sample['image_id'];folder=DATA/'points'/f'{iid:06d}';a,hashes,rt=reconstruct(folder)
        original=CACHE/f'{iid:06d}.npz';ref=original;reference_kind='cached-reference'
        if not ref.exists():
            reference_kind='newly-built-reference';tmp=OUT/'new_reference';tmp.mkdir(exist_ok=True);ref=prepare_native(folder,tmp)
        with np.load(ref) as c:
            assert str(c['version'])==VERSION
            support_equal=np.array_equal(a['support'],c['support']);fields_equal=np.array_equal(a['fields'],c['fields'])
            diff=dict(support_mismatch=int((a['support']!=c['support']).sum()),fields_mismatch=int((a['fields']!=c['fields']).sum()),max_abs_field_difference=float(np.max(np.abs(a['fields'].astype('float32')-c['fields'].astype('float32')))))
            report=json.loads(str(c['report']));assert all(hashes[k]==v for k,v in report['source_hashes'].items())
        dump(OUT/f'{iid:06d}_cache_check.json',dict(reference_kind=reference_kind,reference_path=str(ref),sha256=file_sha(ref),**diff))
        assert support_equal and fields_equal, (iid,diff)
        np.savez_compressed(OUT/'sources'/f'{iid:06d}.npz',**a)
        sampleann=[x for x in anns[sample['split']]['annotations'] if x['image_id']==iid];masks=[decode(x) for x in sampleann]
        union=np.logical_or.reduce(masks) if masks else np.zeros((512,512),bool);boundary=np.zeros_like(union)
        for m in masks:boundary|=m&~cv2.erode(m.astype('uint8'),np.ones((3,3),np.uint8),iterations=2,borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)
        points=[];missing=[]
        for gi,(gn,group) in enumerate((('interior',union&~boundary),('boundary',boundary),('background',~union))):
            valid=np.flatnonzero(group&a['support'])
            if not len(valid):missing.append(gn);continue
            p=int(np.random.Generator(np.random.PCG64(np.random.SeedSequence([SEED,iid,gi]))).choice(valid));y,x=divmod(p,512);idx=int(a['selected_index_map'][y,x])
            near=np.flatnonzero((a['pix']==p)&a['near_1m']);byview={str(vi):[ray(a,int(j)) for j in near[a['view'][near]==vi]] for vi in range(5)}
            neighbors={}
            for direction,(ny,nx) in {'dx':(y,x+1),'dy':(y+1,x)}.items():
                ok=ny<512 and nx<512 and a['support'][ny,nx]
                neighbors[direction]=dict(valid=bool(ok),neighbor_target_uv=[nx,ny],neighbor_ray=ray(a,int(a['selected_index_map'][ny,nx])) if ok else None,
                    stored_field=float(a['fields'][1 if direction=='dx' else 2,y,x]))
            uv=a['source_uv'][idx];dep=cv2.imread(str(folder/f"uav/yaw_{YAWS[a['view'][idx]]:03d}/depth_camera_z.exr"),cv2.IMREAD_UNCHANGED)[:,:,2].astype('float64')/100
            patch=dep[max(0,uv[1]-2):uv[1]+3,max(0,uv[0]-2):uv[0]+3];patch=np.where(np.isfinite(patch),patch,np.nan)
            points.append(dict(group=gn,group_index=gi,valid_group_population=int(len(valid)),target_uv=[x,y],height_ray=ray(a,idx),stored_fields=a['fields'][:,y,x].astype('float32').tolist(),
                gradient_sources=neighbors,near1m_views=byview,near1m_distinct_views=sum(bool(v) for v in byview.values()),source_depth_patch_m=[[float(v) if np.isfinite(v) else None for v in row] for row in patch],visual_status='pending original RGB inspection'))
        contact(folder,a,points,masks,OUT/'contacts'/f'{iid:06d}.png')
        areas=np.array([m.sum() for m in masks]);summary=dict(**sample,instances=len(masks),density_group='<=50' if len(masks)<=50 else '51-100' if len(masks)<=100 else '>100',
            true_mask_area_groups={'small':int((areas<32**2).sum()),'medium':int(((areas>=32**2)&(areas<96**2)).sum()),'large':int((areas>=96**2).sum())},
            native_support_fraction=float(a['support'].mean()),gt_union_support_fraction=float((union&a['support']).sum()/max(1,union.sum())),reference_kind=reference_kind,cache_check=diff,
            source_hashes=hashes,source_camera_roundtrip_max_px=rt,missing_groups=missing,points=points,contact_path=str(OUT/'contacts'/f'{iid:06d}.png'))
        dump(OUT/f'{iid:06d}_points.json',summary);results.append(summary);print('COMPLETE',iid,diff,flush=True)
    dump(OUT/'audit.json',dict(manifest=manifest,images=results,points=sum(len(r['points']) for r in results),all_cache_equal=True))
if __name__=='__main__':main()
