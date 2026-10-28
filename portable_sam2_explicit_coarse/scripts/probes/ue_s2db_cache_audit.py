"""Independent original-depth -> world -> satellite replay of retained anchors."""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import ue_s2db_probe as run
import numpy as np
from rsprompter.uav.local_adapter_probe import camera_arrays,build_candidates
from rsprompter.uav.s2db_geometry import prepare_point,load_depth,VERSION


def check(folder,cache):
    p=prepare_point(folder,cache)
    with np.load(p) as a:
        ix,g,anchor=a['index'],a['geom'],a['anchor'];report=json.loads(str(a['report']))
        assert str(a['version'])==VERSION and anchor.dtype==np.float64
    for name,digest in report['source_hashes'].items():assert run.file_sha(folder/name)==digest
    old=Path('/data/wangcheng/cache/ue_v1_local_adapter_geometry')/p.name
    if old.exists():
        with np.load(old) as a:oi,og=a['index'],a['geom']
    else:oi,og,_=build_candidates(folder)
    assert np.array_equal(ix,oi) and np.array_equal(g,og)
    c=json.loads((folder/'cameras.json').read_text());sk,se,_=camera_arrays(c['satellite'])
    depth=load_depth(folder);error=0.;zerror=0.;n=0
    for v,yaw in enumerate(run.YAWS):
        take=(ix>=0)&(ix//16384==v);a=anchor[take]
        if not len(a):continue
        k,_,t=camera_arrays(c['uav'][f'yaw_{yaw:03d}'])
        u,w,z=a[:,0],a[:,1],a[:,2]
        raw=depth[v,w.astype(int),u.astype(int)]
        assert np.isfinite(raw).all() and (raw>0).all()
        zerror=max(zerror,float(np.max(np.abs(raw-z))));assert zerror<1e-3
        xyz=np.stack(((u-k[0,2])*z/k[0,0],(w-k[1,2])*z/k[1,1],z),1)
        world=xyz@t[:3,:3].T+t[:3,3];sat=world@se[:3,:3].T+se[:3,3]
        proj=sat@sk.T;uv=proj[:,:2]/proj[:,2:]
        error=max(error,float(np.abs(uv-a[:,3:5]).max()))
        assert np.allclose(sat[:,2],a[:,5],atol=1e-8,rtol=0)
        assert np.array_equal(ix[take]%16384,(w.astype(int)//4)*128+u.astype(int)//4)
        n+=len(a)
    assert error<1e-5
    return dict(point=folder.name,candidates=n,legacy_index_geom_exact=True,metadata_projection_error_px=error,
        source_z_float32_load_error_m=zerror,cache_sha256=run.file_sha(p),source_hashes_checked=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,default=run.DATA)
    p.add_argument('--cache',type=Path,default=Path('/data/wangcheng/cache/ue_v1_s2db_geometry_v2'))
    p.add_argument('--split',choices=['trainval','test'],required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();splits=['train','val'] if a.split=='trainval' else ['test']
    folders=[run.point_folder(ds,i,a.data_root) for ds in [run.MinimalSatelliteDataset(a.data_root,s) for s in splits] for i in range(len(ds))]
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows=[]
        for i,r in enumerate(pool.map(lambda f:check(f,a.cache),folders)):
            rows.append(r)
            if i%100==0:print('AUDIT',i,len(folders),flush=True)
    run.dump(a.out,dict(split=a.split,points=len(rows),all_passed=True,rows=rows,version=VERSION,
        max_metadata_projection_error_px=max(r['metadata_projection_error_px'] for r in rows)))

if __name__=='__main__':main()
