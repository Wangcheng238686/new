"""Train-only retained-anchor/source-color/observed-surface audit.

Raw RGB proxy only: does not evaluate learned P0 features, attention or AP.
No proposed filtering is applied to training; 1m uses existing native viewcount
convention. Candidate depth spread is not automatically a surface mistake.
"""
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'
from pathlib import Path
import hashlib
import json
import sys

import cv2
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
import ue_rgb_sat_image_correspondence_20261025 as raw

OUT=Path('/data/wangcheng/checkpoint/ue_compact_joint_20261024/retained_rgb_read_audit')
CACHE=Path('/data/wangcheng/cache/ue_v1_s2db_geometry_v2')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def block_average(image):
    return image.reshape(128,4,128,4,3).mean((1,3)).reshape(16384,3)


def main():
    assert not OUT.exists()
    OUT.mkdir()
    ids=json.loads((raw.DATA/'splits/train.json').read_text());idx=np.linspace(0,len(ids)-1,8,dtype=int).tolist()
    selected=[int(ids[i]) for i in idx]
    protocol=dict(split='train',fixed_indices=idx,image_ids=selected,source_sha256=sha(__file__),
        projector_source_sha256=sha(raw.__file__),candidate_limit=8,per_view_limit=2,
        source='actual retained anchor source RGB pixels; not learned features',
        front_rule='candidate SAT camera-Z minus nearest observed full512 SAT-Z at own floor-projected pixel in [-.01,1]m',
        scope='observed surface proxy, not complete satellite occlusion; cell supports fixed across color methods',
        color_methods=['retained_mean8','front_filtered_mean','nearest_anchor_in_cell','dense_observed_warp'],
        common_cells='all16 dense observed pixels, at least1 retained/front candidate, central40px crop',
        limitations='raw-color proxies and differing samples inside each same4x4 cell; no feature-retention/AP proof or training mutation')
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    (OUT/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    (OUT/Path(raw.__file__).name).write_bytes(Path(raw.__file__).read_bytes())
    rows=[];sources={}
    for ident in selected:
        folder=raw.DATA/'points'/('%06d'%ident);cp=CACHE/('%06d.npz'%ident)
        for p in [folder/'cameras.json',folder/'satellite/rgb.png',cp]:sources[str(p)]=sha(p)
        c=json.loads((folder/'cameras.json').read_text());sk=raw.matrix(c['satellite']);se=np.array(c['satellite']['world_to_camera_cv'])
        with np.load(cp) as z:
            assert str(z['version'])=='s2db_anchor_metadata_v2_fp64'
            index=z['index'].astype(np.int64);anchor=z['anchor'].copy()
        assert anchor.dtype==np.float64
        valid=index>=0
        colors=np.zeros((*index.shape,3),np.float64)
        dense=np.zeros((512,512,3),np.uint8);near=np.full((512,512),np.inf)
        errors=dict(source_Z_m=0.,SAT_uv_px=0.,SAT_Z_m=0.)
        for vi,yaw in enumerate(raw.YAWS):
            rp=folder/('uav/yaw_%03d/rgb.png'%yaw);dp=folder/('uav/yaw_%03d/depth_camera_z.exr'%yaw)
            for p in [rp,dp]:sources[str(p)]=sha(p)
            rgb=cv2.imread(str(rp))[:,:,::-1].copy();depth=cv2.imread(str(dp),cv2.IMREAD_UNCHANGED)[:,:,2].astype(np.float64)
            image,zd=raw.project(rgb,depth,c['uav']['yaw_%03d'%yaw],c['satellite'])
            take=zd<near;dense[take]=image[take];near[take]=zd[take]
            take=valid&(index//16384==vi)
            uv=anchor[take][:,:2];uv_int=np.rint(uv).astype(int)
            assert np.max(np.abs(uv-uv_int))<1e-9
            u,v=uv_int[:,0],uv_int[:,1]
            assert np.array_equal(index[take],vi*16384+(v//4)*128+u//4)
            colors[take]=rgb[v,u]
            actual_depth=depth[v,u]
            k=raw.matrix(c['uav']['yaw_%03d'%yaw]);t=np.array(c['uav']['yaw_%03d'%yaw]['camera_to_world_cv'])
            ray=np.stack(((u-k[0,2])/k[0,0],(v-k[1,2])/k[1,1],np.ones(len(u))),1)
            world=(ray*actual_depth[:,None])@t[:3,:3].T+t[:3,3]
            sat=world@se[:3,:3].T+se[:3,3];proj=sat@sk.T;puv=proj[:,:2]/proj[:,2:]
            errors['source_Z_m']=max(errors['source_Z_m'],float(np.max(np.abs(actual_depth/100-anchor[take][:,2]))))
            errors['SAT_uv_px']=max(errors['SAT_uv_px'],float(np.max(np.abs(puv-anchor[take][:,3:5]))))
            errors['SAT_Z_m']=max(errors['SAT_Z_m'],float(np.max(np.abs(sat[:,2]/100-anchor[take][:,5]))))
        assert max(errors.values())<1e-6,errors
        uv=anchor[:,:,3:5]
        xx=np.clip(np.floor(uv[:,:,0]).astype(int),0,511);yy=np.clip(np.floor(uv[:,:,1]).astype(int),0,511)
        residual=anchor[:,:,5]-near[yy,xx]/100
        front=valid&np.isfinite(residual)&(residual>=-.01)&(residual<=1.)
        behind=valid&np.isfinite(residual)&(residual>1.)
        negative=valid&np.isfinite(residual)&(residual<-.01)
        sat_rgb=cv2.imread(str(folder/'satellite/rgb.png'))[:,:,::-1].copy()
        target_at_anchor=sat_rgb[yy,xx].astype(np.float64)
        candidate_errors={}
        for name,mask in [('all',valid),('front',front),('behind',behind)]:
            candidate_errors[name]=dict(candidates=int(mask.sum()),
                RGB_MAE=float(np.abs(colors[mask]-target_at_anchor[mask]).mean()) if mask.any() else None,
                RGB_ZNCC=raw.zncc(colors[mask],target_at_anchor[mask]) if mask.any() else None)
        counts=valid.sum(1);fc=front.sum(1)
        def average(mask):return (colors*mask[:,:,None]).sum(1)/np.maximum(mask.sum(1,keepdims=True),1)
        choice=np.where(valid,anchor[:,:,5],np.inf).argmin(1)
        proxy=dict(retained_mean8=average(valid),front_filtered_mean=average(front),
            nearest_anchor_in_cell=colors[np.arange(16384),choice],dense_observed_warp=block_average(dense))
        full=np.isfinite(near).reshape(128,4,128,4).all((1,3)).reshape(-1)
        cy,cx=np.mgrid[:128,:128];central=((cx>=10)&(cx<118)&(cy>=10)&(cy<118)).reshape(-1)
        common=full&central&(counts>0)&(fc>0)
        reference=block_average(sat_rgb)
        color_metrics={name:dict(cells=int(common.sum()),ZNCC=raw.zncc(value[common],reference[common]),
            RGB_MAE=float(np.abs(value[common]-reference[common]).mean())) for name,value in proxy.items()}
        low=np.where(valid,anchor[:,:,5],np.inf).min(1);high=np.where(valid,anchor[:,:,5],-np.inf).max(1)
        spread=high[counts>0]-low[counts>0]
        row=dict(image_id=ident,anchor_replay_max_errors=errors,valid_candidates=int(valid.sum()),
            behind_nearest_observed_1m=int(behind.sum()),negative_residual_candidates=int(negative.sum()),
            supported_cells=int((counts>0).sum()),cells_without_front=int(((counts>0)&(fc==0)).sum()),
            mixed_front_and_behind_cells=int((front.any(1)&behind.any(1)).sum()),
            camera_Z_spread_cells=dict(le1=int((spread<=1).sum()),gt1_le5=int(((spread>1)&(spread<=5)).sum()),gt5=int((spread>5).sum())),
            candidate_color=candidate_errors,common_cell_color=color_metrics)
        rows.append(row)
        (OUT/'rows.json').write_text(json.dumps(rows,indent=2)+'\n')
        print('READ AUDIT',ident,row['valid_candidates'],row['behind_nearest_observed_1m'],color_metrics,flush=True)
    assert all(sha(p)==v for p,v in sources.items())
    assert sha(__file__)==protocol['source_sha256'] and sha(raw.__file__)==protocol['projector_source_sha256']
    summary=dict(images=8,valid_candidates=sum(r['valid_candidates'] for r in rows),
        behind_nearest_observed_1m=sum(r['behind_nearest_observed_1m'] for r in rows),
        negative_residual_candidates=sum(r['negative_residual_candidates'] for r in rows),
        supported_cells=sum(r['supported_cells'] for r in rows),cells_without_front=sum(r['cells_without_front'] for r in rows),
        mixed_front_and_behind_cells=sum(r['mixed_front_and_behind_cells'] for r in rows),
        mean_image_ZNCC={name:float(np.mean([r['common_cell_color'][name]['ZNCC'] for r in rows])) for name in proxy})
    (OUT/'complete.json').write_text(json.dumps(dict(complete=True,protocol=protocol,summary=summary,source_sha256=sources),indent=2)+'\n')


if __name__=='__main__':main()
