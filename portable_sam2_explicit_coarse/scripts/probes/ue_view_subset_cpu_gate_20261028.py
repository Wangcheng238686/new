"""Real cache/selected-I/O audit only; not a real model or four-GPU gate."""
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch
WORK=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(WORK.parent/'sam2'),str(WORK)]
import cv2
import numpy as np
import torch
from data.ue_view_subset_inputs import SubsetUAVInputs,code_hashes
from rsprompter.uav.local_adapter_probe import file_sha
from rsprompter.uav.view_subset import PRESETS,subset_key,check_index_views


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('/data/wangcheng/dataset/CitySample_SatUAV_TrainMinimal_v1_20261021'))
    p.add_argument('--cache-root',type=Path,required=True)
    p.add_argument('--image-id',type=int,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();rows=[];read=cv2.imread
    for count,views in PRESETS.items():
        inputs=SubsetUAVInputs(args.data,args.cache_root,views,require_full_dataset=False)
        paths=[]
        def track(path,*a,**k):
            paths.append(str(path));return read(path,*a,**k)
        with patch.object(cv2,'imread',side_effect=track):value=inputs.load(args.image_id,device='cpu')
        assert len(paths)==count and all(any('/yaw_%03d/'%v in name for v in views) for name in paths)
        removed=[i for i,v in enumerate((0,72,144,216,288)) if v not in views]
        if removed:assert not value['rgb'][removed].any()
        check_index_views(value['index'],views)
        rows.append(dict(view_ids=list(views),RGB_files_read=paths,candidates=int((value['index']>=0).sum()),
                         supported_pixels=int(value['geometry'][:,4].sum()),
                         cache_manifest_sha256=inputs.complete['manifest_sha256']))
        if count==5:
            old_anchor=Path('/data/wangcheng/cache/ue_v1_s2db_geometry_v2')/('%06d.npz'%args.image_id)
            old_native=Path('/data/wangcheng/cache/ue_v1_native_geometry')/('%06d.npz'%args.image_id)
            with np.load(old_anchor) as old:
                assert np.array_equal(value['index'].numpy(),old['index'])
                assert np.array_equal(value['anchor'].numpy(),old['anchor'])
            with np.load(old_native) as old:
                expected=np.concatenate((old['fields'].astype(np.float32),old['support'][None].astype(np.float32)))[None]
                assert np.array_equal(value['geometry'].numpy(),expected)
            rows[-1].update(old_candidate_index_exact=True,old_anchor_exact=True,old_native_geometry_exact=True,
                            old_anchor_cache_sha256=file_sha(old_anchor),old_native_cache_sha256=file_sha(old_native))
    # Production construction must reject this partial engineering cache.
    for views in PRESETS.values():
        try:SubsetUAVInputs(args.data,args.cache_root,views)
        except ValueError as e:
            assert 'all train/val/test' in str(e)
        else:raise AssertionError('Partial engineering cache accepted for formal training')
    report=dict(passed=True,scope='one_real_train_image_cache_and_selected_RGB_IO_CPU_no_model_forward_no_AP',
                image_id=args.image_id,rows=rows,formal_partial_cache_rejected=True,source_code_sha256=code_hashes(),
                gate_sha256=file_sha(Path(__file__)))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
