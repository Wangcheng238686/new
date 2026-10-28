"""CPU explicit 1/3/5-view cache builder, isolated by exact camera subset."""
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import argparse
import fcntl
import json
from pathlib import Path
import sys
WORK=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(WORK.parent/'sam2'),str(WORK)]
import numpy as np
from data.ue_view_subset_inputs import VERSION,code_hashes,build_image
from rsprompter.uav.local_adapter_probe import file_sha
from rsprompter.uav.view_subset import PRESETS,canonical_views,subset_key


def dump(path,value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');temp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('/data/wangcheng/dataset/CitySample_SatUAV_TrainMinimal_v1_20261021'))
    p.add_argument('--cache-root',type=Path,required=True)
    choice=p.add_mutually_exclusive_group(required=True)
    choice.add_argument('--view-count',type=int,choices=sorted(PRESETS))
    choice.add_argument('--view-ids',type=int,nargs='+')
    p.add_argument('--ids',type=int,nargs='+',help='Engineering subset only; omit for the full dataset')
    args=p.parse_args()
    views=canonical_views(PRESETS[args.view_count] if args.view_count else args.view_ids)
    folder=args.cache_root/subset_key(views);folder.mkdir(parents=True,exist_ok=True)
    with (folder/'build.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        splits={s:json.loads((args.data/'splits'/(s+'.json')).read_text()) for s in ('train','val','test')}
        full=[int(i) for values in splits.values() for i in values]
        if len(full)!=len(set(full)):raise ValueError('Overlapping split image IDs')
        ids=sorted(set(args.ids) if args.ids else full)
        if not set(ids)<=set(full):raise ValueError('Image ID outside dataset splits')
        manifest=dict(version=VERSION,view_ids=list(views),data_root=str(args.data.resolve()),image_ids=ids,
            source_code_sha256=code_hashes(),builder_sha256=file_sha(Path(__file__)),
            split_sha256={s:file_sha(args.data/'splits'/(s+'.json')) for s in splits},
            purpose='engineering_subset' if args.ids else 'full_dataset',view_count_denominator=5,
            canonical_source_slots=[0,72,144,216,288])
        target=folder/'manifest.json'
        if target.exists():
            if json.loads(target.read_text())!=manifest:raise ValueError('Existing subset cache binds a different contract')
        else:dump(target,manifest)
        for name in code_hashes():
            destination=folder/'source'/name;destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_bytes((WORK/name).read_bytes())
        rows=[]
        for ident in ids:
            path=folder/('%06d.npz'%ident)
            # Recompute even on resume: verifies current raw sources and all old
            # tensor bytes, rather than trusting stale five-view derived fields.
            arrays,report=build_image(args.data,ident,views)
            if path.exists():
                with np.load(path,allow_pickle=False) as old:
                    if str(old['version'])!=VERSION or json.loads(str(old['report']))!=report or any(not np.array_equal(old[k],v) for k,v in arrays.items()):
                        raise ValueError('Existing cache differs from fresh raw rebuild '+str(ident))
            else:
                temp=path.with_suffix('.tmp')
                with temp.open('wb') as stream:np.savez_compressed(stream,version=VERSION,report=json.dumps(report,sort_keys=True),**arrays)
                temp.replace(path)
            rows.append(dict(image_id=ident,cache_sha256=file_sha(path),report=report))
            if len(rows)%25==0:print('SUBSET CACHE',subset_key(views),len(rows),len(ids),flush=True)
        if manifest['source_code_sha256']!=code_hashes() or manifest['builder_sha256']!=file_sha(Path(__file__)):
            raise ValueError('Source changed while building')
        dump(folder/'BUILD_COMPLETE.json',dict(complete=True,images=len(rows),manifest_sha256=file_sha(target),rows=rows))
        print('SUBSET COMPLETE',subset_key(views),len(rows),flush=True)


if __name__=='__main__':main()
