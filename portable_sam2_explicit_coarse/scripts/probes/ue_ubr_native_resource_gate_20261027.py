"""Four-rank A/B native-model resource audit, two formal accumulated updates.

No experiment training. Forced250 boxes stress the original mask consumer;
these fixtures are not an AP evaluation or claims of 250 true detections.
"""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

WORK=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(WORK),str(WORK/'scripts/probes')]


def write(path,value):
    path=Path(path);temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');temp.replace(path)


def memory():
    torch.cuda.synchronize()
    text=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
    rows=[row.strip().split(',') for row in text.splitlines()]
    resident=[float(row[1].strip()) for row in rows if int(row[0].strip())==os.getpid()]
    return dict(allocated_mib=torch.cuda.memory_allocated()/2**20,
                reserved_mib=torch.cuda.memory_reserved()/2**20,
                peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                nvidia_resident_mib=sum(resident),nvidia_pid_records=len(resident))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm',choices=['A_nohead','B_zero','C_correct'],required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    assert int(os.environ.get('WORLD_SIZE','0'))==4
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local);torch.set_num_threads(2)
    dist.init_process_group('nccl',timeout=timedelta(minutes=30))
    cpu=dist.new_group(backend='gloo',timeout=timedelta(minutes=30))
    if rank==0:args.output.mkdir(parents=True,exist_ok=False)
    dist.barrier()
    report=dict(arm=args.arm,rank=rank,world_size=4,pid=os.getpid(),updates=[],
                scope='resource_engineering_only_two_formal_global8_updates_no_AP')
    try:
        import ue_candidate_detail_ddp_20261026 as previous
        from ue_ubr_native_runtime_20261027 import build_runtime,base
        from ue_ubr_native_gate_20261027 import source_inventory
        rt=build_runtime(args.arm);rt.bare=rt.network
        rt.network=DistributedDataParallel(rt.bare,device_ids=[local],output_device=local,
                                          find_unused_parameters=True,broadcast_buffers=False)
        rt.generator=torch.Generator().manual_seed(44+rank)
        base.core.C.seed(44+rank)
        report['initialization']=rt.initialization_audit
        ds=base.core.C.MinimalSatelliteDataset(base.core.previous.DATA,'train',augment=False)
        ordered=sorted(range(len(ds)),key=lambda i:len(ds.samples[i]['annotations']),reverse=True)
        indices=[ordered[rank],ordered[rank+4]]
        batches=[(base.core.C.rtmdet_collate_fn([ds[i]]),int(ds.samples[i]['image_id'])) for i in indices]
        report['image_ids']=[ident for batch,ident in batches]
        report['GT_counts']=[len(ds.samples[i]['annotations']) for i in indices]
        report['source_hashes']=source_inventory()
        rt.network.train();rt.set_epoch(1)
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        for step in (1,2):
            torch.cuda.synchronize();start=time.perf_counter()
            result=previous.accumulate(rt,batches,step)
            torch.cuda.synchronize()
            report['updates'].append(dict(step=step,seconds=time.perf_counter()-start,
                                           result=result,memory=memory(),optimizer_states=len(rt.optimizer.state)))
            write(args.output/('rank%d_progress.json'%rank),report)
        rt.network.eval();rt.clear();rt.optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        sample=base.core.C.processed(rt.satellite,batches[0][0],False)
        with torch.no_grad():
            payload=rt.payload(batches[0][1]);rt.joint.prepare_uav(**payload)
            features=rt.satellite.extract_feat(sample['inputs'])
            head=rt.satellite.roi_head;xx=head._inject_extra_pe(features[0]);meta=sample['data_samples'][0].metainfo
            boxes=torch.tensor([[16.25,20.5,83.75,93.25]],device='cuda').repeat(250,1)
            scores=torch.full((250,),.1,device='cuda')
            torch.cuda.synchronize();start=time.perf_counter()
            result=base.core.consumer.masks(head,xx,features,meta,boxes,scores)
            torch.cuda.synchronize()
            assert len(result['scores'])==250
            report['inference250']=dict(seconds=time.perf_counter()-start,instances=250,
                forced_box_fixture=True,precision='FP32',original_consumer=True,memory=memory())
        report['sources_after']=source_inventory()
        assert all(report['sources_after'].get(k)==v for k,v in report['source_hashes'].items())
        report['passed']=True
        write(args.output/('rank%d.json'%rank),report)
        rows=[None]*4;dist.all_gather_object(rows,report,group=cpu)
        if rank==0:write(args.output/'complete.json',dict(passed=True,arm=args.arm,world_size=4,ranks=rows))
    except Exception:
        report['failed']=traceback.format_exc();write(args.output/('rank%d_failed.json'%rank),report);raise
    finally:
        dist.destroy_process_group()


if __name__=='__main__':main()
