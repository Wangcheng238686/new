"""Resource-only wrapper: same J1 math, allocator ceiling verified by GPU gate."""
import argparse
import sys
import torch


def main():
    p=argparse.ArgumentParser();p.add_argument('--limit-mib',type=int,required=True)
    args,remaining=p.parse_known_args()
    assert 0<args.limit_mib<=5500
    assert remaining and remaining[0] in ('train','eval')
    total=torch.cuda.get_device_properties(0).total_memory/1024**2
    torch.cuda.set_per_process_memory_fraction(args.limit_mib/total)
    import ue_joint_encoding_probe as run
    sys.argv=[run.__file__]+remaining
    run.main()


if __name__=='__main__':main()
