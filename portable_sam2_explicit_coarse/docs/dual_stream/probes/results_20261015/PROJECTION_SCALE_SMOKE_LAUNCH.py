"""Root-authorized shared-GPU smoke only. Preserve UE rendering priority."""
import runpy,sys,torch
# Applied before model allocation; hard per-process allocator cap ~4.9GiB.
torch.cuda.set_per_process_memory_fraction(.20,0)
sys.argv=['/tmp/ue_projection_scale_probe/extract_projection_scale.py','--smoke']
runpy.run_path(sys.argv[0],run_name='__main__')
