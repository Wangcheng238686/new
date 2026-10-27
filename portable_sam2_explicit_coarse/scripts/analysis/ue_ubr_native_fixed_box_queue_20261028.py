"""One-shot four-rank-per-model fixed-C-box follow-up; no training or selection."""
import json,os,subprocess,time
from pathlib import Path
WORK=Path(__file__).resolve().parents[2]
ROOT=Path('/data/wangcheng/checkpoint/ue_g_ubr_native_short_20261027')
OUT=ROOT/'analysis_native_20261028'
PYTHON='/data/wangcheng/envs/cvt2/bin/python'

def main():
 while not (OUT/'interventions/segments.json').exists():
  failed=list((OUT/'interventions').glob('*failed*'))
  if failed:raise RuntimeError(str(failed))
  time.sleep(15)
 segments=json.loads((OUT/'interventions/segments.json').read_text());assert len(segments)==459 and len({r['image_id'] for r in segments})==459
 children=[];streams=[]
 for arm in ('A_nohead','B_zero'):
  folder=OUT/'fixed_C_boxes'/arm
  assert not folder.exists(),'Preserve started fixed-box work'
  log=OUT/('fixed_C_boxes_'+arm+'.log');stream=log.open('x');streams.append(stream)
  argv=[PYTHON,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=4',str(WORK/'scripts/analysis/ue_ubr_native_attribution_20261028.py'),
        '--action','fixed_boxes','--arm',arm,'--root',str(ROOT),'--output',str(folder)]
  child=subprocess.Popen(argv,cwd=WORK,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1,2,3',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
  children.append((arm,child));print('START',arm,child.pid,flush=True)
 results={a:c.wait() for a,c in children}
 for s in streams:s.close()
 (OUT/'fixed_box_queue_complete.json').write_text(json.dumps(dict(complete=all(v==0 for v in results.values()),returncodes=results),indent=2)+'\n')
 assert all(v==0 for v in results.values()),results

if __name__=='__main__':main()
