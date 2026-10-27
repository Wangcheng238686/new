"""One-shot CPU follow-up queue, maximum two processes, fixed existing records."""
from pathlib import Path
import json,os,subprocess,time
WORK=Path(__file__).resolve().parents[2];ROOT=Path('/data/wangcheng/checkpoint/ue_g_ubr_native_short_20261027');OUT=ROOT/'analysis_native_20261028'
PYTHON='/data/wangcheng/envs/cvt2/bin/python'
PAIRED=WORK/'scripts/analysis/ue_ubr_native_paired_20261028.py';ERRORS=WORK/'scripts/analysis/ue_ubr_native_fixed_box_errors_20261028.py'

def write(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)

def main():
 jobs={};correct=OUT/'interventions/correct/records.json'
 for mode in ('zero','donor'):
  base=OUT/'interventions'/mode/'records.json';label='C_correct_minus_'+mode;output=OUT/(label+'_paired.json')
  jobs[label]=dict(argv=[PYTHON,'-u',str(PAIRED),'--root',str(ROOT),'--baseline-records',str(base),'--candidate-records',str(correct),'--label',label,'--output',str(output)],inputs=[str(base),str(correct)],output=str(output),status='pending')
 for arm in ('A_nohead','B_zero'):
  base=OUT/'fixed_C_boxes'/arm/arm/'records.json';label='fixed_C_boxes_C_minus_'+arm;output=OUT/(label+'_paired.json')
  jobs[label]=dict(argv=[PYTHON,'-u',str(PAIRED),'--root',str(ROOT),'--baseline-records',str(base),'--candidate-records',str(correct),'--label',label,'--output',str(output)],inputs=[str(base),str(correct)],output=str(output),status='pending')
  label='fixed_C_boxes_C_minus_'+arm+'_errors';output=OUT/(label+'.json')
  jobs[label]=dict(argv=[PYTHON,'-u',str(ERRORS),'--baseline-records',str(base),'--candidate-records',str(correct),'--label',label,'--output',str(output)],inputs=[str(base),str(correct)],output=str(output),status='pending')
 children={};streams={}
 while True:
  for label,p in list(children.items()):
   rc=p.poll()
   if rc is None:continue
   streams.pop(label).close();del children[label]
   job=jobs[label];passed=rc==0 and Path(job['output']).exists() and json.loads(Path(job['output']).read_text()).get('complete')
   job.update(status='complete' if passed else 'failed',returncode=rc);print('END',label,rc,flush=True)
  # The two already-running architecture analyses hold the CPU slots until done.
  prerequisites=all((OUT/('paired_C_minus_A_'+view+'.json')).exists() for view in ('combined','original_path'))
  if not any(j['status']=='failed' for j in jobs.values()):
   for label,job in jobs.items():
    if len(children)>=2:break
    if not prerequisites and not label.endswith('_errors'):continue
    if job['status']!='pending' or not all(Path(p).exists() for p in job['inputs']):continue
    assert not Path(job['output']).exists(),'Preserve completed output'
    log=OUT/(label+'.log');s=log.open('x');streams[label]=s
    p=subprocess.Popen(job['argv'],cwd=WORK,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1'),stdout=s,stderr=subprocess.STDOUT,start_new_session=True)
    children[label]=p;job.update(status='running',pid=p.pid,log=str(log));print('START',label,p.pid,flush=True)
  done=all(j['status']=='complete' for j in jobs.values());failed=any(j['status']=='failed' for j in jobs.values())
  write(OUT/'attribution_queue_state.json',dict(complete=done,failed=failed,jobs=jobs))
  if done:return
  if failed and not children:raise RuntimeError('Analysis job failed; outputs preserved')
  time.sleep(15)

if __name__=='__main__':main()
