"""Finish ready CPU AP comparisons after an unrelated offline GPU audit failure."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]
R=Path('/data/wangcheng/checkpoint/ue_post_activation_20261023')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):
 t=p.with_suffix(p.suffix+'.recovery.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
def main():
 s=read(R/'state.json');pid=s['owner_pid']
 assert not Path(f'/proc/{pid}').exists(),'Original owner is still alive'
 assert not any(j['status']=='running' for j in s['jobs'].values())
 dump(R/'state_before_cpu_recovery_20261023.json',s)
 hashes=read(R/'manifest.json')['source_hashes']
 jobs=[j for j in s['jobs'].values() if j['status']=='pending' and not j['gpu_required'] and all(s['jobs'][d]['status']=='complete' for d in j['dependencies'])]
 def work(j):
  assert all(sha(ROOT/n)==v for n,v in hashes.items())
  log=R/'logs'/('cpu_recovery_'+j['key']+'.log')
  with log.open('a') as f:
   rc=subprocess.call(j['cmd'],cwd=ROOT,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1'),stdout=f,stderr=subprocess.STDOUT)
  return rc,str(log)
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  tasks={}
  for j in jobs:
   j.update(status='running',recovery_owner_pid=os.getpid());tasks[pool.submit(work,j)]=j
  dump(R/'state.json',s)
  for task in concurrent.futures.as_completed(tasks):
   j=tasks[task];rc,log=task.result()
   j.update(status='complete' if rc==0 and Path(j['artifact']).exists() else 'failed',returncode=rc,recovery_log=log)
   dump(R/'state.json',s);print(j['key'],j['status'],flush=True)
 dump(R/'cpu_recovery_complete.json',dict(complete=all(j['status']=='complete' for j in jobs),keys=[j['key'] for j in jobs],
  source_unchanged=all(sha(ROOT/n)==v for n,v in hashes.items()),scope='Ready CPU comparisons only; failed offline GPU topology audits remain separately reported'))
if __name__=='__main__':main()
