"""User-authorized shared-GPU orchestration; experiment sources stay frozen."""
import os,json,time,subprocess,hashlib,fcntl
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
PY='/data/wangcheng/envs/cvt2/bin/python'
BASE=Path('/data/wangcheng/checkpoint')
S=BASE/'ue_s2db_probe_20261022';N=BASE/'ue_native_assignment_probe_20261022'
R=BASE/'ue_probe_coresident_20261022';R.mkdir(exist_ok=True)
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):
 p=Path(p);t=p.with_suffix(p.suffix+'.coresident.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def alive(pid):
 try:return Path(f'/proc/{pid}/status').exists() and '\nState:\tZ' not in Path(f'/proc/{pid}/status').read_text()
 except (FileNotFoundError,ProcessLookupError):return False

def memory():
 lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total,uuid','--format=csv,noheader,nounits'],text=True).splitlines()
 g={};uuid={}
 for line in lines:
  ix,u,total,key=[v.strip() for v in line.split(',')];g[int(ix)]=dict(used=int(u),total=int(total));uuid[key]=int(ix)
 pids={}
 lines=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory,gpu_uuid','--format=csv,noheader,nounits'],text=True).splitlines()
 for line in lines:
  p,u,key=[v.strip() for v in line.split(',')]
  if p.isdigit() and u.isdigit() and key in uuid:pids[int(p)]=(uuid[key],int(u))
 return g,pids

def main():
 lock=(R/'owner.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 ss=read(S/'state.json');ns=read(N/'state.json');sm=read(S/'manifest.json');nm=read(N/'plan.json')
 pid=os.getpid();ss['owner_pid']=pid;ns['owner_pid']=pid
 amendment=dict(registered_utc=time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime()),
  authorization='User: pending tasks should start immediately; shared GPUs allowed when memory permits',
  scope='resource scheduling only; no changes to model/data/initialization/optimizer/seeds/checkpoint/evaluation',
  old_s2db_owner=2857887,old_native_owner=2909731,new_owner=pid,
  reserves_MiB=dict(training=6500,evaluation=3500,native_gate=4500,offline_gradient=20000,device_headroom=2048,old_gpu1_evaluation=3500),
  source_sha256=sha(__file__),s2db_frozen_source_hashes=sm['source_hashes'],native_frozen_source_hashes=nm['source_hashes'])
 dump(R/'scheduling_amendment.json',amendment)
 for r in (S,N):dump(r/'coresident_scheduling_amendment.json',amendment)
 nj={};previous=ns.get('jobs',{})
 def add(key,cmd,artifact,gpu=True,deps=(),priority=10):
  nj[key]=previous.get(key,dict(key=key,cmd=cmd,artifact=str(artifact),gpu_required=gpu,dependencies=list(deps),priority=priority,status='complete' if Path(artifact).exists() else 'pending'))
 script=str(ROOT/'scripts/probes/ue_native_assignment_probe.py');analysis=str(ROOT/'scripts/probes/ue_native_assignment_analysis.py')
 add('gpu_gate',[PY,script,'gate','--out',str(N)],N/'gpu_gate.json',priority=0)
 for mode in nm['modes']:
  add('eval_'+mode,[PY,script,'eval','--mode',mode,'--out',str(N/mode)],N/mode/'diagnostic_identity.json',deps=['gpu_gate']+(['eval_'+nm['modes'][nm['modes'].index(mode)-1]] if mode!='correct' else []),priority=12)
 add('delta8',[PY,analysis,'delta8','--root',str(N)],N/'delta8.json',deps=['eval_'+nm['modes'][-1]],priority=15)
 add('analysis',[PY,analysis,'report','--root',str(N)],N/'complete_audit_summary.json',False,['delta8'],50)
 ns['jobs']=nj
 children={};streams={}
 def save():
  ss.update(owner_pid=pid,scheduling='coresident_memory_budget');ns.update(owner_pid=pid,scheduling='coresident_memory_budget')
  dump(S/'state.json',ss);dump(N/'state.json',ns)
 save()
 while True:
  for path,h in {**sm['source_hashes'],**nm['source_hashes']}.items():assert sha(ROOT/path)==h,('Frozen source changed',path)
  for tag,st in [('s2db',ss),('native',ns)]:
   for key,j in st['jobs'].items():
    if j['status']!='running':continue
    token=(tag,key)
    rc=children[token].poll() if token in children else (None if alive(j['pid']) else 0)
    if rc is None:continue
    if token in streams:streams.pop(token).close()
    if token in children:del children[token]
    ok=rc==0 and (not j.get('artifact') or Path(j['artifact']).exists())
    j.update(status='complete' if ok else 'failed',returncode=rc)
    print('END',tag,key,j['status'],flush=True);save()
  ready=[]
  for tag,st in [('s2db',ss),('native',ns)]:
   failures=[k for k,j in st['jobs'].items() if j['status']=='failed']
   if failures:st.update(stage='failure_requires_repair',failed_jobs=failures);continue
   if all(j['status']=='complete' for j in st['jobs'].values()):st['stage']='complete';continue
   st['stage']='running'
   for k,j in st['jobs'].items():
    if j['status']=='pending' and all(st['jobs'][d]['status']=='complete' for d in j['dependencies']):ready.append((j['priority'],tag,k,j,st))
  ready.sort(key=lambda x:x[:3])
  g,pids=memory()
  running=[j for st in (ss,ns) for j in st['jobs'].values() if j['status']=='running']
  cpu_count=sum(not j['gpu_required'] for j in running)
  # Include startup/activation headroom for workers whose CUDA allocations have
  # not yet reached their measured working footprint.
  projected={i:v['used'] for i,v in g.items()}
  for j in running:
   if not j['gpu_required']:continue
   allocation=pids.get(j['pid'],(j.get('gpu'),0))[1]
   projected[j['gpu']]+=max(0,j.get('reserve_MiB',6500 if j['key'].startswith('train_') else 3500)-allocation)
  old_test=read(BASE/'ue_v1_adapter_probe_20261021/test/test_state.json')
  old_pending=any(x['stage']!='complete' for x in old_test['evaluations'])
  # Keep room for an old evaluation that can start as its control checkpoint
  # arrives. This is a memory budget, not an exclusive device reservation.
  if old_pending:projected[1]+=3500
  for _,tag,key,j,st in ready:
   gpu=None
   cost=20000 if key.startswith('gradient_') else (6500 if key.startswith('train_') else (4500 if key=='gpu_gate' else 3500))
   if j['gpu_required']:
    choices=[i for i,v in g.items() if v['total']-projected[i]>=cost+2048]
    if not choices:continue
    gpu=max(choices,key=lambda i:g[i]['total']-projected[i]);projected[gpu]+=cost
   else:
    if cpu_count>=2:continue
    cpu_count+=1
   r=S if tag=='s2db' else N;log=r/'logs'/f'{key}.log';log.parent.mkdir(exist_ok=True)
   stream=log.open('ab');env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
   env['CUDA_VISIBLE_DEVICES']=str(gpu) if gpu is not None else ''
   proc=subprocess.Popen(j['cmd'],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
   token=(tag,key);children[token]=proc;streams[token]=stream
   j.update(status='running',pid=proc.pid,gpu=gpu,reserve_MiB=cost if gpu is not None else 0,log=str(log),
    started_utc=time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime()))
   print('START',tag,key,gpu,proc.pid,flush=True);save()
  save();dump(R/'state.json',dict(stage='complete' if ss['stage']==ns['stage']=='complete' else 'running',owner_pid=pid,
   physical_memory=g,projected_memory_MiB=projected,old_gpu1_memory_allowance=old_pending,
   running=[dict(family=t,key=k,pid=j['pid'],gpu=j.get('gpu')) for t,st in [('s2db',ss),('native',ns)] for k,j in st['jobs'].items() if j['status']=='running']))
  if ss['stage']==ns['stage']=='complete':return
  time.sleep(5)

if __name__=='__main__':main()
