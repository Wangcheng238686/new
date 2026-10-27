"""One-command, immutable-code, isolated-arm full-budget queue. Never trains on import."""
import argparse,fcntl,hashlib,json,os,subprocess,sys,time
from pathlib import Path
WORK=Path(__file__).resolve().parents[2];REPO=WORK.parent
ROOT=Path('/data/wangcheng/checkpoint/ue_g_ubr_native_full100_20261028')
ANALYSIS=Path('/data/wangcheng/checkpoint/ue_g_ubr_native_short_20261027/analysis_native_20261028')
GATE=Path('/data/wangcheng/checkpoint/ue_ubr_native_gate_20261027/attempt5/complete.json')
RESOURCE=Path('/data/wangcheng/checkpoint/ue_ubr_native_resource_20261027/ABC_complete.json')
ARMS=('A_nohead','B_zero','C_correct');ALIASES=dict(A=ARMS[0],B=ARMS[1],C=ARMS[2],**{a:a for a in ARMS})
RESERVATIONS=dict(A_nohead=5988,B_zero=8596,C_correct=8890)
CAPS=dict(A_nohead=5600,B_zero=8200,C_correct=8500)
PYTHON='/data/wangcheng/envs/cvt2/bin/python'


def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def write(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,indent=2,sort_keys=True)+'\n');t.replace(p)

def read(p):return json.loads(Path(p).read_text())

def identity(pid):
 try:
  p=Path('/proc')/str(pid);a=(p/'cmdline').read_bytes().decode().rstrip('\0').split('\0');stat=(p/'stat').read_text().rsplit(') ',1)[1].split()
  return dict(pid=int(pid),start_ticks=stat[19],argv=a,state=stat[0])
 except (OSError,ProcessLookupError):return None

def live(d):
 a=identity(d.get('pid',0));return bool(a and a['state']!='Z' and a['start_ticks']==str(d['start_ticks']))

def modules(source):
 work=Path(source)/WORK.name;sys.path[:0]=[str(Path(source)/'sam2'),str(work),str(work/'scripts/probes')]
 import ue_ubr_native_full_run_20261028 as full
 import ue_ubr_native_owner_20261027 as admission
 return full,admission

def prepare(root):
 root.mkdir(parents=True,exist_ok=True);path=root/'launch_manifest.json'
 with (root/'prepare.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  if path.exists():
   manifest=read(path)
   if subprocess.check_output(['git','rev-parse','HEAD'],cwd=manifest['runtime_source'],text=True).strip()!=manifest['git_commit']:raise RuntimeError('Pinned runtime worktree HEAD changed')
   for n,h in manifest['source_sha256'].items():
    if sha(Path(manifest['runtime_source'])/n)!=h:raise RuntimeError('Pinned runtime bytes changed '+n)
   return manifest
  commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip();source=root/'runtime_source'
  if source.exists():raise RuntimeError('Unbound runtime_source already exists; preserve and audit')
  subprocess.run(['git','worktree','add','--detach',str(source),commit],cwd=REPO,check=True)
  full,admission=modules(source);orders=full.prepare_orders(root)
  sources=full.source_inventory()
  receipt=read(GATE)
  for n,h in full.native.normalize_sources(receipt).items():sources[n]=h
  for n in ('scripts/experiments/ue_native_full100_control.py','scripts/experiments/ue_native_full100.sh'):
   key=str(Path(WORK.name)/n);sources[key]=sha(source/key)
  for n,h in sources.items():
   if sha(source/n)!=h:raise RuntimeError('Source mismatch '+n)
   blob=subprocess.check_output(['git','show',commit+':'+n],cwd=source)
   if hashlib.sha256(blob).hexdigest()!=h:raise RuntimeError('Commit full-budget source first '+n)
  full.native.verify_receipt(GATE,sources)
  resource=read(RESOURCE)
  if resource.get('passed') is not True or resource.get('world_size')!=4:raise RuntimeError('Passed four-rank resource audit required')
  for arm in ARMS:
   peak=resource['arms'][arm]['max_reserved_mib'];resident=resource['arms'][arm]['max_resident_mib']
   if RESERVATIONS[arm]<max(peak+512,resident+32) or CAPS[arm]<peak:raise RuntimeError('Full arm resource assumptions invalid '+arm)
  manifest=dict(version=1,root=str(root),runtime_source=str(source),git_commit=commit,source_sha256=sources,
   protocol=full.PROTOCOL,order_manifest_sha256=sha(root/'orders/manifest.json'),engineering_receipt=str(GATE),engineering_sha256=sha(GATE),
   resource_receipt=str(RESOURCE),resource_sha256=sha(RESOURCE),per_arm_reservation_mib=RESERVATIONS,allocator_caps=CAPS,
   margin_mib=512,dependencies=[str(ANALYSIS/'attribution_queue_state.json'),str(ANALYSIS/'fixed_box_queue_complete.json'),str(ANALYSIS/'interventions/complete.json')],
   initialization='fresh common generic foundation, full100; explicit resume only own same-arm after_validation checkpoints',created=time.time())
  write(path,manifest);return manifest


def command(m,action,arm,resume=False):
 work=Path(m['runtime_source'])/WORK.name
 argv=[PYTHON,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=4',str(work/'scripts/probes/ue_ubr_native_full_run_20261028.py'),action,
       '--arm',arm,'--root',m['root'],'--engineering-receipt',m['engineering_receipt'],'--allocator-cap-mib',str(m['allocator_caps'][arm])]
 if resume:argv+=['--resume',str(Path(m['root'])/arm/'last.pth')]
 return argv


def environment(m):
 return dict(os.environ,CUDA_VISIBLE_DEVICES='0,1,2,3',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1',UE_UBR_NATIVE_SOURCE_COMMIT=m['git_commit'])


def dependencies(m):
 waiting=[]
 for p in m['dependencies']:
  path=Path(p)
  if not path.exists():waiting.append(p);continue
  d=read(path)
  if d.get('failed') or ('returncodes' in d and any(v!=0 for v in d['returncodes'].values())):raise RuntimeError('Preceding attribution failed '+p)
  if not d.get('complete'):waiting.append(p)
 return waiting


def validate_sources(m):
 source=Path(m['runtime_source'])
 for n,h in m['source_sha256'].items():
  if sha(source/n)!=h:raise RuntimeError('Frozen full-budget source drift '+n)
 for key in ('engineering','resource'):
  if sha(m[key+'_receipt'])!=m[key+'_sha256']:raise RuntimeError(key+' receipt drift')
 if sha(Path(m['root'])/'orders/manifest.json')!=m['order_manifest_sha256']:raise RuntimeError('Order manifest drift')


def marker(root,action,arm):
 return root/'resume_gate'/arm/'complete.json' if action=='gate-resume' else root/arm/'training_complete.json' if action=='train' else root/'evaluation/test'/arm/'complete.json'


def accepted(m,job):
 p=Path(job['marker'])
 if not p.exists():return False
 d=read(p)
 if job['action']=='gate-resume':
  if d.get('passed') is not True or d.get('world_size')!=4:return False
  c=d['contract']
  return c['arm']==job['arm'] and c['git_commit']==m['git_commit'] and c['protocol']==m['protocol'] and c['allocator_cap_mib']==m['allocator_caps'][job['arm']] and c['engineering_receipt_sha256']==m['engineering_sha256'] and all(m['source_sha256'].get(n)==h for n,h in c['source_hashes'].items()) and c['source_order_sha256'][str(Path(m['root'])/'orders/manifest.json')]==m['order_manifest_sha256'] and sha(p.parent/'restore_gate.pth')==d['checkpoint_sha256']
 if not d.get('complete') or d.get('world_size')!=4 or d.get('arm')!=job['arm']:return False
 if job['action']=='train':
  return d['epoch']==100 and d['updates']==26700 and d['contract']['git_commit']==m['git_commit'] and sha(p.parent/'best_model.pth')==d['best_checkpoint_sha256'] and sha(p.parent/'last.pth')==d['last_checkpoint_sha256']
 trained=read(Path(m['root'])/job['arm']/'training_complete.json')
 return d['split']=='test' and d['images']==459 and d['precision']=='FP32' and d['checkpoint_sha256']==trained['best_checkpoint_sha256']


def owner(root,recover=False):
 m=read(root/'launch_manifest.json');full,admission=modules(m['runtime_source']);validate_sources(m)
 lock=(root/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 prior=root/'owner_state.json'
 if prior.exists() and not recover:raise RuntimeError('Existing state requires explicit wrapper recover')
 jobs={}
 for action,arms in [('gate-resume',('C_correct',)),('train',ARMS),('eval',ARMS)]:
  for arm in arms:
   name=action+'_'+arm
   jobs[name]=dict(action=action,arm=arm,status='pending',marker=str(marker(root,action,arm)),
     dependencies=[] if action=='gate-resume' else ['gate-resume_C_correct'] if action=='train' else ['train_'+a for a in ARMS])
 if recover:
  previous=read(prior)
  for job in previous['jobs'].values():
   if job.get('launcher') and live(job['launcher']):raise RuntimeError('Owned launcher still live; do not recover')
   rows=job.get('ranks',[])
   if job.get('rank_launch_record') and Path(job['rank_launch_record']).exists():rows=rows+read(job['rank_launch_record']).get('ranks',[])
   for r in rows:
    if live(r):raise RuntimeError('Owned rank still live; do not recover')
  for name,job in jobs.items():
   if accepted(m,job):job['status']='complete'
   elif job['action']=='train' and (root/job['arm']/'last.pth').exists():job['resume']=True
  write(root/'recovery'/str(time.time_ns())/'prior_state.json',previous)
 children={};streams={};failures=[];me=identity(os.getpid());admission.old.RUN=Path(m['runtime_source'])/WORK.name/'scripts/probes/ue_ubr_native_full_run_20261028.py'
 while True:
  for name,p in list(children.items()):
   rc=p.poll()
   if rc is None:continue
   streams.pop(name).close();del children[name];job=jobs[name]
   ok=rc==0 and accepted(m,job);job.update(status='complete' if ok else 'failed',returncode=rc,ended=time.time())
   if not ok:failures.append(name+' failed')
  waiting=[];cards={}
  if not failures:
   try:
    validate_sources(m);waiting=dependencies(m);cards=admission.resources(jobs,m['per_arm_reservation_mib'],m['margin_mib'])
   except Exception as e:failures.append(repr(e))
  if not failures and not waiting:
   for name,job in jobs.items():
    if job['status']!='pending' or any(jobs[d]['status']!='complete' for d in job['dependencies']):continue
    if len(children)>=3:break
    if job['action']!='train' and children:continue
    reserve=m['per_arm_reservation_mib'][job['arm']]
    if any(c['admissible_free_mib']<reserve for c in cards.values()):continue
    log=root/'logs'/name/('attempt_'+str(time.time_ns())+'.log');log.parent.mkdir(parents=True,exist_ok=True);s=log.open('x')
    argv=command(m,job['action'],job['arm'],job.get('resume',False))
    p=subprocess.Popen(argv,cwd=Path(m['runtime_source'])/WORK.name,env=environment(m),stdin=subprocess.DEVNULL,stdout=s,stderr=subprocess.STDOUT,start_new_session=True)
    children[name]=p;streams[name]=s;job.update(status='running',launcher=identity(p.pid),argv=argv,log=str(log),launched=time.time(),ranks=[],rank_launch_record=str(root/'rank_launches'/(job['action']+'_'+job['arm']+'.json')))
    for c in cards.values():c['admissible_free_mib']-=reserve
    print('START',name,p.pid,flush=True)
  done=all(j['status']=='complete' for j in jobs.values())
  write(prior,dict(stage='complete' if done else 'failed' if failures else 'waiting_for_analysis' if waiting else 'running',owner=me,jobs=jobs,failures=failures,waiting_for=waiting,GPU_headroom=cards,updated=time.time(),source_git_commit=m['git_commit']))
  if done:write(root/'owner_complete.json',dict(complete=True,git_commit=m['git_commit'],jobs=len(jobs)));return
  if failures and not children:raise RuntimeError(str(failures))
  time.sleep(15)


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['plan','prepare','queue','status','train','resume','eval','recover','owner']);p.add_argument('--arm',choices=list(ALIASES));p.add_argument('--root',type=Path,default=ROOT);a=p.parse_args();root=a.root.resolve()
 if a.action=='status':
  for n in ('launch_manifest.json','owner_state.json'):
   path=root/n
   if path.exists():
    d=read(path)
    if n=='launch_manifest.json':d={k:d[k] for k in ('root','runtime_source','git_commit','initialization')}
    print(json.dumps(d,indent=2))
  return
 if a.action=='plan':
  print(json.dumps(dict(arms=ARMS,epochs=100,updates=26700,world_size_per_arm=4,root=str(root),selection='val MaskAP best, ties earlier',wait_for=str(ANALYSIS),wrapper=str(Path(__file__).with_name('ue_native_full100.sh'))),indent=2));return
 if a.action=='owner':owner(root,bool(os.environ.get('UE_FULL_RECOVER')));return
 m=prepare(root)
 if a.action=='prepare':print(json.dumps(m,indent=2));return
 source_control=Path(m['runtime_source'])/WORK.name/'scripts/experiments/ue_native_full100_control.py'
 if a.action in ('queue','recover'):
  state=root/'owner_state.json'
  if state.exists() and live(read(state)['owner']):print('Already queued/running',read(state)['owner']['pid']);return
  if state.exists() and a.action!='recover':raise RuntimeError('Dead owner state: use explicit recover after reviewing status')
  log=root/'owner.log'
  with log.open('a') as s:
   env=environment(m)
   if a.action=='recover':env['UE_FULL_RECOVER']='1'
   child=subprocess.Popen([PYTHON,'-u',str(source_control),'owner','--root',str(root)],cwd=source_control.parents[2],env=env,stdin=subprocess.DEVNULL,stdout=s,stderr=subprocess.STDOUT,start_new_session=True)
  print(json.dumps(dict(owner=identity(child.pid),root=str(root),log=str(log),queued=True)));return
 if not a.arm:p.error('--arm A/B/C required')
 state=root/'owner_state.json'
 if state.exists() and live(read(state)['owner']):raise RuntimeError('Queue owner active; use queue/status to avoid duplicate arm launch')
 arm=ALIASES[a.arm];validate_sources(m)
 if dependencies(m):raise RuntimeError('Preceding attribution still pending; use queue')
 gate=root/'resume_gate/C_correct/complete.json'
 if not accepted(m,dict(action='gate-resume',arm='C_correct',marker=str(gate))):raise RuntimeError('This full-budget recovery gate pending/invalid; use queue')
 _,admission=modules(m['runtime_source'])
 cards=admission.resources({},m['per_arm_reservation_mib'],m['margin_mib'])
 if any(c['admissible_free_mib']<m['per_arm_reservation_mib'][arm] for c in cards.values()):raise RuntimeError('Four-card memory insufficient for manual launch; use queue')
 if a.action=='resume' and not (root/arm/'last.pth').exists():raise RuntimeError('No same-arm last checkpoint')
 argv=command(m,'train' if a.action in ('train','resume') else 'eval',arm,a.action=='resume')
 os.execve(PYTHON,argv,environment(m))

if __name__=='__main__':main()
