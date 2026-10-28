"""Read-only validation CPU watcher and resource-admitted four-GPU gradient sidecar."""
import argparse,fcntl,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
WORK=Path(__file__).resolve().parents[2]
ROOT=Path('/data/wangcheng/checkpoint/ue_g_ubr_native_full100_20261028')
PYTHON='/data/wangcheng/envs/cvt2/bin/python'
ARMS=('A_nohead','B_zero','C_correct');ANCHORS=(1,10,25,50,75,100)
SOURCES=('ue_native_telemetry_owner_20261028.py','ue_native_detection_telemetry_20261028.py','ue_native_gradient_telemetry_20261028.py','ue_native_gradient_math_20261028.py')


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2,sort_keys=True)+'\n');t.replace(p)
def identity(pid):
 try:return {'pid':pid,'start_ticks':Path('/proc',str(pid),'stat').read_text().rsplit(') ',1)[1].split()[19]}
 except OSError:return None

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['start','run','status']);p.add_argument('--root',type=Path,default=ROOT);a=p.parse_args();folder=a.root/'telemetry';folder.mkdir(parents=True,exist_ok=True)
 if a.action=='status':print(json.dumps(read(folder/'owner_state.json') if (folder/'owner_state.json').exists() else {'stage':'not_started'},indent=2));return
 if a.action=='start':
  path=folder/'owner_state.json'
  if path.exists():
   state=read(path);owner=state['owner']
   if identity(owner['pid'])==owner:print('Telemetry owner already running',owner['pid']);return
   raise RuntimeError('Prior telemetry owner needs explicit state/child audit; do not overwrite')
  source=folder/'observer_source';source.mkdir(exist_ok=False)
  for name in SOURCES:shutil.copyfile(Path(__file__).with_name(name),source/name)
  manifest=dict(sources={name:sha(source/name) for name in SOURCES},observer_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=WORK,text=True).strip(),root=str(a.root),sampling_epochs=ANCHORS,
   gradient_precision='FP32 train-mode fixed8; no augmentation, no optimizer; not live AMP update',GPU_allocator_cap_mib=14336,GPU_context_reserve_mib=512,all_card_margin_mib=1024,gradient_max_groups=1,CPU_max_workers=2)
  write(folder/'observer_manifest.json',manifest)
  env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
  with (folder/'owner.log').open('x') as log:
   child=subprocess.Popen([PYTHON,'-u',str(source/SOURCES[0]),'run','--root',str(a.root)],cwd=WORK,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  print(json.dumps({'owner':identity(child.pid),'folder':str(folder)}));return
 lock=(folder/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (folder/'owner_state.json').exists(),'Do not overwrite prior observer state'
 manifest=read(folder/'observer_manifest.json');source=folder/'observer_source';children={};streams={};completed={};failed={};owner=identity(os.getpid())
 while True:
  for name,p in list(children.items()):
   rc=p.poll()
   if rc is None:continue
   row=p.telemetry_job;streams.pop(name).close();del children[name]
   output=Path(row['output']);passed=rc==0 and output.exists() and read(output).get('complete') is True
   (completed if passed else failed)[name]=dict(row,returncode=rc)
  for name,digest in manifest['sources'].items():
   if sha(source/name)!=digest:raise RuntimeError('Observer source mutated '+name)
  pending=[]
  # Most recent available validation epochs first; fill every completed epoch.
  for epoch in range(100,0,-1):
   for arm in ARMS:
    name='detection_'+arm+'_E'+str(epoch);out=folder/'detection'/arm/('epoch%d.json'%epoch)
    if out.exists() and name not in children:
     d=read(out);assert d['complete'] and d['arm']==arm and d['epoch']==epoch
     completed.setdefault(name,{'output':str(out),'adopted_existing':True});continue
    if name in children or name in failed:continue
    marker=a.root/arm/('validation_epoch%d'%epoch)/'complete.json'
    if not marker.exists():continue
    pending.append(dict(name=name,kind='CPU',arm=arm,epoch=epoch,output=str(out),argv=[PYTHON,'-u',str(source/SOURCES[1]),'--root',str(a.root),'--arm',arm,'--epoch',str(epoch),'--output',str(out)]))
  # First C/E25 is the genuine-model pilot, then identical anchors for all arms.
  priority=[(25,'C_correct')]+[(e,arm) for e in ANCHORS for arm in ARMS if (e,arm)!=(25,'C_correct')]
  for epoch,arm in priority:
   name='gradient_'+arm+'_E'+str(epoch);out=folder/'gradients'/arm/('epoch%d'%epoch)
   if name in children or name in completed or name in failed:continue
   if (out/'complete.json').exists():
    d=read(out/'complete.json');assert d['complete'] and d['arm']==arm and d['epoch']==epoch;completed[name]={'output':str(out/'complete.json'),'adopted_existing':True};continue
   if not (a.root/arm/('epoch%d.pth'%epoch)).exists():continue
   pending.append(dict(name=name,kind='GPU',arm=arm,epoch=epoch,output=str(out/'complete.json'),argv=[PYTHON,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=4',str(source/SOURCES[2]),'--root',str(a.root),'--arm',arm,'--epoch',str(epoch),'--output',str(out),'--allocator-cap-mib','14336']))
  raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True);free={int(x.split(',')[0]):int(x.split(',')[1]) for x in raw.splitlines()}
  for row in pending:
   active=[p.telemetry_job['kind'] for p in children.values()]
   if row['kind']=='CPU' and active.count('CPU')>=2:continue
   if row['kind']=='GPU':
    if active.count('GPU') or any(n.startswith('gradient_') for n in failed):continue
    if any(free.get(g,0)<14336+512+1024 for g in range(4)):continue
   path=folder/'logs'/(row['name']+'.log');path.parent.mkdir(exist_ok=True);log=path.open('x')
   env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1,2,3',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
   child=subprocess.Popen(row['argv'],cwd=WORK,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);child.telemetry_job=row
   children[row['name']]=child;streams[row['name']]=log;print('START',row['name'],child.pid,flush=True)
  write(folder/'owner_state.json',dict(stage='observing' if not failed else 'failed_diagnostics_retained',owner=owner,completed=completed,failed=failed,running={n:dict(p.telemetry_job,pid=p.pid) for n,p in children.items()},pending=[{k:v for k,v in j.items() if k!='argv'} for j in pending if j['name'] not in children],GPU_free_mib=free,updated=time.time()))
  if len(completed)==300+18 and not children:write(folder/'complete.json',dict(complete=True,validation_epochs=100,arms=ARMS,gradient_anchors=ANCHORS));return
  time.sleep(15)

if __name__=='__main__':main()
