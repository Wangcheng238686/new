"""Independent CPU paired native-tail analysis; frozen training sources untouched.

One process per view, all three arms matched once per metric. No checkpoint
selection; full459 fixedE10 development-test. Generic explicit record pairs are
supported for later fixed-checkpoint/fixed-box interventions with identical GT.
"""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
WORK=Path(__file__).resolve().parents[2];REPO=WORK.parent
sys.path[:0]=[str(WORK),str(WORK/'scripts/probes'),str(WORK/'scripts/analysis')]
import ue_joint_encoding_fast_analysis as fast
from ue_compact_joint_paired_20261024 import point_metrics
ROOT=Path('/data/wangcheng/checkpoint/ue_g_ubr_native_short_20261027')
ARMS=('A_nohead','B_zero','C_correct')
CONTRASTS={'C-A':('A_nohead','C_correct'),'C-B':('B_zero','C_correct'),'B-A':('A_nohead','B_zero')}
METRICS=[('MaskAP','segm',None),('BBoxAP','bbox',None),('BoundaryAP_2pct','segm',.02),('BoundaryAP_1pct','segm',.01)]


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');tmp.replace(path)


def audit_root(root):
 study=read(root/'study_contract.json');shared=study['shared'];sources=shared['source_hashes'];commit=study['git_commit']
 assert len(sources)==98,'This study is the frozen98-source native run'
 evidence={str(root/'study_contract.json'):sha(root/'study_contract.json')}
 for name,expected in sources.items():
  assert sha(REPO/name)==expected,('current source drift',name)
  assert sha(root/'frozen_source'/name)==expected,('frozen source drift',name)
  blob=subprocess.check_output(['git','show',commit+':'+name],cwd=REPO)
  assert hashlib.sha256(blob).hexdigest()==expected,('commit source mismatch',name)
 for path,expected in shared['source_order_sha256'].items():assert sha(path)==expected
 return shared,sources,commit,evidence


def load_arm(root,arm,view,shared,commit):
 folder=root/arm;contract=read(folder/'contract.json');done=read(folder/'training_complete.json');evaldir=root/'evaluation/test'/arm;complete=read(evaldir/'complete.json')
 assert done['complete'] and done['epoch']==10 and done['updates']==2670 and done['world_size']==4 and done['arm']==arm
 assert done['contract']==contract and contract['git_commit']==commit and contract['arm']==arm and contract['epochs']==10
 assert {k:v for k,v in contract.items() if k not in ('arm','allocator_cap_mib','native_evidence_port','git_commit')}==shared
 assert contract['initialization']['task_weights_loaded'] is False and contract['initialization']['main_mask_size']==[512,512]
 assert complete['complete'] and complete['split']=='test' and complete['images']==459 and complete['world_size']==4 and complete['precision']=='FP32' and complete['arm']==arm
 assert complete['checkpoint_sha256']==done['checkpoint_sha256']==sha(folder/'epoch10.pth')
 path=evaldir/'records.json';records=read(path)
 assert len(records)==len({r['image_id'] for r in records})==459
 if view=='original_path':records=[dict(r,dt=r['original_path']) for r in records]
 evidence={str(p):sha(p) for p in (folder/'contract.json',folder/'training_complete.json',evaldir/'complete.json',path)}
 evidence[str(folder/'epoch10.pth')]=done['checkpoint_sha256']
 orders=[]
 for epoch in range(1,11):
  path=folder/('epoch%d.json'%epoch);row=read(path)
  assert row['epoch']==epoch and row['step']==epoch*267 and row['world_size']==4
  assert len(row['sample_order'])==len(set(row['sample_order']))==2136
  orders.append(row['sample_order']);evidence[str(path)]=sha(path)
 return records,orders,evidence


def validate_pair(baseline,candidate):
 assert len(baseline)==len(candidate)==459
 assert [r['image_id'] for r in baseline]==[r['image_id'] for r in candidate]
 assert len({r['image_id'] for r in baseline})==459
 assert all(a['gt']==b['gt'] for a,b in zip(baseline,candidate)),'GT or image correspondence changed'


def run(args):
 shared,sources,commit,evidence=audit_root(args.root)
 data={};orders=None
 if args.baseline_records or args.candidate_records:
  assert args.baseline_records and args.candidate_records and args.label and args.output
  data={'baseline':read(args.baseline_records),'candidate':read(args.candidate_records)};validate_pair(data['baseline'],data['candidate'])
  contrasts={args.label:('baseline','candidate')};outputs={args.label:args.output}
  for path in (args.baseline_records,args.candidate_records):evidence[str(path.resolve())]=sha(path)
  # Validate E10 identity even for external intervention records; record metadata
  # and checkpoints remain a bound reference, not an implicit input assertion.
  for arm in ARMS:
   _,_,extra=load_arm(args.root,arm,'combined',shared,commit);evidence.update(extra)
  mode='explicit_pair; intervention semantics supplied by parent metadata, not inferred from records'
 else:
  contrasts=CONTRASTS if args.comparison=='all' else {args.comparison:CONTRASTS[args.comparison]}
  for arm in sorted({a for pair in contrasts.values() for a in pair}):
   rows,order,extra=load_arm(args.root,arm,args.view,shared,commit);data[arm]=rows;evidence.update(extra)
   if orders is None:orders=order
   else:assert order==orders,'Matched training sample order differs'
  for rows in data.values():validate_pair(next(iter(data.values())),rows)
  out=args.root/'analysis_native_20261028'
  outputs={label:out/('paired_'+label.replace('-','_minus_')+'_'+args.view+'.json') for label in contrasts}
  mode='three matched native architecture arms; G remains enabled in ALL arms'
 for path in outputs.values():assert not path.exists(),'Preserve completed analysis: '+str(path)
 analysis_sources={str(p.resolve()):sha(p) for p in (Path(__file__),Path(fast.__file__),Path(fast.original.__file__),WORK/'scripts/analysis/ue_compact_joint_paired_20261024.py',WORK/'utils/coco_eval_utils.py')}
 reports={label:dict(complete=False,label=label,baseline_arm=pair[0],candidate_arm=pair[1],view=args.view,
  images=459,split='test',precision='FP32',max_dets=250,selection='fixed E10 after2670 updates; no test selection',
  bootstrap_samples=500,bootstrap_seed=20261021,git_commit=commit,frozen_training_source_count=len(sources),
  input_sha256=evidence,analysis_source_sha256=analysis_sources,comparison_scope=mode,metrics={},
  limitations='single training seed, development-exposed v1 test; nominal image bootstrap does not cover training variance, geographic dependence or multiple comparisons; B_zero retains upstream true G/RGB',
  metric_definition='COCO all-area AP IoU .50:.95 step.05 maxDets250; BoundaryAP uses min(maskIoU,boundaryIoU), boundary radius max(1,round(ratio*image diagonal)); 512square radii2%=14px,1%=7px') for label,pair in contrasts.items()}
 samples=np.random.default_rng(20261021).integers(0,459,(500,459));fast.audits.clear();started=time.time()
 for name,kind,ratio in METRICS:
  points={};boot={};audit_start=len(fast.audits)
  for arm,rows in data.items():
   print('MATCH',args.view,name,arm,flush=True);ev=fast.original.evaluator(rows,kind=kind,ratio=ratio);point=point_metrics(ev)
   identity=fast.fast_sample_ap(ev,np.arange(459));assert abs(identity-point['AP'])<1e-12
   draws=np.empty(500,np.float64)
   for index,indices in enumerate(samples):
    draws[index]=fast.fast_sample_ap(ev,indices)
    if (index+1)%100==0:print('BOOTSTRAP',args.view,name,arm,index+1,flush=True)
   points[arm]=point;boot[arm]=draws;del ev
  for label,(baseline,candidate) in contrasts.items():
   delta=points[candidate]['AP']-points[baseline]['AP'];draws=boot[candidate]-boot[baseline]
   reports[label]['metrics'][name]=dict(baseline=points[baseline],candidate=points[candidate],delta=delta,
    paired_image_bootstrap95=np.quantile(draws,[.025,.975]).tolist(),bootstrap_samples=500,seed=20261021)
   reports[label]['accumulation_reference_checks']=list(fast.audits)
   reports[label]['elapsed_seconds']=time.time()-started
   write(outputs[label].with_suffix('.partial.json'),reports[label])
   print('RESULT',args.view,label,name,json.dumps(reports[label]['metrics'][name]),flush=True)
  assert len(fast.audits)-audit_start==3*len(data)
 assert all(r['max_abs_error']<1e-12 for r in fast.audits)
 for path,expected in evidence.items():assert sha(path)==expected,('Input changed during analysis',path)
 for path,expected in analysis_sources.items():assert sha(path)==expected,('Analysis source changed',path)
 for label,report in reports.items():
  mask=report['metrics']['MaskAP'];report['signal']=dict(positive_interval=mask['paired_image_bootstrap95'][0]>0,
   reference_gate=mask['delta']>=.002 and mask['paired_image_bootstrap95'][0]>0,gate_scope='numeric .002 reference; C-B tail evidence content, C-A combined architecture, not pure-satellite net gain')
  report['complete']=True;write(outputs[label],report)


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--view',choices=['combined','original_path'],default='combined')
 p.add_argument('--comparison',choices=['all',*CONTRASTS],default='all');p.add_argument('--baseline-records',type=Path);p.add_argument('--candidate-records',type=Path);p.add_argument('--label');p.add_argument('--output',type=Path)
 args=p.parse_args();run(args)

if __name__=='__main__':main()
