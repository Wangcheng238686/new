"""Prioritize the four predeclared geometry content contrasts, Mask AP only.

No extra GPU process or changed estimator. Full Boundary analyses remain with
the original cohort owner. Fixed pair order, no result-dependent scheduling.
"""
import json,sys,time,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/probes'))
import numpy as np
import torch
from ue_geometry_mechanism_collect import audit_training,paired
from ue_v1_adapter_analysis import evaluator,precision_ap,sample_ap
from ue_geometry_mechanism_schedule import read,sha,dump
R=Path('/data/wangcheng/checkpoint/ue_geometry_mechanism_20261022')
OUT=Path('/data/wangcheng/checkpoint/ue_geometry_content_priority_20261023')


def main():
 OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
 manifest=read(R/'manifest.json');init=torch.load(R/'common_init.pth',map_location='cpu',weights_only=False)
 names=[s+'_'+f for s in ['bilinear','nearest'] for f in ['concat','difference']]
 plan=dict(pairs=names,primary='MaskAP',bootstrap=500,seed=20261021,selection='existing fixed E10 last',no_new_training=True,manifest_sha256=sha(R/'manifest.json'),runner_sha256=sha(__file__))
 if (OUT/'plan.json').exists():assert read(OUT/'plan.json')==plan
 else:dump(OUT/'plan.json',plan)
 done=[]
 while len(done)<4:
  for name in names:
   if name in done:continue
   arms=[next(a for a in manifest['arms'] if a['name']==name+'_'+c) for c in ['geometry','constant']]
   paths=[R/'test'/arms[i]['name']/mode for i,mode in enumerate(['correct','constant'])]
   if not all((p/'evaluation_identity.json').exists() for p in paths):continue
   audits=[audit_training(R,a,manifest,init) for a in arms]
   assert audits[0]['sample_order_epoch_hashes']==audits[1]['sample_order_epoch_hashes']
   records=[read(p/'records.json') for p in paths];paired(*records)
   for p,arm,audit in zip(paths,arms,audits):
    ident=read(p/'evaluation_identity.json')
    assert ident['split']=='test' and ident['images']==459
    if 'training_content' in ident:assert ident['training_content']==arm['content']
    assert ident['adapter_sha256']==audit['checkpoint_sha256'] and ident['base_sha256']==manifest['base_sha256']
    assert ident['inference_precision']=='fp32' and ident['score_key']=='scores' and ident['max_dets']==[1,10,250]
   candidate=evaluator(records[0]);baseline=evaluator(records[1]);vc=precision_ap(candidate);vb=precision_ap(baseline)
   assert abs(sample_ap(candidate,np.arange(459))-vc)<1e-12 and abs(sample_ap(baseline,np.arange(459))-vb)<1e-12
   for p,v in zip(paths,[vc,vb]):assert abs(read(p/'metrics.json')['segm/mAP']-v)<1e-12
   draws=np.random.default_rng(20261021).integers(0,459,(500,459))
   delta=np.array([sample_ap(candidate,ix)-sample_ap(baseline,ix) for ix in draws])
   ci=np.quantile(delta,[.025,.975]).tolist()
   dump(OUT/(name+'.json'),dict(images=459,correct=vc,constant=vb,delta=vc-vb,paired_image_bootstrap95=ci,bootstrap=500,seed=20261021,
    passes_positive_content_evidence=vc>vb and ci[0]>0,training_audits=audits,record_sha256=[sha(p/'records.json') for p in paths],
    caveats='single seed; nominal image interval; constant retains UAV support; not full cohort audit'))
   done.append(name);print('CONTENT_COMPLETE',name,100*(vc-vb),[100*x for x in ci],flush=True)
  dump(OUT/'state.json',dict(stage='complete' if len(done)==4 else 'running',owner_pid=os.getpid(),completed=done,pending=[n for n in names if n not in done]))
  if len(done)<4:time.sleep(15)

if __name__=='__main__':main()
