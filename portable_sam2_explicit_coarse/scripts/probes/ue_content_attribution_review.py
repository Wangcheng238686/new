"""Immediate attribution verdict for already complete content-paired S2DB arms.

Reuses their fixed, identical 500 image resamples. No new training, selection,
changed prediction scores or mathematical estimator.
"""
import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/probes'))
import numpy as np
from ue_s2db_collect import audit_training
from ue_s2db_analysis import assert_paired
from ue_v1_adapter import file_sha,dump
R=Path('/data/wangcheng/checkpoint/ue_s2db_probe_20261022')
OUT=Path('/data/wangcheng/checkpoint/ue_content_attribution_review_20261023')
METRICS=['MaskAP','BoundaryAP_2pct','BoundaryAP_1pct']


def main():
 OUT.mkdir(parents=True,exist_ok=True);manifest=json.loads((R/'manifest.json').read_text())
 names=['A','B','C','D','A_constant','B_constant','C_constant','D_constant','C_zero']
 arms={a['name']:a for a in manifest['arms']};audits={};reports={};vectors={};records={};interventions={}
 for n in ['baseline']+names:
  mode='disabled' if n=='baseline' else 'constant' if n.endswith('_constant') else 'correct'
  folder=R/'test'/n/mode;path=folder/'records.json';identity=json.loads((folder/'evaluation_identity.json').read_text())
  assert identity['images']==459 and identity['split']=='test' and identity['mode']==mode
  assert identity['base_sha256']==manifest['base_sha256'] and identity['score_key']=='scores' and identity['max_dets']==[1,10,250] and identity['inference_precision']=='fp32'
  if n!='baseline':
   audits[n]=audit_training(R,arms[n],manifest)
   assert identity['adapter_sha256']==audits[n]['checkpoint_sha256']
  rows=json.loads(path.read_text());assert len(rows)==459
  if records:assert_paired(records['baseline'],rows)
  report=json.loads((R/'analysis'/n/'metrics.json').read_text())
  assert report['records_sha256']==file_sha(path) and report['images']==459 and report['bootstrap_samples']==500 and report['seed']==20261021
  assert abs(report['metrics']['MaskAP']['AP']-json.loads((folder/'metrics.json').read_text())['segm/mAP'])<1e-12
  arrays=dict(np.load(R/'analysis'/n/'bootstrap.npz'))
  assert set(arrays)==set(METRICS) and all(v.shape==(500,) and np.isfinite(v).all() for v in arrays.values())
  reports[n]=report;vectors[n]=arrays;records[n]=rows
  if n in ['A','B','C','D']:
   interventions[n]={}
   for alt in ['constant','donor']:
    f=R/'test'/n/alt;rr=json.loads((f/'records.json').read_text());assert_paired(rows,rr)
    ii=json.loads((f/'evaluation_identity.json').read_text());assert ii['adapter_sha256']==audits[n]['checkpoint_sha256'] and ii['mode']==alt
    interventions[n][alt]=json.loads((f/'metrics.json').read_text())['segm/mAP']
  print('AUDITED',n,flush=True)
 assert len({tuple(a['sample_order_epoch_hashes']) for a in audits.values()})==1
 # Explicitly verify the fixed math which generated all vectors and sample order.
 for source in ['scripts/probes/ue_s2db_analysis.py','scripts/probes/ue_v1_adapter_analysis.py']:
  assert file_sha(ROOT/source)==manifest['source_hashes'][source]==file_sha(R/'frozen_source'/source)
 comparisons={}
 pairs=[(n,n+'_constant','RGB_content_'+n) for n in 'ABCD']+[(n,'baseline','net_'+n) for n in 'ABCD']+[
  ('C','C_zero','aggregated_content_C'),('C_constant','C_zero','constantRGB_aggregated_content_C')]
 for cand,base,key in pairs:
  comparisons[key]={}
  for metric in METRICS:
   point=reports[cand]['metrics'][metric]['AP']-reports[base]['metrics'][metric]['AP']
   ci=np.quantile(vectors[cand][metric]-vectors[base][metric],[.025,.975]).tolist()
   comparisons[key][metric]=dict(candidate=cand,baseline=base,delta=point,paired_image_bootstrap95=ci,bootstrap_samples=500,seed=20261021)
 assert all(comparisons['RGB_content_'+n]['MaskAP']['paired_image_bootstrap95'][0]<=0 for n in 'ABCD')
 result=dict(status='audited_completed_pairs',scope='A/B/C/D current E10 fixed v1 development-test predictions; excludes unfinished auxiliary/Dzero/other cohorts',
  training_audits=audits,reports=reports,comparisons=comparisons,fixed_checkpoint_interventions=interventions,
  verdict='All four tested routes fail to establish correct UAV RGB contribution to overall Mask AP. Constant-RGB controls reproduce the C/D model net improvement. Strict Boundary1% has partial A/B content evidence. Geometry/support versus capacity is not identified by RGB constant controls.',
  caveats=['not proof of exactly zero RGB effect','constant RGB retains target geometry and support',
   'Czero comparison removes aggregate nonzero features, changing effective capacity and training; not isolated correct-geometry causal proof',
   'single training seed, spatial dependence, nominal intervals/multiple exploratory comparisons; v1 test is development data'],
  runner_sha256=file_sha(__file__),source_manifest_sha256=file_sha(R/'manifest.json'))
 dump(OUT/'attribution_audit.json',result)
 print('ATTRIBUTION_AUDIT_COMPLETE',flush=True)

if __name__=='__main__':main()
