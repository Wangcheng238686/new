"""Final cross-artifact audit for the native geometry/query conclusion."""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
S=Path('/data/wangcheng/checkpoint/ue_native_spatial_diagnostic_20261022')
Q=Path('/data/wangcheng/checkpoint/ue_native_query_factorial_20261022')

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 assert read(S/'complete_audit_summary.json')['status']=='passed'
 assert read(S/'state.json')['stage']=='complete' and read(Q/'state.json')['stage']=='complete'
 sp=read(S/'plan.json');qp=read(Q/'plan.json')
 assert sha(ROOT/'scripts/probes/ue_native_query_factorial.py')==qp['runner_sha256']
 assert sha(qp['source_checkpoint'])==sp['adapter_sha256']
 cases={'query_geometry':S/'correct','query_only':S/'zero','geometry_only':Q/'correct','bias_support_only':Q/'zero'}
 reference=read(S/'correct/records.json');results={}
 for name,path in cases.items():
  rows=read(path/'records.json');metrics=read(path/'metrics.json');identity=read(path/'evaluation_identity.json')
  assert len(rows)==459 and identity['split']=='test' and identity['images']==459
  assert identity['inference_precision']=='fp32' and identity['max_dets']==[1,10,250] and identity['score_key']=='scores'
  assert identity['adapter_sha256']==sp['adapter_sha256'] and identity['base_sha256']==sp['base_sha256']
  assert metrics['bbox/mAP']==.6854287640342888
  count=0
  for x,y in zip(reference,rows):
   assert x['image_id']==y['image_id'] and x['gt']==y['gt']
   for k in ['scores','labels','bboxes']:assert x['dt'][k]==y['dt'][k]
   assert len(y['dt']['rles'])==len(y['dt']['scores']);count+=len(y['dt']['scores'])
  assert count==40932
  if path.parent==Q:
   i=read(path/'factorial_identity.json');assert i['runner_sha256']==qp['runner_sha256'] and i['satellite_state_unchanged']
   assert i['fields']==('correct' if name=='geometry_only' else 'zero')
  results[name]=dict(MaskAP=metrics['segm/mAP'],images=len(rows),detections=count,records_sha256=sha(path/'records.json'))
 paired=read(Q/'paired_mask_analysis.json')
 mapping={'geometry_only_vs_bias_only':('geometry_only','bias_support_only'),'satellite_query_vs_bias_only':('query_only','bias_support_only')}
 for name,(cand,base) in mapping.items():
  v=paired[name];assert v['images']==459 and v['bootstrap_samples']==500 and v['seed']==20261021
  assert abs(v['candidate']-results[cand]['MaskAP'])<1e-12 and abs(v['baseline']-results[base]['MaskAP'])<1e-12
  assert abs(v['delta']-(v['candidate']-v['baseline']))<1e-12
 result=dict(status='passed',scope='current fixed E10 function on v1 development test; not training-capacity causal proof',cases=results,paired=paired,
  adapter_sha256=sp['adapter_sha256'],base_sha256=sp['base_sha256'],query_runner_sha256=qp['runner_sha256'],spatial_audit_passed=True)
 (Q/'complete_conclusion_audit.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2))

if __name__=='__main__':main()
