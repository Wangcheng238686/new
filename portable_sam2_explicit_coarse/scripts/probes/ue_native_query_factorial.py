"""Complete frozen-query/geometry 2x2 inference diagnostic, no training."""
import argparse,json,os,sys,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'scripts/probes'),str(ROOT)]

def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)

def evaluate(a):
 import ue_v1_adapter as c
 import numpy as np
 import torch
 from rsprompter.uav.native_geometry_probe import NativeGeometryAdapter,prepare_native
 a.base=Path('/data/wangcheng/checkpoint/ue_minimal_single_20261021_EXP-0017/checkpoints/best_model_epoch73.pth')
 a.adapter=Path('/data/wangcheng/checkpoint/ue_v1_adapter_probe_20261021/native_geometry/last.pth')
 a.data_root=Path('/data/wangcheng/dataset/CitySample_SatUAV_TrainMinimal_v1_20261021');a.cache=Path('/data/wangcheng/cache/ue_v1_native_geometry')
 a.source_mode='fine';a.route='highres';a.seed=44;a.max_images=0;a.appearance='correct';a.eval_split='test'
 c.seed(44);model,old,snap=c.make_model(a)
 for h in old.handles:h.remove()
 del old;torch.cuda.empty_cache()
 adapter=NativeGeometryAdapter().cuda().eval();adapter.attach(model)
 cp=torch.load(a.adapter,map_location='cpu',weights_only=False);assert cp['contract']['base_sha256']==c.file_sha(a.base)
 adapter.load_state_dict(cp['adapter'],strict=True);del cp
 # Remove all 32 query output channels, including the query projection bias.
 hook=adapter.query.register_forward_hook(lambda module,args,output:torch.zeros_like(output))
 before=c.state_digest(model)
 def payload(folder,cache,appearance='correct',donor=None):
  with np.load(prepare_native(folder,cache)) as d:
   support=torch.from_numpy(d['support'].astype(np.float32))[None,None].cuda()
   fields=torch.from_numpy(d['fields'].astype(np.float32))[None].cuda()
  if a.fields=='zero':fields.zero_()
  return None,support,fields
 c.payload=payload
 c.evaluate(a,model,adapter,c.MinimalSatelliteDataset(a.data_root,'test'),'correct',a.out)
 assert c.state_digest(model)==before
 old=json.loads(Path('/data/wangcheng/checkpoint/ue_native_spatial_diagnostic_20261022/correct/records.json').read_text())
 rows=json.loads((a.out/'records.json').read_text());assert len(rows)==459
 for b,d in zip(old,rows):
  assert b['image_id']==d['image_id'] and b['gt']==d['gt']
  for k in ['scores','bboxes','labels']:assert b['dt'][k]==d['dt'][k]
 c.dump(a.out/'factorial_identity.json',dict(query_channels='zero including query projection bias',fields=a.fields,
  shared_support=True,mask_mixer_bias_retained=True,detector_gt_exact=True,satellite_state_unchanged=True,
  checkpoint_sha256=c.file_sha(a.adapter),runner_sha256=c.file_sha(__file__),images=459))
 print('FACTORIAL_COMPLETE',a.fields,flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['run','eval']);p.add_argument('--out',type=Path,required=True);p.add_argument('--fields',choices=['correct','zero'],default='correct');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 if a.action=='eval':evaluate(a);return
 import hashlib
 assert not (a.out/'plan.json').exists()
 write(a.out/'plan.json',dict(scope='two additional fixed-weight cells, query-zero geometry-on/off; no training or model selection',
  registered_utc=time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime()),
  runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),gpus=[0,2],
  source_checkpoint='/data/wangcheng/checkpoint/ue_v1_adapter_probe_20261021/native_geometry/last.pth',
  existing_cells='/data/wangcheng/checkpoint/ue_native_spatial_diagnostic_20261022/{correct,zero}',
  shared_support=True,limitations='query-zero is out-of-distribution; bias-only still uses UAV support; train capacity causality not established'))
 (a.out/'runner.source.py').write_text(Path(__file__).read_text());jobs=[]
 for gpu,field in [(0,'correct'),(2,'zero')]:
  log=(a.out/f'{field}.log').open('a');env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2')
  p=subprocess.Popen([sys.executable,__file__,'eval','--fields',field,'--out',str(a.out/field)],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close();jobs.append((field,p))
 write(a.out/'state.json',dict(stage='running',owner_pid=os.getpid(),jobs=[dict(fields=f,pid=p.pid) for f,p in jobs]))
 for field,p in jobs:
  assert p.wait()==0,field
 write(a.out/'state.json',dict(stage='inference_complete',owner_pid=os.getpid()))
 # Only the two decisive Mask AP comparisons; original spatial job reports all Boundary AP.
 import ue_v1_adapter_analysis as an
 import numpy as np
 import contextlib,io
 comparisons=[('geometry_only_vs_bias_only',a.out/'zero/records.json',a.out/'correct/records.json'),
  ('satellite_query_vs_bias_only',a.out/'zero/records.json',Path('/data/wangcheng/checkpoint/ue_native_spatial_diagnostic_20261022/zero/records.json'))]
 result={}
 samples=np.random.default_rng(20261021).integers(0,459,(500,459))
 for name,b,c in comparisons:
  br=json.loads(b.read_text());cr=json.loads(c.read_text())
  for x,y in zip(br,cr):
   assert x['image_id']==y['image_id'] and x['gt']==y['gt']
   for k in ['scores','labels','bboxes']:assert x['dt'][k]==y['dt'][k]
  eb=an.evaluator(br);ec=an.evaluator(cr);vb=an.precision_ap(eb);vc=an.precision_ap(ec)
  assert abs(an.sample_ap(eb,np.arange(459))-vb)<1e-12 and abs(an.sample_ap(ec,np.arange(459))-vc)<1e-12
  vals=[an.sample_ap(ec,s)-an.sample_ap(eb,s) for s in samples]
  result[name]=dict(baseline=vb,candidate=vc,delta=vc-vb,paired_image_bootstrap95=np.quantile(vals,[.025,.975]).tolist(),images=459,bootstrap_samples=500,seed=20261021)
  write(a.out/'paired_mask_analysis.json',result);print(name,result[name],flush=True)
 write(a.out/'state.json',dict(stage='complete',owner_pid=os.getpid()))

if __name__=='__main__':main()
