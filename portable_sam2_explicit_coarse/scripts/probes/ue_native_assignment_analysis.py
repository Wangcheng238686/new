"""Offline analysis of the predeclared native assignment sensitivity probe."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import ue_native_assignment_probe as probe
import numpy as np
from ue_v1_adapter_analysis import evaluator,precision_ap,sample_ap
from pycocotools import mask as mu
from scipy.optimize import linear_sum_assignment
import cv2
cv2.setNumThreads(1)

def describe_ev(ev):
 p=ev.eval['precision'];r=ev.eval['recall']
 def avg(x):return float(x[x>=0].mean()) if (x>=0).any() else None
 return dict(AP=precision_ap(ev),AP75=avg(p[5,:,:,0,-1]),AP_small=avg(p[:,:,:,1,-1]),
  AP_medium=avg(p[:,:,:,2,-1]),AP_large=avg(p[:,:,:,3,-1]),AR250=avg(r[:,:,0,-1]))


def metrics(records,out):
 if (out/'metrics.json').exists():return
 out.mkdir(parents=True,exist_ok=True)
 rows=probe.read(records);assert len(rows)==459
 e=evaluator(rows);point=describe_ev(e)
 assert abs(sample_ap(e,np.arange(459))-point['AP'])<1e-12
 samples=np.random.default_rng(20261021).integers(0,459,(500,459))
 values=[]
 for i,ix in enumerate(samples):
  values.append(sample_ap(e,ix))
  if (i+1)%100==0:print('BOOTSTRAP',out.name,i+1,flush=True)
 result=dict(records_sha256=probe.sha(records),images=459,bootstrap=500,seed=20261021,MaskAP=point,
  BoundaryAP_2pct=describe_ev(evaluator(rows,ratio=.02)),BoundaryAP_1pct=describe_ev(evaluator(rows,ratio=.01)),bbox=describe_ev(evaluator(rows,kind='bbox')))
 # Explicit GT-mask area supplement, keeping other historical evaluator rules.
 from pycocotools.cocoeval import COCOeval
 for ann in e.cocoGt.dataset['annotations']:ann['area']=float(mu.area(ann['segmentation']))
 corrected=COCOeval(e.cocoGt,e.cocoDt,'segm');corrected.params.maxDets=[1,10,250]
 with contextlib.redirect_stdout(io.StringIO()):corrected.evaluate();corrected.accumulate()
 result['GT_mask_area_grouping_supplement']=describe_ev(corrected)
 assert abs(result['GT_mask_area_grouping_supplement']['AP']-point['AP'])<1e-12
 np.save(out/'MaskAP_bootstrap.npy',np.array(values))
 probe.dump(out/'metrics.json',result)


def delta8(root):
 import torch
 a=argparse.Namespace(out=root,mode='correct');c,model,adapter=probe.build(a)
 before=c.state_digest(model);points=probe.read('/data/wangcheng/checkpoint/ue_native_source_audit_20261022/manifest.json')['samples'];rows=[]
 for sample in points:
  ds=c.MinimalSatelliteDataset(probe.DATA,sample['split']);i=next(i for i,x in enumerate(ds.samples) if x['image_id']==sample['image_id'])
  p=c.processed(model,c.rtmdet_collate_fn([ds[i]]));folder=c.point_folder(ds,i,probe.DATA);fields,support=probe.load_fields(folder)
  raw=(None,torch.from_numpy(support.astype(np.float32))[None,None].cuda(),torch.from_numpy(fields)[None].cuda())
  capture=[]
  h=adapter.handles[-1];h.remove()
  def capture_hook(module,args,base):
   support_t,fields_t,high=adapter.context;q=adapter.query(high)
   original=torch.cat([adapter._chunk(q,fields_t,support_t,b,base.shape[-1]) for b in adapter.boxes.split(4)])
   data={}
   for mode in probe.MODES:
    changed,_=probe.intervene(fields,support,mode,sample['image_id']);f=torch.from_numpy(changed)[None].cuda()
    alt=original if mode=='correct' else torch.cat([adapter._chunk(q,f,support_t,b,base.shape[-1]) for b in adapter.boxes.split(4)])
    x,y=original.double().flatten(),alt.double().flatten();nx,ny=x.norm(),y.norm()
    data[mode]=dict(elements=x.numel(),reference_square_sum=float(x.square().sum()),altered_square_sum=float(y.square().sum()),
     difference_square_sum=float((x-y).square().sum()),dot=float(x@y),cosine=float(x@y/(nx*ny)) if nx>0 and ny>0 else None)
   capture.append(data);return base+adapter.alpha*original
  handle=model.roi_head.mask_head.mask_decoder.decoder.conv_s0.register_forward_hook(capture_hook)
  adapter.handles[-1]=handle
  with torch.inference_mode():
   features=c.bind(model,adapter,p,raw)
   with c.cached_extract(model,features):model.predict(p['inputs'],p['data_samples'],rescale=True)
  rows.append(dict(**sample,calls=capture));adapter.context=None
 assert c.state_digest(model)==before
 probe.dump(root/'delta8.json',dict(images=8,samples=rows,precision='fp32',satellite_state_sha256=before,
  scope='all mask calls in fixed8 trainval; no fitting or model selection',zero_norm_cosine=None))


def mechanism_records(rows,correct):
 totals={z:dict(pixels=0,corrected=0,introduced=0,reference_errors=0) for z in ('boundary','interior','background')}
 images=[];unmatched_gt=unmatched_dt=changed_pixels=changed_instances=0
 for ref,row in zip(correct,rows):
  assert row['image_id']==ref['image_id'] and row['gt']==ref['gt']
  assert all(row['dt'][k]==ref['dt'][k] for k in ('bboxes','scores','labels'))
  g=ref['gt'];d=ref['dt'];G=len(g['rles']);D=len(d['rles'])
  gb=np.array(g['bboxes'],float).reshape(-1,4);db=np.array(d['bboxes'],float).reshape(-1,4)
  lo=np.maximum(gb[:,None,:2],db[None,:,:2]);hi=np.minimum(gb[:,None,2:],db[None,:,2:]);inter=np.maximum(hi-lo,0).prod(-1)
  ga=np.maximum(gb[:,2:]-gb[:,:2],0).prod(-1);da=np.maximum(db[:,2:]-db[:,:2],0).prod(-1)
  iou=inter/(ga[:,None]+da[None,:]-inter).clip(1e-9)
  gi,di=linear_sum_assignment((iou>=.5)*1000+iou,maximize=True)
  pairs=[(int(x),int(y)) for x,y in zip(gi,di) if iou[x,y]>=.5];unmatched_gt+=G-len(pairs);unmatched_dt+=D-len(pairs)
  gm=np.moveaxis(mu.decode(g['rles']).astype(bool),2,0)
  rm=np.moveaxis(mu.decode(d['rles']).astype(bool),2,0) if D else np.zeros((0,512,512),bool)
  cm=np.moveaxis(mu.decode(row['dt']['rles']).astype(bool),2,0) if D else np.zeros_like(rm)
  change=np.count_nonzero(rm!=cm,axis=(1,2));changed_pixels+=int(change.sum());changed_instances+=int((change>0).sum())
  for x,y in pairs:
   mask=gm[x];er=cv2.erode(mask.astype(np.uint8),np.ones((3,3),np.uint8),iterations=2,borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)
   dil=cv2.dilate(mask.astype(np.uint8),np.ones((3,3),np.uint8),iterations=2,borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)
   low=np.maximum(np.floor(np.minimum(gb[x,:2],db[y,:2])).astype(int)-8,0);high=np.minimum(np.ceil(np.maximum(gb[x,2:],db[y,2:])).astype(int)+8,512)
   sl=np.s_[low[1]:high[1],low[0]:high[0]];old_error=(rm[y]!=mask)[sl];new_error=(cm[y]!=mask)[sl]
   for zone,z in [('boundary',dil&~er),('interior',er),('background',~dil)]:
    z=z[sl];v=totals[zone];v['pixels']+=int(z.sum());v['corrected']+=int((z&old_error&~new_error).sum())
    v['introduced']+=int((z&~old_error&new_error).sum());v['reference_errors']+=int((z&old_error).sum())
  with np.load(probe.CACHE/f"{row['image_id']:06d}.npz") as cache:support=cache['support']
  union=gm.any(0);coverage=float((union&support).sum()/max(int(union.sum()),1))
  images.append(dict(image_id=row['image_id'],coverage=coverage,coverage_group='low' if coverage<.5 else ('mid' if coverage<.8 else 'high'),
   density='le50' if G<=50 else ('51to100' if G<=100 else 'gt100'),matched=len(pairs),gt=G))
 # Descriptive image groups reaccumulate the same fixed all-area matches, without
 # replacing the primary all-image metric or claiming additive AP contributions.
 er=evaluator(correct);ec=evaluator(rows);groups={}
 def subset_ap(ev,indices):
  if not indices:return None
  old=ev.evalImgs;params=ev._paramsEval
  import copy
  n=459;ev.evalImgs=[old[j*n+i] for j in range(len(ev.params.areaRng)) for i in indices]
  ev._paramsEval=copy.deepcopy(params);ev._paramsEval.imgIds=list(range(len(indices)))
  p=copy.deepcopy(ev._paramsEval)
  with contextlib.redirect_stdout(io.StringIO()):ev.accumulate(p)
  value=precision_ap(ev);ev.evalImgs=old;ev._paramsEval=params;return value
 for column,labels in [('coverage_group',['low','mid','high']),('density',['le50','51to100','gt100'])]:
  for label in labels:
   ids=[i for i,r in enumerate(images) if r[column]==label]
   groups[column+'/'+label]=dict(images=len(ids),correct_AP=subset_ap(er,ids),altered_AP=subset_ap(ec,ids))
 return dict(zones=totals,matched_instances=sum(r['matched'] for r in images),unmatched_gt=unmatched_gt,unmatched_dt=unmatched_dt,
  matching='bbox IoU>=.5 max-cardinality then max-IoU Hungarian fixed from correct',
  zone_denominator='matched instance-pixel counts; overlaps can repeat; boundary is GT dilate minus erode 2px; ROI union bounds +8px',
  changed_instance_pixels=changed_pixels,changed_instances=changed_instances,images=images,groups=groups)


def report(root):
 plan=probe.read(root/'plan.json');assert all(probe.sha(probe.ROOT/p)==h for p,h in plan['source_hashes'].items())
 assert probe.read(root/'gpu_gate.json')['passed']
 correct=probe.read(root/'correct/records.json');assert len(correct)==459
 original=probe.read(probe.OLD/'test/native_geometry/correct/records.json')
 assert [r['dt'] for r in correct]==[r['dt'] for r in original]
 result={};vectors={}
 for mode in probe.MODES:
  out=root/mode;identity=probe.read(out/'diagnostic_identity.json');assert identity['mode']==mode and identity['images']==459
  assert identity['satellite_state_sha256']==probe.FROZEN and identity['satellite_state_unchanged'] and identity['detector_gt_exact']
  assert identity['base_sha256']==plan['base_sha256'] and identity['adapter_sha256']==plan['adapter_sha256']
  assert identity['source_hashes']==plan['source_hashes']
  for path,h in plan['source_hashes'].items():assert probe.sha(root/'source'/path)==h
  metrics(out/'records.json',root/'analysis'/mode)
  measured=probe.read(root/'analysis'/mode/'metrics.json');assert measured['records_sha256']==probe.sha(out/'records.json')
  result[mode]=dict(metrics=measured);vectors[mode]=np.load(root/'analysis'/mode/'MaskAP_bootstrap.npy');assert len(vectors[mode])==500
  telemetry=probe.read(out/'all_call_squares.json');rows=probe.read(out/'records.json')
  assert len(telemetry)==len(rows)==459
  for record,obs in zip(rows,telemetry):assert record['image_id']==obs['image_id'] and obs['rois']==len(record['dt']['scores'])
  count=sum(v['elements'] for v in telemetry)
  result[mode]['all_call_telemetry']=dict(elements=count,rois=sum(v['rois'] for v in telemetry),calls=sum(len(v['calls']) for v in telemetry),
   **{name:(sum(v[key] for v in telemetry)/count)**.5 for name,key in [('base_RMS','base_square_sum'),('residual_RMS','residual_square_sum'),('injected_RMS','injected_square_sum')]},
   image_rms_ratios=[v['injected_to_base_rms_ratio'] for v in telemetry])
  mapping=probe.read(out/'input_mapping_identity.json');assert len(mapping)==459
  reference_mapping=probe.read(root/'correct/input_mapping_identity.json')
  for row,ident,reference_ident in zip(rows,mapping,reference_mapping):
   assert row['image_id']==ident['image_id'] and ident['mode']==mode
   for key in ('support_sha256','original_fields_sha256','native_cache_sha256'):assert ident[key]==reference_ident[key]
   if mode.startswith('permute'):assert ident['joint_vector_multiset_exact'] and ident['seed']==probe.SEEDS[mode]
  path=root/'analysis'/mode/'mechanisms.json'
  if not path.exists():probe.dump(path,mechanism_records(rows,correct))
  result[mode]['mechanisms_path']=str(path)
 contrasts={}
 for name,others in [('correct-minus-mean',['mean_eval']),('correct-minus-permutation-average',['permute_s1','permute_s2','permute_s3'])]:
  point=result['correct']['metrics']['MaskAP']['AP']-np.mean([result[m]['metrics']['MaskAP']['AP'] for m in others])
  samples=vectors['correct']-np.mean([vectors[m] for m in others],axis=0)
  contrasts[name]=dict(delta=float(point),nominal95=np.quantile(samples,[.025,.975]).tolist(),bonferroni97_5=np.quantile(samples,[.0125,.9875]).tolist(),bootstrap=500,seed=20261021)
 zero_samples=vectors['correct']-vectors['zero_eval']
 extra=dict(correct_minus_zero_eval=dict(delta=result['correct']['metrics']['MaskAP']['AP']-result['zero_eval']['metrics']['MaskAP']['AP'],nominal95=np.quantile(zero_samples,[.025,.975]).tolist()))
 assert len(probe.read(root/'delta8.json')['samples'])==8
 complete=dict(passed=True,plan=plan,modes=result,primary_contrasts=contrasts,secondary=extra,
  interpretation='fixed model assignment sensitivity only; not unique evidence of correct geometry, exact alignment, or repeatable UAV gains',
  limitations=['OOD, frequency, query correlation and amplitude are not separately identified','500 resamples: about 6 tail samples per side for 97.5% CI',
   'single training checkpoint; image bootstrap does not measure training variance, spatial dependence, or all permutation randomness',
   'three permutations are not training repeats; secondary metrics are descriptive','v1 test previously used for development'])
 lines=['# Native空间赋值敏感性：已审核六条件诊断','',
  '固定native E10、459 test、FP32、相同GT/框/分数、完整卫星冻结审计；没有训练或RMS匹配。','',
  '|模式|Mask AP250|Boundary AP2%|Boundary AP1%|实际注入RMS|','|---|---:|---:|---:|---:|']
 for mode,v in result.items():
  m=v['metrics'];lines.append(f"|{mode}|{m['MaskAP']['AP']:.6f}|{m['BoundaryAP_2pct']['AP']:.6f}|{m['BoundaryAP_1pct']['AP']:.6f}|{v['all_call_telemetry']['injected_RMS']:.6f}|")
 lines+=['','## 两项Mask AP主对比','']
 for name,v in contrasts.items():lines.append(f"- {name}: Δ={v['delta']:+.6f}；95% {v['nominal95']}；Bonferroni 97.5% {v['bonferroni97_5']}。")
 lines+=['','correct比置换好只支持对空间赋值敏感；不能唯一分开正确几何、频谱、分布外、query相关性和幅度。',
  'mean只保留通道均值；不显著不等于等效。所有Boundary/AP75/尺寸/分组及纠错为描述性次要结果。',
  '500次97.5%区间每侧约6个尾部样本，端点粗糙；图像bootstrap不证明训练重复性。',
  '旧五条件空间诊断与本轮六条件的模式、种子及统计不同，产物独立，不混作同一预注册实验。',
  '', '完整逐模式指标、输入映射SHA、全调用平方和、固定bbox错误计数和固定8图残差方向见本目录。']
 (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
 probe.dump(root/'complete_audit_summary.json',complete)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['delta8','report']);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 (delta8 if a.action=='delta8' else report)(a.root)
