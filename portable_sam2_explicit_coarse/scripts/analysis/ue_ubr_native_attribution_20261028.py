"""Fixed E10 C tail interventions and cross-model C-box mask attribution. No training.

Donor replaces only post-G pre-fusion aligned RGB/support and native geometry
bundle. Target G features, query, detections, instance outputs and weights stay
unchanged. Donor bundle includes donor support; this is a deliberately mismatched
scene diagnostic, not a physical counterfactual or pure RGB necessity test.
"""
import argparse
from copy import deepcopy
from pathlib import Path
import hashlib,json,os,sys,time
import numpy as np
import torch
import torch.distributed as dist
from datetime import timedelta
WORK=Path(__file__).resolve().parents[2];REPO=WORK.parent
sys.path[:0]=[str(WORK),str(WORK/'scripts/probes')]
import ue_ubr_native_run_20261027 as run


def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()


def load_json(p):return json.loads(Path(p).read_text())


def finish(rt,rows,folder,cpu,receipt):
 gathered=[None]*4 if dist.get_rank()==0 else None
 dist.gather_object(rows,gathered,dst=0,group=cpu)
 if dist.get_rank()==0:
  records=sorted([r for part in gathered for r in part],key=lambda r:r['index'])
  assert [r.pop('index') for r in records]==list(range(459))
  C=run.engine.core.C;gt=[];pred=[]
  for r in records:
   g=deepcopy(r['gt']);g['bboxes']=np.asarray(g['bboxes'],np.float32).reshape(-1,4);g['labels']=np.asarray(g['labels'],np.int64);gt.append(g)
   pred.append(run.engine.core.consumer.arrays(r['dt']))
  cg,cd=C.build_coco_gt_and_dt(gt,pred,[dict(img_shape=(512,512))]*459,score_key='scores');metrics={}
  for kind in ('segm','bbox'):metrics.update(C.run_coco_eval(cg,cd,kind,max_dets=[1,10,250]))
  C.dump(folder/'records.json',records);C.dump(folder/'metrics.json',metrics)
  C.dump(folder/'complete.json',dict(receipt,complete=True,images=459,world_size=4,precision='FP32',records_sha256=sha(folder/'records.json')))
 dist.barrier()


def render(rt,features,meta,segments,reference):
 head=rt.satellite.roi_head;xx=head._inject_extra_pe(features[0]);rles=[]
 for segment in segments:
  boxes=torch.tensor(segment['boxes'],device='cuda',dtype=torch.float32).reshape(-1,4)
  scores=torch.tensor(segment['scores'],device='cuda',dtype=torch.float32)
  dt=run.engine.core.consumer.masks(head,xx,features,meta,boxes,scores)
  assert np.array_equal(np.asarray(dt['bboxes']),np.asarray(segment['boxes'],np.float32).reshape(-1,4))
  rles+=run.engine.core.C.serialized(dt)['rles']
 assert len(rles)==len(reference['bboxes'])
 result=deepcopy(reference);result['rles']=rles;result['mask_count']=len(rles)
 from pycocotools import mask as mu
 result['mask_fill_sum']=float(np.sum(mu.area(rles))/512**2) if rles else 0.
 return result


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('/data/wangcheng/checkpoint/ue_g_ubr_native_short_20261027'))
 p.add_argument('--action',choices=['interventions','fixed_boxes'],required=True);p.add_argument('--arm',choices=['A_nohead','B_zero','C_correct'],default='C_correct')
 p.add_argument('--output',type=Path,required=True);args=p.parse_args()
 assert int(os.environ.get('WORLD_SIZE',0))==4
 rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);torch.set_num_threads(2)
 dist.init_process_group('nccl',timeout=timedelta(minutes=90));cpu=dist.new_group(backend='gloo',timeout=timedelta(minutes=90))
 try:
  if rank==0:args.output.mkdir(parents=True,exist_ok=False)
  dist.barrier();run.configure();arm='C_correct' if args.action=='interventions' else args.arm
  contract=load_json(args.root/arm/'contract.json');path=args.root/arm/'epoch10.pth';done=load_json(args.root/arm/'training_complete.json')
  assert done['complete'] and done['epoch']==10 and done['updates']==2670 and sha(path)==done['checkpoint_sha256']
  # Source fidelity to trained model; new driver is an explicitly separate observer.
  for name,digest in contract['source_hashes'].items():assert sha(REPO/name)==digest,name
  rt=run.runtime(arm,False);run.engine.restore(rt,path,contract,False);rt.bare.eval();before=run.engine.core.C.state_digest(rt.bare)
  ds=run.engine.core.C.MinimalSatelliteDataset(run.engine.core.previous.DATA,'test');ids=[int(s['image_id']) for s in ds.samples];assert len(ids)==459
  official=load_json(args.root/'evaluation/test/C_correct/records.json');assert [int(r['image_id']) for r in official]==ids
  donor_map={ident:ids[(i+459//2)%459] for i,ident in enumerate(ids)};index_by_id={v:i for i,v in enumerate(ids)}
  modes=('correct','zero','donor') if args.action=='interventions' else (arm,)
  rows={m:[] for m in modes};segments_rows=[];started=time.time()
  csegments=None
  if args.action=='fixed_boxes':
   csegments=load_json(args.root/'analysis_native_20261028/interventions/segments.json');assert [r['image_id'] for r in csegments]==ids
  with torch.inference_mode():
   for index in range(rank,459,4):
    ident=ids[index]
    if args.action=='interventions':
     di=index_by_id[donor_map[ident]];db=run.engine.core.C.rtmdet_collate_fn([ds[di]]);sample_d=run.engine.core.C.processed(rt.satellite,db)
     rt.joint.prepare_uav(**rt.payload(donor_map[ident]));fd=rt.satellite.extract_feat(sample_d['inputs']);ad=rt.joint.uav
     donor=(ad.native_aligned_rgb.clone(),ad.native_rgb_valid.clone(),ad.native_geometry.clone());del fd,sample_d;rt.clear()
    batch=run.engine.core.C.rtmdet_collate_fn([ds[index]]);sample=run.engine.core.C.processed(rt.satellite,batch)
    payload=rt.payload(ident);rt.joint.prepare_uav(**payload);features=rt.satellite.extract_feat(sample['inputs']);meta=sample['data_samples'][0].metainfo
    gt=run.engine.core.C.serialized(run.engine.core.C.gt_arrays(sample['data_samples'][0].gt_instances));assert gt==official[index]['gt']
    if args.action=='interventions':
     adapter=rt.joint.uav;real=(adapter.native_aligned_rgb,adapter.native_rgb_valid,adapter.native_geometry)
     consumer=run.engine.core.consumer;original=consumer.masks;segments=[]
     def capture(head,xx,f,met,boxes,scores):
      segments.append(dict(boxes=boxes.detach().cpu().tolist(),scores=scores.detach().cpu().tolist()))
      return original(head,xx,f,met,boxes,scores)
     consumer.masks=capture
     try:prefix,combined,accepted=run.engine.core.inference_outputs(rt.satellite,adapter,sample,features,payload['geometry'])
     finally:consumer.masks=original
     correct=run.engine.core.C.serialized(combined)
     for key in ('bboxes','scores','labels','rles'):assert correct[key]==official[index]['dt'][key],(index,key,'correct reference mismatch')
     assert len(segments)==3
     # Same original consumer segmentation/order, not a regrouped RoI batch.
     rec=dict(index=index,image_id=ident,gt=gt,dt=correct);rows['correct'].append(rec)
     segments_rows.append(dict(index=index,image_id=ident,segments=segments))
     for mode in ('zero','donor'):
      rt.shim.zero_evidence=mode=='zero'
      if mode=='donor':adapter.native_aligned_rgb,adapter.native_rgb_valid,adapter.native_geometry=donor
      dt=render(rt,features,meta,segments,correct)
      for key in ('bboxes','scores','labels'):assert dt[key]==correct[key]
      rows[mode].append(dict(index=index,image_id=ident,gt=gt,dt=dt))
     adapter.native_aligned_rgb,adapter.native_rgb_valid,adapter.native_geometry=real;rt.shim.zero_evidence=False
     del donor,real
    else:
     segments=csegments[index]['segments'];dt=render(rt,features,meta,segments,official[index]['dt'])
     if arm=='C_correct':assert dt['rles']==official[index]['dt']['rles']
     rows[arm].append(dict(index=index,image_id=ident,gt=gt,dt=dt))
    rt.clear();del features
    if (index//4+1)%10==0:
     run.engine.core.C.dump(args.output/('rank%d_progress.json'%rank),dict(images=index//4+1,total_rank=len(range(rank,459,4)),elapsed=time.time()-started))
  after=run.engine.core.C.state_digest(rt.bare);assert before==after and sha(path)==done['checkpoint_sha256']
  receipt=dict(action=args.action,arm=arm,checkpoint_sha256=done['checkpoint_sha256'],model_state_unchanged=True,split='test',selection='fixedE10_no_selection',
    driver_sha256=sha(__file__),training_source_hashes=contract['source_hashes'],reference_records_sha256=sha(args.root/'evaluation/test/C_correct/records.json'),
    fixed='same C boxes/scores/labels/order and original source decoding chunks; own-model satellite/G features for cross-model fixed boxes',
    donor='cyclic donor (index+229)%459; donor pre-fusion aligned RGB/support + geometry5, conditioned on donor satellite query; target upstream G/detections unchanged; joint content/support mismatch, not RGB-only',
    donor_map=donor_map if args.action=='interventions' else None)
  if args.action=='interventions':
   gathered=[None]*4 if rank==0 else None;dist.gather_object(segments_rows,gathered,dst=0,group=cpu)
   if rank==0:
    ss=sorted([r for rr in gathered for r in rr],key=lambda r:r['index']);assert [r.pop('index') for r in ss]==list(range(459));run.engine.core.C.dump(args.output/'segments.json',ss)
  for mode in modes:finish(rt,rows[mode],args.output/mode,cpu,dict(receipt,mode=mode))
  if rank==0:run.engine.core.C.dump(args.output/'complete.json',dict(receipt,complete=True,world_size=4,images=459,modes=list(modes)))
 finally:dist.destroy_process_group()


if __name__=='__main__':main()
