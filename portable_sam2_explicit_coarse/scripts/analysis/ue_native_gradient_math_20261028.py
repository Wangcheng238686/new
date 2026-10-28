"""Pure gradient telemetry math. Fixed parameter coordinates; no loss reweighting."""
import math
import torch
DETECTION=('loss_rpn_cls','loss_rpn_bbox','loss_cls','loss_bbox')
MASK=('loss_mask','loss_shape_prior')
AUX=('loss_uav_instances',)
SHARED_GROUPS={'satellite_lora','satellite_encoder_other','fpn','uav_lora','uav_encoder_other','geometry_encoder','rgb_alignment_fusion','uav_interaction_writeback'}


def parameter_group(name):
 if name.startswith('satellite.backbone.') and 'lora_' in name:return 'satellite_lora'
 if name.startswith('satellite.backbone.'):return 'satellite_encoder_other'
 if name.startswith('satellite.neck.'):return 'fpn'
 if name.startswith(('satellite.rpn_head.','satellite.roi_head.bbox_head.')):return 'detection_heads'
 if '.native_refiner.' in name:return 'native_refiner'
 if '.mask_decoder.' in name:return 'mask_decoder'
 if name.startswith('satellite.'):return 'mask_prompt_other'
 if name.startswith('uav.rgb_encoder.'):return 'uav_lora' if 'lora_' in name else 'uav_encoder_other'
 if name.startswith(('uav.a.','uav.b.','uav.c.','uav.d.','uav.up.')):return 'geometry_encoder'
 if name.startswith(('uav.rgb_project.','uav.rgb_fusion.')):return 'rgb_alignment_fusion'
 if name.startswith(('uav.head_satellite.','uav.head_fusion.','uav.heat.','uav.reg.')):return 'instance_head'
 if name.startswith('uav.'):return 'uav_interaction_writeback'
 return 'unassigned'


def partition_losses(losses):
 def tensors(x):
  if torch.is_tensor(x):return [x]
  if isinstance(x,(list,tuple)):return [t for v in x for t in tensors(v)]
  raise TypeError('Unexpected loss value type')
 actual={k:sum(tensors(v)) for k,v in losses.items() if 'loss' in k}
 unknown=set(actual)-set(DETECTION+MASK+AUX)
 if unknown or not set(DETECTION+AUX+('loss_mask',)).issubset(actual):raise ValueError('Unregistered task losses: '+str(sorted(actual)))
 keys={'detection':[k for k in DETECTION if k in actual],'mask':[k for k in MASK if k in actual],'instance_auxiliary':[k for k in AUX if k in actual]}
 return {task:sum(actual[k] for k in ks) for task,ks in keys.items()},keys


def direction(a,b):
 a=a.double();b=b.double();aa=float(a.square().sum());bb=float(b.square().sum());dot=float((a*b).sum());na=math.sqrt(aa);nb=math.sqrt(bb)
 return dict(detection_norm=na,mask_norm=nb,dot=dot,cosine=dot/(na*nb) if na and nb else None,mask_over_detection_norm=nb/na if na else None,
             status='defined' if na and nb else 'undefined_zero_task_norm')


def summarize(named,vectors,support):
 """Vectors already global mean8; support[3,P] global graph connectivity flags."""
 layout=[];offset=0
 for i,(name,p) in enumerate(named):
  layout.append((i,name,offset,offset+p.numel(),parameter_group(name)));offset+=p.numel()
 assert vectors.shape==(3,offset) and support.shape==(3,len(named))
 groups=sorted({x[4] for x in layout});result={}
 for group in ['all_trainable']+groups+['shared_representation']:
  shared=SHARED_GROUPS
  selected=[x for x in layout if group=='all_trainable' or (group=='shared_representation' and x[4] in shared) or x[4]==group]
  indices=torch.cat([torch.arange(s,e,device=vectors.device) for _,_,s,e,_ in selected]) if selected else torch.empty(0,dtype=torch.long,device=vectors.device)
  v=vectors[:,indices];row=direction(v[0],v[1]);row['instance_auxiliary_norm']=float(v[2].double().norm())
  row.update(parameter_tensors=len(selected),parameter_elements=len(indices),support={k:dict(tensors=0,elements=0) for k in ('both','det_only','mask_only','neither')})
  joint=[]
  for i,name,s,e,g in selected:
   d,m=bool(support[0,i]),bool(support[1,i]);key='both' if d and m else 'det_only' if d else 'mask_only' if m else 'neither'
   row['support'][key]['tensors']+=1;row['support'][key]['elements']+=e-s
   if d and m:joint.append(torch.arange(s,e,device=vectors.device))
  ji=torch.cat(joint) if joint else torch.empty(0,dtype=torch.long,device=vectors.device)
  row['joint_connected_only']=direction(vectors[0,ji],vectors[1,ji]);result[group]=row
 return result
