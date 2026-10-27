"""Independent foundation G + same-pass native refiner. No global UBR constructed."""
import json
from pathlib import Path
import sys
WORK=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(WORK),str(WORK/'scripts/probes')]
import torch
import ue_compact_joint_run_20261024 as base
from rsprompter.uav.compact_joint import CompactJoint
from rsprompter.uav.candidate_detail_interaction import install as install_g
from rsprompter.uav.ubr_native_integration import native_model_config,install_native_refiner
from data.ue_joint_inputs import JointUAVInputs
from data.ue_candidate_evidence import load_candidate_geometry
core=base.core
ARMS=('A_nohead','B_zero','C_correct')


class Runtime(base.Runtime):
 def __init__(self,arm='C_correct'):
  if arm not in ARMS:raise ValueError(arm)
  self.root=base.ROOT;self.arm='rgbd';self.native_arm=arm
  path=self.root/'common_init.pth';receipt=json.loads((self.root/'initialization.json').read_text())
  assert receipt['common_init_sha256']==core.C.file_sha(path)
  assert receipt['source_hashes']==base.sources()
  cp=torch.load(path,map_location='cpu',weights_only=False)
  assert cp['task_weights_loaded'] is False and cp['foundation_sha256']==core.C.file_sha(base.FOUNDATION)
  core.C.seed(base.PROTOCOL['seed']);config=native_model_config(cp['model_config'])
  satellite=core.C._register_and_build(config)
  retired=[k for k in cp['satellite'] if k.startswith('roi_head.mask_head.decoder_tail_refiner.')]
  satellite.load_state_dict({k:v for k,v in cp['satellite'].items() if k not in retired},strict=True)
  self.joint=CompactJoint(satellite,base.FOUNDATION,source_seed=2045);self.joint.uav.load_state_dict(cp['uav'],strict=True);self.joint.cuda()
  install_g(self.joint.uav,'detail',satellite)
  self.satellite=satellite;self.shim=None
  if arm!='A_nohead':self.shim=install_native_refiner(satellite,self.joint.uav,zero_evidence=arm=='B_zero')
  self.network=self.joint;self.bare=self.joint
  satellite.rpn_head.test_cfg.nms_pre=base.PROTOCOL['RPN_pre']
  self.uav_inputs=JointUAVInputs(core.previous.DATA,'/data/wangcheng/cache/ue_v1_native_geometry','/data/wangcheng/cache/ue_v1_s2db_geometry_v2',base.PLANE)
  self.generator=torch.Generator().manual_seed(44)
  self.optimizer=torch.optim.AdamW([p for p in self.network.parameters() if p.requires_grad],lr=base.PROTOCOL['lr'],weight_decay=base.PROTOCOL['weight_decay'])
  self.scaler=torch.amp.GradScaler('cuda',init_scale=base.PROTOCOL['amp_init_scale'])
  self.initialization_audit=dict(arm=arm,common_init_sha256=receipt['common_init_sha256'],foundation_sha256=cp['foundation_sha256'],task_weights_loaded=False,
   udpr_never_instantiated=True,global_UBR_never_instantiated=True,retired_tail_state_keys=retired,retained_state_strict=True,main_mask_size=list(config['train_cfg']['rcnn']['mask_size']),head_seed=2080,
   RGB='same G reduce_rgb output before rgb_fusion',RGB_support='retained candidate index.any, independent geometry support',native='128 decoder output interpreted ROI-local; original512 mask supervision preserved')
  core.C.seed(base.PROTOCOL['seed'])
 def payload(self,ident):
  value=super().payload(ident);self.joint.uav.candidate_geometry=load_candidate_geometry(core.previous.DATA,ident,value['index'],value['anchor']);return value
 def load_engineering_G(self,path):
  """Gate-only task checkpoint, never used by an experiment initializer."""
  cp=torch.load(path,map_location='cpu',weights_only=False)
  state={}
  for name,value in cp['model'].items():
   if 'decoder_tail_refiner' in name:continue
   if self.shim is not None and name.startswith('satellite.roi_head.mask_head.mask_decoder.'):
    name=name.replace('satellite.roi_head.mask_head.mask_decoder.','satellite.roi_head.mask_head.mask_decoder.base.',1)
   state[name]=value
  expected=self.joint.state_dict();missing=set(expected)-set(state);unexpected=set(state)-set(expected)
  assert not unexpected
  assert all(name.startswith('satellite.roi_head.mask_head.native_refiner.') for name in missing)
  self.joint.load_state_dict({**expected,**state},strict=True)
  return dict(checkpoint_sha256=core.C.file_sha(path),role='engineering_only_not_training_initialization',native_state_retained=sorted(missing))


def build_runtime(arm='C_correct'):return Runtime(arm)
