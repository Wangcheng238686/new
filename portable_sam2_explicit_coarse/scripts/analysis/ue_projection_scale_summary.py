"""CPU verification and fixed readout gates. No fit/model selection."""
import json,hashlib
from pathlib import Path
import numpy as np
O=Path('/tmp/ue_projection_scale_probe');NAMES=['SG','S64','C','CX','F','FX'];PAIRS=[('C','S64'),('C','CX'),('F','S64'),('F','FX')]
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def ck(b,s):
 if not b:raise AssertionError(s)
def mean(a,sel):return float(np.mean(a[sel])) if sel.any() else None
def main():
 d=O/'fit';m=read(d/'fit_manifest.json');ck(m['complete'] and m['validation_evaluated'] and not m['timing_smoke'],'Full completed fit');ck(m['steps']==1000 and m['selection']=='last fixed step; no validation selection','Fixed last');ck(m['seed']==1831 and m['batch_rois']==64,'Fixed seed/batch');ck(sha(O/'PREREGISTRATION.md')==m['protocol_sha256'] and sha('/tmp/ue_projection_scale_fit.py')==m['script_sha256'],'Fit source provenance');ck(sha(d/'sample_sequence.npy')==m['sample_sha256'],'Paired sample SHA');ck(sha(d/'normalization.json')==m['normalization_sha256'],'Normalization SHA')
 for name,model in m['models'].items():ck(model['last_step']==1000 and sha(model['path'])==model['sha256'],'Model weights SHA')
 metrics={};raw={}
 for split in ['train','validation']:
  p=d/(split+'_predictions.npz');ck(sha(p)==m['prediction_sha256'][split],'Prediction SHA');a=dict(np.load(p));raw[split]=a;target=a['target'];ck(target.shape[1:]==(16,16),'16grid target');n=len(target);metrics[split]={};boundary=a['boundary_weight'];den=boundary.sum((1,2));ck(np.isfinite(target).all() and np.all((target>=0)&(target<=1)),'Target legal')
  area=(a['boxes'][:,2]-a['boxes'][:,0])*(a['boxes'][:,3]-a['boxes'][:,1]);support=a['common_coverage'].mean((1,2))>0;groups={'all':np.ones(n,dtype=bool),'support_positive':support,'support_zero':~support,'small':area<1024,'medium':(area>=1024)&(area<9216),'large':area>=9216}
  metrics[split]['uncorrected']=dict(brier=float(np.mean((a['prob']-target)**2)))
  for arch in ['linear','nonlinear']:
   for name in NAMES:
    key=arch+'_'+name;v=a[key];ck(np.isfinite(v).all(),'Finite predictions');p=np.clip(v,0,1);err=(p-target)**2;brier=err.mean((1,2));boundary_err=(err*boundary).sum((1,2))/np.maximum(den,1e-12);bp=p>=.5;truth=target>=.5;iou=(bp&truth).sum((1,2))/np.maximum((bp|truth).sum((1,2)),1);buildings=np.unique(a['global_id']);metrics[split][key]=dict(brier=float(brier.mean()),unclipped_brier=float(np.mean((v-target)**2)),clamp_fraction=float(((v<0)|(v>1)).mean()),boundary_brier=float(boundary_err[den>0].mean()),boundary_rois=int((den>0).sum()),binary16_iou=float(iou.mean()),building_equal_brier=float(np.mean([brier[a['global_id']==g].mean() for g in buildings])),groups={g:dict(n=int(sel.sum()),brier=mean(brier,sel)) for g,sel in groups.items()})
    a[key+'_brier_per_roi']=brier
 val=raw['validation'];ids=np.unique(val['image_id']);ck(len(ids)==153 and len(val['target'])==4496 and len(np.unique(val['global_id']))==237,'Fixed validation');train=raw['train'];ck(len(np.unique(train['image_id']))==128 and len(train['target'])==12621,'Fixed train');ck(not set(train['global_id'])&set(val['global_id']),'Building isolation')
 counts=np.array([(val['image_id']==i).sum() for i in ids]);draw=np.random.default_rng(1830).integers(0,len(ids),(1000,len(ids)));denom=counts[draw].sum(1)
 def contrast(stage,control,arch,main):
  delta=val[arch+'_'+control+'_brier_per_roi']-val[arch+'_'+stage+'_brier_per_roi'];sums=np.array([delta[val['image_id']==i].sum() for i in ids]);boot=sums[draw].sum(1)/denom;q=[.00625,.99375] if main else [.025,.975];ci=np.quantile(boot,q).tolist();out=dict(mean=float(delta.mean()),interval=ci,quantiles=q,building_equal_mean=float(np.mean([delta[val['global_id']==g].mean() for g in np.unique(val['global_id'])])))
  if main:out['gate']=dict(minimum_effect=.001,passed=bool(delta.mean()>=.001 and ci[0]>0))
  return out
 primary={stage+'_vs_'+control:contrast(stage,control,'nonlinear',True) for stage,control in PAIRS};linear={stage+'_vs_'+control:contrast(stage,control,'linear',False) for stage,control in PAIRS};direct={arch:contrast('F','C',arch,False) for arch in ['linear','nonlinear']}
 result=dict(validation='PASS',scope=dict(train_images=128,val_images=153,train_rois=12621,val_rois=4496,task='new16x16softoccupancyBrier',exploratory_reused_development_subset=True),metrics=metrics,primary_nonlinear=primary,stage_support={stage:all(primary[stage+'_vs_'+control]['gate']['passed'] for control in ['S64',stage+'X']) for stage in ['C','F']},fine_conditionally_better=bool(all(primary['F_vs_'+c]['gate']['passed'] for c in ['S64','FX']) and direct['nonlinear']['interval'][0]>0),linear_companion=linear,Fine_minus_coarse_descriptive=dict(direction='C_Brier minus F_Brier; positive favors fine',results=direct),bootstrap=dict(seed=1830,draws=1000,method='paired image sum/count, four primary Bonferroni98.75% intervals, descriptive95%'),source_sha256={str(p):sha(p) for p in [Path(__file__),d/'fit_manifest.json']},limitations='Fixed32 random feature compression, Same learned CVT at coarse/fine production projection scales, different support/sampler limitations require exact matching. Same support/projection/readout budget, not proof of irreversible information loss. Six cells and both architectures fully reported; no fit extension or val tuning if gates fail. Shape Brier not SAMIoU/AP/UAV net model gain.')
 (O/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(primary=primary,stage_support=result['stage_support'],direct=direct),indent=2))
if __name__=='__main__':main()
