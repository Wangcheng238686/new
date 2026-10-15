"""Independent post-fit metrics and gates from fixed predictions; no model fit."""
import json,hashlib
from pathlib import Path
import numpy as np
R=Path('/tmp/ue_stage_shape_probe');D=R/'fit'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 s=json.loads((R/'summary.json').read_text());m=json.loads((D/'fit_manifest.json').read_text());assert m['complete'] and m['steps']==1000 and not m['validation_targets_loaded_before_fits'];out={}
 for split in ['train','validation']:
  assert sha(D/(split+'_predictions.npz'))==m['prediction_sha256'][split]
  a=dict(np.load(D/(split+'_predictions.npz')));t=a['target'];b=a['boundary_weight'];n=len(t);calc={}
  for arch in ['linear','nonlinear']:
   for cell in ['SG','S64','C','CX','U','UX']:
    k=arch+'_'+cell;p=np.clip(a[k],0,1);e=(p-t)**2;r=e.mean((1,2));expected=s['metrics'][split][k];assert abs(float(r.mean())-expected['brier'])<1e-12
    den=b.sum((1,2));bv=(e*b).sum((1,2))/np.maximum(den,1e-12);assert abs(float(bv[den>0].mean())-expected['boundary_brier'])<1e-12
    gid=a['global_id'];be=np.mean([r[gid==g].mean() for g in np.unique(gid)]);assert abs(float(be)-expected['building_equal_brier'])<1e-12
    calc[k]=r
  out[split]={'n':n,'brier':{k:float(v.mean()) for k,v in calc.items()}}
  if split=='validation':val=a;values=calc
 ids=np.unique(val['image_id']);assert len(ids)==153 and len(val['target'])==4496;ix=val['image_id'];ct=np.array([(ix==i).sum() for i in ids]);draw=np.random.default_rng(1830).integers(0,len(ids),size=(1000,len(ids)));den=ct[draw].sum(axis=1);gates={}
 for stage,control in [('C','S64'),('C','CX'),('U','S64'),('U','UX')]:
  delta=values['nonlinear_'+control]-values['nonlinear_'+stage];summed=np.array([delta[ix==i].sum() for i in ids]);boot=summed[draw].sum(1)/den;ci=np.quantile(boot,[.00625,.99375]);name=stage+'_vs_'+control;expected=s['primary_nonlinear'][name];assert abs(float(delta.mean())-expected['mean'])<1e-12;assert np.max(np.abs(ci-expected['interval']))<1e-12;passed=bool(delta.mean()>=.001 and ci[0]>0);assert passed==expected['gate']['passed'];gates[name]=dict(mean=float(delta.mean()),interval=ci.tolist(),passed=passed)
 out.update(validation='PASS',scope='Root independent Brier, boundary/building means and four main paired bootstrap gates; not independent model replay',primary=gates,source_sha256={str(p):sha(p) for p in [Path(__file__),R/'summary.json',D/'fit_manifest.json']})
 (R/'root_numeric_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(gates,indent=2))
if __name__=='__main__':main()
