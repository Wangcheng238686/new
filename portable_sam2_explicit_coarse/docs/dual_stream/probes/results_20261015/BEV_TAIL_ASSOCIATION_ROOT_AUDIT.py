"""Independent ROI correlation reproduction and fixed FP32 sensitivity; CPU only."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr,spearmanr
R=Path('/tmp/ue_bev_tail_association');IDS=[1218,950,306,1216,948,505];XS=['abs_dz_excess','height_incompatible_mass'];YS=['degradation_single_minus_dual','dual_absolute_error']
def corr(x,y):
 x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float);ok=np.isfinite(x)&np.isfinite(y);x=x[ok];y=y[ok]
 if len(x)<3 or not np.ptp(x) or not np.ptp(y):return dict(n=len(x),pearson=None,spearman=None)
 return dict(n=len(x),pearson=float(pearsonr(x,y).statistic),spearman=float(spearmanr(x,y).statistic))
def main():
 rows=json.loads((R/'roi_records.json').read_text());s=json.loads((R/'association_summary.json').read_text());maps={};summary={};maxerr=0.;maxxdiff={k:0. for k in XS}
 for size in [32,64]:
  for layer in [0,1]:
   tag=f's{size}_layer{layer}';summary[tag]={}
   for iid in IDS:
    rs=[r for r in rows if r['image_id']==iid and r['size']==size and r['layer']==layer];a=dict(np.load(R/f'maps_{iid}_layer{layer}_s{size}.npz'));stride=512//size
    p=Path('/tmp/ue_uav_experiments/20261014_ad_seed42/final_analysis/full_fixed_box_audit/dual_correct.json');d=json.loads(p.read_text());boxes=next(r['boxes_xyxy'] for r in d['records'] if r['box_type']=='gt' and r['image_id']==iid)
    prod={k:[] for k in XS}
    for r in rs:
     x1,y1,x2,y2=boxes[r['gt_slot']];axis=np.arange(512);wx=np.clip(np.minimum(axis+1,x2)-np.maximum(axis,x1),0,None);wy=np.clip(np.minimum(axis+1,y2)-np.maximum(axis,y1),0,None);weights=(wy[:,None]*wx).reshape(size,stride,size,stride).sum(axis=(1,3));ok=(~a['missing_mask'])&(weights>0)
     for key in XS:
      v=float(np.average(a[key][ok],weights=weights[ok])) if ok.any() else None
      assert (v is None)==(r[key] is None)
      if v is not None:assert abs(v-r[key])<1e-12
      pv=float(np.average(a['production_fp32_'+key][ok],weights=weights[ok])) if ok.any() else None;prod[key].append(pv)
      if v is not None:maxxdiff[key]=max(maxxdiff[key],abs(pv-v))
    summary[tag][str(iid)]={}
    for key in XS:
     summary[tag][str(iid)][key]={}
     for y in YS:
      c=corr([r[key] for r in rs],[r[y] for r in rs]);expected=s['per_image'][tag][str(iid)]['associations'][key][y];assert c['n']==expected['n']
      for method in ['pearson','spearman']:
       if c[method] is None:assert expected[method] is None
       else:
        delta=abs(c[method]-expected[method]);maxerr=max(maxerr,delta);assert delta<1e-12
      summary[tag][str(iid)][key][y]=corr(prod[key],[r[y] for r in rs])
 equal={}
 for tag,per in summary.items():
  equal[tag]={x:{y:{method:float(np.mean([per[str(i)][x][y][method] for i in IDS if per[str(i)][x][y][method] is not None])) for method in ['pearson','spearman']} for y in YS} for x in XS}
 out=dict(validation='PASS',scope='Independent bbox overlap exposure and scipy correlation reproduction; FP32 numerical sensitivity is companion only',main_correlation_max_abs_error=maxerr,max_roi_fp32_vs_fp64_exposure_difference=maxxdiff,production_fp32_per_image_correlations=summary,production_fp32_image_equal_correlations=equal)
 (R/'root_association_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k!='production_fp32_per_image_correlations'},indent=2))
if __name__=='__main__':main()
