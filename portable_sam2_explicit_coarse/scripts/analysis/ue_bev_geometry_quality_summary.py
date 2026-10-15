from pathlib import Path
import json,numpy as np
O=Path('/tmp/ue_bev_geometry_quality');m=json.load(open(O/'summary.json'));assert m['complete'];r=m['records'];out={'images':m['selected_ids'],'n_images':len(r),'elapsed_seconds':m['elapsed_seconds'],'all_cache_exact':all(z['cache'].get('all_arrays_exact',False) for z in r),'point_counts':{k:sum(z['point_counts'][k] for z in r) for k in r[0]['point_counts']},'cell_counts':{k:sum(z['cell_counts'][k] for z in r) for k in r[0]['cell_counts']}}
ds=[np.load(O/f'raw_geometry_{i}.npz') for i in m['selected_ids']];out['displacements']={}
for name in ['valid_height_all','valid_height_satframe','selected_bin_all','selected_bin_satframe']:
 vals={k:[] for k in ['z_change_px','xy_center_px','total_px']}
 for d in ds:
  s=d['common'].copy()
  if name.startswith('selected'):s&=d['selected_bin']
  if name.endswith('satframe'):s&=d['in_satellite_frame']
  for k in vals:vals[k].append(d[k][s])
 result={}
 for k in vals:
  a=np.concatenate(vals[k]);result[k]=dict(n=len(a),mean=float(a.mean()),p50=float(np.quantile(a,.5)),p90=float(np.quantile(a,.9)),p99=float(np.quantile(a,.99)),gt1_fraction=float((a>1).mean()),gt4_fraction=float((a>4).mean()),gt8_fraction=float((a>8).mean()))
 out['displacements'][name]=result
for key in ['candidates_eligible','candidates_projected']:
 n=sum(z[key]['candidate_votes'] for z in r);out[key]=dict(cells=sum(z[key]['cells'] for z in r),candidate_votes=n,height_compatible_fraction=sum(z[key]['candidate_votes']*z[key]['height_compatible_fraction'] for z in r)/n,xy_cell_core_fraction=sum(z[key]['candidate_votes']*z[key]['xy_cell_core_fraction'] for z in r)/n)
(O/'aggregate.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
