from pathlib import Path
import hashlib,json
import numpy as np
O=Path('/tmp/ue_bev_geometry_quality');m=json.loads((O/'summary.json').read_text());assert m['complete'] and len(m['records'])==6
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(O/'audit.py')==m['script_sha256'];assert sha(O/'PREREGISTRATION.md')==m['protocol_sha256']
for p,h in m['source_sha256'].items():assert sha(p)==h
out={'complete':True,'groups':{},'source_checks':True,'records':[]};pooled={}
for r in m['records']:
 f=np.load(O/f"raw_geometry_{r['image_id']}.npz");assert np.isfinite(f['z_change_px']).all() and np.isfinite(f['xy_center_px']).all() and np.isfinite(f['total_px']).all()
 assert int(f['height_valid'].sum())==r['point_counts']['valid_height'];assert int(f['selected_bin'].sum())==r['point_counts']['selected_bin'];assert len(f['flat'])==r['point_counts']['in_bev'];assert r['point_counts']['in_bev']+r['point_counts']['outside_bev']==r['point_counts']['raw_valid']
 assert (f['common']<=f['height_valid']).all();assert (f['selected_bin']<=f['height_valid']).all()
 observed=np.bincount(f['flat'],minlength=16384)>0;assert int(observed.sum())==r['cell_counts']['observed'];assert int(((f['cell_zbin_count']>1)&observed).sum())==r['cell_counts']['multibin_observed']
 for group,sel in [('valid_height_all',f['common']),('valid_height_satframe',f['common']&f['in_satellite_frame']),('selected_bin_all',f['common']&f['selected_bin']),('selected_bin_satframe',f['common']&f['selected_bin']&f['in_satellite_frame'])]:
  assert int(sel.sum())==r['displacement'][group]['n']
  for key in ['z_change_px','xy_center_px','total_px']:
   a=f[key][sel];s=r['displacement'][group][key];assert len(a)==s['n'];assert abs(float(a.mean())-s['mean'])<1e-12
   for k,q in [('p50',.5),('p90',.9),('p99',.99)]:assert abs(float(np.quantile(a,q))-s[k])<1e-12
   for t in [1,4,8]:assert abs(float((a>t).mean())-s[f'gt{t}_fraction'])<1e-12
   pooled.setdefault((group,key),[]).append(a)
 c=r['cache'];assert c['exists'] and c['all_arrays_exact'] and sha(c['path'])==c['sha256'];p=np.load(c['path']);candidate=p['candidate_index'];projected=np.zeros(16384,bool);projected[p['project_source']]=True;eligible=p['observation_support']&p['height_valid']&(candidate>=0).any(1)
 assert (projected<=eligible).all()
 for group,sel in [('eligible',eligible),('projected',projected)]:
  ix=candidate[sel];valid=ix>=0;delta=p['anchor_xyz'][np.maximum(ix,0)]-p['reference_xyz'][sel,None];dz=np.abs(delta[...,2])[valid];core=(np.abs(delta[...,:2])<=2).all(-1)[valid];s=r['candidates_'+group];assert int(valid.sum())==s['candidate_votes'];assert abs(float((dz<=2).mean())-s['height_compatible_fraction'])<1e-12;assert abs(float(core.mean())-s['xy_cell_core_fraction'])<1e-12
 out['records'].append({'image_id':r['image_id'],'all_saved_displacement_stats_exact':True,'cache_sha_and_candidate_stats_exact':True,'common_equals_height_valid':bool(np.array_equal(f['common'],f['height_valid']))})
for (group,key),arrays in pooled.items():
 a=np.concatenate(arrays);out['groups'].setdefault(group,{})[key]={'n':len(a),'p50':float(np.quantile(a,.5)),'p90':float(np.quantile(a,.9)),'p99':float(np.quantile(a,.99)),'gt4_fraction':float((a>4).mean())}
Path('/tmp/ue_bev_geometry_quality_independent_review.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
