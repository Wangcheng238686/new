"""Independent CPU cache acceptance; no model or fitting."""
import hashlib,json
from pathlib import Path
import numpy as np
O=Path('/tmp/ue_stage_shape_probe')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
m=json.loads((O/'features/run_manifest.json').read_text())
assert m['complete'] and m['processed_images']==281 and m['parameter_versions_unchanged']
assert m['strict_load']==dict(missing_keys=[],unexpected_keys=[])
assert not m['training'] and not m['optimizer'] and m['weights']=='ema'
assert sha(O/'extract_stage_pair.py')==m['script_sha256']
assert sha(O/'PREREGISTRATION_DRAFT.md')==m['protocol_sha256']==sha(O/'PREREGISTRATION.md')
assert sha(O/'stage_pair_extraction_draft.py')==m['stage_helper_sha256']
assert sha(m['helper_path'])==m['helper_sha256']
for p,h in m['source_sha256'].items():assert sha(p)==h
seen={'train':set(),'validation':set()};tot={'train':0,'validation':0};images={'train':[],'validation':[]};size=0
for r in m['records']:
 assert sha(r['path'])==r['sha256'];z=dict(np.load(r['path']));n=r['n_rois'];size+=Path(r['path']).stat().st_size
 assert all(r[k] for k in ['source_features_unchanged','payload_rgb_satellite_unchanged','hooks_removed','uniform_cvt_projection_quality_coverage_exact'])
 for k in ['sat32','sat_extra32','cvt32','cvt_shift32','uniform32','uniform_shift32']:assert z[k].shape==(n,256,32)
 assert z['geometry'].shape==(n,256,7)
 for k in ['prob','target','boundary_weight','original_coverage','common_coverage']:assert z[k].shape==(n,256) and np.all((z[k]>=0)&(z[k]<=1+1e-6))
 assert np.array_equal(z['geometry'][...,0],z['prob']) and np.array_equal(z['geometry'][...,-1],z['common_coverage'])
 assert np.all(z['common_coverage']<=z['original_coverage']+1e-6)
 assert all(np.isfinite(a).all() for a in z.values())
 assert np.array_equal(z['gt_slot'],np.arange(n)) and np.all(z['image_id']==r['image_id']) and len(set(z['global_id']))==n
 zero=z['common_coverage'].sum(1)==0
 for k in ['cvt32','cvt_shift32','uniform32','uniform_shift32']:assert np.all(z[k][zero]==0)
 seen[r['split']].update(map(int,z['global_id']));tot[r['split']]+=n;images[r['split']].append(r['image_id'])
assert tot==dict(train=12621,validation=4496)
assert images==m['selected_image_ids'] and len(images['train'])==128 and len(images['validation'])==153
assert not seen['train']&seen['validation']
for k,v in seen.items():assert v==set(m['split_global_ids'][k])
assert len(seen['validation'])==237 and m['visible_buildings_disjoint']
assert m['elapsed_seconds']<900 and m['peak_cuda_allocated_bytes']<20*1024**3 and size<5*1024**3
for k,v in m['train_full256_covariance'].items():assert v['n']==12621*256
print(json.dumps(dict(validation='PASS',files=281,rois=tot,bytes=size,seconds=m['elapsed_seconds'],peak_bytes=m['peak_cuda_allocated_bytes'],visible_buildings={k:len(v) for k,v in seen.items()})))
