"""CPU-only extraction smoke integrity; no readout fit or outcome evaluation."""
import json,hashlib
from pathlib import Path
import numpy as np
O=Path('/tmp/ue_projection_scale_probe');P=O/'smoke';m=json.loads((P/'run_manifest.json').read_text());old=json.loads(Path('/tmp/ue_stage_shape_probe/features/run_manifest.json').read_text());oldr={(r['split'],r['image_id']):r for r in old['records']}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert m['protocol_sha256']==sha(O/'PREREGISTRATION_DRAFT.md')
assert m['old_manifest_sha256']==sha('/tmp/ue_stage_shape_probe/features/run_manifest.json')
assert m['selected_image_ids']==old['selected_image_ids'] and m['architecture_contract']==old['architecture_contract']
assert m['checkpoint_sha256']==old['checkpoint_sha256'] and m['parameter_versions_unchanged'] and m['weights']=='ema' and not m['training'] and not m['optimizer']
assert m['processed_images']==6 and m['n_rois']==240
assert m['actual_pipeline']['roi_spatial_scale']==.25 and m['actual_pipeline']['roi_output']==[16,16]
for p,s in m['source_sha256'].items():assert sha(p)==s,p
records=[]
for r in m['records']:
 assert sha(r['path'])==r['sha256'];prev=oldr[(r['split'],r['image_id'])];assert sha(prev['path'])==r['source_cache_sha256']==prev['sha256']
 a=dict(np.load(r['path']));b=dict(np.load(prev['path']));n=r['n_rois']
 for k in ('sat32','sat_extra32','prob','target','boundary_weight','global_id','boxes','image_id','gt_slot'):assert np.array_equal(a[k],b[k]),k
 assert np.array_equal(a['geometry'][...,:5],b['geometry'][...,:5])
 for k in ('sat32','sat_extra32','cvt32','cvt_shift32','fine32','fine_shift32'):assert a[k].shape==(n,256,32)
 assert a['geometry'].shape==(n,256,7) and np.array_equal(a['common_coverage'],a['geometry'][...,-1])
 for k in ('prob','target','boundary_weight','original_coverage','common_coverage'):assert a[k].shape==(n,256) and ((a[k]>=0)&(a[k]<=1)).all()
 for v in a.values():assert len(v)==n and np.isfinite(v).all()
 zero=a['common_coverage']==0
 for k in ('cvt32','cvt_shift32','fine32','fine_shift32'):assert (a[k][zero]==0).all(),k
 assert r['reused_arrays_exact'] and r['projection_payload_same'] and r['common_fine_support'] and r['hookon_off_features_exact'] and r['original_inputs_unchanged'] and r['hooks_removed']
 records.append(dict(image_id=r['image_id'],split=r['split'],n=n,zero_cells=int(zero.sum()),zero_rois=int((a['common_coverage'].mean(1)==0).sum()),cache_bytes=Path(r['path']).stat().st_size,native_reconstruction_roi_delta=r['native_reconstruction_roi_delta']))
assert not(set(m['split_global_ids']['train'])&set(m['split_global_ids']['validation']))
out=dict(verdict='PASS',gpu_used=False,fit=False,records=records,cache_bytes=sum(r['cache_bytes'] for r in records),actual_roi_flags=m['actual_pipeline'],source_manifest_sha256=sha(P/'run_manifest.json'),script_sha256=sha(__file__))
(O/'cpu_smoke_check.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
