import sys
sys.path.insert(0,'/tmp')
import ue_bev_tail_map as m
import numpy as np
import torch,json
from types import SimpleNamespace
r=m.read('/tmp/ue_bev_geometry_quality/summary.json')['records'][0];p=dict(np.load(r['cache']['path']));a=dict(np.load(m.A/'attention_1218_layer0.npz'));src=p['project_source'];pix=p['project_pixel'];w=p['project_mass'].astype(float)*p['project_quality'].astype(float);target=(pix//512//16)*32+pix%512//16;d=np.bincount(target,weights=w,minlength=1024);v=np.stack([np.where(a['has_candidate'],a[k],0) for k in m.FIELDS],1);ref=np.stack([np.bincount(target,weights=w*v[src,j],minlength=1024)/np.maximum(d,1e-6) for j in range(len(m.FIELDS))]);out,_,_=m.BEVToSatellite(512)(torch.from_numpy(v.astype('float32').T.copy()).reshape(1,len(m.FIELDS),128,128),[SimpleNamespace(**{k:torch.from_numpy(z) for k,z in p.items()})],(32,32));actual=out.numpy().reshape(len(m.FIELDS),-1);diff=np.abs(ref-actual);tol=2e-5+2e-6*np.abs(actual);bad=np.argwhere(diff>tol);rows=[]
for j,t in bad:
 mask=target==t;weights=w[mask];vals=v[src[mask],j];f=np.sum(weights*vals,dtype=np.float64)/weights.sum();rows.append(dict(field=m.FIELDS[j],target=int(t),numpy=float(ref[j,t]),production=float(actual[j,t]),absdiff=float(diff[j,t]),tol=float(tol[j,t]),votes=int(mask.sum()),source_unique=int(np.unique(src[mask]).size),denominator=float(d[t]),local_float64=float(f)))
print(json.dumps(dict(max_abs=float(diff.max()),fail_count=len(rows),failures=rows),indent=2))
rows2=[]
for j,t in bad:
 mask=target==t;mm=p['project_mass'][mask];qq=p['project_quality'][mask];x=v[src[mask],j].astype(np.float32);ww=(mm*qq).astype(np.float32);num_terms=(ww*x).astype(np.float32)
 den32=np.float32(0);num32=np.float32(0)
 for z,y in zip(ww,num_terms):den32=np.float32(den32+z);num32=np.float32(num32+y)
 val32=np.float32(num32/max(den32,np.float32(1e-6)));den64=(mm.astype(float)*qq.astype(float)).sum();num64=(mm.astype(float)*qq.astype(float)*x.astype(float)).sum();quantden=np.sum(ww,dtype=np.float64);quantnum=np.sum(num_terms,dtype=np.float64)
 rows2.append(dict(field=m.FIELDS[j],target=int(t),production=float(actual[j,t]),numpy_sequential_float32=float(val32),sequential_bitexact=bool(val32==actual[j,t]),denominator_float64=float(den64),denominator_float32=float(den32),denominator_product_quantization_error=float(quantden-den64),denominator_sum_order_error=float(float(den32)-quantden),numerator_float64=float(num64),numerator_float32=float(num32),numerator_product_quantization_error=float(quantnum-num64),numerator_sum_order_error=float(float(num32)-quantnum),absolute_error=float(diff[j,t]),relative_error=float(diff[j,t]/abs(ref[j,t]))))
m.dump(m.O/'roundoff_sequential_detail.json',rows2)
