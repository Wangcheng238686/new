import json, collections
from pathlib import Path
import numpy as np, cv2
from pycocotools import mask as M
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
ROOT=Path('/tmp/ue_uav_experiments/20261014_ad_seed42/final_analysis')
OUT=Path('/tmp/ue_adjacent_pair_20261016/outputs')
names=['single_best','dual_correct','dual_zero','dual_shift']
def read(p): return json.loads(p.read_text())
def group(rows):
 d=collections.defaultdict(list)
 for r in rows:d[r['image_id']].append(r)
 return d
def rle(rows):
 return [dict(size=r['segmentation']['size'],counts=r['segmentation']['counts'].encode()) for r in rows]
def assignment(a,t):
 n,p=a.shape
 cost=np.zeros((n,p+n));cost[:,:p]=np.where(a>=t,-(n+1)-a,1e6)
 rr,cc=linear_sum_assignment(cost)
 ans=np.full(n,-1)
 for i,j in zip(rr,cc):
  if j<p and a[i,j]>=t:ans[i]=j
 assert np.count_nonzero(ans>=0)==np.count_nonzero(maximum_bipartite_matching(csr_matrix(a>=t),perm_type='column')>=0)
 return ans
G=group(read(ROOT/'single_best/gt_records.json'))
image_ids=[r['image_id'] for r in read(ROOT/'single_best/images.json')]
D={n:group(read(ROOT/n/'dt_records.json')) for n in names}
S={n:group(read(ROOT/'full_stage_audit'/f'{n}.json')['instances']) for n in names}
F={n:{r['image_id']:r for r in read(ROOT/'full_fixed_box_audit'/f'{n}.json')['records'] if r['box_type']=='gt'} for n in names}
y,x=np.mgrid[-5:6,-5:6]; disk=((x*x+y*y)<=25).astype('uint8')
expected_pairs={}; assignments={}; fixed_maxdiff=0; stage_maxdiff=0
for image_id in image_ids:
 gr=G[image_id]; gs=rle(gr); masks=M.decode(gs).transpose(2,0,1); label=np.zeros(masks.shape[1:],np.int32)
 for i,m in enumerate(masks): label[m!=0]=i+1
 for i,m in enumerate(masks):
  dil=cv2.dilate(m,disk)
  for value in np.unique(label[dil!=0]):
   j=int(value)-1
   if j<=i:continue
   # exact contact is a single 3x3 square dilation.
   contact=np.any(cv2.dilate(m,np.ones((3,3),np.uint8))[masks[j]!=0])
   expected_pairs[(image_id,i,j)]='contact' if contact else 'near'
 for name in names:
  dr=D[name][image_id]; ds=rle(dr)
  mi=np.asarray(M.iou(gs,ds,[0]*len(ds))).reshape(len(gr),len(dr))
  bi=np.asarray(M.iou([r['bbox'] for r in gr],[r['bbox'] for r in dr],[0]*len(dr))).reshape(len(gr),len(dr))
  stage_maxdiff=max(stage_maxdiff,float(np.max(np.abs(bi.max(1)-[r['final_best_iou'] for r in S[name][image_id]]))))
  fm=F[name][image_id]; fs=[dict(size=r['size'],counts=r['counts'].encode()) for r in fm['masks']]
  fi=np.asarray(M.iou(gs,fs,[0]*len(fs))).diagonal()
  fixed_maxdiff=max(fixed_maxdiff,float(np.max(np.abs(fi-fm['mask_iou']))))
  for t in [.5,.75]:assignments[(image_id,name,str(t))]=(assignment(mi,t),assignment(bi,t))
 print('verified',image_id,flush=True)
rows=read(OUT/'pairs.json'); actual={(r['image_id'],*r['slots']):r['stratum'] for r in rows}
assert actual==expected_pairs,(len(actual),len(expected_pairs))
for r in rows:
 for n in names:
  for t in ['0.5','0.75']:
   ma,ba=assignments[r['image_id'],n,t]; i,j=r['slots']; v=r['modes'][n][t]
   assert v['mask_pair_success']==bool(ma[i]>=0 and ma[j]>=0)
   assert v['box_pair_success']==bool(ba[i]>=0 and ba[j]>=0)
summary=read(OUT/'summary.json')
counts={}
for stratum in ['contact','near','all_adjacent','largest_components_near']:
 chosen=[r for r in rows if stratum=='all_adjacent' or (stratum=='largest_components_near' and r['largest_components_near']) or r['stratum']==stratum]
 counts[stratum]={'n':len(chosen),'physical':len(set(tuple(r['physical_pair']) for r in chosen))}
 assert summary['strata'][stratum]['appearances']==len(chosen)
 for n in names:
  for t in ['0.5','0.75']:
   c=collections.Counter(k for r in chosen for k,v in r['modes'][n][t].items() if isinstance(v,bool) and v)
   for k,v in summary['strata'][stratum]['modes'][n][t]['counts'].items(): assert c[k]==v,(stratum,n,t,k,c[k],v)
   counts[stratum][n+'@'+t]={k:c[k] for k in ['mask_pair_success','box_pair_success','mask_fail_box_success','mask_fail_box_fail','nms_coverage_loss_among_mask_fail','rpn_lacks_coverage_among_mask_fail']}
for stratum in counts:
 chosen=[r for r in rows if stratum=='all_adjacent' or (stratum=='largest_components_near' and r['largest_components_near']) or r['stratum']==stratum]
 for t in ['0.5','0.75']:
  c=collections.Counter((r['modes']['single_best'][t]['mask_pair_success'],r['modes']['dual_correct'][t]['mask_pair_success']) for r in chosen)
  tr=summary['strata'][stratum]['transitions'][t]
  assert tr['improved']==c[False,True] and tr['regressed']==c[True,False]
  assert tr['both_success']==c[True,True] and tr['both_fail']==c[False,False]
  for name in names:
   phy=collections.defaultdict(list)
   for r in chosen: phy[tuple(r['physical_pair'])].append(r['modes'][name][t]['mask_pair_success'])
   mean=np.mean([np.mean(v) for v in phy.values()])
   assert abs(mean-summary['strata'][stratum]['modes'][name][t]['physical_pair_equal_mean'])<1e-12
assert stage_maxdiff < 1e-5, stage_maxdiff
assert fixed_maxdiff < 1e-5, fixed_maxdiff
report=dict(verdict='PASS',adjacency_implementation='independent Euclidean-disk morphology and 3x3 contact dilation',matching='independent cost assembly; sparse maximum matching validates cardinality for 2448 image/mode/threshold mask-or-box matrices',source_stage_max_abs_error=stage_maxdiff,source_fixed_mask_max_abs_error=fixed_maxdiff,counts=counts)
Path('/tmp/ue_adjacent_pair_20261016/independent_numbers.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
