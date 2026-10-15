"""CPU-only preregistered ROI association, existing final masks, no fitting."""
import hashlib,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from scipy.ndimage import binary_dilation,binary_erosion
from pycocotools import mask as maskutils
O=Path('/tmp/ue_bev_tail_association');R=Path('/tmp/ue_uav_experiments/20261014_ad_seed42/final_analysis');IDS=[1218,950,306,1216,948,505];XS=['abs_dz_excess','height_incompatible_mass'];YS=['degradation_single_minus_dual','dual_absolute_error']
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ck(b,s):
 if not b:raise AssertionError(s)
def decode(r):return maskutils.decode(dict(size=r['size'],counts=r['counts'].encode())).astype(bool)
def corr(x,y):
 x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float);s=np.isfinite(x)&np.isfinite(y);x=x[s];y=y[s];n=len(x)
 if n<3 or np.ptp(x)==0 or np.ptp(y)==0:return dict(n=n,pearson=None,spearman=None)
 return dict(n=n,pearson=float(np.corrcoef(x,y)[0,1]),spearman=float(np.corrcoef(rankdata(x,method='average'),rankdata(y,method='average'))[0,1]))
def mean(v):
 a=[x for x in v if x is not None and np.isfinite(x)];return float(np.mean(a)) if a else None
def stat(v):
 a=np.array([x for x in v if x is not None and np.isfinite(x)]);return dict(n=len(a),mean=float(a.mean()) if len(a) else None,quantiles=np.quantile(a,[0,.25,.5,.75,1]).tolist() if len(a) else None)
def weighted(v,w):
 sel=(w>0)&np.isfinite(v);return float(np.sum(v[sel]*w[sel])/w[sel].sum()) if sel.any() else None
def main():
 m=read(O/'mapping_manifest.json');ck(m['complete'] and m['image_ids']==IDS,'Mapping complete fixed6');
 for p,h in m['source_sha256'].items():ck(sha(p)==h,'Mapping source SHA '+p)
 fixed={};source_paths=[O/'mapping_manifest.json',O/'PREREGISTRATION.md',Path(__file__)]
 for name in ['single_best','dual_correct']:
  p=R/'full_fixed_box_audit'/(name+'.json');source_paths.append(p);s=read(p);ck(s['weights']=='ema' and s['strict'] and s['training'] is False,'Existing frozen EMA outcomes');fixed[name]={r['image_id']:r for r in s['records'] if r['box_type']=='gt' and r['image_id'] in IDS}
 gp=R/'dual_correct/gt_records.json';ip=R/'full_fixed_box_audit/per_gt.json';source_paths +=[gp,ip];gt=defaultdict(list)
 for r in read(gp):
  if r['image_id'] in IDS:gt[r['image_id']].append(r)
 identity={(r['image_id'],r['gt_slot']):r for r in read(ip) if r['box_type']=='gt' and r['image_id'] in IDS}
 maps={(r['image_id'],r['layer'],r['size']):r for r in m['records']};rows=[];regions=[]
 for iid in IDS:
  single=fixed['single_best'][iid];dual=fixed['dual_correct'][iid];boxes=np.array(single['boxes_xyxy'],dtype=np.float32);ck(single['boxes_sha256']==dual['boxes_sha256']==hashlib.sha256(boxes.tobytes()).hexdigest(),'Common box SHA');ck(single['boxes_xyxy']==dual['boxes_xyxy'],'Common boxes');ck(len(boxes)==len(gt[iid]),'GT count')
  for j,box in enumerate(boxes):
   g=gt[iid][j];expected=np.array([g['bbox'][0],g['bbox'][1],g['bbox'][0]+g['bbox'][2],g['bbox'][1]+g['bbox'][3]],dtype=np.float32);ck(np.array_equal(box,expected),'GT slot boxes');truth=decode(g['segmentation']);sm=decode(single['masks'][j]);dm=decode(dual['masks'][j]);si=float((sm&truth).sum()/max((sm|truth).sum(),1));di=float((dm&truth).sum()/max((dm|truth).sum(),1));ck(si==single['mask_iou'][j] and di==dual['mask_iou'][j],'Exact RLE IoU');ident=identity[iid,j];ck(ident['single_best']==si and ident['dual_correct']==di,'Historical identity IoU')
   boundary=binary_dilation(truth,structure=np.ones((7,7),bool),border_value=0)^binary_erosion(truth,structure=np.ones((7,7),bool),border_value=0);interior=truth&~boundary;background=~truth&~boundary;dualerr=dm^truth;singleerr=sm^truth
   x1,y1,x2,y2=map(float,box);# exact pixel-overlap with bbox, supports fractional coordinates
   axis=np.arange(512,dtype=float);pw=np.maximum(0,np.minimum(axis+1,x2)-np.maximum(axis,x1));ph=np.maximum(0,np.minimum(axis+1,y2)-np.maximum(axis,y1));pixel_area=ph[:,None]*pw[None,:];area=pixel_area.sum();ck(area==ident['bbox_area'],'Historical area')
   for layer in [0,1]:
    for size in [32,64]:
     rec=maps[iid,layer,size];ck(sha(rec['path'])==rec['sha256'],'Map NPZ SHA');a=dict(np.load(rec['path']));stride=512//size;supported=~a['missing_mask'];w=pixel_area.reshape(size,stride,size,stride).sum((1,3));covered_area=float(w[supported].sum());covfrac=covered_area/area
     def poolmask(mask):return (pixel_area*mask).reshape(size,stride,size,stride).sum((1,3))
     tw=poolmask(truth);bweights=poolmask(boundary);base=dict(image_id=iid,gt_slot=j,global_id=ident['global_id'],layer=layer,size=size,stride=stride,bbox_area=area,clipped=bool(x1<=0 or y1<=0 or x2>=512 or y2>=512),covered_area=covered_area,covered_fraction=covfrac,mean_projection_coverage=weighted(a['coverage'],w),mean_projection_quality=weighted(a['mean_quality'],w*supported),gt_boundary_fraction=float(bweights.sum()/area),gt_foreground_fraction=float(tw.sum()/area),single_iou=si,dual_iou=di,degradation_single_minus_dual=si-di,dual_absolute_error=1-di)
     for key in XS:base[key]=weighted(a[key],w);base[key+'_GT_inside']=weighted(a[key],tw)
     rows.append(base)
     for name,mask in [('boundary3px',boundary),('GT_interior',interior),('nonboundary_background',background)]:
      rw=poolmask(mask);support_area=float(rw[supported].sum());de=poolmask(mask&dualerr);se=poolmask(mask&singleerr);total=float(rw.sum());r=dict(image_id=iid,gt_slot=j,layer=layer,size=size,region=name,total_area=total,covered_area=support_area,covered_dual_error_rate=float(de[supported].sum()/support_area) if support_area else None,covered_single_error_rate=float(se[supported].sum()/support_area) if support_area else None,uncovered_area=float(rw[~supported].sum()),uncovered_dual_error_rate=float(de[~supported].sum()/rw[~supported].sum()) if rw[~supported].sum() else None)
      r['covered_error_delta']=r['covered_dual_error_rate']-r['covered_single_error_rate'] if support_area else None
      for key in XS:r[key]=weighted(a[key],rw)
      regions.append(r)
 per_image={};pooled={};equal={};building={};confounds=['bbox_area','clipped','covered_fraction','mean_projection_coverage','mean_projection_quality','gt_boundary_fraction','gt_foreground_fraction']
 for size in [32,64]:
  for layer in [0,1]:
   tag=f's{size}_layer{layer}';rr=[r for r in rows if r['size']==size and r['layer']==layer];per_image[tag]={}
   def associations(rs):return {x:{y:corr([np.nan if r[x] is None else r[x] for r in rs],[r[y] for r in rs]) for y in YS} for x in XS}
   for iid in IDS:
    rs=[r for r in rr if r['image_id']==iid];per_image[tag][str(iid)]=dict(n=len(rs),no_covered_roi=sum(r['covered_area']==0 for r in rs),associations=associations(rs),distributions={k:stat([r[k] for r in rs]) for k in XS+YS+confounds},confound_diagnostics={x:{c:corr([np.nan if r[x] is None else r[x] for r in rs],[r[c] for r in rs]) for c in confounds} for x in XS},GT_inside_companion={x:associations([{**r,x:r[x+'_GT_inside']} for r in rs])[x] for x in XS})
   equal[tag]={x:{y:{method:stat([per_image[tag][str(i)]["associations"][x][y][method] for i in IDS]) for method in ['spearman','pearson']} for y in YS} for x in XS};pooled[tag]=associations(rr)
   bids=sorted(set(r['global_id'] for r in rr));br=[{k:mean([r[k] for r in rr if r['global_id']==bid]) for k in XS+YS} for bid in bids];building[tag]=dict(n_buildings=len(bids),associations=associations(br))
 region_summary={}
 for size in [32,64]:
  for layer in [0,1]:
   for name in ['boundary3px','GT_interior','nonboundary_background']:
    rs=[r for r in regions if r['size']==size and r['layer']==layer and r['region']==name];per={str(i):{k:stat([r[k] for r in rs if r['image_id']==i]) for k in XS+['covered_error_delta','covered_dual_error_rate','covered_single_error_rate','covered_area','uncovered_area','uncovered_dual_error_rate']} for i in IDS};region_summary[f's{size}_layer{layer}_{name}']=dict(per_image=per,image_equal={k:stat([per[str(i)][k]['mean'] for i in IDS]) for k in XS+['covered_error_delta','covered_dual_error_rate']})
 result=dict(validation='PASS',scope=dict(images=IDS,unique_roi_appearances=len(identity),layers=[0,1],primary_size=32,companion_size=64,no_GPU=True,no_fit=True),source_sha256={str(p):sha(p) for p in source_paths},mapping_manifest_sha256=sha(O/'mapping_manifest.json'),per_image=per_image,image_equal_correlations=equal,pooled_roi_descriptive=pooled,building_aggregated_descriptive=building,region_companion=region_summary,limitations='Six fixed image clusters, possibly repeated buildings; no p/CI/causal or AP claims. GT boxes exclude detection error. Different independently trained dual/single models; this is not a branch intervention. Tail maps expose mixed source metrics, not complete BEV/SAM residual contribution. Zero support excluded, binary RLE only, no probability reconstruction.')
 (O/'roi_records.json').write_text(json.dumps(rows,indent=2));(O/'region_records.json').write_text(json.dumps(regions,indent=2));(O/'association_summary.json').write_text(json.dumps(result,indent=2));print('PASS',len(identity),'ROI appearances,',len(rows),'scale-layer rows')
if __name__=='__main__':main()
