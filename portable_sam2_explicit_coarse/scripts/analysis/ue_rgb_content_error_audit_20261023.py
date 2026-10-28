"""Read-only fixed-checkpoint RGB intervention error attribution (no AP estimation)."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='2'
import json,hashlib,sys
from pathlib import Path
import numpy as np
import cv2
from pycocotools import mask as mu
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/probes'))
from ue_s2db_analysis import bbox_matching
from rsprompter.uav.s2db_geometry import observed_support
cv2.setNumThreads(1)
SOURCE=Path('/data/wangcheng/checkpoint/ue_s2db_probe_20261022')
CACHE=Path('/data/wangcheng/cache/ue_v1_s2db_geometry_v2')
OUT=Path('/data/wangcheng/checkpoint/ue_rgb_content_error_audit_20261023')
THRESHOLDS=np.arange(.5,1,.05)
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def masks(row,key):
    r=row[key]['rles']
    return mu.decode(r).astype(bool) if r else np.zeros((512,512,0),bool)
def bucket():return {'pixels':0,'corrupted_errors':0,'correct_errors':0,'corrected':0,'introduced':0,'changed':0}
def run(arm):
    paths={m:SOURCE/'test'/arm/m/'records.json' for m in ('correct','constant','donor')}
    data={m:json.loads(p.read_text()) for m,p in paths.items()}
    assert all(len(v)==459 for v in data.values())
    result={m:{'zones':{f'{z}/{s}':bucket() for z in ('boundary','interior','background') for s in ('all','cell_supported','cell_unsupported','anchor_observed','anchor_unobserved')},'instances':[],'unmatched_gt':0,'unmatched_dt':0,'cross_up':[0]*10,'cross_down':[0]*10} for m in ('constant','donor')}
    cache_hashes={}
    for i,b in enumerate(data['correct']):
        others={m:data[m][i] for m in result}
        for m,c in others.items():
            assert b['image_id']==c['image_id'] and b['gt']==c['gt']
            assert all(b['dt'][k]==c['dt'][k] for k in ('bboxes','labels','scores'))
        pairs,ug,ud=bbox_matching(b['gt']['bboxes'],b['dt']['bboxes'])
        gm=masks(b,'gt');cm=masks(b,'dt')
        path=CACHE/f"{b['image_id']:06d}.npz";cache_hashes[str(path)]=sha(path)
        with np.load(path) as a:
            cells=np.repeat(np.repeat((a['index']>=0).any(-1).reshape(128,128),4,0),4,1)
            observed=observed_support(a['index'],a['anchor']).astype(bool)
        for mode,c in others.items():
            r=result[mode];r['unmatched_gt']+=ug;r['unmatched_dt']+=ud
            dm=masks(c,'dt')
            for gi,di in pairs:
                g=gm[:,:,gi];correct=cm[:,:,di];corrupt=dm[:,:,di]
                gc=np.count_nonzero(g&correct);gd=np.count_nonzero(g&corrupt)
                ic=gc/max(np.count_nonzero(g|correct),1);id_=gd/max(np.count_nonzero(g|corrupt),1)
                up=(id_<THRESHOLDS)&(ic>=THRESHOLDS);down=(id_>=THRESHOLDS)&(ic<THRESHOLDS)
                r['cross_up']=(np.array(r['cross_up'])+up).tolist();r['cross_down']=(np.array(r['cross_down'])+down).tolist()
                r['instances'].append({'image_id':b['image_id'],'gt':gi,'dt':di,'correct_iou':ic,'corrupted_iou':id_,'delta':ic-id_,'gt_pixels':int(g.sum()),'gt_cell_support':float(cells[g].mean()) if g.any() else 0.,'gt_anchor_support':float(observed[g].mean()) if g.any() else 0.})
                box=np.array([b['gt']['bboxes'][gi],b['dt']['bboxes'][di]])
                lo=np.maximum(np.floor(box[:,:2].min(0)).astype(int)-8,0);hi=np.minimum(np.ceil(box[:,2:].max(0)).astype(int)+8,512)
                sl=np.s_[lo[1]:hi[1],lo[0]:hi[0]]
                # Morphology on full GT preserves exact existing-zone convention.
                inner=cv2.erode(g.astype(np.uint8),np.ones((3,3),np.uint8),iterations=2,borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)
                outer=cv2.dilate(g.astype(np.uint8),np.ones((3,3),np.uint8),iterations=2,borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)
                zones={'boundary':(outer&~inner)[sl],'interior':inner[sl],'background':(~outer)[sl]}
                ce=(correct!=g)[sl];de=(corrupt!=g)[sl]
                supports={'all':np.ones_like(ce),'cell_supported':cells[sl],'cell_unsupported':~cells[sl],'anchor_observed':observed[sl],'anchor_unobserved':~observed[sl]}
                for zname,z in zones.items():
                    for sname,s in supports.items():
                        region=z&s;q=r['zones'][zname+'/'+sname]
                        q['pixels']+=int(region.sum());q['corrupted_errors']+=int((region&de).sum());q['correct_errors']+=int((region&ce).sum());q['corrected']+=int((region&de&~ce).sum());q['introduced']+=int((region&~de&ce).sum());q['changed']+=int((region&(de!=ce)).sum())
            del dm
        if (i+1)%25==0:print(arm,i+1,flush=True)
    for m,r in result.items():
        d=np.array([v['delta'] for v in r['instances']]);r['summary']={'matched_instances':len(d),'positive':int((d>0).sum()),'negative':int((d<0).sum()),'unchanged':int((d==0).sum()),'mean_delta_iou':float(d.mean()),'median_delta_iou':float(np.median(d)),'delta_iou_quantiles':np.quantile(d,[0,.05,.25,.5,.75,.95,1]).tolist()}
        for q in r['zones'].values():q['net_corrected']=q['corrected']-q['introduced']
    report={'arm':arm,'checkpoint':str(SOURCE/arm/'last.pth'),'checkpoint_sha256':sha(SOURCE/arm/'last.pth'),'sources':{str(p):sha(p) for p in paths.values()},'cache_sha256':cache_hashes,'comparisons':result}
    (OUT/f'{arm}.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    reports=[run(a) for a in ('A','D')]
    protocol={'script':str(Path(__file__).resolve()),'script_sha256':sha(__file__),'matching':'one-to-one Hungarian bbox IoU >=0.5 using unchanged detector boxes; same assignments across input conditions','instance_iou':'full 512x512 masks for matched GT/detection pairs; instance means and threshold crossings are not COCO AP','thresholds':THRESHOLDS.tolist(),'zones':'GT two-iteration 3x3 erosion/dilation band; cropped to union of GT/detection bbox plus 8 pixels, clipped to image; instance overlaps count repeatedly','support':'cell support is >=1 retained valid anchor per 4x4 satellite cell, expanded nearest; anchor observed uses unique satellite pixels of retained anchors; neither certifies visibility; unsupported complements reported separately','denominators':'zone instance-pixels; instance IoU uses each full predicted/GT mask union; unmatched GT/DT reported and excluded from matched-instance attribution','interventions':'constant RGB normalized channel mean; donor cyclic successor RGB with original depth/cameras, an inconsistent input','inference':'existing complete 459 records only; no new inference/training; fixed checkpoint per arm','statistics':'descriptive mechanism analysis only, no new threshold selection or inferential AP claims'}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    lines=['# 固定权重RGB内容净纠错诊断','','正确输入减去同一检查点的常量/循环错配RGB。只读A/D完整459图，不新增训练或推理。','','|臂|对照|匹配实例|IoU改善/下降|平均IoU差|边界净纠错像素|内部净纠错|背景净纠错|','','']
    lines[4]='|臂|对照|匹配实例|IoU改善/下降|平均IoU差|边界净纠错像素|内部净纠错|背景净纠错|'
    lines[5]='|---|---|---:|---:|---:|---:|---:|---:|'
    for report in reports:
        for mode,r in report['comparisons'].items():
            s=r['summary'];z=r['zones'];lines.append(f"|{report['arm']}|{mode}|{s['matched_instances']}|{s['positive']}/{s['negative']}|{s['mean_delta_iou']:+.7f}|{z['boundary/all']['net_corrected']:+d}|{z['interior/all']['net_corrected']:+d}|{z['background/all']['net_corrected']:+d}|")
            lines.append('')
    lines+=['## 解释边界','','此处是固定框匹配后的像素/实例描述，不是AP分解或新增显著性检验。阈值跨越统计不含COCO分数排序、全局匹配、重复检出影响；不可将跨越净数转换为AP。完整每实例记录、0.50–0.95各阈值跨越和所有支持分区分母见A.json、D.json。','', '区域沿用既有：GT两像素形态学边界，限制于GT/预测框并集外扩8像素；像素可能随实例重复。格支持几乎覆盖整图不意味着精确像素有观测；精确锚点支持亦不是遮挡认证。未匹配实例数逐对报告，未纳入配对IoU/像素判断。','', 'donor仅替换RGB而保留目标几何，是输入不一致诊断；正确内容相对该输入的优势不独立证明真实数据分布下的净收益。']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    (OUT/'complete.json').write_text(json.dumps({'complete':True,'arms':['A','D'],'protocol_sha256':sha(OUT/'protocol.json')},indent=2)+'\n')
if __name__=='__main__':main()
