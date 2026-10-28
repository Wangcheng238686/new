"""COCO matching evidence for newly consumed UAV hypotheses, no fitting."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'probes'))
from ue_joint_encoding_analysis import evaluator

ROOT=Path('/data/wangcheng/checkpoint/ue_geometry_conditional_online_20261024')
OUT=Path('/data/wangcheng/checkpoint/ue_geometry_new_instance_audit_20261024')


def main():
    OUT.mkdir(exist_ok=True)
    path=ROOT/'geometry/test/prefix_records.json';base=json.loads(path.read_text())
    eb=evaluator(base,kind='segm');assert len(base)==459
    result={};image_rows={}
    for arm in ['geometry','constant']:
        rows=json.loads((ROOT/arm/'test/combined_records.json').read_text())
        ec=evaluator(rows,kind='segm')
        groups={key:dict(GT=0,new50=0,new75=0,new75_previously_matched50=0,new75_previously_missing50=0)
                for key in ['all','bbox_small','bbox_medium','bbox_large','density_le50','density_51to100','density_gt100']}
        images=[];added=0;new_tp={'.5':0,'.75':0}
        for i,(b,c) in enumerate(zip(base,rows)):
            assert b['image_id']==c['image_id'] and b['gt']==c['gt']
            bb,cc=eb.evalImgs[i],ec.evalImgs[i];n=len(bb['dtIds'])
            assert np.array_equal(bb['dtMatches'],cc['dtMatches'][:,:n]),'Original matching changed'
            assert np.array_equal(bb['dtIgnore'],cc['dtIgnore'][:,:n]),'Original ignore flags changed'
            old=bb['gtMatches']>0;new=cc['gtMatches']>0
            assert not (old&~new).any()
            t50=int(np.flatnonzero(np.isclose(eb.params.iouThrs,.5))[0]);t75=int(np.flatnonzero(np.isclose(eb.params.iouThrs,.75))[0])
            born50=new[t50]&~old[t50];born75=new[t75]&~old[t75]
            assert not np.asarray(bb['gtIgnore']).any(),'Unexpected ignored GT in all-area'
            boxes=np.asarray(b['gt']['bboxes']);area=np.prod(boxes[:,2:]-boxes[:,:2],1);ng=len(boxes)
            selections={'all':np.ones(ng,bool),'bbox_small':area<1024,'bbox_medium':(area>=1024)&(area<9216),'bbox_large':area>=9216,
                'density_le50':np.full(ng,ng<=50),'density_51to100':np.full(ng,50<ng<=100),'density_gt100':np.full(ng,ng>100)}
            for name,select in selections.items():
                g=groups[name];g['GT']+=int(select.sum());g['new50']+=int((born50&select).sum());g['new75']+=int((born75&select).sum())
                g['new75_previously_matched50']+=int((born75&old[t50]&select).sum());g['new75_previously_missing50']+=int((born75&~old[t50]&select).sum())
            extra=len(cc['dtIds'])-n;added+=extra;new_tp['.5']+=int(born50.sum());new_tp['.75']+=int(born75.sum())
            images.append(dict(image_id=b['image_id'],GT=ng,added=extra,new50=int(born50.sum()),new75=int(born75.sum())))
        result[arm]=dict(groups=groups,additional_detections=added,additional_TP=new_tp,
            additional_unmatched_at50=added-new_tp['.5'],additional_unmatched_at75=added-new_tp['.75'])
        image_rows[arm]=images;print(arm,json.dumps(result[arm]),flush=True)
    (OUT/'per_image.json').write_text(json.dumps(image_rows))
    (OUT/'summary.json').write_text(json.dumps(dict(complete=True,images=459,results=result,
        original_matches_unchanged=True,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Exact COCO all-area segm one-to-one matching at .5/.75,max250. Added unmatched includes duplicates/background/localization failures; not all background FP. '
              'Counts are actual new GT mask matches, not box coverage proxies. Existing masks unchanged; boundary shape correction of old instances is not claimed. '
              'Scores are priority bands, not calibrated common probabilities; raw extra count is not an operating-point precision.'),indent=2))


if __name__=='__main__':main()
