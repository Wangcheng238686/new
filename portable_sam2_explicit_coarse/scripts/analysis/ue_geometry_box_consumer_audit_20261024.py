"""Read-only geometry raw-box consumption attrition and refinement upper bound."""
import json
from pathlib import Path
import numpy as np
from ue_geometry_proposal_complement_20261024 import overlaps

P=Path('/data/wangcheng/checkpoint/ue_geometry_proposal_pilot_20261024')
C=Path('/data/wangcheng/checkpoint/ue_geometry_proposal_consume_20261024')
B=Path('/data/wangcheng/checkpoint/ue_rpn_geometry_20261023/val/budget/records.json')


def best(ov):
    return ov.max(1) if ov.shape[1] else np.zeros(len(ov))


def main():
    baseline=json.loads(B.read_text());report={}
    for arm in ['geometry','constant']:
        raw=json.loads((P/arm/'val_records.json').read_text())
        final=json.loads((C/f'{arm}_raw/val/records.json').read_text())
        total=dict(gt=0,potential_final_missed_gt=0,after_novelty_filter_missed_gt=0,accepted_missed_gt=0,
            raw_proposals=0,after_novelty_filter_proposals=0,accepted_proposals=0,
            matched_sat_gt=0,oracle_refinement_gt_improve_gt05=0,oracle_refinement_gt_cross75=0,
            nearest_geo_match_gt=0,nearest_geo_improve_gt05=0,nearest_geo_worse_gt05=0)
        deltas=[]
        for b,r,f in zip(baseline,raw,final):
            assert b['image_id']==r['image_id']==f['image_id'] and b['gt']==f['gt']
            gt=np.asarray(b['gt']['bboxes'],float).reshape(-1,4)
            sat=np.asarray(b['dt']['bboxes'],float).reshape(-1,4)
            geo=np.asarray(r['boxes'],float).reshape(-1,4)
            accepted=np.asarray(f['dt']['bboxes'][len(sat):],float).reshape(-1,4)
            oldov=overlaps(gt,sat);old=best(oldov);missed=old<.5
            relation=overlaps(geo,sat)
            novel=best(relation)<=.5
            go=overlaps(gt,geo);potential=best(go)
            total['gt']+=len(gt);total['potential_final_missed_gt']+=int((missed&(potential>=.5)).sum())
            total['after_novelty_filter_missed_gt']+=int((missed&(best(go[:,novel])>=.5)).sum())
            total['accepted_missed_gt']+=int((missed&(best(overlaps(gt,accepted))>=.5)).sum())
            total['raw_proposals']+=len(geo);total['after_novelty_filter_proposals']+=int(novel.sum());total['accepted_proposals']+=len(accepted)
            if not len(sat) or not len(geo):continue
            chosen=oldov.argmax(1);matched=old>=.5
            # Candidate must overlap the existing best SAT box by >=.5.
            allowed=relation[:,chosen].T>=.5
            oracle=np.maximum(np.where(allowed,go,0).max(1),old)
            total['matched_sat_gt']+=int(matched.sum())
            total['oracle_refinement_gt_improve_gt05']+=int((matched&(oracle-old>.05)).sum())
            total['oracle_refinement_gt_cross75']+=int((matched&(old<.75)&(oracle>=.75)).sum())
            nearest=relation[:,chosen].argmax(0)
            corresponding=relation[nearest,chosen]>=.5
            use=matched&corresponding
            new=go[np.arange(len(gt)),nearest]
            total['nearest_geo_match_gt']+=int(use.sum())
            total['nearest_geo_improve_gt05']+=int((use&(new-old>.05)).sum())
            total['nearest_geo_worse_gt05']+=int((use&(old-new>.05)).sum())
            deltas.extend((new-old)[use].tolist())
        total['nearest_geo_mean_iou_delta']=float(np.mean(deltas))
        report[arm]=total
    (C/'box_consumer_audit.json').write_text(json.dumps(dict(complete=True,images=459,report=report,
        scope='CPU float64 IoU diagnostic. GT-based oracle is an upper bound, not executable inference or AP. '
              'Novelty uses the fixed .5 admission threshold; remaining difference includes classifier, self-NMS, final cap. '
              'Nearest-Geo correspondence is a diagnostic heuristic, not a selected model.'),indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
