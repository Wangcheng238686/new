"""Collect image-cumulative sparse correction counts, without GPU or training."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);r=p.parse_args().root
    paths={f'{content}_{mode}':r/'test'/content/mode for content,modes in
        [('correct',['correct','constant','donor']),('constant',['constant'])] for mode in modes}
    complete={}
    while len(complete)<len(paths):
        for key,folder in paths.items():
            if key in complete or not (folder/'evaluation_identity.json').exists():continue
            records_path=folder/'records.json';rows=json.loads(records_path.read_text())
            assert len(rows)==459
            totals={};images=0
            for row in rows:
                counts=row['read_observation'].get('point_diagnostics')
                if counts is None:continue
                assert counts['new_correct']-counts['old_correct']==counts['corrected']-counts['damaged']
                images+=1
                for k,v in counts.items():totals[k]=totals.get(k,0)+v
            complete[key]=dict(images_with_matched_rois=images,images=459,counts=totals,
                net_corrected=totals.get('corrected',0)-totals.get('damaged',0),
                records_sha256=hashlib.sha256(records_path.read_bytes()).hexdigest())
        out=dict(complete=len(complete)==len(paths),arms=complete,
            scope='Repeated matched RoIs/selected points; same-image original UDPR counterfactual; not AP or image-independent statistical proof')
        target=r/'point_summary.json';tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(out,indent=2));tmp.replace(target)
        if not out['complete']:time.sleep(30)


if __name__=='__main__':main()
