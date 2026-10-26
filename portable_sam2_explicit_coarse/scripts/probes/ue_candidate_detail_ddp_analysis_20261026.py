"""Same four-GPU contract for every comparison; fixed E10, no test selection."""
import argparse
import json
from pathlib import Path

import numpy as np
from ue_candidate_detail_ddp_20261026 import ROOT, PROTOCOL, hashes, base
from ue_candidate_detail_analysis_20261026 import difficult, size_and_density
from ue_joint_encoding_analysis import evaluator, precision_ap
from ue_joint_encoding_fast_analysis import fast_sample_ap, audits


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--split', choices=['val','test'], required=True)
    args = p.parse_args()
    data, evidence = {}, {}
    for variant,content in PROTOCOL['training_jobs']:
        modes = ['matched','constant','donor'] if (variant,content) == ('both','correct') else ['matched']
        done_path = args.root/variant/content/'training_complete.json'
        done = json.loads(done_path.read_text())
        assert done['epoch'] == 10 and done['updates'] == 2670 and done['world_size'] == 4
        assert done['contract']['source_hashes'] == hashes()
        assert done['checkpoint_sha256'] == base.core.C.file_sha(args.root/variant/content/'epoch10.pth')
        evidence[str(done_path)] = base.core.C.file_sha(done_path)
        for mode in modes:
            folder = args.root/'evaluation'/args.split/variant/content/mode
            meta = json.loads((folder/'complete.json').read_text())
            assert meta['checkpoint_sha256'] == done['checkpoint_sha256']
            assert meta['source_hashes'] == hashes() and meta['world_size'] == 4
            assert meta['images'] == 459 and meta['split'] == args.split and meta['precision'] == 'FP32'
            key = variant+'_'+content if mode == 'matched' else 'both_'+mode+'_input'
            data[key] = json.loads((folder/'records.json').read_text())
            evidence[str(folder/'records.json')] = base.core.C.file_sha(folder/'records.json')
    reference = data['baseline_correct']
    for rows in data.values():
        assert len(rows) == 459
        assert [r['image_id'] for r in rows] == [r['image_id'] for r in reference]
        assert all(r['gt'] == b['gt'] for r,b in zip(rows, reference))
    contrasts = {'baseline_content':{'baseline_correct':1,'baseline_constant':-1}}
    for v in ['query','detail','both']:
        contrasts[v+'_content'] = {v+'_correct':1, v+'_constant':-1}
        contrasts[v+'_net'] = {v+'_correct':1, 'satellite_correct':-1}
        contrasts[v+'_vs_baseline'] = {v+'_correct':1, 'baseline_correct':-1}
        contrasts[v+'_content_interaction'] = {v+'_correct':1, v+'_constant':-1,
                                               'baseline_correct':-1,'baseline_constant':1}
    contrasts.update(both_depth={'both_correct':1,'both_plane':-1},
        both_donor_dependency={'both_correct':1,'both_donor_input':-1},
        both_constant_dependency={'both_correct':1,'both_constant_input':-1},
        query_effect_with_detail={'both_correct':1,'detail_correct':-1},
        detail_effect_with_query={'both_correct':1,'query_correct':-1},
        interaction={'both_correct':1,'query_correct':-1,'detail_correct':-1,'baseline_correct':1})
    samples = np.random.default_rng(20261021).integers(0,459,(500,459))
    out = args.root/'analysis'/args.split
    result = dict(split=args.split, world_size=4, effective_batch=8, selection='fixed E10',
        images=459, bootstrap=500, bootstrap_seed=20261021, paths={}, evidence_sha256=evidence,
        limitation='single-seed DDP prefix, native distributed normalizers, development-exposed v1 test; no single-GPU trajectory equivalence')
    for path in ['combined','original_path']:
        mapped = {key:([dict(r, dt=r['original_path']) for r in rows] if path == 'original_path' else rows)
                  for key,rows in data.items()}
        result['paths'][path] = {}
        for name,kind,ratio in [('MaskAP','segm',None),('BBoxAP','bbox',None),
                                ('BoundaryAP2','segm',.02),('BoundaryAP1','segm',.01)]:
            points,boot = {},{}
            for key,rows in mapped.items():
                ev = evaluator(rows,kind=kind,ratio=ratio)
                points[key] = precision_ap(ev)
                assert abs(fast_sample_ap(ev,np.arange(459))-points[key]) < 1e-12
                boot[key] = np.asarray([fast_sample_ap(ev,ix) for ix in samples])
                print(path,name,key,points[key],flush=True)
            effects = {}
            for key,weights in contrasts.items():
                delta = sum(points[k]*v for k,v in weights.items())
                draws = sum(boot[k]*v for k,v in weights.items())
                effects[key] = dict(coefficients=weights, delta=delta,
                    paired_image_bootstrap95=np.quantile(draws,[.025,.975]).tolist())
            result['paths'][path][name] = dict(points=points,comparisons=effects)
            base.core.C.dump(out/'summary.partial.json', result)
        diagnostics = {}
        for key,rows in mapped.items():
            diagnostics[key] = dict(**size_and_density(rows),errors=difficult(rows))
            print('difficult',path,key,flush=True)
        base.core.C.dump(out/(path+'_difficult.json'), diagnostics)
    result['calibration'] = audits
    base.core.C.dump(out/'summary.json', result)


if __name__ == '__main__':
    main()
