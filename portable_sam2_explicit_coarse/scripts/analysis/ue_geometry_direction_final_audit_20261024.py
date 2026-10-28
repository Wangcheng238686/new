"""Reproducible completion audit of the effective geometry proposal direction.

Checks actual checkpoints, paired epoch streams, fixed prediction/analysis
artifacts, input controls and budget controls. Does not fit or infer models.
"""
import hashlib
import json
import math
from pathlib import Path
import torch

W=Path(__file__).resolve().parents[2]
B=Path('/data/wangcheng/checkpoint')
OUT=B/'ue_geometry_direction_final_audit_20261024'


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fingerprint(state):
    h=hashlib.sha256()
    for k,t in state.items():
        h.update(k.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    frozen=read(B/'ue_rpn_geometry_20261023/final_source_freeze.json')['source_hashes']
    assert len(frozen)==49 and all(sha(W/p)==h for p,h in frozen.items())
    data=read(B/'ue_probe_input_identity_audit_20261024/summary.json')
    assert data['passed'] and data['archive_files']==36734 and data['cache_files']==3060
    assert read(B/'ue_geometry_confirmation_analysis_20261024/complete.json')['complete']
    assert read(B/'ue_geometry_conditional_repeat_seed45_20261024/complete.json')['complete']
    prefix_sha=sha(B/'ue_geometry_conditional_online_20261024/geometry/test/prefix_records.json')
    seeds={};initials=[];first_orders=[]
    for seed,train,run,analysis in [
        (44,'ue_geometry_proposal_pilot_20261024','ue_geometry_conditional_online_20261024','ue_geometry_confirmation_analysis_20261024'),
        (45,'ue_geometry_proposal_repeat_seed45_20261024','ue_geometry_conditional_repeat_seed45_20261024','ue_geometry_conditional_repeat_seed45_20261024/analysis')]:
        tr,rr,ar=B/train,B/run,B/analysis
        init=torch.load(tr/'common_init.pth',map_location='cpu',weights_only=False)
        assert fingerprint(init['model'])==init['fingerprint'];initials.append(init['fingerprint'])
        params=sum(t.numel() for t in init['model'].values());assert params==202581
        first_orders.append(read(tr/'geometry/epoch1.json')['sample_order'])
        for arm in ['geometry','constant']:
            cp=torch.load(tr/arm/'last.pth',map_location='cpu',weights_only=False)
            assert cp['epoch']==10 and cp['step']==2670 and cp['content']==arm
            assert cp['initial_fingerprint']==init['fingerprint']
            assert set(cp['rng_state'])=={'python','numpy','torch','cuda','loader_generator'}
            assert {int(x['step']) for x in cp['optimizer']['state'].values()}=={2670}
            identity=read(rr/arm/'test/identity.json')
            assert identity['complete'] and identity['images']==459 and identity['satellite_state_unchanged'] and identity['max_dets']==250
            assert identity['checkpoint_sha256']==sha(tr/arm/'last.pth')
            assert sha(rr/arm/'test/prefix_records.json')==prefix_sha
            source='ue_geometry_conditional_online_20261024.py' if seed==44 else 'ue_geometry_conditional_configured_20261024.py'
            assert identity['source_sha256']==sha(W/'scripts/probes'/source)
        for epoch in range(1,11):
            a,b=[read(tr/arm/f'epoch{epoch}.json') for arm in ['geometry','constant']]
            assert a['sample_order']==b['sample_order'] and len(a['sample_order'])==2136
            assert a['step']==b['step']==267*epoch
            assert math.isfinite(a['mean_batch_loss']) and math.isfinite(b['mean_batch_loss'])
        net=read(ar/('conditional_net.json' if seed==44 else 'net.json'))
        content=read(ar/('conditional_content.json' if seed==44 else 'content.json'))
        for result in [net,content]:
            assert result['images']==459 and result['gt_exact']
            checks=result['accumulation_equivalence']['reference_checks']
            assert len(checks)==24 and max(c['max_abs_error'] for c in checks)<1e-12
        n,c=net['metrics']['MaskAP'],content['metrics']['MaskAP']
        assert n['delta']>=.002 and n['paired_image_bootstrap95'][0]>0
        assert c['delta']>0 and c['paired_image_bootstrap95'][0]>0
        points={arm:read(rr/arm/'test/combined_metrics.json') for arm in ['geometry','constant']}
        prefix=read(rr/'geometry/test/prefix_metrics.json')
        assert abs(n['candidate']-points['geometry']['segm/mAP'])<1e-12
        assert abs(n['baseline']-prefix['segm/mAP'])<1e-12
        assert abs(c['baseline']-points['constant']['segm/mAP'])<1e-12
        seeds[str(seed)]=dict(parameters=params,initial_fingerprint=init['fingerprint'],
            mask_AP=points['geometry']['segm/mAP'],constant_mask_AP=points['constant']['segm/mAP'],
            strong_SAT_mask_AP=prefix['segm/mAP'],net=net['metrics'],content=content['metrics'])
    assert len(set(initials))==2 and first_orders[0]!=first_orders[1]
    controls=read(B/'ue_geometry_intervention_analysis_20261024/summary.json');assert controls['complete']
    assert set(controls['comparisons'])=={'constant','shuffle','whole_map_donor'}
    for record in controls['comparisons'].values():
        assert record['same_checkpoint'] and record['same_satellite_prefix'] and record['images']==459
        assert record['delta']>0 and record['paired_image_bootstrap95'][0]>0
        assert max(c['max_abs_error'] for c in record['reference_function_calibration'])<1e-12
    count=read(B/'ue_geometry_count_matched_20261024/summary.json')
    assert count['complete'] and count['equal_extra_count_per_image'] and count['total_extra_count']==50838
    assert count['metrics']['geometry']['segm/mAP']==seeds['44']['mask_AP']
    assert count['metrics']['constant']['segm/mAP']==seeds['44']['constant_mask_AP']
    g,k=[read(B/'ue_geometry_count_matched_20261024'/f'{a}_records.json') for a in ['geometry','constant']]
    assert all(a['image_id']==b['image_id'] and len(a['dt']['scores'])==len(b['dt']['scores']) for a,b in zip(g,k))
    new=read(B/'ue_geometry_new_instance_audit_20261024/summary.json')
    assert new['complete'] and new['original_matches_unchanged'] and new['results']['geometry']['groups']['all']['GT']==44988
    assert new['results']['geometry']['additional_TP']['.5']==811 and new['results']['constant']['additional_TP']['.5']==476
    report=dict(passed=True,audit_source_sha256=sha(Path(__file__)),
        scope='Effective direction verified within fixed v1/E73 development protocol, not a final joint model or unseen real-world guarantee',
        old49_sources_unchanged=True,archive_identity_passed=True,all3060_cache_sources_bound=True,
        independent_initialization_and_data_order=True,two_independent_training_seeds=seeds,
        runtime_input_controls=controls['comparisons'],equal_count_control=count['total_extra_count'],actual_new_instance_matches=new['results'],
        limits=['UAV RGB unused; depth/pose-derived geometry only','v1 test repeatedly exposed; bootstrap nominal; same-city spatial correlation',
                'Two seeds are repetitions, not training-variance CI','Priority scores not unified calibrated probabilities',
                'Existing masks unchanged; gains via supplemental instances','Final joint model from common initialization is separate development'])
    (OUT/'audit.json').write_text(json.dumps(report,indent=2))
    print('PASS: sources/data/4 checkpoints/20 paired epochs/2 seeds/4 primary reports/3 input controls/count matching/actual mask matches')


if __name__=='__main__':main()
