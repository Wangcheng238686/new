"""Audit the complete, predeclared test matrix against weights and prediction records."""
import argparse
import json
from pathlib import Path
from ue_v1_adapter_collect import audit_training,normalized_metrics,read,sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    r=parser.parse_args().root;t=r/'test';state=read(t/'test_state.json');plan=read(t/'frozen_test_plan.json')
    assert state['stage']=='analysis_complete' and state['plan']==plan
    assert plan['split']=='test' and plan['selection']=='epoch10 last'
    assert plan['evaluation_count']==19 and plan['comparison_count']==15 and len(plan['arms'])==10
    assert len(state['evaluations'])==19 and len(state['analyses'])==15
    assert all(v['stage']=='complete' for v in state['evaluations']+state['analyses'])
    assert read(r/'test_geometry_cache_audit.json')['checked_points']==459
    assert read(r/'test_native_cache_audit.json')['points']==459
    base=read(t/'baseline/disabled/records.json');bm=read(t/'baseline/disabled/metrics.json')
    base_identity=read(t/'baseline/disabled/evaluation_identity.json')
    sealed=read(r/'correct/frozen_training_contract.json')
    data=Path(sealed['data_root']);ids=read(data/'splits/test.json')
    assert len(base)==len(ids)==459 and {v['image_id'] for v in base}==set(ids)
    assert not set(ids)&set(read(data/'splits/train.json'))
    assert not set(ids)&set(read(data/'splits/val.json'))
    assert sha(Path(sealed['base']))==sealed['base_sha256']==base_identity['base_sha256']
    assert base_identity['mode']=='disabled' and base_identity['adapter_sha256'] is None
    summary=dict(primary_split='test',selection='epoch10 last',plan=plan,
                 images=len(base),gt_instances=sum(len(v['gt']['bboxes']) for v in base),
                 baseline=normalized_metrics(bm),arms={},content_comparisons={},all_complete=True,
                 caveats=['single training seed; image bootstrap excludes training variance and spatial dependence',
                          'multiple exploratory routes; nominal per-comparison intervals',
                          'test designated primary by user before new probe test results; no test-based retraining or epoch selection'])
    for arm in plan['arms']:
        name=arm['name'];out=r/arm['folder'];c,h=audit_training(out,sealed['base_sha256'],sealed['archive_hashes'])
        expected_route='early' if name in ('early','early_control') else 'highres'
        expected_source='contextual' if name in ('contextual','contextual_control') else ('unit' if name in ('unit','unit_geometry') else 'fine')
        expected_appearance='constant' if name in ('geometry','unit_geometry','early_control','contextual_control') else 'correct'
        assert c['args']['route']==expected_route and c['args']['source_mode']==expected_source
        assert c['args']['appearance']==expected_appearance
        if arm['family']=='nativegeo':
            assert not c['rgb_encoder_used']
            assert sha(out/'ue_v1_native_geometry.py.source')==c['native_runner_sha256']
            assert sha(out/'native_geometry_probe.py.source')==c['native_source_sha256']
        cp_sha=sha(out/'last.pth');metrics={};identities={}
        for mode in arm['modes']:
            p=t/name/mode;identity=read(p/'evaluation_identity.json');rows=read(p/'records.json')
            assert identity['split']=='test' and identity['images']==len(rows)==459
            assert identity['base_sha256']==sealed['base_sha256'] and identity['adapter_sha256']==cp_sha
            assert identity['mode']==mode and identity['score_key']=='scores'
            assert identity['max_dets']==[1,10,250] and identity['inference_precision']=='fp32'
            for b,d in zip(base,rows):
                assert b['image_id']==d['image_id'] and b['gt']==d['gt']
                assert all(b['dt'][k]==d['dt'][k] for k in ('bboxes','scores','labels'))
            metrics[mode]=normalized_metrics(read(p/'metrics.json'));identities[mode]=identity
        result=read(t/name/'paired_analysis.json')
        assert result['evaluation_split']=='test' and result['analysis_sha256']==plan['analysis_sha256']
        assert result['images']==459 and result['detector_and_gt_exact'] and result['selection']=='last'
        assert all(v['bootstrap_samples']==500 and v['seed']==20261021 for v in result['metrics'].values())
        assert abs(result['metrics']['MaskAP']['baseline']-bm['segm/mAP'])<1e-12
        assert abs(result['metrics']['MaskAP']['candidate']-metrics['correct']['segm/mAP'])<1e-12
        assert metrics['correct']['bbox/mAP']==bm['bbox/mAP']
        summary['arms'][name]=dict(config=c['args'],training_updates=len(h),last_checkpoint_sha256=cp_sha,
            satellite_state_sha256=c['satellite_state_sha256'],source_snapshots_match=True,
            detector_and_gt_unchanged_all_modes=True,metrics=metrics,identities=identities,paired_analysis=result)
    pairs=[('correct','geometry'),('early','early_control'),('contextual','contextual_control'),
           ('unit','unit_geometry'),('native_geometry','native_zero')]
    for first,second in pairs:
        result=read(t/first/'content_paired_analysis.json')
        assert result['evaluation_split']=='test' and result['analysis_sha256']==plan['analysis_sha256']
        assert result['images']==459 and result['detector_and_gt_exact']
        for key,values in result['metrics'].items():
            assert abs(values['baseline']-summary['arms'][second]['paired_analysis']['metrics'][key]['candidate'])<1e-12
            assert abs(values['candidate']-summary['arms'][first]['paired_analysis']['metrics'][key]['candidate'])<1e-12
            assert values['bootstrap_samples']==500 and values['seed']==20261021
        summary['content_comparisons'][first+' minus '+second]=result
    # Every comparison must independently reproduce the same baseline boundary APs.
    reference=summary['arms']['correct']['paired_analysis']['metrics']
    for arm in summary['arms'].values():
        for key,values in arm['paired_analysis']['metrics'].items():
            assert abs(values['baseline']-reference[key]['baseline'])<1e-12
    p=r/'complete_test_audit_summary.json';tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(summary,indent=2)+'\n');tmp.replace(p)
    print(json.dumps({name:dict(AP=arm['metrics']['correct']['segm/mAP'],
        delta=arm['paired_analysis']['metrics']['MaskAP']['delta']) for name,arm in summary['arms'].items()},indent=2))


if __name__=='__main__':main()
