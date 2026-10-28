"""Collect completed parallel probes with explicit invariant/coverage audits."""
import argparse
import hashlib
import json
from pathlib import Path

import torch


def read(p):return json.loads(p.read_text())


def normalized_metrics(values):
    # Canonical evaluator retains AR@100 as its historical name even at a custom cap.
    return {k.replace('AR@100','AR@250'):v for k,v in values.items()}


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def audit_recovery(out,contract,history):
    info=contract.get('recovery')
    assert info,'Recovered experiment must retain its parent contract'
    assert sha(Path(info['parent_checkpoint']))==info['parent_sha256']
    prefix=info['parent_step'];epoch=info['parent_epoch']
    for row in history[prefix:]:
        assert row['recovery_parent_epoch']==epoch
        assert row['epoch_seed']==1000000+contract['args']['seed']*1000+row['epoch']
    assert read(out/'parameter_changes.json')['satellite_state_sha256']==contract['satellite_state_sha256']


def audit_training(out,base_sha,archive_hashes):
    """Inspect actual terminal weights and every training row, including fresh controls."""
    contract=read(out/'frozen_training_contract.json')
    history=[json.loads(line) for line in (out/'train.jsonl').read_text().splitlines()]
    assert [v['update'] for v in history]==list(range(1,2671))
    assert all(sum(v['epoch']==e for v in history)==267 for e in range(1,11))
    assert contract['base_sha256']==base_sha and contract['archive_hashes']==archive_hashes
    assert contract['satellite_frozen'] and not contract['ema']
    assert contract['checkpoint_selection']=='last' and contract['effective_batch']==8
    changes=read(out/'parameter_changes.json')
    assert changes['satellite_unchanged']
    expected='c6921c30416595fb994facd3d4d927408be9bdd636399fd669cfbe5968329536'
    assert changes['satellite_state_sha256']==contract['satellite_state_sha256']==expected
    for filename,key in [('ue_v1_adapter.source.py','script_sha256'),
                         ('local_adapter_probe.source.py','adapter_sha256'),
                         ('local_adapter_variants.source.py','variants_sha256')]:
        assert sha(out/filename)==contract[key]
    # A terminal state file alone is insufficient: open the trained checkpoint.
    checkpoint=torch.load(out/'last.pth',map_location='cpu',weights_only=False)
    assert checkpoint['epoch']==10 and checkpoint['step']==2670
    # JSON snapshots convert configuration tuples to lists.
    assert json.loads(json.dumps(checkpoint['contract']))==contract and checkpoint['adapter']
    del checkpoint
    if contract.get('recovery'):audit_recovery(out,contract,history)
    else:
        reseed=contract['args'].get('reseed_from_epoch',0)
        for row in history:
            expected_seed=1000000+contract['args']['seed']*1000+row['epoch'] if reseed and row['epoch']>=reseed else None
            assert row['epoch_seed']==expected_seed and row['recovery_parent_epoch'] is None
    return contract,history


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
    state=read(r/'run_state.json')
    assert state['stage']=='analysis_complete','Complete all authorized arms before collecting final results'
    assert read(r/'contextual_control_state.json')['stage']=='analysis_complete','Test-primary amendment requires the contextual control'
    assert read(r/'test/test_state.json')['stage']=='analysis_complete','Primary test matrix must finish before final collection'
    base=read(r/'evaluation/disabled/records.json')
    bm=read(r/'evaluation/disabled/metrics.json')
    assert len(base)==459
    identity=read(r/'correct/frozen_training_contract.json')
    assert sha(Path(identity['base']))==identity['base_sha256']
    summary=dict(baseline=normalized_metrics(bm),arms={},all_complete=True,scope='v1 validation diagnostics; test is primary, see complete_test_audit_summary.json',
        caveats=['single training seed','image bootstrap does not cover spatial dependence or training variance',
                 'multiple exploratory routes; nominal per-comparison intervals'])
    for arm in state['arms']:
        name=arm['name'];out=r/name
        audit_training(out,identity['base_sha256'],identity['archive_hashes'])
        log=[json.loads(line) for line in (out/'train.jsonl').read_text().splitlines()]
        assert len(log)==2670 and [x['update'] for x in log]==list(range(1,2671))
        assert log[-1]['epoch']==10
        assert all(sum(x['epoch']==e for x in log)==267 for e in range(1,11))
        assert read(out/'parameter_changes.json')['satellite_unchanged']
        contract=read(out/'frozen_training_contract.json')
        audit_recovery(out,contract,log)
        assert sha(out/'ue_v1_adapter.source.py')==contract['script_sha256']
        assert sha(out/'local_adapter_probe.source.py')==contract['adapter_sha256']
        if 'variants_sha256' in contract:
            assert sha(out/'local_adapter_variants.source.py')==contract['variants_sha256']
        modes=['correct'] if arm['appearance']=='constant' else ['correct','constant','donor']
        metrics={}
        for mode in modes:
            rows=read(out/f'evaluation/{mode}/records.json')
            assert len(rows)==len(base)
            for b,c in zip(base,rows):
                assert b['image_id']==c['image_id'] and b['gt']==c['gt']
                assert all(b['dt'][k]==c['dt'][k] for k in ('bboxes','scores','labels'))
            metrics[mode]=normalized_metrics(read(out/f'evaluation/{mode}/metrics.json'))
        result=read(out/'paired_analysis.json')
        assert abs(result['metrics']['MaskAP']['baseline']-bm['segm/mAP'])<1e-12
        assert abs(result['metrics']['MaskAP']['candidate']-metrics['correct']['segm/mAP'])<1e-12
        last=log[-1]['latest_forward']
        summary['arms'][name]=dict(config={k:contract['args'].get(k,default) for k,default in
            [('route','highres'),('source_mode','fine'),('appearance','correct')]},
            training_updates=len(log),detector_gt_unchanged_all_modes=True,source_snapshot_match=True,
            final_checkpoint_sha256=sha(out/'last.pth'),metrics=metrics,paired_analysis=result,
            last_forward=last,applied_to_base_rms_ratio=last['applied_delta_rms']/max(last['base_rms'],1e-12))
        appearance=out/'appearance_paired_analysis.json'
        if appearance.is_file():summary['arms'][name]['appearance_analysis']=read(appearance)
        needs_control=name in ('early','contextual')
        if needs_control:
            ctrl,control_history=audit_training(out/'control',identity['base_sha256'],identity['archive_hashes'])
            assert ctrl['args']['appearance']=='constant' and ctrl['args']['reseed_from_epoch']==5
            for key in ('route','source_mode'):
                assert ctrl['args'].get(key,{'route':'highres','source_mode':'fine'}[key])==contract['args'].get(key,{'route':'highres','source_mode':'fine'}[key])
            rows=read(out/'control_evaluation/correct/records.json');assert len(rows)==459
            for b,d in zip(base,rows):
                assert b['image_id']==d['image_id'] and b['gt']==d['gt']
                assert all(b['dt'][k]==d['dt'][k] for k in ('bboxes','scores','labels'))
            assert appearance.is_file()
            comparison=read(appearance)['metrics']['MaskAP']
            assert abs(comparison['baseline']-read(out/'control_evaluation/correct/metrics.json')['segm/mAP'])<1e-12
            assert abs(comparison['candidate']-metrics['correct']['segm/mAP'])<1e-12
            summary['arms'][name]['matched_constant_training_complete']=True
    native_state=read(r/'native_state.json')
    assert native_state['stage']=='analysis_complete','Native geometry pair is also part of this authorized probe'
    for arm in native_state['arms']:
        name=arm['name'];out=r/name
        audit_training(out,identity['base_sha256'],identity['archive_hashes'])
        history=[json.loads(line) for line in (out/'train.jsonl').read_text().splitlines()]
        assert len(history)==2670 and [x['update'] for x in history]==list(range(1,2671))
        assert history[-1]['epoch']==10
        assert read(out/'parameter_changes.json')['satellite_unchanged']
        c=read(out/'frozen_training_contract.json')
        audit_recovery(out,c,history)
        assert not c['rgb_encoder_used'] and c['probe_family']=='native_geometry'
        assert sha(out/'ue_v1_native_geometry.py.source')==c['native_runner_sha256']
        assert sha(out/'native_geometry_probe.py.source')==c['native_source_sha256']
        rows=read(out/'evaluation/correct/records.json');assert len(rows)==len(base)
        for b,d in zip(base,rows):
            assert b['image_id']==d['image_id'] and b['gt']==d['gt']
            assert all(b['dt'][k]==d['dt'][k] for k in ('bboxes','scores','labels'))
        result=read(out/'paired_analysis.json')
        metrics=normalized_metrics(read(out/'evaluation/correct/metrics.json'))
        assert abs(result['metrics']['MaskAP']['baseline']-bm['segm/mAP'])<1e-12
        assert abs(result['metrics']['MaskAP']['candidate']-metrics['segm/mAP'])<1e-12
        summary['arms'][name]=dict(config=dict(family='native_geometry',content=c['content']),
            training_updates=2670,detector_gt_unchanged_all_modes=True,source_snapshot_match=True,
            final_checkpoint_sha256=sha(out/'last.pth'),metrics={'correct':metrics},paired_analysis=result,
            trainable_parameters=c['trainable_parameters'],last_forward=history[-1]['latest_forward'])
    summary['native_content_analysis']=read(r/'native_geometry/content_paired_analysis.json')
    unit_state=read(r/'unit_state.json')
    assert unit_state['stage']=='analysis_complete','Unit-normalized pair is part of the fixed final matrix'
    for arm in unit_state['arms']:
        name=arm['name'];out=r/name
        audit_training(out,identity['base_sha256'],identity['archive_hashes'])
        history=[json.loads(line) for line in (out/'train.jsonl').read_text().splitlines()]
        assert len(history)==2670 and [v['update'] for v in history]==list(range(1,2671))
        assert history[-1]['epoch']==10 and all(v['latest_forward']['source_delta_rms']<=1.01 for v in history)
        c=read(out/'frozen_training_contract.json');assert c['args']['source_mode']=='unit' and c['args']['reseed_from_epoch']==5
        assert read(out/'parameter_changes.json')['satellite_state_sha256']==c['satellite_state_sha256']
        for v in history:
            assert v['epoch_seed']==(1000000+44*1000+v['epoch'] if v['epoch']>=5 else None)
        assert sha(out/'ue_v1_adapter.source.py')==c['script_sha256']
        assert sha(out/'local_adapter_variants.source.py')==c['variants_sha256']
        metrics={}
        for mode in (['correct','constant','donor'] if arm['appearance']=='correct' else ['correct']):
            rows=read(out/f'evaluation/{mode}/records.json');assert len(rows)==len(base)
            for b,d in zip(base,rows):
                assert b['image_id']==d['image_id'] and b['gt']==d['gt']
                assert all(b['dt'][k]==d['dt'][k] for k in ('bboxes','scores','labels'))
            metrics[mode]=normalized_metrics(read(out/f'evaluation/{mode}/metrics.json'))
        result=read(out/'paired_analysis.json')
        assert abs(result['metrics']['MaskAP']['baseline']-bm['segm/mAP'])<1e-12
        assert abs(result['metrics']['MaskAP']['candidate']-metrics['correct']['segm/mAP'])<1e-12
        summary['arms'][name]=dict(config=dict(source_mode='unit',appearance=arm['appearance']),
            training_updates=2670,source_snapshot_match=True,detector_gt_unchanged_all_modes=True,
            final_checkpoint_sha256=sha(out/'last.pth'),metrics=metrics,paired_analysis=result,last_forward=history[-1]['latest_forward'])
    summary['unit_appearance_analysis']=read(r/'unit/appearance_paired_analysis.json')
    summary['interim']={}
    for filename,expected in [('interim_state.json',8),('native_interim_state.json',4),('unit_interim_state.json',4)]:
        observations=read(r/filename)
        assert len(observations)==expected and all(v['stage']=='complete' for v in observations.values())
        summary['interim'][filename]=observations
        for label,obs in observations.items():
            name,epoch=label.rsplit('_e',1)
            rows=read(r/name/f'interim_epoch{epoch}/correct/records.json')
            assert len(rows)==len(base)
            for b,d in zip(base,rows):
                assert b['image_id']==d['image_id'] and b['gt']==d['gt']
                assert all(b['dt'][k]==d['dt'][k] for k in ('bboxes','scores','labels'))
    (r/'complete_audit_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:dict(AP=v['metrics']['correct']['segm/mAP'],delta=v['paired_analysis']['metrics']['MaskAP']['delta'])
        for k,v in summary['arms'].items()},indent=2))


if __name__=='__main__':main()
