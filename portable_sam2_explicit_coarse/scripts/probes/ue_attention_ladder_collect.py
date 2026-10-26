"""Strict actual-artifact audit for all fourteen fixed attention-ladder arms.

L6 low-rank branches live inside the adapter and hook original projections;
the complete original satellite state must therefore remain byte-identical.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import torch
from ue_attention_ladder_schedule import read,sha,dump,arms,FIRST

FROZEN='c6921c30416595fb994facd3d4d927408be9bdd636399fd669cfbe5968329536'
DATA=Path('/data/wangcheng/dataset/CitySample_SatUAV_TrainMinimal_v1_20261021')
METRICS=('MaskAP','BoundaryAP_2pct','BoundaryAP_1pct')


def tensor_digest(state):
    h=hashlib.sha256()
    for key,value in state.items():
        h.update(key.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def paired(base,candidate):
    assert len(base)==len(candidate)==459
    for b,c in zip(base,candidate):
        assert b['image_id']==c['image_id'] and b['gt']==c['gt']
        assert all(b['dt'][k]==c['dt'][k] for k in ('bboxes','scores','labels'))


def audit_training(root,arm,manifest,init):
    folder=root/arm['name'];cp=torch.load(folder/'last.pth',map_location='cpu',weights_only=False)
    assert cp['epoch']==10 and cp['step']==2670
    assert all(k in cp for k in ('adapter','optimizer','scaler','rng_state','contract'))
    assert 'loader_generator' in cp['rng_state']
    c=cp['contract'];assert json.loads(json.dumps(c))==read(folder/'frozen_training_contract.json')
    for key in ('direction','content'):
        assert c[key]==c['args'][key]==arm[key]
    assert c['base_sha256']==manifest['base_sha256']
    assert c['satellite_state_sha256']==FROZEN and c['original_satellite_parameters_frozen']
    assert c['init_file_sha256']==manifest['init_sha256']
    assert c['initial_fingerprint']==init['fingerprint']==tensor_digest(init['adapter'])
    assert c['epochs']==10 and c['updates']==2670 and c['batch']==1 and c['accum']==8
    assert c['epoch_reseed_from']==1 and c['epoch_seed_formula']=='1044000+epoch'
    assert c['initialization_seed']==45 and c['data_seed']==44
    assert c['selection']=='E10 last' and not c['ema']
    assert c['max_dets']==[1,10,250] and c['eval_precision']=='fp32'
    l6=arm['direction']=='L6'
    assert c['decoder_lora']['active']==l6
    assert c['decoder_lora']['rank']==4 and c['decoder_lora']['alpha']==4 and c['decoder_lora']['dropout']==0
    assert c['decoder_lora']['paths']==['transformer.layers.0.cross_attn_token_to_image.q_proj',
                                      'transformer.layers.0.cross_attn_token_to_image.v_proj']
    for name,digest in c['source_hashes'].items():
        assert digest==manifest['source_hashes'][name]==sha(folder/'source'/name)
    rows=[json.loads(line) for line in (folder/'train.jsonl').read_text().splitlines()]
    assert len(rows)==2670 and [x['update'] for x in rows]==list(range(1,2671))
    train_ids={str(x).zfill(6) for x in read(DATA/'splits/train.json')};orders=[]
    for epoch in range(1,11):
        part=rows[(epoch-1)*267:epoch*267]
        assert all(x['epoch']==epoch and x['epoch_seed']==1044000+epoch for x in part)
        assert [x['images'] for x in part]==list(range(8,2137,8))
        summary=read(folder/f'epoch{epoch}_summary.json')
        assert summary['epoch']==epoch and summary['update']==epoch*267 and summary['microbatches']==2136
        assert summary['satellite_state_sha256']==FROZEN
        assert len(summary['sample_order'])==len(set(summary['sample_order']))==2136
        assert set(summary['sample_order'])<=train_ids
        assert summary['sample_order_sha256']==hashlib.sha256(json.dumps(summary['sample_order']).encode()).hexdigest()
        orders.append(summary['sample_order_sha256'])
    for row in rows:
        assert np.isfinite(row['loss']) and np.isfinite(row['grad_norm'])
        lr=1e-4*min(1.,row['update']/50)*(.1+.9*.5*(1+np.cos(np.pi*row['update']/2670)))
        assert abs(row['lr']-lr)<1e-14
    assert set(cp['adapter'])==set(init['adapter'])
    deltas={k:float((v.detach().cpu()-init['adapter'][k]).float().norm()) for k,v in cp['adapter'].items()}
    changes=read(folder/'parameter_changes.json')
    assert changes['satellite_unchanged'] and changes['satellite_state_sha256']==FROZEN
    assert all(abs(value-changes['adapter_changes'][k])<1e-6 for k,value in deltas.items())
    decoder={k:v for k,v in deltas.items() if k.startswith(('decoder_lora_q.','decoder_lora_v.'))}
    assert len(decoder)==4
    if l6:
        assert decoder['decoder_lora_q.B.weight']>0 and decoder['decoder_lora_v.B.weight']>0
    else: assert all(v==0 for v in decoder.values())
    assert any(v>0 for v in deltas.values())
    return dict(checkpoint_sha256=sha(folder/'last.pth'),updates=len(rows),sample_order_epoch_hashes=orders,
        initial_fingerprint=init['fingerprint'],archive_hashes=c['archive_hashes'],adapter_changes=deltas,
        original_satellite_epoch_audits_exact=True,decoder_lora_active=l6,
        frozen_evidence_scope='Actual original satellite state hashed after each epoch; all decoder LoRA tensors are adapter-owned')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);root=p.parse_args().root
    m=read(root/'manifest.json');assert m['arms']==arms()
    assert m['primary_split']=='test' and m['no_test_selection'] and m['selection']=='epoch10 last'
    assert len(m['evaluations'])==42 and len(m['comparisons'])==41
    assert sha(root/'common_init.pth')==m['init_sha256']
    assert sha(root/'cpu_gate.json')==m['cpu_gate_sha256']
    cpu=read(root/'cpu_gate.json')
    assert cpu['status']=='passed' and cpu['device']=='cpu' and len(cpu['gates'])==9
    assert cpu['directions']==['L0','L1','L2','L3','L4','L5','L6']
    assert cpu['source_coordinate_contract'].startswith('grid=(source_uv+2)/256-1;')
    assert sha(root/'existing_cache_audit_identity.json')==m['cache_audit_identity_sha256']
    cache_identity=read(root/'existing_cache_audit_identity.json');cache_points=set()
    for audit in cache_identity['audits']:
        assert sha(audit['path'])==audit['sha256']
        actual=read(audit['path'])
        assert actual['all_passed'] and actual['points']==audit['points']==len(actual['rows'])
        assert actual['version']==audit['version']=='s2db_anchor_metadata_v2_fp64'
        for row in actual['rows']:
            assert row['source_hashes_checked'] and row['legacy_index_geom_exact']
            assert sha(Path(cache_identity['cache'])/(row['point']+'.npz'))==row['cache_sha256']
            cache_points.add(int(row['point']))
    init=torch.load(root/'common_init.pth',map_location='cpu',weights_only=False)
    assert init['seed']==45 and init['fingerprint']==tensor_digest(init['adapter'])
    for name,digest in m['source_hashes'].items():assert sha(root/'frozen_source'/name)==digest
    ids={s:set(read(DATA/'splits'/(s+'.json'))) for s in ('train','val','test')}
    assert len(ids['train'])==2142 and len(ids['val'])==len(ids['test'])==459
    assert not ids['train']&ids['test'] and not ids['train']&ids['val'] and not ids['val']&ids['test']
    assert cache_points==ids['train']|ids['val']|ids['test']
    for split in ids:assert sha(DATA/'splits'/(split+'.json'))==m['split_sha256'][split]
    state=read(root/'state.json');assert state['stage'] in ('running','complete')
    jobs=state['jobs']
    for key,job in jobs.items():
        if key=='collect':continue
        assert job['status']=='complete' and job['returncode']==0 and Path(job['artifact']).is_file()
        if not job['gpu_required']:continue
        assert job['gpu'] in (1,2,3)
        if job.get('resource_measurement_job'):
            actual=read(job['artifact'])
            assert job['admission']['required_free_mib']==(12288 if job['direction']=='L6' else 8192)
            assert actual['resource_rois']==250 and actual['resource_chunk']==32
            assert actual['satellite_parameters_buffers_exact'] and actual['satellite_no_grad']
            assert actual['satellite_state_sha256']==FROZEN
            assert all(m['source_hashes'][n]==h for n,h in actual['source_hashes'].items())
            continue
        gate=job['admission']['gate'];g=gate['configuration'];actual=read(g['gate_path'])
        assert sha(g['gate_path'])==g['gate_sha256'] and g['all_passed']
        assert g['fp32_resource_rois']==250 and actual['resource_rois']==250 and actual['resource_chunk']==32
        assert actual['satellite_parameters_buffers_exact'] and actual['satellite_no_grad']
        assert actual['satellite_state_sha256']==FROZEN
        measured=max(actual['training_amp_peak']['reserved_MiB'],actual['fp32_250roi_peak']['reserved_MiB'])
        assert g['peak_reserved_mib']>=measured
        assert all(m['source_hashes'][n]==h for n,h in actual['source_hashes'].items())
        for report in actual['identity'].values():assert report['zero_alpha_logits_exact'] and report['no_support_nonzero_alpha_logits_exact']
    for gpu,first in FIRST.items():
        gpu_jobs=[j for j in jobs.values() if j.get('gpu')==gpu]
        assert min(gpu_jobs,key=lambda j:j['started_utc'])['key']==first
    audits={a['name']:audit_training(root,a,m,init) for a in m['arms']}
    assert len({tuple(x['sample_order_epoch_hashes']) for x in audits.values()})==1
    assert len({json.dumps(x['archive_hashes'],sort_keys=True) for x in audits.values()})==1
    base_dir=root/'test/baseline/disabled';base=read(base_dir/'records.json');bi=read(base_dir/'evaluation_identity.json')
    assert bi['split']=='test' and bi['mode']=='disabled' and bi['adapter_sha256'] is None
    assert bi['base_sha256']==m['base_sha256'] and bi['init_sha256']==m['init_sha256']
    assert bi['images']==len(base)==459 and {r['image_id'] for r in base}==ids['test']
    assert bi['max_dets']==[1,10,250] and bi['score_key']=='scores' and bi['inference_precision']=='fp32'
    bm=read(base_dir/'metrics.json');assert abs(bm['segm/mAP']-.6455098296975914)<1e-12
    assert abs(bm['bbox/mAP']-.6854287640342888)<1e-12
    evaluations={};val_base=None
    for ev in m['evaluations']:
        folder=Path(ev['folder']);i=read(folder/'evaluation_identity.json');records=read(folder/'records.json')
        assert i['split']==ev['split'] and i['mode']==ev['mode'] and i['images']==len(records)==459
        assert {r['image_id'] for r in records}==ids[ev['split']]
        arm=next(a for a in m['arms'] if a['name']==ev['arm'])
        assert i['direction']==arm['direction'] and i['training_content']==arm['content']
        assert i['adapter_sha256']==audits[ev['arm']]['checkpoint_sha256']
        assert i['base_sha256']==m['base_sha256'] and i['init_sha256']==m['init_sha256']
        assert i['max_dets']==[1,10,250] and i['score_key']=='scores' and i['inference_precision']=='fp32'
        assert all(m['source_hashes'][n]==h for n,h in i['source_hashes'].items())
        if ev['split']=='test':paired(base,records)
        else:
            if val_base is None:val_base=records
            paired(val_base,records)
        metrics=read(folder/'metrics.json')
        if ev['split']=='test':assert abs(metrics['bbox/mAP']-bm['bbox/mAP'])<1e-12
        evaluations[ev['key']]=dict(identity=i,metrics=metrics,records_sha256=sha(folder/'records.json'))
    comparisons={};points={}
    for comp in m['comparisons']:
        result=read(comp['output'])
        assert result['evaluation_split']=='test' and result['images']==459 and result['detector_and_gt_exact']
        assert result['analysis_sha256']==m['source_hashes']['scripts/probes/ue_v1_adapter_analysis.py']
        paired(read(comp['baseline']),read(comp['candidate']))
        for metric in METRICS:
            v=result['metrics'][metric]
            assert v['bootstrap_samples']==500 and v['seed']==20261021
            assert abs(v['delta']-(v['candidate']-v['baseline']))<1e-12
            assert len(v['paired_image_bootstrap95'])==2 and np.isfinite(v['paired_image_bootstrap95']).all()
            for side in ('baseline','candidate'):
                key=(comp[side],metric)
                if key in points:assert abs(points[key]-v[side])<1e-12
                else:points[key]=v[side]
        for side in ('baseline','candidate'):
            assert abs(result['metrics']['MaskAP'][side]-read(Path(comp[side]).parent/'metrics.json')['segm/mAP'])<1e-12
        comparisons[comp['key']]=dict(kind=comp['kind'],result=result,
            baseline_records_sha256=sha(comp['baseline']),candidate_records_sha256=sha(comp['candidate']))
    caveats=['single training seed; image bootstrap excludes training variance and spatial dependence',
        'nominal intervals for multiple exploratory comparisons, not multiplicity corrected',
        'v1 test already used for mechanism development; not a fresh final benchmark',
        'constant RGB preserves target depth, poses, candidates and UAV support',
        'donor RGB changes appearance while retaining target geometry; not a coherent alternate RGB-D sample',
        'L2 adds output projection capacity and L6 adds decoder adaptation; matched training content comparisons are essential',
        'all directions are single-factor changes from this fresh common L0, not combined changes']
    dump(root/'complete_audit_summary.json',dict(all_complete=True,manifest=m,training_audits=audits,
        evaluations=evaluations,comparisons=comparisons,caveats=caveats))
    lines=['# Fixed L0–L6 attention ladder: complete audit','','All 14 E10 arms, 29 test evaluations, 14 val diagnostics and 41 comparisons audited.','',
        '|Direction|Correct Mask AP|Constant-trained Mask AP|Content delta (pp)|Nominal 95% interval (pp)|',
        '|---|---:|---:|---:|---|']
    for n in range(7):
        d=f'L{n}';v=comparisons['content_'+d]['result']['metrics']['MaskAP']
        lines.append(f"|{d}|{100*v['candidate']:.4f}|{100*v['baseline']:.4f}|{100*v['delta']:+.4f}|{[100*x for x in v['paired_image_bootstrap95']]}|")
    lines+=['','A positive net change alone does not establish UAV content contribution.','', '## Limits','']+['- '+x for x in caveats]
    (root/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n')
    print('Complete audit: 14 E10 checkpoints, 43 evaluations including baseline, 41 comparisons.')


if __name__=='__main__':main()
