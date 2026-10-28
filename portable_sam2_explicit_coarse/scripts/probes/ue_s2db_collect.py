"""Evidence-based stage review and final audit; no test-based model choice."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import torch
from ue_v1_adapter import dump,file_sha
from ue_s2db_analysis import assert_paired

FROZEN='c6921c30416595fb994facd3d4d927408be9bdd636399fd669cfbe5968329536'


def read(p):return json.loads(Path(p).read_text())


def audit_training(root,arm,manifest):
    out=root/arm['name'];cp=torch.load(out/'last.pth',map_location='cpu',weights_only=False)
    assert cp['epoch']==10 and cp['step']==2670
    c=cp['contract'];del cp
    assert json.loads(json.dumps(c))==read(out/'frozen_training_contract.json')
    assert c['base_sha256']==manifest['base_sha256'] and c['satellite_state_sha256']==FROZEN
    assert c['init_file_sha256']==file_sha(root/'common_init.pth')
    assert c['initial_fingerprint']==read(root/'initialization.json')['fingerprint']
    for name in ('read','fusion','content','aux_weight'):assert c['args'][name]==arm[name]
    for path,digest in c['source_hashes'].items():
        assert digest==manifest['source_hashes'][path] and file_sha(out/'source'/path)==digest
    rows=[json.loads(s) for s in (out/'train.jsonl').read_text().splitlines()]
    assert len(rows)==2670 and [v['update'] for v in rows]==list(range(1,2671))
    summaries=[]
    for epoch in range(1,11):
        part=rows[(epoch-1)*267:epoch*267]
        assert all(v['epoch']==epoch for v in part)
        assert [v['images'] for v in part]==list(range(8,2137,8))
        assert all(v['epoch_seed']==(1044000+epoch if epoch>=5 else None) for v in part)
        assert all(np.isfinite(v['loss']) and np.isfinite(v['grad_norm']) for v in part)
        e=read(out/f'epoch{epoch}_summary.json')
        assert e['satellite_state_sha256']==FROZEN and e['microbatches']==2136
        assert e['sample_order_sha256']==hashlib.sha256(json.dumps(e['sample_order']).encode()).hexdigest()
        assert len(e['sample_order'])==len(set(e['sample_order']))==2136
        summaries.append(e['sample_order_sha256'])
    changes=read(out/'parameter_changes.json')
    assert changes['satellite_unchanged'] and changes['satellite_state_sha256']==FROZEN
    for key,value in changes['adapter_changes'].items():
        if key.startswith('encoder.') and 'lora_' not in key:assert value==0,key
        if key.startswith('offset.') and arm['read']=='fixed':assert value==0,key
        if key.startswith('joint.') and arm['fusion']=='normalized':assert value==0,key
        if key.startswith('output.') and arm['fusion']=='concat':assert value==0,key
        if key.startswith('aux_head.') and arm['aux_weight']==0:assert value==0,key
    assert abs(rows[-1]['lr']-1e-5)<1e-14
    return dict(checkpoint_sha256=file_sha(out/'last.pth'),epochs=10,updates=len(rows),
        initial_fingerprint=c['initial_fingerprint'],source_snapshots_match=True,frozen_satellite_sha256=FROZEN,
        sample_order_epoch_hashes=summaries)


def stage1(root,manifest):
    arms=[a for a in manifest['arms'] if a['stage']==1];assert len(arms)==10
    audits={a['name']:audit_training(root,a,manifest) for a in arms}
    assert len({tuple(a['sample_order_epoch_hashes']) for a in audits.values()})==1
    summaries={a['name']:read(root/'analysis'/a['name']/'metrics.json') for a in arms}
    effects={}
    vectors={a['name']:dict(np.load(root/'analysis'/a['name']/'bootstrap.npz')) for a in arms}
    for title,coeff in {'read_B-A':{'B':1,'A':-1},'read_D-C':{'D':1,'C':-1},
        'fusion_C-A':{'C':1,'A':-1},'fusion_D-B':{'D':1,'B':-1},'interaction':{'D':1,'C':-1,'B':-1,'A':1}}.items():
        effects[title]={}
        for metric in ('MaskAP','BoundaryAP_2pct','BoundaryAP_1pct'):
            point=sum(c*summaries[n]['metrics'][metric]['AP'] for n,c in coeff.items())
            samples=sum(c*vectors[n][metric] for n,c in coeff.items())
            effects[title][metric]=dict(delta=point,paired_image_bootstrap95=np.quantile(samples,[.025,.975]).tolist())
    mechanism={a['name']:read(root/'analysis'/a['name']/'mechanisms.json') for a in arms}
    # Review produces diagnostics. D and auxiliary weight were fixed beforehand.
    dump(root/'stage1_mechanism_review.json',dict(stage1_complete=True,audits=audits,metrics=summaries,factorial_effects=effects,
        mechanism_summary={n:dict(zones=v['zones'],groups=v['groups'],unmatched_gt=v['unmatched_gt'],unmatched_dt=v['unmatched_dt']) for n,v in mechanism.items()},
        next_structure='D',aux_weight=.1,selection_from_test=False,
        reason='D fixed in manifest before this cohort test evaluation'))


def final(root,manifest):
    for path,digest in manifest['source_hashes'].items():assert file_sha(root/'frozen_source'/path)==digest
    cpu=read(root/'cpu_gate.json');assert all(cpu[k] for k in ('actual_train_augmentation_resume_prefix_exact','zero_offset_gather_exact','invalid_depth_fallback',
        'bounds_fallback','valid_depth_jump_retained','unsupported_concat_and_normalized_exact','dataloader_resume_order_and_worker_augmentation_exact'))
    for path,digest in cpu['source_hashes'].items():assert manifest['source_hashes'][path]==digest
    for name in ('A','B','C','D','D_aux'):
        g=read(root/'gates'/name/'gpu_gate.json')
        assert all(g['zero_alpha_logits_exact'].values()) and g['zero_offset_gpu_exact']
        assert g['no_support_nonzero_alpha_logits_exact'] and g['satellite_frozen_parameters_buffers_exact']
        assert g['fp32_resource_rois']==250 and g['lora_offset_fusion_gradient_reachable']
        assert g['all_rng_streams_replay_exact']
        for path,digest in g['source_hashes'].items():assert manifest['source_hashes'][path]==digest
    assert read(root/'gates/D_aux/gpu_gate.json')['aux_query_gradient_blocked']
    for split,count in [('trainval',2601),('test',459)]:
        a=read(root/f'cache_audit_{split}.json');assert a['all_passed'] and a['points']==count
    baseline=read(root/'test/baseline/disabled/records.json');assert len(baseline)==459
    dataset=Path('/data/wangcheng/dataset/CitySample_SatUAV_TrainMinimal_v1_20261021')
    ids={s:set(read(dataset/f'splits/{s}.json')) for s in ('train','val','test')}
    assert len(ids['train'])==2142 and len(ids['val'])==len(ids['test'])==459
    assert not ids['train']&ids['val'] and not ids['train']&ids['test'] and not ids['val']&ids['test']
    assert {r['image_id'] for r in baseline}==ids['test']
    baseline_metrics=read(root/'analysis/baseline/metrics.json');allarms={}
    for arm in manifest['arms']:
        name=arm['name'];audit=audit_training(root,arm,manifest)
        metrics=read(root/'analysis'/name/'metrics.json');assert metrics['bootstrap_samples']==500
        with np.load(root/'analysis'/name/'bootstrap.npz') as a:assert all(len(a[k])==500 for k in a)
        primary='constant' if arm['content']=='constant' else 'correct';interventions={};zero_reference=None
        for mode in ('correct','constant','donor'):
            p=root/'test'/name/mode
            identity=read(p/'evaluation_identity.json');records=read(p/'records.json')
            assert identity['split']=='test' and identity['images']==459 and identity['mode']==mode
            assert identity['adapter_sha256']==audit['checkpoint_sha256'] and identity['base_sha256']==manifest['base_sha256']
            assert identity['inference_precision']=='fp32' and identity['max_dets']==[1,10,250]
            assert_paired(baseline,records)
            if arm['content']=='zero':
                current=[r['dt']['rles'] for r in records]
                if zero_reference is None:zero_reference=current
                else:assert current==zero_reference,'zero pooled control leaked RGB'
            assert set(identity['source_hashes']).issubset(manifest['source_hashes'])
            assert all(manifest['source_hashes'][k]==v for k,v in identity['source_hashes'].items())
            im=metrics if mode==primary else read(root/'analysis'/name/mode/'metrics.json')
            assert im['records_sha256']==file_sha(p/'records.json')
            interventions[mode]=im['metrics']
        vi=read(root/'val'/name/primary/'evaluation_identity.json')
        assert vi['split']=='val' and vi['images']==459 and vi['adapter_sha256']==audit['checkpoint_sha256']
        mech=read(root/'analysis'/name/'mechanisms.json')
        assert mech['baseline_sha256']==file_sha(root/'test/baseline/disabled/records.json')
        if arm['aux_weight']:
            for epoch in (1,3,10):
                g=read(root/'offline'/name/f'E{epoch}/gradients.json');assert g['epoch']==epoch and g['model_state_unchanged']
                assert len(g['samples'])==4 and g['checkpoint_sha256']==file_sha(root/name/f'epoch{epoch}.pth')
            aux=read(root/'offline'/name/'test/aux_iou.json');assert aux['split']=='test' and aux['images']==459
        allarms[name]=dict(config=arm,audit=audit,metrics=metrics,interventions=interventions,
            primary_input=primary,mechanisms=dict(zones=mech['zones'],groups=mech['groups'],unmatched_gt=mech['unmatched_gt'],unmatched_dt=mech['unmatched_dt']),
            content_changes={mode:read(root/'analysis'/name/(mode+'_prediction_changes.json')) for mode in ('constant','donor')})
    assert len(allarms)==13
    assert len({tuple(a['audit']['sample_order_epoch_hashes']) for a in allarms.values()})==1
    contrasts=read(root/'contrasts.json');stage=read(root/'stage1_mechanism_review.json')
    assert stage['next_structure']=='D' and not stage['selection_from_test']
    summary=dict(all_complete=True,manifest=manifest,baseline=baseline_metrics,arms=allarms,contrasts=contrasts,
        evidence_levels=dict(wiring='engineering gates passed; all weights and buffers audited',
            content_dependency='see fixed-checkpoint correct/constant/donor metrics and actual prediction changes',
            net_gain='see correct-content vs baseline and matched content/capacity contrasts; no proxy substitutes'),
        caveats=contrasts['limitations'])
    dump(root/'complete_audit_summary.json',summary)
    lines=['# S²DB机制探针：固定E10完整结论','',
        '13个独立训练、全部固定干预、来源与冻结审计已完成。test为主要评价；val仅诊断。','',
        '|臂|主输入|Mask AP250|ΔAP|名义95% CI|Boundary AP2%|Boundary AP1%|',
        '|---|---|---:|---:|---|---:|---:|']
    for name,a in allarms.items():
        m=a['metrics']['metrics'];d=contrasts['contrasts'][name+'-baseline']['metrics']['MaskAP']
        lines.append(f"|{name}|{a['primary_input']}|{m['MaskAP']['AP']:.6f}|{d['delta']:+.6f}|{d['paired_image_bootstrap95']}|{m['BoundaryAP_2pct']['AP']:.6f}|{m['BoundaryAP_1pct']['AP']:.6f}|")
    lines+=['','## 机制效应','']
    for key in ('read_B-A','read_D-C','fusion_C-A','fusion_D-B','interaction','aux_correct','aux_constant','aux_zero'):
        m=contrasts['contrasts'][key]['metrics']['MaskAP'];lines.append(f"- {key}：Mask AP {m['delta']:+.6f}，名义95% CI {m['paired_image_bootstrap95']}。")
    passed=[n for n in 'ABCD' if contrasts['contrasts'][n+'-baseline']['clear_gain_reference']]
    lines+=['','明确净收益参考门槛通过主臂：'+(', '.join(passed) if passed else '无')+'。',
        '', '读取效应只检验保留锚点邻域，不能排除无锚点/远距离遗漏。融合效应包含联合内容与卷积容量；真实UAV贡献须同时检查匹配常量及零特征对照。',
        '辅助IoU及梯度只回答可解码性/训练动力，不能代替实例AP。',
        '单训练seed、图像相关、多路线探索及已暴露v1 test限制保留；图像bootstrap不证明训练重复性。',
        '', '完整数值、分组、纠错/新增错误分母及初始化/源码/冻结证据见complete_audit_summary.json和analysis/offline目录。']
    (root/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage1','final']);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();m=read(a.root/'manifest.json')
    (stage1 if a.action=='stage1' else final)(a.root,m)
