"""Audit and summarize all five fixed native spatial interventions."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import math


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
    plan=read(r/'plan.json');state=read(r/'state.json');assert state['stage']=='complete'
    repo=Path(__file__).resolve().parents[2]
    assert sha(repo/'scripts/probes/ue_native_spatial_diagnostic.py')==plan['runner_sha256']
    for path,digest in plan['source_hashes'].items():assert sha(repo/path)==digest
    correct=read(r/'correct/records.json');assert len(correct)==459
    baseline=read('/data/wangcheng/checkpoint/ue_v1_adapter_probe_20261021/test/baseline/disabled/metrics.json')['segm/mAP']
    result={};lines=['# Native固定检查点空间诊断结果','',
        '固定test459、FP32、相同检测候选/分数及同一native E10。各干预均为推理诊断，不是新增模型或选模。',
        '', '| 输入 | Mask AP | 对卫星Δ pp | 对正确几何Δ pp [95%区间] |','|---|---:|---:|---|']
    for mode in plan['modes']:
        ident=read(r/mode/'diagnostic_identity.json');assert ident['satellite_state_unchanged'] and ident['detector_gt_exact']
        assert ident['checkpoint_sha256']==plan['adapter_sha256'] and ident['base_sha256']==plan['base_sha256']
        assert ident['runner_sha256']==plan['runner_sha256']
        rows=read(r/mode/'records.json');assert len(rows)==459
        for b,c in zip(correct,rows):
            assert b['image_id']==c['image_id'] and b['gt']==c['gt']
            for k in ['bboxes','scores','labels']:assert b['dt'][k]==c['dt'][k]
        telemetry=read(r/mode/'all_call_telemetry.json');assert len(telemetry)==459
        calls=[]
        for row,t in zip(rows,telemetry):
            assert row['image_id']==t['image_id']
            assert sum(x['rois'] for x in t['calls'])==len(row['dt']['scores'])
            calls.extend(t['calls'])
        count=sum(c['rois'] for c in calls)
        global_rms=lambda k:math.sqrt(sum(c[k]**2*c['rois'] for c in calls)/count)
        fields=['source_delta_rms','applied_delta_rms','base_rms','raw_intervention_rms','reference_delta_rms','rms_match_scale','residual_cosine_to_correct','residual_change_rms']
        stats=dict(calls=len(calls),rois=count,call_medians={k:statistics.median(c[k] for c in calls) for k in fields},
                   roi_pixel_weighted_rms={k:global_rms(k) for k in fields if k.endswith('_rms')})
        if mode.endswith('_rms'):
            error=max(abs(c['source_delta_rms']-c['reference_delta_rms'])/max(c['reference_delta_rms'],1e-8) for c in calls)
            assert error<1e-5;stats['max_relative_rms_matching_error']=error
        metrics=read(r/mode/'metrics.json');ap=metrics['segm/mAP'];assert metrics['bbox/mAP']==.6854287640342888
        comparison=read(r/('correct_vs_satellite_paired_analysis.json' if mode=='correct' else f'{mode}_paired_analysis.json'))
        assert comparison['evaluation_split']=='test' and comparison['detector_and_gt_exact']
        assert comparison['analysis_sha256']==plan['source_hashes']['scripts/probes/ue_v1_adapter_analysis.py']
        assert abs(comparison['metrics']['MaskAP']['candidate']-ap)<1e-12
        ci=comparison['metrics']['MaskAP']['paired_image_bootstrap95'];delta=comparison['metrics']['MaskAP']['delta']
        compare='—' if mode=='correct' else f'{100*delta:+.4f} [{100*ci[0]:+.4f}, {100*ci[1]:+.4f}]'
        lines.append(f'| {mode} | {100*ap:.4f} | {100*(ap-baseline):+.4f} | {compare} |')
        result[mode]=dict(metrics=metrics,telemetry=stats,comparison=comparison)
    assert read(r/'correct/diagnostic_identity.json')['correct_predictions_exact_to_archive']
    lines+=['','## 完整调用遥测','',
        '下表为1500个实际mask调用的中位数，不是末调用跨图中位数；完整审计另存按RoI/像素加权RMS。',
        '', '| 输入 | 残差RMS | 实际注入RMS | 匹配scale | 对正确残差余弦 | 残差差值RMS |',
        '|---|---:|---:|---:|---:|---:|']
    for mode,v in result.items():
        d=v['telemetry']['call_medians'];lines.append('| '+mode+' | '+' | '.join(f'{d[k]:.6f}' for k in ['source_delta_rms','applied_delta_rms','rms_match_scale','residual_cosine_to_correct','residual_change_rms'])+' |')
    lines+=['','## 解释边界','',
        'zero改变输入分布；shuffle保留支持及四字段向量多重集，但破坏空间自相关和高度差的物理一致性。',
        'RMS匹配只控制每调用单个二阶幅度，不控制局部幅度、方向或高阶统计；其结果只用于诊断。',
        '名义图像bootstrap区间不覆盖训练种子、空间相关及多重比较。正确内容优势应结合训练内容对照解释。',
        '同一权重下干预不降不推翻字段训练对照数值，但会削弱“正确局部几何驱动推理收益”的解释。',
        '', '原始计划、全部预测、完整调用遥测、配对分析及身份核对均在本目录。']
    (r/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    (r/'complete_audit_summary.json').write_text(json.dumps(dict(status='passed',plan=plan,modes=result),indent=2)+'\n')
    print(r/'RESULTS.md')

if __name__=='__main__':main()
