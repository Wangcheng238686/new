"""Wait for verified execution, audit identities, and summarize the fixed study.

This does not train, select checkpoints, or change the frozen execution plan.
Completion here still requires the foreground completion review.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import ue_instance_interaction_owner_20261024 as owner


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def proc_identity(pid):
    directory = Path('/proc') / str(pid)
    try:
        stat = (directory / 'stat').read_text().split()
        command = (directory / 'cmdline').read_bytes().replace(b'\x00', b' ').decode()
    except FileNotFoundError:
        return None
    if stat[2] == 'Z':
        return None
    return dict(pid=pid, start_ticks=stat[21], command=command)


def additional_audit(root):
    import torch
    manifest = json.loads((root / 'study_manifest.json').read_text())
    handoff_file = root / 'capacity_handoff.json'
    if handoff_file.exists():
        handoff = json.loads(handoff_file.read_text())
        name = 'scripts/probes/ue_instance_interaction_capacity_owner_20261024.py'
        assert sha(owner.WORK / name) == sha(root / 'source' / name) == handoff['source_sha256']
        assert handoff['original_jobs'] == 119 and handoff['frozen_scientific_sources_unchanged']
    cache_file = root / 'input_cache_sha256.json'
    assert sha(cache_file) == manifest['cache_sha256_manifest']
    cache = json.loads(cache_file.read_text())
    counts = {}
    for label, folder in [('native', '/data/wangcheng/cache/ue_v1_native_geometry'),
                          ('anchors', '/data/wangcheng/cache/ue_v1_s2db_geometry_v2')]:
        reference = cache['caches'][label]
        actual = list(Path(folder).glob('*.npz'))
        assert len(actual) == len(reference) == 3060
        for path in actual:
            assert sha(path) == reference[path.name], ('Input cache changed', path)
        counts[label] = len(actual)
    print('FULL INPUT CACHE SHA PASS', counts, flush=True)
    bindings = {}
    active_parameters = {}
    for arm in owner.arms():
        directory = root / arm['name']
        cp_path = directory / 'last.pth'
        bindings[arm['name']] = sha(cp_path)
        cp = torch.load(cp_path, map_location='cpu', weights_only=False)
        assert cp['contract'] == json.loads((directory / 'contract.json').read_text())
        active_parameters[arm['name']] = len(cp['optimizer']['state'])
        del cp
    evaluations = []
    for job in owner.build_jobs(root):
        if job['type'] != 'eval':
            continue
        path = Path(job['marker'])
        identity = json.loads(path.read_text())
        arm_name = path.relative_to(root).parts[0]
        assert identity['checkpoint_sha256'] == bindings[arm_name], ('Evaluation weight identity', job['name'])
        assert all(value == manifest['sources'][key] for key, value in identity['source_hashes'].items())
        evaluations.append(dict(job=job['name'], marker_sha256=sha(path),
            satellite_records_sha256=sha(path.parent / 'satellite_records.json'),
            combined_records_sha256=sha(path.parent / 'combined_records.json')))
    owner.dump(root / 'additional_identity_audit.json', dict(passed=True, input_cache_counts=counts,
        checkpoint_bindings=bindings, actual_optimizer_state_parameters=active_parameters,
        evaluation_bindings=evaluations, full_training_contract_json_exact=True,
        capacity_controller_identity_verified=handoff_file.exists(),
        finish_source_sha256=sha(__file__)))


def positive(metric):
    return metric['delta'] > 0 and metric['paired_image_bootstrap95'][0] > 0


def summarize(root):
    analyses = {}
    for job in owner.build_jobs(root):
        if job['type'] == 'analysis':
            analyses[job['name']] = json.loads(Path(job['marker']).read_text())
    metrics = {}
    main_rows = []
    for arm in owner.arms():
        directory = root / arm['name'] / 'test' / arm['content']
        combined = json.loads((directory / 'combined_metrics.json').read_text())
        satellite = json.loads((directory / 'satellite_metrics.json').read_text())
        metrics[arm['name']] = dict(combined=combined, conditioned_satellite=satellite)
        main_rows.append('|{}|{:.4f}|{:.4f}|{:.4f}|{:.4f}|'.format(arm['name'],
            combined['segm/mAP'] * 100, combined['segm/mAP_75'] * 100,
            combined['bbox/mAP'] * 100, satellite['segm/mAP'] * 100))
    claims = {}
    for seed in (44, 45):
        prefix = f'geom_s{seed}_internal_'
        claims[f'geometry_seed{seed}'] = dict(
            combined_content_positive=positive(analyses[prefix + 'content']['metrics']['MaskAP']),
            original_path_content_positive=positive(analyses[prefix + 'content_satellite']['metrics']['MaskAP']),
            internal_vs_head_positive=positive(analyses[prefix + 'vs_head']['metrics']['MaskAP']),
            original_path_internal_vs_head_positive=positive(analyses[prefix + 'vs_head_satellite']['metrics']['MaskAP']),
            net_clear_reference=analyses[prefix + 'net']['gate']['clear_v1_checkpoint_gain'],
            correct_over_constant_input=positive(analyses[prefix + 'constant_input']['metrics']['MaskAP']),
            correct_over_geometry_donor=positive(analyses[prefix + 'donor_input']['metrics']['MaskAP']))
    for route in ('head', 'internal'):
        prefix = f'rgb_{route}_'
        claims[f'rgb_{route}'] = dict(
            combined_RGB_content_positive=positive(analyses[prefix + 'content']['metrics']['MaskAP']),
            original_path_RGB_content_positive=positive(analyses[prefix + 'content_satellite']['metrics']['MaskAP']),
            combined_depth_value_positive=positive(analyses[prefix + 'depth_values']['metrics']['MaskAP']),
            original_path_depth_value_positive=positive(analyses[prefix + 'depth_values_satellite']['metrics']['MaskAP']),
            net_clear_reference=analyses[prefix + 'net']['gate']['clear_v1_checkpoint_gain'],
            correct_over_constant_RGB_input=positive(analyses[prefix + 'constant_rgb_input']['metrics']['MaskAP']),
            correct_over_RGB_donor=positive(analyses[prefix + 'donor_rgb_input']['metrics']['MaskAP']))
    output = dict(complete=True, fixed_arms=14, paired_analyses=len(analyses), metrics=metrics,
        comparisons={key: value['metrics'] for key, value in analyses.items()}, evidence_flags=claims,
        flag_scope='Positive means nominal image-bootstrap lower bound >0; not training-variance or multiplicity corrected',
        limitations=['development-exposed v1 test; spatial dependence',
            'geometry two training seeds, RGB one seed', 'head also contains full interaction',
            'constant geometry retains depth-derived correspondence/support',
            'geometry donor in RGB arms is a mixed inconsistent RGB-D input',
            'frozen E73 diagnostic; not full-budget common-initialization joint training'])
    owner.dump(root / 'study_summary.json', output)
    lines = ['# 实例监督内部回写：固定实验完整结果', '',
        '全部14臂固定E10、FP32/max250评价与严格来源审计完成。数值为AP×100；当前仅是冻结E73开发实验。', '',
        '## 完整测试点值', '',
        '|臂|合并Mask AP|AP75|bbox AP|条件化原路径Mask AP|',
        '|---|---:|---:|---:|---:|', *main_rows, '',
        '## 配对比较', '',
        '单位为百分点；每项500次同图配对bootstrap，名义95%区间。head也有交互，internal−head只检验继续编码的作用。', '',
        '|比较|Mask差值 [95%区间]|bbox差值|Boundary2%差值|Boundary1%差值|',
        '|---|---|---:|---:|---:|']
    for name, result in analyses.items():
        m = result['metrics']['MaskAP']
        lo, hi = m['paired_image_bootstrap95']
        lines.append('|{}|{:+.4f} [{:+.4f},{:+.4f}]|{:+.4f}|{:+.4f}|{:+.4f}|'.format(
            name, m['delta'] * 100, lo * 100, hi * 100,
            result['metrics']['BBoxAP']['delta'] * 100,
            result['metrics']['BoundaryAP_2pct']['delta'] * 100,
            result['metrics']['BoundaryAP_1pct']['delta'] * 100))
    lines += ['', '## 证据解释约束', '',
        '- 非零梯度只证明接通；正确内容优于训练常量及固定权重干预，才支持内容贡献。',
        '- 原路径的内容差与入口差必须单独看；合并AP上涨不能自动归为卫星表征改善。',
        '- 图像bootstrap不等于多seed或独立空间区域验证；RGB只有一个训练seed。',
        '- 常量字段不是无深度：深度对应/support仍在。RGB臂几何donor不是物理一致场景反事实。',
        '- `segm/AR@100`是既有evaluator旧键名；本轮maxDets=[1,10,250]，该统计实际为AR250。',
        '- 原高分辨率stride4分支不被block4直接更新；P0外观读取与可达LoRA范围有限。',
        '- 本轮没有验证共同基础初始化的全预算联合训练，不据test再改结构/轮次。', '',
        '完整四指标区间、尺寸组点值和证据布尔项见study_summary.json；最终方法判断仍需人工审阅。']
    (root / 'RESULTS.md').write_text('\n'.join(lines) + '\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=owner.OUT)
    p.add_argument('--owner-pid', type=int, required=True)
    p.add_argument('--audit-only', action='store_true')
    args = p.parse_args()
    root = args.root
    initial = proc_identity(args.owner_pid)
    if not args.audit_only:
        allowed_owners = ('ue_instance_interaction_owner_20261024.py',
                          'ue_instance_interaction_capacity_owner_20261024.py')
        assert initial and any(name in initial['command'] for name in allowed_owners)
        owner.dump(root / 'finish_owner_identity.json', initial)
        while not (root / 'owner_complete.json').exists():
            current = proc_identity(args.owner_pid)
            assert current and current['start_ticks'] == initial['start_ticks'], 'Owner ended without completion'
            state = json.loads((root / 'state.json').read_text())
            if state['stage'] == 'failure_requires_audit':
                raise RuntimeError(('Execution failure requires foreground review', state.get('failed')))
            owner.dump(root / 'finish_state.json', dict(stage='waiting_verified_owner',
                owner_pid=args.owner_pid, finish_pid=os.getpid(), observed=time.time(),
                done=len(state.get('done', [])), running=len(state.get('running', {}))))
            time.sleep(20)
    jobs = owner.build_jobs(root)
    assert all(owner.valid_marker(j) for j in jobs)
    owner.dump(root / 'finish_state.json', dict(stage='strict_final_audit', finish_pid=os.getpid()))
    subprocess.run([owner.PYTHON, '-u', 'scripts/analysis/ue_instance_interaction_audit_20261024.py',
                    '--root', str(root)], cwd=owner.WORK, check=True)
    additional_audit(root)
    summarize(root)
    owner.dump(root / 'finish_complete.json', dict(complete=True, foreground_review_still_required=True,
        final_audit_sha256=sha(root / 'final_audit.json'),
        additional_audit_sha256=sha(root / 'additional_identity_audit.json'),
        summary_sha256=sha(root / 'study_summary.json'), report_sha256=sha(root / 'RESULTS.md'),
        finish_source_sha256=sha(__file__)))
    print('FINISH COMPLETE; FOREGROUND REVIEW REQUIRED', flush=True)


if __name__ == '__main__':
    main()
