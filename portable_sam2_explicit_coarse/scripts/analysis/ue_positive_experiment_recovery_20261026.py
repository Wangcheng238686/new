"""Preserve two positive experiment lines without touching live training.

Copies only fixed checkpoints and completed artifacts. Rolling joint last/best
files are intentionally excluded; no model loading, GPU work or retraining.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time


WORK = Path(__file__).resolve().parents[2]
CHECKPOINT = Path('/data/wangcheng/checkpoint')
STUDIES = ['ue_instance_interaction_20261024', 'ue_compact_joint_20261024']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def preserve(src, dst, records, expected=None):
    before = sha(src)
    if expected:
        assert before == expected, str(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    assert not dst.exists(), str(dst)
    shutil.copyfile(src, dst)
    assert sha(dst) == before == sha(src), 'Input changed while archiving: '+str(src)
    records.append(dict(original=str(src), archived=str(dst), sha256=before,
                        bytes=dst.stat().st_size))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    records, sources, assets = [], {}, []
    for study in STUDIES:
        root, target = CHECKPOINT/study, args.output/study
        manifest = read(root/('study_manifest.json' if 'instance' in study else 'initialization.json'))
        declared = manifest.get('sources', manifest.get('source_hashes'))
        for name, expected in declared.items():
            assert sha(root/'source'/name) == expected
        for src in sorted((root/'source').rglob('*')):
            if not src.is_file():
                continue
            name = str(src.relative_to(root/'source'))
            assert sha(src) == sha(WORK/name), 'Current source differs: '+name
            preserve(src, target/'source'/name, records)
            sources.setdefault(name, sha(src))
            assert sources[name] == sha(src), 'Experiments need distinct source versions'
        # Capture completed, small metadata only; skip rolling state/metrics.
        for src in sorted(root.glob('*.json')):
            if src.name in ['state.json', 'execution_state.json', 'matched_validation_observation.json']:
                continue
            preserve(src, target/'metadata'/src.name, records)
        if 'instance' in study:
            for name, expected in manifest['initializations'].items():
                preserve(root/name, target/'weights'/name, records, expected)
            audit = read(root/'final_audit.json')
            assert audit['passed']
            for row in audit['checkpoints']:
                arm = row['arm']
                receipt = read(root/arm/'training_complete.json')
                assert receipt['complete'] and receipt['updates'] == 2670
                preserve(root/arm/'epoch10.pth', target/'weights'/arm/'epoch10.pth', records, row['sha256'])
                for src in sorted((root/arm).glob('*.json')):
                    if src.name != 'progress.json':
                        preserve(src, target/'metadata'/arm/src.name, records)
            for folder in ['analysis', 'baseline/test',
                           'seed44_rgb_internal_geometry_correct/test',
                           'seed44_rgb_internal_geometry_constant/test']:
                for src in sorted((root/folder).rglob('*')):
                    if src.is_file() and src.suffix in ['.json', '.md']:
                        preserve(src, target/'evidence'/src.relative_to(root), records)
            preserve(root/'RESULTS.md', target/'RESULTS.md', records)
            base = CHECKPOINT/'ue_minimal_single_20261021_EXP-0017/checkpoints/best_model_epoch73.pth'
            contract = read(root/'seed44_rgb_internal_geometry_correct/contract.json')
            preserve(base, target/'weights/satellite_E73.pth', records, contract['base_sha256'])
            assets.append(dict(dataset_identity=contract['archive_hashes'], native_cache=contract['native_cache']))
        else:
            preserve(root/'common_init.pth', target/'weights/common_init.pth', records, manifest['common_init_sha256'])
            for arm in manifest['protocol']['arms']:
                preserve(root/arm/'epoch50.pth', target/'weights'/arm/'epoch50.pth', records)
                for src in sorted((root/arm).glob('epoch*.json')):
                    preserve(src, target/'metadata'/arm/src.name, records)
                preserve(root/arm/'contract.json', target/'metadata'/arm/'contract.json', records)
            assert not list(root.glob('*/epoch60.pth')), 'E60 availability changed; update recovery scope'
    # External inputs stay external: verify metadata identities, not every raw file.
    dataset = Path('/data/wangcheng/dataset/CitySample_SatUAV_TrainMinimal_v1_20261021')
    for item in assets:
        for name, expected in item['dataset_identity'].items():
            assert sha(dataset/name) == expected
    foundation = Path('/data/wangcheng/pretrained-models/sam2/sam2_hiera_base_plus.pt')
    expected = read(CHECKPOINT/STUDIES[1]/'initialization.json')['foundation_sha256']
    assert sha(foundation) == expected
    assets.append(dict(foundation=str(foundation), sha256=expected,
                       scope='raw dataset/cache bytes not duplicated or fully rescanned'))
    freeze = subprocess.check_output(['/data/wangcheng/envs/cvt2/bin/python','-m','pip','freeze'], text=True)
    (args.output/'current_pip_freeze.txt').write_text(freeze)
    result = dict(complete=True, created=time.time(),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=WORK,text=True).strip(),
        recovery_source_sha256=sha(__file__), original_sources=sources, files=records,
        external_assets=assets, environment_scope='current environment, not launch-time proof',
        limitations=['No E60 checkpoint; historical E60 metrics only',
                     'No clean-checkout training replay or AP rerun performed',
                     'Backup is on the same filesystem, not an offsite copy',
                     'Joint experiment is still training; rolling files excluded'])
    (args.output/'recovery.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(complete=True,files=len(records),source_files=len(sources),
                          bytes=sum(r['bytes'] for r in records),output=str(args.output)),indent=2),flush=True)


if __name__ == '__main__':
    main()
