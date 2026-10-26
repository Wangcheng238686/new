"""Inventory actual local imports, enforce committed bytes, pin runtime inputs."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

from ue_candidate_detail_ddp_20261026 import ROOT, WORK, PROTOCOL, hashes, base
import ue_candidate_detail_ddp_owner_20261026
import ue_candidate_detail_ddp_analysis_20261026


def inventory(root):
    # Build the real CPU architecture to include registry/lazy module imports.
    from rsprompter.uav.compact_joint import build_foundation_satellite
    model,_ = build_foundation_satellite(base.core.C._register_and_build,
        WORK/'configs/ablations/ue_minimal_single.py', base.FOUNDATION)
    del model
    repository = WORK.parent
    files = {}
    for module in list(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if not name:
            continue
        path = Path(name).resolve()
        if path.suffix == '.py' and path.is_file() and repository in path.parents:
            files[str(path.relative_to(repository))] = base.core.C.file_sha(path)
    for name,sha in hashes().items():
        files[str((WORK/name).relative_to(repository))] = sha
    base.core.C.dump(root/'runtime_source_inventory.json', dict(local_source_sha256=files,
        scope='actual generic model construction and analysis/owner imports plus declared numerical sources'))
    return files


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['inventory','prepare'])
    p.add_argument('--root', type=Path, default=ROOT)
    args = p.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    files = inventory(args.root)
    print('runtime source files',len(files),flush=True)
    if args.action == 'inventory':
        return
    repository = WORK.parent
    for name,expected in files.items():
        blob = subprocess.check_output(['git','show','HEAD:'+name],cwd=repository)
        assert hashlib.sha256(blob).hexdigest() == expected, 'Commit required source bytes: '+name
    evidence = {}
    peak = 0
    for variant in ['query','detail','both','baseline','satellite']:
        path = args.root/'gates'/variant/'complete.json'
        gate = json.loads(path.read_text())
        assert gate['passed'] and gate['source_hashes'] == hashes()
        assert gate['world_size'] == 4 and gate['successful_updates'] == 4
        assert gate['replicas_identical'] and gate['nonempty_optimizer_restore']
        for row in gate['rank_peaks']:
            assert row['restored_all_five_rng'] and row['forced_decoder_instances'] == 250
            peak = max(peak, row['peak_reserved_MiB'])
        evidence[str(path)] = base.core.C.file_sha(path)
    assert peak <= PROTOCOL['allocator_cap_MiB']
    base.core.C.dump(args.root/'engineering_complete.json', dict(passed=True,
        source_hashes=hashes(), world_size=4, gate_sha256=evidence, max_reserved_MiB=peak,
        per_GPU_reservation_MiB=int(peak)+1024,
        scope='real four-rank update/backoff/replica/frozen/restore/FP32/250-instance engineering, not AP benefit'))
    versions = {name:importlib.metadata.version(name) for name in
                ['torch','numpy','mmcv','mmengine','mmdet','peft','pycocotools']}
    (args.root/'environment').mkdir(exist_ok=True)
    freeze = subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)
    (args.root/'environment/pip_freeze.txt').write_text(freeze)
    gpu = subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid,driver_version,memory.total',
        '--format=csv'],text=True)
    (args.root/'environment/gpus.csv').write_text(gpu)
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=repository,text=True).strip()
    status = subprocess.check_output(['git','status','--porcelain'],cwd=repository,text=True)
    initialization = json.loads((base.ROOT/'initialization.json').read_text())
    assert initialization['task_weights_loaded'] is False
    orders = {}
    for epoch in range(1,11):
        path = base.ROOT/'rgbd'/f'epoch{epoch}.json'
        order = json.loads(path.read_text())['sample_order']
        assert len(order) == len(set(order)) == 2136
        target = args.root/'reproduction_inputs/epoch_orders'/path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        orders[str(path)] = base.core.C.file_sha(path)
    for name in ['common_init.pth','initialization.json']:
        source = base.ROOT/name
        target = args.root/'reproduction_inputs'/name
        shutil.copyfile(source, target)
    base.core.C.dump(args.root/'reproducibility.json',dict(git_commit=commit,git_sources_exact=True,
        local_source_sha256=files, source_hashes=hashes(), versions=versions,
        python=sys.executable, python_version=sys.version, platform=platform.platform(),
        pip_freeze_sha256=base.core.C.file_sha(args.root/'environment/pip_freeze.txt'),
        GPU_manifest_sha256=base.core.C.file_sha(args.root/'environment/gpus.csv'),
        initialization=initialization,common_init_sha256=base.core.C.file_sha(base.ROOT/'common_init.pth'),
        foundation_sha256=base.core.C.file_sha(base.FOUNDATION),
        epoch_order_source_sha256=orders,
        data_identity=base.core.C.archive_identity(base.core.previous.DATA),
        git_status_unrelated=status, observed=time.time(),
        scope='all loaded project source bytes equal committed tree; other dirty files not implicitly included'))
    print('prepared reproducible launch',commit,'per GPU reservation',int(peak)+1024,flush=True)


if __name__ == '__main__':
    main()
