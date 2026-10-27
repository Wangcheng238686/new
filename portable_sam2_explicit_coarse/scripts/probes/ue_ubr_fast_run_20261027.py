"""Independent fast G/UBR run. Reuses the sealed short engine in THIS process.

No old file is changed; imported engine globals below affect only this new
process, never existing jobs. Both engineering and efficiency receipts are
mandatory. Full training remains fixed10/2670; no one-card fallback.
"""
import argparse
import fcntl
import hashlib
import os
import subprocess
import time
import json
from pathlib import Path
import sys

WORK = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(WORK), str(WORK / 'scripts/probes')]
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import ue_ubr_short_run_20261027 as engine
import ue_ubr_fast_runtime_20261027 as fast_runtime

CONFIG = WORK / 'configs/ablations/ue_g_ubr_fast_protocol_20261027.json'
ROOT = Path('/data/wangcheng/checkpoint/ue_g_ubr_fast_20261027')
PROTOCOL = json.loads(CONFIG.read_text())
OLD_CRITICAL = list(engine.CRITICAL)
NEW_SOURCES = [
    'rsprompter/uav/ubr_detector_fast.py',
    'scripts/probes/ue_ubr_fast_runtime_20261027.py',
    'scripts/probes/ue_ubr_fast_run_20261027.py',
    'configs/ablations/ue_g_ubr_fast_protocol_20261027.json',
]
EFFICIENCY_RECEIPT = None
_BASE_IDENTITY = engine.identity


def normalize_sources(receipt):
    """Full-repository scope, including vendored SAM2 lazy dependencies."""
    result = {}
    containers = [receipt] + list(receipt.get('ranks', []))
    for value in containers:
        for key in ('source_sha256', 'source_hashes', 'sources_after', 'local_source_sha256'):
            for name, digest in value.get(key, {}).items():
                path = Path(name)
                if path.is_absolute():
                    full = path.resolve()
                elif (engine.REPOSITORY / path).is_file():
                    full = (engine.REPOSITORY / path).resolve()
                else:
                    full = (WORK / path).resolve()
                if engine.REPOSITORY not in full.parents or not full.is_file():
                    raise RuntimeError('Unresolved efficiency source: ' + name)
                relative = str(full.relative_to(engine.REPOSITORY))
                if relative in result and result[relative] != digest:
                    raise RuntimeError('Conflicting efficiency source hash: ' + relative)
                result[relative] = digest
    return result


def efficiency_check(path, sources):
    receipt = json.loads(path.read_text())
    if receipt.get('passed') is not True or receipt.get('world_size') != 4:
        raise RuntimeError('PASSED four-rank numerical/resource efficiency gate required')
    ranks = receipt.get('ranks', [])
    if len(ranks) != 4:
        raise RuntimeError('Efficiency gate must contain four actual rank reports')
    declared = normalize_sources(receipt)
    critical = OLD_CRITICAL + ['rsprompter/uav/ubr_detector_fast.py']
    critical += ['../sam2/sam2/modeling/sam/prompt_encoder.py',
                 '../sam2/sam2/modeling/sam/transformer.py']
    for name in critical:
        relative = str((WORK / name).resolve().relative_to(engine.REPOSITORY))
        if relative not in declared or declared[relative] != sources.get(relative):
            raise RuntimeError('Efficiency gate numerical source absent/changed: ' + relative)
    for name, digest in declared.items():
        if engine.core.C.file_sha(engine.REPOSITORY / name) != digest:
            raise RuntimeError('Efficiency-gate source drift: ' + name)
    return dict(sha256=engine.core.C.file_sha(path), source_sha256=declared,
                scope='four-card real loss/gradient/evaluation equivalence and resource gate')


def runtime(arm, training):
    rt = fast_runtime.build_runtime(content='correct', variant='detail', enabled=arm != 'G_no_tail')
    rt.bare = engine.TailPortModel(rt.joint, rt.controller, arm)
    rt.bare.pool_callback = rt.select_pool
    rt.network = rt.bare
    rt.optimizer = torch.optim.AdamW([p for p in rt.bare.parameters() if p.requires_grad],
        lr=PROTOCOL['lr'], weight_decay=PROTOCOL['weight_decay'])
    if training:
        rt.network = DistributedDataParallel(rt.bare,
            device_ids=[torch.cuda.current_device()], output_device=torch.cuda.current_device(),
            find_unused_parameters=True, broadcast_buffers=False)
    rt.generator = torch.Generator().manual_seed(PROTOCOL['seed'] + dist.get_rank())
    engine.core.C.seed(PROTOCOL['seed'] + dist.get_rank())
    return rt


def identity(rt, args, sources):
    result = _BASE_IDENTITY(rt, args, sources)
    if EFFICIENCY_RECEIPT is None:
        raise RuntimeError('--efficiency-receipt required')
    result['efficiency_gate'] = efficiency_check(EFFICIENCY_RECEIPT, sources)
    result['efficiency_receipt'] = str(EFFICIENCY_RECEIPT)
    return result


def freeze_root(args, frozen):
    """Pin the numerical owner commit even if a later docs-only commit exists."""
    shared = {key: value for key, value in frozen.items() if key != 'arm'}
    args.root.mkdir(parents=True, exist_ok=True)
    with (args.root / 'study.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = args.root / 'study_contract.json'
        requested = os.environ.get('UE_UBR_FAST_SOURCE_COMMIT')
        if path.exists():
            receipt = json.loads(path.read_text())
            if receipt['shared'] != shared:
                raise RuntimeError('Root binds different numerical/input contract')
            commit = receipt['git_commit']
            if requested and requested != commit:
                raise RuntimeError('Owner numerical commit differs from existing study')
        else:
            commit = requested or subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], cwd=engine.REPOSITORY, text=True).strip()
        for name, expected in shared['source_hashes'].items():
            blob = subprocess.check_output(['git', 'show', commit + ':' + name], cwd=engine.REPOSITORY)
            if hashlib.sha256(blob).hexdigest() != expected:
                raise RuntimeError('Commit exact numerical source before launch: ' + name)
        if not path.exists():
            engine.core.C.dump(path, dict(shared=shared, git_commit=commit, created=time.time()))
            for name in shared['source_hashes']:
                target = args.root / 'frozen_source' / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((engine.REPOSITORY / name).read_bytes())
    return commit


def configure(efficiency_receipt):
    """Explicit process-local engine binding; also used by the independent owner."""
    global EFFICIENCY_RECEIPT
    EFFICIENCY_RECEIPT = Path(efficiency_receipt).resolve(strict=True)
    engine.CONFIG, engine.ROOT, engine.PROTOCOL = CONFIG, ROOT, PROTOCOL
    # Base gate predates fast sources. Verify old baseline critical files there;
    # identity() separately verifies the complete new efficiency gate closure.
    engine.CRITICAL = OLD_CRITICAL
    engine.runtime, engine.identity = runtime, identity
    engine.freeze_root = freeze_root


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--efficiency-receipt', type=Path)
    options, remaining = parser.parse_known_args()
    if options.efficiency_receipt is None:
        if '--help' in remaining or '-h' in remaining:
            print('FAST REQUIRED OPTION: --efficiency-receipt PATH (passed full four-card efficiency gate)')
            sys.argv = [sys.argv[0]] + remaining
            engine.main()
            return
        parser.error('--efficiency-receipt is required')
    configure(options.efficiency_receipt)
    sys.argv = [sys.argv[0]] + remaining
    engine.main()


if __name__ == '__main__':
    main()
