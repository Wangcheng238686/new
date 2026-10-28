"""Explicit staged, source-pinned, four-GPU experiments. Import and plan are CPU-only.

Queue runs ONLY the named stage. Later stages require an explicit validation and
engineering based selection lock; this controller never ranks test metrics.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

WORK = Path(__file__).resolve().parents[2]
REPO = WORK.parent
MATRIX = 'configs/ablations/ue_staged_uav_experiments_20261028.json'
ROOT = Path(json.loads((WORK / MATRIX).read_text())['root'])
RUNNER = 'scripts/probes/ue_staged_uav_run_20261028.py'
LEGACY_CACHE = Path('/data/wangcheng/checkpoint/ue_instance_interaction_20261024/input_cache_sha256.json')
CACHE_DIRS = {'native': Path('/data/wangcheng/cache/ue_v1_native_geometry'),
              'anchors': Path('/data/wangcheng/cache/ue_v1_s2db_geometry_v2')}
DATA_ROOT = Path('/data/wangcheng/dataset/CitySample_SatUAV_TrainMinimal_v1_20261021')
PYTHON = os.environ.get('UE_EXPERIMENT_PYTHON', '/data/wangcheng/envs/cvt2/bin/python')
GPUS = (0, 1, 2, 3)
UNMEASURED_RESERVATION = 15872
DEVICE_MARGIN = 1024


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')
    os.replace(temporary, path)


def identity(pid):
    try:
        folder = Path('/proc') / str(pid)
        fields = (folder / 'stat').read_text().rsplit(') ', 1)[1].split()
        return dict(pid=int(pid), start_ticks=fields[19],
                    argv=(folder / 'cmdline').read_bytes().decode().rstrip('\0').split('\0'),
                    state=fields[0])
    except (OSError, ProcessLookupError):
        return None


def live(record):
    actual = identity(record.get('pid', 0)) if record else None
    return bool(actual and actual['state'] != 'Z' and
                actual['start_ticks'] == str(record['start_ticks']))


def helper(source):
    path = Path(source) / WORK.name / 'scripts/experiments/ue_staged_uav_matrix.py'
    spec = importlib.util.spec_from_file_location('staged_matrix_' + sha(path)[:12], path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_setup(source):
    api = helper(source)
    matrix_path = Path(source) / WORK.name / MATRIX
    matrix = api.load_matrix(matrix_path)
    api.audit_matrix(matrix)
    protocol_path = Path(source) / WORK.name / matrix['protocol_file']
    return api, matrix, read(protocol_path)


def recipe_digest(recipe):
    return canonical_sha({k: v for k, v in recipe.items() if k != 'recipe_sha256'})


def checked_recipes(api, matrix, base=None):
    recipes = api.resolve_recipes(matrix, base_arm=base)
    for name, item in recipes.items():
        if item.get('name') != name or item.get('recipe_sha256') != recipe_digest(item):
            raise RuntimeError('Recipe canonical identity mismatch: ' + name)
    return recipes


def prepare(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = root / 'launch_manifest.json'
        if path.exists():
            manifest = read(path)
            validate(manifest)
            return manifest
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
        required = [Path(WORK.name) / p for p in (MATRIX, RUNNER,
                    'scripts/analysis/ue_staged_uav_analysis_20261028.py',
                    'scripts/experiments/ue_staged_uav_control.py',
                    'scripts/experiments/ue_staged_uav_matrix.py', 'scripts/experiments/ue_staged_uav.sh')]
        for relative in required:
            blob = subprocess.check_output(['git', 'show', commit + ':' + str(relative)], cwd=REPO)
            if hashlib.sha256(blob).hexdigest() != sha(REPO / relative):
                raise RuntimeError('Commit exact staged source before prepare: ' + str(relative))
        source = root / 'runtime_source'
        if source.exists():
            raise RuntimeError('Unbound runtime_source exists; preserve and audit it')
        subprocess.run(['git', 'worktree', 'add', '--detach', str(source), commit], cwd=REPO, check=True)
        api, matrix, protocol = load_setup(source)
        recipes = checked_recipes(api, matrix)
        names = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', commit], cwd=source,
                                        text=True).splitlines()
        sources = {n: sha(source / n) for n in names
                   if Path(n).suffix in ('.py', '.json', '.sh', '.yaml', '.yml')}
        # Isolated process ensures editable SAM2 cannot supply the old checkout.
        snippet = ('import sys;from pathlib import Path;'
                   'r=Path(sys.argv[1]);w=r/"portable_sam2_explicit_coarse";'
                   'sys.path[:0]=[str(r/"sam2"),str(w),str(w/"scripts/probes")];'
                   'import ue_ubr_native_full_run_20261028 as f;'
                   'f.source_inventory();f.prepare_orders(Path(sys.argv[2]))')
        subprocess.run([PYTHON, '-c', snippet, str(source), str(root)], cwd=source, check=True)
        protocol_path = matrix['protocol_file']
        manifest = dict(version=1, root=str(root), runtime_source=str(source), git_commit=commit,
                        source_sha256=sources, matrix_path=MATRIX,
                        matrix_sha256=sha(source / WORK.name / MATRIX), protocol_path=protocol_path,
                        protocol_sha256=canonical_sha(protocol), protocol=protocol,
                        recipes=recipes, stages=matrix['stages'], base_lock=None, cache_receipts={},
                        order_manifest_sha256=sha(root / 'orders/manifest.json'),
                        legacy_cache_manifest=str(LEGACY_CACHE),
                        legacy_cache_manifest_sha256=sha(LEGACY_CACHE),
                        created=time.time(), selection='validation MaskAP only; test is never a selector')
        write(path, manifest)
        return manifest


def validate(manifest):
    source = Path(manifest['runtime_source'])
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()
    if head != manifest['git_commit']:
        raise RuntimeError('Pinned worktree HEAD changed')
    for name, digest in manifest['source_sha256'].items():
        if sha(source / name) != digest:
            raise RuntimeError('Pinned source drift: ' + name)
    if sha(source / WORK.name / manifest['matrix_path']) != manifest['matrix_sha256']:
        raise RuntimeError('Matrix drift')
    if canonical_sha(read(source / WORK.name / manifest['protocol_path'])) != manifest['protocol_sha256']:
        raise RuntimeError('Protocol drift')
    order_path = Path(manifest['root']) / 'orders/manifest.json'
    if sha(order_path) != manifest['order_manifest_sha256']:
        raise RuntimeError('Order manifest drift')
    for item in read(order_path)['epochs'].values():
        if sha(order_path.parent / item['path']) != item['sha256']:
            raise RuntimeError('Per-epoch order drift')
    if sha(manifest['legacy_cache_manifest']) != manifest['legacy_cache_manifest_sha256']:
        raise RuntimeError('Original cache identity drift')
    base = manifest.get('base_lock')
    if base:
        path = Path(manifest['root']) / 'selection_lock.json'
        if sha(path) != manifest['base_lock_sha256'] or read(path) != base:
            raise RuntimeError('Base selection lock drift')
    api, matrix, _ = load_setup(source)
    expected = checked_recipes(api, matrix, base['base_arm'] if base else None)
    if expected != manifest['recipes']:
        raise RuntimeError('Resolved recipe identities changed')


def stage_recipes(manifest, stage):
    names = manifest['stages'][stage]
    if stage != 'structure' and not manifest.get('base_lock'):
        raise RuntimeError('Explicit validation-based lock --base-arm U/T/C required first')
    if any(n not in manifest['recipes'] for n in names):
        raise RuntimeError('Stage has unresolved recipes')
    return list(names)


def _stamp(path):
    stat = Path(path).stat()
    return dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def _verify_cache_receipt(manifest, name, deep=False):
    item = manifest['cache_receipts'].get(name)
    if not item or sha(item['path']) != item['sha256']:
        raise RuntimeError('Missing or changed cache receipt; run cache first: ' + name)
    receipt = read(item['path'])
    if (receipt['recipe_sha256'] != manifest['recipes'][name]['recipe_sha256'] or
            receipt['protocol_sha256'] != manifest['protocol_sha256'] or
            receipt['source_sha256'] != manifest['source_sha256']):
        raise RuntimeError('Cache receipt binds a different experiment')
    for path, info in receipt['files'].items():
        if _stamp(path) != info['stat'] or (deep and sha(path) != info['sha256']):
            raise RuntimeError('Cache bytes or file identity changed: ' + path)
    return receipt


def cache_stage(manifest, stage):
    names = stage_recipes(manifest, stage)
    source = Path(manifest['runtime_source'])
    for name in names:
        recipe = manifest['recipes'][name]
        print('CACHE_VERIFY', name, 'existing_receipt' if name in manifest['cache_receipts'] else 'prepare_identity', flush=True)
        if name in manifest['cache_receipts']:
            _verify_cache_receipt(manifest, name, deep=True)
            print('CACHE_READY', name, flush=True)
            continue
        files = {}
        subset_root = recipe.get('subset_cache_root')
        pure_satellite = recipe['arm'] == 'S'
        if pure_satellite:
            # Hash test annotation bytes for identity only, never decode labels.
            # S has no model-side UAV dependency and must not audit UAV samples.
            paths = [DATA_ROOT / n for n in ('dataset.json', 'SHA256SUMS.txt')]
            paths += [DATA_ROOT / family / (split + '.json')
                      for family in ('splits', 'annotations') for split in ('train', 'val', 'test')]
            declared = {str(path): sha(path) for path in paths}
        elif subset_root:
            argv = [PYTHON, '-u', str(source / WORK.name /
                    'scripts/probes/ue_build_view_subset_cache_20261028.py'),
                    '--cache-root', subset_root, '--view-ids'] + list(map(str, recipe['uav_view_ids']))
            subprocess.run(argv, cwd=source / WORK.name, check=True)
            key = 'views_' + '_'.join('%03d' % v for v in recipe['uav_view_ids'])
            folder = Path(subset_root) / key
            marker = folder / 'BUILD_COMPLETE.json'
            complete = read(marker)
            if not complete.get('complete') or complete['images'] != 3060:
                raise RuntimeError('Full3060 subset cache required')
            contract = read(folder / 'manifest.json')
            if (contract['purpose'] != 'full_dataset' or contract['view_ids'] != recipe['uav_view_ids'] or
                    sha(folder / 'manifest.json') != complete['manifest_sha256']):
                raise RuntimeError('Subset cache identity mismatch')
            declared = {str(marker): sha(marker), str(folder / 'manifest.json'): complete['manifest_sha256']}
            declared.update({str(folder / ('%06d.npz' % row['image_id'])): row['cache_sha256']
                             for row in complete['rows']})
        else:
            legacy = read(manifest['legacy_cache_manifest'])
            if not legacy.get('complete'):
                raise RuntimeError('Original cache audit incomplete')
            declared = {manifest['legacy_cache_manifest']: manifest['legacy_cache_manifest_sha256']}
            for family, folder in CACHE_DIRS.items():
                if len(legacy['caches'][family]) != 3060:
                    raise RuntimeError('Original cache must cover3060 images')
                declared.update({str(folder / n): digest for n, digest in legacy['caches'][family].items()})
        for count, (path, digest) in enumerate(declared.items(), 1):
            if sha(path) != digest:
                raise RuntimeError('Cache content differs from producer receipt: ' + path)
            files[path] = dict(sha256=digest, stat=_stamp(path))
            if count % 500 == 0:
                print('CACHE_HASHED', name, count, len(declared), flush=True)
        receipt = dict(complete=True, recipe=name, recipe_sha256=recipe['recipe_sha256'],
                       source_sha256=manifest['source_sha256'], protocol_sha256=manifest['protocol_sha256'],
                       files=files, images=3060, created=time.time(),
                       kind='pure_satellite' if pure_satellite else 'uav_subset' if subset_root else 'uav_five_view',
                       sample_uav_cache_dependency=not pure_satellite)
        out = Path(manifest['root']) / 'cache_receipts' / (name + '.json')
        if out.exists():
            raise RuntimeError('Unbound cache receipt exists; preserve and audit')
        write(out, receipt)
        manifest['cache_receipts'][name] = dict(path=str(out), sha256=sha(out))
        write(Path(manifest['root']) / 'launch_manifest.json', manifest)
        print('CACHE_READY', name, len(files), 'identity_files', flush=True)


def marker(manifest, action, name):
    root = Path(manifest['root'])
    if action == 'gate':
        return root / 'gates' / name / 'complete.json'
    if action == 'train':
        return root / name / 'training_complete.json'
    return root / 'evaluation/test' / name / 'complete.json'


def accepted(manifest, action, name):
    path = marker(manifest, action, name)
    if not path.exists():
        return False
    value = read(path)
    if not (value.get('passed') if action == 'gate' else value.get('complete')):
        return False
    if (value.get('world_size') != 4 or value.get('recipe') != name or
            value.get('recipe_sha256') != manifest['recipes'][name]['recipe_sha256'] or
            value.get('source_sha256') != manifest['source_sha256'] or
            value.get('protocol_sha256') != manifest['protocol_sha256']):
        return False
    if action == 'gate':
        return value.get('max_reserved_mib', 0) > 0 and value.get('max_resident_mib', 0) > 0
    if action == 'train':
        return (value.get('epoch') == 100 and value.get('updates') == 26700 and
                sha(path.parent / 'last.pth') == value.get('last_checkpoint_sha256') and
                sha(path.parent / 'best_model.pth') == value.get('best_checkpoint_sha256'))
    trained = read(marker(manifest, 'train', name))
    return (value.get('split') == 'test' and value.get('images') == 459 and
            value.get('precision') == 'FP32' and
            value.get('checkpoint_sha256') == trained.get('best_checkpoint_sha256'))


def lock_base(manifest, base_arm, reason):
    root = Path(manifest['root'])
    if manifest.get('base_lock') or (root / 'selection_lock.json').exists():
        raise RuntimeError('Base already locked; create a new study to change selection')
    for name in stage_recipes(manifest, 'structure'):
        if not accepted(manifest, 'train', name):
            raise RuntimeError('All structure trainings must finish before lock')
    chosen = read(marker(manifest, 'train', base_arm))
    lock = dict(base_arm=base_arm, reason=reason, basis='explicit user selection using validation and engineering only; no test metrics read',
                selected_best_checkpoint_sha256=chosen['best_checkpoint_sha256'],
                selected_validation_best=chosen.get('best'),
                structure_receipt_sha256={n: sha(marker(manifest, 'train', n))
                                          for n in manifest['stages']['structure']},
                source_sha256=manifest['source_sha256'], protocol_sha256=manifest['protocol_sha256'],
                matrix_sha256=manifest['matrix_sha256'], created=time.time())
    api, matrix, _ = load_setup(manifest['runtime_source'])
    recipes = checked_recipes(api, matrix, base_arm)
    if any(recipes.get(n) != r for n, r in manifest['recipes'].items()):
        raise RuntimeError('Lock must not alter any existing recipe')
    write(root / 'selection_lock.json', lock)
    manifest.update(base_lock=lock, base_lock_sha256=sha(root / 'selection_lock.json'), recipes=recipes)
    write(root / 'launch_manifest.json', manifest)


def reservation(manifest, name):
    if not accepted(manifest, 'gate', name):
        return UNMEASURED_RESERVATION
    gate = read(marker(manifest, 'gate', name))
    return int(max(gate['max_reserved_mib'] + 1024, gate['max_resident_mib'] + 256))


def verify_initialization_family(manifest, name):
    """Compare actual state hashes, never descriptive initialization metadata.

    All gates of this stage precede any full training. S/U/T/C share satellite
    and task heads; U/T/C additionally share RGB. Follow-ups must be byte-equal
    to their locked base's FULL initial model, despite different evidence input.
    """
    stages = [stage for stage, names in manifest['stages'].items() if name in names]
    if len(stages) != 1:
        raise RuntimeError('Recipe has no unique scientific stage: ' + name)
    stage = stages[0]
    names = stage_recipes(manifest, stage)
    if not all(accepted(manifest, 'gate', n) for n in names):
        raise RuntimeError('Every gate in this stage must pass before training')
    initial = {n: read(marker(manifest, 'gate', n))['initialization'] for n in names}
    def same(keys, selected, reference):
        for key in keys:
            expected = reference.get(key)
            if not isinstance(expected, str) or not expected:
                raise RuntimeError('Missing initial state fingerprint: ' + key)
            for recipe in selected:
                if initial[recipe].get(key) != expected:
                    raise RuntimeError('Initialization family differs: ' + recipe + ' / ' + key)
    if stage == 'structure':
        same(('satellite_sha256', 'head_sha256'), names, initial['S'])
        non_satellite = [n for n in names if n != 'S']
        same(('rgb_sha256',), non_satellite, initial['U'])
    else:
        base = manifest['base_lock']['base_arm']
        if not accepted(manifest, 'gate', base):
            raise RuntimeError('Locked base gate is missing or changed')
        reference = read(marker(manifest, 'gate', base))['initialization']
        same(('satellite_sha256', 'head_sha256', 'rgb_sha256', 'full_model_sha256'), names, reference)
    return dict(passed=True, stage=stage, recipe=name, compared_recipes=names,
                scope='initial model state only; audit metadata excluded')


def resources(manifest, running):
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free',
                                   '--format=csv,noheader,nounits'], text=True)
    cards = {}
    for line in raw.splitlines():
        index, uuid, free = [x.strip() for x in line.split(',')]
        if int(index) in GPUS:
            cards[int(index)] = dict(uuid=uuid, actual_free_mib=int(free), admissible_free_mib=int(free) - DEVICE_MARGIN)
    if set(cards) != set(GPUS):
        raise RuntimeError('Four physical GPUs0..3 required')
    usage = {}
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
                                   '--format=csv,noheader,nounits'], text=True)
    for line in raw.splitlines():
        fields = [x.strip() for x in line.split(',')]
        if len(fields) == 3 and fields[1].isdigit() and fields[2].isdigit():
            usage[(fields[0], int(fields[1]))] = int(fields[2])
    expected_runner = str(Path(manifest['runtime_source']) / WORK.name / RUNNER)
    for job in running:
        receipt = Path(job['rank_launch_record'])
        record = read(receipt) if receipt.exists() else None
        rows = record['ranks'] if record else []
        if record and (record.get('world_size') != 4 or len(rows) != 4):
            raise RuntimeError('Rank launch receipt must contain all four ranks')
        seen = set()
        for row in rows:
            local = int(row['local_rank'])
            if local not in GPUS or local in seen:
                raise RuntimeError('Invalid rank launch receipt')
            seen.add(local)
            if live(row):
                actual = identity(row['pid'])
                if expected_runner not in actual['argv'] or int(row['start_ticks']) < int(job['launcher']['start_ticks']):
                    raise RuntimeError('Rank receipt does not belong to this launch')
                argv = actual['argv']
                if ('--root' not in argv or '--recipe' not in argv or
                        argv[argv.index('--root') + 1] != manifest['root'] or
                        argv[argv.index('--recipe') + 1] != job['recipe']):
                    raise RuntimeError('Rank PID belongs to another root or recipe')
        job['ranks'] = rows
        reserve = reservation(manifest, job['recipe'])
        for index, card in cards.items():
            row = next((r for r in rows if int(r['local_rank']) == index and live(r)), None)
            used = usage.get((card['uuid'], row['pid']), 0) if row else 0
            card['admissible_free_mib'] -= max(0, reserve - used)
    return cards


def command(manifest, action, name, resume=False):
    argv = [PYTHON, '-u', '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4',
            str(Path(manifest['runtime_source']) / WORK.name / RUNNER), action,
            '--root', manifest['root'], '--recipe', name, '--allocator-cap-mib', '0']
    if resume:
        argv += ['--resume', str(Path(manifest['root']) / name / 'last.pth')]
    return argv


def analysis_command(manifest, stage):
    """Explicit CPU analysis, never invoked by the training queue."""
    return [PYTHON, '-u', str(Path(manifest['runtime_source']) / WORK.name /
            'scripts/analysis/ue_staged_uav_analysis_20261028.py'),
            '--root', manifest['root'], '--stage', stage]


def require_analysis_inputs(manifest, stage):
    names = stage_recipes(manifest, stage)
    if not manifest.get('base_lock'):
        raise RuntimeError('Lock validation-selected base before inspecting test analysis')
    if stage != 'structure':
        names = names + [manifest['base_lock']['base_arm']]
    for name in names:
        if not accepted(manifest, 'train', name) or not accepted(manifest, 'eval', name):
            raise RuntimeError('Complete matched full training and test required for analysis: ' + name)
    return names


def environment():
    return dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3', OPENBLAS_NUM_THREADS='1',
                OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONUNBUFFERED='1')


def job_plan(manifest, stage, evaluation_only=False):
    names = stage_recipes(manifest, stage)
    if evaluation_only and stage == 'structure' and not manifest.get('base_lock'):
        raise RuntimeError('Lock validation-selected base before exposing structure test')
    actions = ('eval',) if evaluation_only else ('gate', 'train') if stage == 'structure' else ('gate', 'train', 'eval')
    jobs = {}
    for action in actions:
        for name in names:
            deps = [] if evaluation_only or action == 'gate' else ['gate_' + n for n in names] if action == 'train' else ['train_' + n for n in names]
            jobs[action + '_' + name] = dict(action=action, recipe=name, status='pending', dependencies=deps,
                                            marker=str(marker(manifest, action, name)))
    return jobs


def owner(root, stage, recover=False, evaluation_only=False):
    manifest = read(root / 'launch_manifest.json')
    validate(manifest)
    names = stage_recipes(manifest, stage)
    if evaluation_only:
        if stage == 'structure' and not manifest.get('base_lock'):
            raise RuntimeError('Lock validation-selected base before exposing structure test')
        if not all(accepted(manifest, 'train', name) for name in names):
            raise RuntimeError('All stage trainings must finish before evaluation')
    queue_name = stage + ('_eval' if evaluation_only else '')
    folder = root / 'queue' / queue_name
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / 'owner.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # A second stage owner must not compete over shared mutable manifests.
    global_lock = (root / 'active_stage.lock').open('a')
    fcntl.flock(global_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = folder / 'state.json'
    if state_path.exists() and not recover:
        raise RuntimeError('Prior queue state requires explicit recover and child audit')
    jobs = job_plan(manifest, stage, evaluation_only)
    if recover:
        old = read(state_path)
        for job in old['jobs'].values():
            rows = list(job.get('ranks', []))
            receipt = job.get('rank_launch_record')
            if receipt and Path(receipt).exists():
                rows += read(receipt).get('ranks', [])
            if live(job.get('launcher')) or any(live(row) for row in rows):
                raise RuntimeError('Prior launcher/rank still live; refusing recovery')
        write(folder / 'recovery' / (str(time.time_ns()) + '.json'), old)
    for job in jobs.values():
        if accepted(manifest, job['action'], job['recipe']):
            job['status'] = 'complete'
        elif recover and job['action'] == 'train' and (root / job['recipe'] / 'last.pth').exists():
            job['resume'] = True
    children, logs, failures = {}, {}, []
    me = identity(os.getpid())
    while True:
        for key, process in list(children.items()):
            rc = process.poll()
            if rc is None:
                continue
            logs.pop(key).close()
            del children[key]
            job = jobs[key]
            error = None
            try:
                ok = rc == 0 and accepted(manifest, job['action'], job['recipe'])
            except Exception as exc:
                ok, error = False, repr(exc)
            job.update(status='complete' if ok else 'failed', returncode=rc, ended=time.time(),
                       receipt_error=error)
            if not ok:
                failures.append(key + ' failed; preserve partial artifacts and audit')
        cards = {}
        if not failures:
            try:
                validate(manifest)
                for name in names:
                    _verify_cache_receipt(manifest, name)
                cards = resources(manifest, [jobs[k] for k in children])
            except Exception as exc:
                failures.append(repr(exc))
        if not failures:
            for key, job in jobs.items():
                if job['status'] != 'pending' or any(jobs[d]['status'] != 'complete' for d in job['dependencies']):
                    continue
                # Evaluation is serialized; training groups may co-rent all four cards.
                if job['action'] == 'eval' and children:
                    continue
                reserve = reservation(manifest, job['recipe'])
                if any(cards[g]['admissible_free_mib'] < reserve for g in GPUS):
                    continue
                if job['action'] == 'train':
                    try:
                        verify_initialization_family(manifest, job['recipe'])
                    except Exception as exc:
                        failures.append(repr(exc))
                        break
                log = folder / 'logs' / (key + '_' + str(time.time_ns()) + '.log')
                log.parent.mkdir(parents=True, exist_ok=True)
                stream = log.open('x')
                argv = command(manifest, job['action'], job['recipe'], job.get('resume', False))
                process = subprocess.Popen(argv, cwd=Path(manifest['runtime_source']) / WORK.name,
                                           env=environment(), stdin=subprocess.DEVNULL, stdout=stream,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                children[key], logs[key] = process, stream
                job.update(status='running', argv=argv, launcher=identity(process.pid),
                           rank_launch_record=str(root / 'rank_launches' / (job['action'] + '_' + job['recipe'] + '.json')),
                           log=str(log), ranks=[])
                for card in cards.values():
                    card['admissible_free_mib'] -= reserve
        done = all(j['status'] == 'complete' for j in jobs.values())
        write(state_path, dict(stage=stage, evaluation_only=evaluation_only,
                               status='complete' if done else 'failed' if failures else 'running_or_waiting_for_resources',
                               owner=me, jobs=jobs, failures=failures, GPU_headroom=cards, updated=time.time()))
        if done:
            write(folder / 'complete.json', dict(complete=True, stage=stage, recipes=names,
                  git_commit=manifest['git_commit'], no_automatic_transition_to_next_stage=True,
                  test_deferred_until_explicit_base_lock=stage == 'structure' and not evaluation_only))
            return
        if failures and not children:
            raise RuntimeError(str(failures))
        time.sleep(15)


def ensure_no_owner(root):
    for path in (root / 'queue').glob('*/state.json'):
        if live(read(path).get('owner')):
            raise RuntimeError('Stage owner active; use status/queue instead')


def queue_stage(manifest, stage, recover=False, evaluation_only=False):
    root = Path(manifest['root'])
    names = stage_recipes(manifest, stage)
    queue_name = stage + ('_eval' if evaluation_only else '')
    folder = root / 'queue' / queue_name
    state = folder / 'state.json'
    if state.exists() and live(read(state).get('owner')):
        if recover:
            raise RuntimeError('Live owner cannot be recovered')
        return dict(already_queued=True, stage=stage, owner=read(state)['owner'])
    ensure_no_owner(root)
    if state.exists() and not recover:
        raise RuntimeError('Prior state exists; explicit recover after audit required')
    if recover and not state.exists():
        raise RuntimeError('No prior queue state to recover')
    if evaluation_only:
        if stage == 'structure' and not manifest.get('base_lock'):
            raise RuntimeError('Lock validation-selected base before exposing structure test')
        if not all(accepted(manifest, 'train', name) for name in names):
            raise RuntimeError('Every training in this stage must finish before test')
    for name in names:
        _verify_cache_receipt(manifest, name, deep=True)
    folder.mkdir(parents=True, exist_ok=True)
    controller = Path(manifest['runtime_source']) / WORK.name / 'scripts/experiments/ue_staged_uav_control.py'
    argv = [PYTHON, '-u', str(controller), '_owner', '--root', str(root), '--stage', stage]
    if recover:
        argv += ['--recover-owner']
    if evaluation_only:
        argv += ['--evaluation-only']
    with (folder / 'owner.log').open('a') as stream:
        process = subprocess.Popen(argv, cwd=controller.parents[2], env=environment(), stdin=subprocess.DEVNULL,
                                   stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    return dict(queued=True, stage=stage, owner=identity(process.pid))


def start_stage(root, stage):
    """One explicit stage: CPU prepare/cache, then a detached GPU queue owner."""
    manifest = prepare(Path(root))
    stage_recipes(manifest, stage)  # Later stages never infer a scientific base.
    state = Path(manifest['root']) / 'queue' / stage / 'state.json'
    if state.exists():
        prior = read(state)
        if live(prior.get('owner')):
            return dict(already_queued=True, stage=stage, owner=prior['owner'])
        raise RuntimeError('Prior dead/completed stage state requires explicit recover; start never overwrites it')
    ensure_no_owner(Path(manifest['root']))
    # Hold the same lock as the owner while preparing caches; release before
    # spawning the child so it can acquire the stage lock without a race.
    with (Path(manifest['root']) / 'active_stage.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print('START_STAGE_CPU_PREPARATION', stage, flush=True)
        cache_stage(manifest, stage)
    return queue_stage(manifest, stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plan', 'audit', 'prepare', 'start', 'lock', 'cache', 'gate', 'train', 'eval', 'resume', 'queue', 'analyze', 'status', 'recover', '_owner'])
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--stage', choices=['structure', 'modality', 'views'], default='structure')
    parser.add_argument('--recipe')
    parser.add_argument('--base-arm', choices=['U', 'T', 'C'])
    parser.add_argument('--reason')
    parser.add_argument('--recover-owner', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--evaluation-only', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action in ('plan', 'audit'):
        api, matrix, protocol = load_setup(REPO)
        report = api.audit_matrix(matrix)
        print(json.dumps(dict(matrix=matrix, audit=report, protocol=protocol,
                              resolved_structure=checked_recipes(api, matrix), stage=args.stage,
                              all_jobs_four_gpu=True, automatic_stage_transition=False), indent=2))
        return
    if args.action == 'status':
        for path in [root / 'launch_manifest.json'] + sorted((root / 'queue').glob('*/state.json')):
            if path.exists():
                print(json.dumps(dict(path=str(path), data=read(path)), indent=2))
        return
    if args.action == '_owner':
        owner(root, args.stage, args.recover_owner, args.evaluation_only)
        return
    if args.action == 'start':
        print(json.dumps(start_stage(root, args.stage)))
        return
    manifest = prepare(root)
    if args.action == 'prepare':
        print(json.dumps(manifest, indent=2))
        return
    if args.action == 'analyze':
        require_analysis_inputs(manifest, args.stage)
        argv = analysis_command(manifest, args.stage)
        # CPU analyses can coexist with unrelated healthy training groups.
        os.execve(PYTHON, argv, dict(environment(), CUDA_VISIBLE_DEVICES=''))
    evaluation_only = args.evaluation_only or (args.action == 'eval' and args.recipe is None)
    if args.action == 'eval' and args.stage == 'structure' and not manifest.get('base_lock'):
        raise RuntimeError('Lock validation-selected base before exposing structure test')
    queue_name = args.stage + ('_eval' if evaluation_only else '')
    own_state = root / 'queue' / queue_name / 'state.json'
    if args.action == 'queue' and own_state.exists() and live(read(own_state).get('owner')):
        print(json.dumps(dict(already_queued=True, stage=args.stage, owner=read(own_state)['owner'])))
        return
    ensure_no_owner(root)
    if args.action == 'lock':
        if not args.base_arm or not args.reason:
            parser.error('lock requires explicit --base-arm U/T/C and --reason from validation/engineering evidence')
        with (root / 'active_stage.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            manifest = read(root / 'launch_manifest.json')
            validate(manifest)
            lock_base(manifest, args.base_arm, args.reason)
        return
    if args.action == 'cache':
        with (root / 'active_stage.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            manifest = read(root / 'launch_manifest.json')
            validate(manifest)
            cache_stage(manifest, args.stage)
        return
    names = stage_recipes(manifest, args.stage)
    for name in names:
        _verify_cache_receipt(manifest, name, deep=True)
    if args.action in ('queue', 'recover') or (args.action == 'eval' and args.recipe is None):
        print(json.dumps(queue_stage(manifest, args.stage, args.action == 'recover', evaluation_only)))
        return
    if args.recipe not in names:
        parser.error('--recipe must belong to the explicitly selected stage: ' + str(names))
    action = 'train' if args.action == 'resume' else args.action
    if action in ('train', 'eval') and not accepted(manifest, 'gate', args.recipe):
        raise RuntimeError('Matching four-GPU recipe gate required')
    if action == 'train':
        verify_initialization_family(manifest, args.recipe)
    if action == 'eval' and not all(accepted(manifest, 'train', name) for name in names):
        raise RuntimeError('Every training in this stage must finish before test')
    if args.action == 'resume' and not (root / args.recipe / 'last.pth').exists():
        raise RuntimeError('No same-recipe last checkpoint')
    cards = resources(manifest, [])
    if any(cards[g]['admissible_free_mib'] < reservation(manifest, args.recipe) for g in GPUS):
        raise RuntimeError('Insufficient common four-GPU headroom; use queue for background waiting')
    argv = command(manifest, action, args.recipe, args.action == 'resume')
    os.execve(PYTHON, argv, environment())


if __name__ == '__main__':
    main()
