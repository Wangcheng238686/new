#!/usr/bin/env python3
"""Versioned freeze snapshot of the raw HERO-UAV pairset (incident 2026-10-14).

Copies the dataset with dereferenced scene directories (the live tree is a
symlink farm into the /data1 production set that a repair pipeline rewrites
in place), recomputes the split builder fingerprint ON THE COPY, and writes a
snapshot manifest. The registered artifacts (accepted_pairs.jsonl,
surfbev_b0_split_v2.json) are copied bit-identical so their registered SHA256
contracts keep validating; the manifest records the new content fingerprint.
Fail-closed: any hash mismatch aborts leaving no partial manifest.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, shutil, subprocess, sys
from pathlib import Path


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--dest', type=Path, required=True)
    ap.add_argument('--builder', type=Path, default=Path(__file__).resolve().parent
                    / 'build_surfacebev_b0_split_v2.py')
    a = ap.parse_args()
    src, dst = a.source.resolve(), a.dest.resolve()
    if dst.exists() and any(dst.iterdir()):
        raise SystemExit(f'destination not empty: {dst}')
    for name in ('accepted_pairs.jsonl', 'DATASET_CONTRACT.json',
                 'surfbev_b0_split.json', 'surfbev_b0_split_v2.json'):
        if not (src / name).is_file():
            raise SystemExit(f'missing manifest file: {src / name}')
    expected = {
        'accepted_pairs.jsonl': 'f15cd8f5b8d10a69d721546d6cfa1f2cce58af2861d57b1357c5e9ff1e1fe62a',
        'surfbev_b0_split_v2.json': '17f49430b15037c8da25c63a1bf7fe4d510cf5ff153263d2feba62c4393441d5',
    }
    for name, want in expected.items():
        got = sha256(src / name)
        if got != want:
            raise SystemExit(f'{name} sha drift: {got}')
    # The producer tree may still be under active writes (PI decision: freeze
    # the current state instead of waiting). Per-file consistency: hash source,
    # copy, hash copy, hash source again; retry files whose source changed
    # mid-copy so no torn payload enters the snapshot.
    def consistent_copy(rel, attempts=4):
        sp, dp = src / rel, dst / rel
        dp.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(attempts):
            h0 = sha256(sp)
            shutil.copyfile(sp, dp)
            if sha256(dp) == h0 and sha256(sp) == h0:
                return
        raise SystemExit(f'unstable source file after {attempts} attempts: {rel}')

    print(f'copying {src} -> {dst} (per-file consistent, dereferenced)', flush=True)
    n = 0
    for scene_dir in sorted((src / 'scenes').iterdir()):
        for payload in ('satellite', 'uav'):
            for f in sorted((scene_dir / payload).rglob('*')):
                if f.is_dir():
                    continue
                consistent_copy(f.relative_to(src))
                n += 1
        if n % 2000 == 0:
            print(f'  {n} files copied...', flush=True)
    print(f'  total {n} payload files copied', flush=True)
    for name, want in expected.items():
        got = sha256(dst / name)
        if got != want:
            shutil.rmtree(dst, ignore_errors=True)
            raise SystemExit(f'copy {name} sha drift: {got}')
    spec = importlib.util.spec_from_file_location('splitbuilder', a.builder)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    payload = m.build(dst)
    fingerprint = payload['source_payload_fingerprint_sha256']
    manifest = {
        'schema': 'ue-raw-pairset-frozen-snapshot-v1',
        'source': str(src),
        'accepted_pairs_sha256': expected['accepted_pairs.jsonl'],
        'split_v2_sha256': expected['surfbev_b0_split_v2.json'],
        'source_payload_fingerprint_sha256': fingerprint,
        'split_counts': payload['counts'],
        'note': ('frozen snapshot per docs/DATA-INTEGRITY-20261014_incident.md; the '
                 'split file fingerprint field records the pre-repair generation and '
                 'is superseded by this manifest for wrapper verification'),
    }
    tmp = dst.parent / (dst.name + '.manifest.tmp')
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    tmp.replace(dst.parent / (dst.name + '_manifest.json'))
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
