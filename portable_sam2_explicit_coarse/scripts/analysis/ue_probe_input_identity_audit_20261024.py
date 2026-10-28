"""Read-only archive checksum and native-cache source binding census."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import numpy as np

DATA=Path('/data/wangcheng/dataset/CitySample_SatUAV_TrainMinimal_v1_20261021')
CACHE=Path('/data/wangcheng/cache/ue_v1_native_geometry')
OUT=Path('/data/wangcheng/checkpoint/ue_probe_input_identity_audit_20261024')


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    expected={}
    for line in (DATA/'SHA256SUMS.txt').read_text().splitlines():
        h,name=line.split('  ',1);assert len(h)==64 and not Path(name).is_absolute() and '..' not in Path(name).parts
        assert name not in expected;expected[name]=h
    def check(item):
        name,h=item;p=DATA/name
        actual=sha(p) if p.is_file() else None
        return name,actual,actual==h
    with ThreadPoolExecutor(max_workers=8) as pool:checks=list(pool.map(check,expected.items()))
    actual={name:h for name,h,_ in checks}
    mismatches=[dict(path=n,expected=expected[n],actual=h) for n,h,ok in checks if not ok]
    ids={s:json.loads((DATA/'splits'/f'{s}.json').read_text()) for s in ['train','val','test']}
    flattened=[i for group in ids.values() for i in group];assert len(flattened)==len(set(flattened))==3060
    def cache_check(ident):
        p=CACHE/f'{ident:06d}.npz'
        if not p.is_file():return dict(image_id=ident,error='missing_cache')
        with np.load(p) as z:
            version=str(z['version']);report=json.loads(str(z['report']))
        bindings=[]
        for name,h in report['source_hashes'].items():
            relative=f'points/{ident:06d}/{name}'
            if h!=expected.get(relative) or actual.get(relative)!=h:
                bindings.append(dict(source=relative,cache_expected=h,archive_expected=expected.get(relative),actual=actual.get(relative)))
        return dict(image_id=ident,version=version,cache_sha256=sha(p),sources=len(report['source_hashes']),source_mismatches=bindings)
    with ThreadPoolExecutor(max_workers=8) as pool:cache_checks=list(pool.map(cache_check,flattened))
    errors=[r for r in cache_checks if 'error' in r or r['version']!='native512_height_dx_dy_viewcount_v1' or r['sources']!=6 or r['source_mismatches']]
    summary=dict(complete=True,passed=not mismatches and not errors,
        archive_files=len(expected),archive_mismatches=mismatches,
        split_counts={s:len(v) for s,v in ids.items()},cache_files=len(cache_checks),cache_errors=errors,
        manifest_sha256=sha(DATA/'SHA256SUMS.txt'),script_sha256=sha(Path(__file__)),
        scope='Archive bytes checked against frozen checksums, all native cache camera+5 depth hashes bound to same archive. '
              'Does not recompute every geometric field or certify external camera calibration, visibility, or label semantics. '
              'No modification of dataset/cache; no label decoding or training use.')
    (OUT/'archive_file_checks.json').write_text(json.dumps(checks))
    (OUT/'cache_checks.json').write_text(json.dumps(cache_checks))
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
