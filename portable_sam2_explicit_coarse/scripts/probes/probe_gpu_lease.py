"""Cooperative lock for a single in-flight interim validation on GPU2."""
import json
import os
from pathlib import Path


def acquire(path):
    path=Path(path)
    try:
        fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    except FileExistsError:
        try:owner=json.loads(path.read_text())['pid']
        except (ValueError,KeyError,FileNotFoundError):return False
        try:os.kill(owner,0)
        except ProcessLookupError:
            try:path.unlink()
            except FileNotFoundError:pass
        return False
    with os.fdopen(fd,'w') as f:json.dump(dict(pid=os.getpid()),f)
    return True


def release(path):
    path=Path(path)
    if path.is_file() and json.loads(path.read_text())['pid']==os.getpid():path.unlink()
