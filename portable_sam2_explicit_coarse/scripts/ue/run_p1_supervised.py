#!/usr/bin/env python3
"""Detached supervisor for the explicit-identity UE-WHU P1 runner."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent

def stamp():
    return datetime.datetime.now().astimezone().isoformat()

def supervise():
    output = Path(sys.argv[2]).resolve()
    arguments = sys.argv[3:]
    if '--output-dir' not in arguments or Path(arguments[arguments.index('--output-dir') + 1]).resolve() != output:
        raise RuntimeError('supervisor and runner output directories must match')
    output.mkdir(parents=True, exist_ok=True)
    status = output.parent / (output.name + '_evaluation_runtime.json')
    argv = [sys.executable, '-u', str(HERE / 'probe_surfacebev_p1.py'), *arguments]
    start = stamp()
    status.write_text(json.dumps({'state': 'starting', 'started_at': start, 'argv': argv}))
    child = subprocess.Popen(argv)
    while True:
        code = child.poll()
        record = {'timestamp': stamp(), 'started_at': start, 'argv': argv,
                  'supervisor_pid': os.getpid(), 'worker_pid': child.pid,
                  'state': 'running' if code is None else 'exited', 'exit_code': code,
                  'test_scenes_cached': len(list((output / 'cache/test').glob('*.npz'))),
                  'test_lock_exists': (output / 'cache/test_opened_once.json').exists()}
        tmp = status.with_suffix('.tmp')
        try:
            tmp.write_text(json.dumps(record, indent=2))
            os.replace(tmp, status)
        except Exception:
            child.terminate()
            child.wait()
            raise
        print(f'{stamp()} HEARTBEAT state={record["state"]} test={record["test_scenes_cached"]}/151', flush=True)
        if code is not None:
            print(f'{stamp()} WORKER_EXIT code={code}', flush=True)
            sys.exit(code if code >= 0 else 128 - code)
        time.sleep(10)

if __name__ == '__main__':
    supervise()
