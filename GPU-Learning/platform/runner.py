"""Native CUDA runner. No user-provided command strings are executed."""
import json
import os
from pathlib import Path
import subprocess
import time
import uuid
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'records'
NVCC = Path(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin\nvcc.exe')
VCVARS = r'C:\BuildTools2022\VC\Auxiliary\Build\vcvars64.bat'
EXPERIMENTS = {'vector-add': ROOT / '01-vector-add' / 'main.cu'}


def process(args, timeout=120, env=None):
    start = time.perf_counter()
    proc = subprocess.Popen(args, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        subprocess.run(['taskkill', '/PID', str(proc.pid), '/T', '/F'], capture_output=True,
                       creationflags=subprocess.CREATE_NO_WINDOW)
        out, err = proc.communicate()
        raise RuntimeError(f'Process timed out after {timeout}s; process tree terminated')
    return {'exit_code': proc.returncode, 'stdout': out.decode(errors='replace'),
            'stderr': err.decode(errors='replace'), 'process_ms': (time.perf_counter()-start)*1000}


def environment():
    # Only a fixed trusted script enters cmd.exe; GUI inputs never enter shell text.
    result = process(['cmd.exe', '/d', '/s', '/c',
                      'platform\\environment.cmd'])
    env = os.environ.copy()
    for line in result['stdout'].splitlines():
        key, sep, value = line.partition('=')
        if sep and key:
            env[key] = value
    return env


def execute(action, experiment='vector-add', repeats=5):
    if experiment not in EXPERIMENTS or action not in ('build', 'run', 'benchmark'):
        raise ValueError('Unknown action or experiment')
    if type(repeats) is not int or not 1 <= repeats <= 20:
        raise ValueError('Repeat count must be 1..20')
    build = ROOT / 'build' / 'platform'
    build.mkdir(parents=True, exist_ok=True)
    exe = build / 'vector_add.exe'
    record = {'id': uuid.uuid4().hex, 'time': datetime.now(timezone.utc).isoformat(),
              'experiment': experiment, 'action': action, 'samples': [], 'passed': False,
              'source': EXPERIMENTS[experiment].read_text(encoding='utf-8')}
    try:
        compiled = process([str(NVCC), '-arch=sm_120', '-std=c++17',
                            str(EXPERIMENTS[experiment]), '-o', str(exe)], env=environment())
        record['compile'] = compiled
        if compiled['exit_code']:
            raise RuntimeError('nvcc compilation failed; see compiler output')
        if action != 'build':
            for _ in range(repeats if action == 'benchmark' else 1):
                result = process([str(exe)], timeout=30)
                result['passed'] = result['exit_code'] == 0 and 'mismatches=0' in result['stdout'] and 'PASS:' in result['stdout']
                record['samples'].append(result)
                if not result['passed']:
                    raise RuntimeError('GPU correctness check failed')
        record['passed'] = True
    except Exception as error:
        record['error'] = str(error)
    DATA.mkdir(exist_ok=True)
    target = DATA / (record['id']+'.json')
    tmp = target.with_suffix('.tmp')
    tmp.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding='utf-8')
    tmp.replace(target)
    return record

