"""Reproducible experiment runner. All generated artifacts stay in a unique run."""
import argparse
import csv
import hashlib
import itertools
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from stats import summarize, effective_gbps

ROOT = Path(__file__).resolve().parents[1]
PHASE1 = ROOT.parent
TZ = timezone(timedelta(hours=8))
FIELDS = ['id', 'op', 'n', 'algorithm', 'threads', 'items', 'grid', 'layout', 'allocation',
          'pattern', 'rounds', 'warmups', 'repeats', 'group']
PARAMS = ['op', 'algorithm', 'threads', 'items', 'grid', 'layout']
SEED = 20261117
SIZES = [33, 257, 4096, 65537, 1048579, 8388611, 33554435]
TUNE_SIZES = [257, 65537, 1048579, 8388611]


def stamp():
    return datetime.now(TZ).isoformat()


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, data, fields=None):
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields or list(data[0]))
        w.writeheader()
        w.writerows(data)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def phase1_hashes():
    files = [PHASE1 / 'build.cmd', PHASE1 / 'run-tests.ps1']
    for folder in ['src', 'include', 'tests']:
        files += sorted((PHASE1 / folder).rglob('*'))
    return {str(p.relative_to(PHASE1)): sha(p) for p in files if p.is_file()}


def command(args, logfile, timeout=1800, check=True):
    started = stamp()
    if logfile.exists():
        previous = logfile.with_name(logfile.stem + '.prior-' + uuid.uuid4().hex[:8] + logfile.suffix)
        logfile.rename(previous)
        prior_command = logfile.with_suffix('.command.json')
        if prior_command.exists():
            prior_command.rename(previous.with_suffix('.command.json'))
    with logfile.open('w', encoding='utf-8') as f:
        child_env = os.environ.copy()
        if Path(str(args[0])).name.lower() == 'powershell.exe':
            # Codex can inherit PowerShell 7 module paths; Windows PowerShell 5
            # must resolve its own Utility module (Get-FileHash) first.
            child_env['PSModulePath'] = str(Path(os.environ.get('SystemRoot','C:/Windows')) / 'System32/WindowsPowerShell/v1.0/Modules')
        p = subprocess.run([str(x) for x in args], stdout=f, stderr=subprocess.STDOUT,
                           cwd=ROOT, timeout=timeout, shell=False, env=child_env)
    save(logfile.with_suffix('.command.json'), dict(argv=[str(x) for x in args], started=started,
         finished=stamp(), exit_code=p.returncode,
         child_powershell_module_path=child_env.get('PSModulePath') if Path(str(args[0])).name.lower()=='powershell.exe' else None))
    if check and p.returncode:
        raise RuntimeError(f'Command exit {p.returncode}: {logfile}')
    return p.returncode


def discover_profiler():
    candidates = []
    if shutil.which('ncu'):
        candidates.append(Path(shutil.which('ncu')))
    roots = [Path('C:/Program Files/NVIDIA Corporation'), Path('C:/Program Files/NVIDIA GPU Computing Toolkit'),
             Path('C:/Program Files (x86)/NVIDIA Corporation')]
    for base in [Path('C:/Program Files'), Path('C:/Program Files (x86)')]:
        roots += list(base.glob('*Nsight*'))
    errors = []
    for root in roots:
        if root.exists():
            try:
                candidates += list(root.rglob('ncu.exe'))
            except OSError as exc:
                errors.append(str(exc))
    return dict(path=str(candidates[0]) if candidates else None,
                searched=['PATH'] + [str(x) for x in roots], errors=errors,
                status='FOUND' if candidates else 'NOT_FOUND',
                missing_metrics=[] if candidates else ['achieved_occupancy', 'dram_throughput', 'warp_stall',
                                                     'profiled_register_usage', 'profiled_shared_usage', 'memory_access_efficiency'])


def telemetry(run, stop):
    path = run / 'telemetry.csv'
    query = 'name,uuid,pstate,temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu,utilization.memory'
    with path.open('a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        if path.stat().st_size == 0:
            writer.writerow(['timestamp', 'query_result'])
        while not stop.is_set():
            try:
                p = subprocess.run(['nvidia-smi', f'--query-gpu={query}', '--format=csv,noheader'],
                                   capture_output=True, text=True, timeout=10)
                writer.writerow([stamp(), p.stdout.strip() or p.stderr.strip()]); f.flush()
            except Exception as exc:
                writer.writerow([stamp(), str(exc)]); f.flush()
            stop.wait(3)


def create_run():
    run = ROOT / 'runs' / (datetime.now(TZ).strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8])
    run.mkdir(parents=True)
    save(run / 'phase1-before.json', phase1_hashes())
    save(run / 'environment.json', dict(started=stamp(), python=sys.version, os=platform.platform(),
         cpu=platform.processor(), seed=SEED, warmups=10, rounds=15, repeats=10,
         sizes=SIZES, tune_sizes=TUNE_SIZES, cache_policy='warmed_reused_no_flush', compiler_flags='-arch=sm_120 -std=c++17 -O3 -lineinfo'))
    snapshot = run / 'source'
    for folder in ['src', 'scripts', 'tests']:
        shutil.copytree(ROOT / folder, snapshot / 'phase2' / folder, ignore=shutil.ignore_patterns('__pycache__'))
    for name in ['build.cmd', 'DESIGN.md']:
        shutil.copy2(ROOT / name, snapshot / 'phase2' / name)
    for folder in ['src', 'include', 'tests']:
        shutil.copytree(PHASE1 / folder, snapshot / 'phase1' / folder)
    save(run / 'source-sha256.json', {str(p.relative_to(snapshot)): sha(p) for p in snapshot.rglob('*') if p.is_file()})
    save(run / 'profiler.json', discover_profiler())
    command(['nvidia-smi', '-q'], run / 'nvidia-smi.txt')
    return run


def job(op='sum', n=1048579, algorithm='shuffle', threads=256, items=1, grid=0,
        layout='striped', allocation='pageable_reuse', pattern='mixed_exact', rounds=15,
        warmups=10, repeats=10, group=''):
    return dict(id='', op=op, n=n, algorithm=algorithm, threads=threads, items=items, grid=grid,
                layout=layout, allocation=allocation, pattern=pattern, rounds=rounds,
                warmups=warmups, repeats=repeats, group=group)


def configs(sm):
    for op, algorithm, threads, items, grid, layout in itertools.product(
            ['sum', 'max', 'min'], ['shared', 'shuffle'], [128, 256, 512], [1, 4, 8],
            [0, 2 * sm, 8 * sm], ['striped', 'contiguous']):
        yield dict(op=op, algorithm=algorithm, threads=threads, items=items, grid=grid, layout=layout)


def key(row):
    return tuple(str(row[k]) for k in PARAMS)


def run_plan(run, name, plan, mode='bench'):
    directory = run / name
    directory.mkdir()  # Never silently overwrite an existing stage.
    for i, j in enumerate(plan):
        j['id'] = f'{name}-{i:06d}'
    write_csv(directory / 'plan.csv', plan, FIELDS)
    print(f'{name}: {len(plan)} jobs', flush=True)
    command([run / 'build' / 'whalegpu_bench.exe', mode, directory / 'plan.csv', directory], directory / 'execution.log')
    check_rows = rows(directory / 'correctness.csv')
    if len(check_rows) != len(plan) or any(x['correct'] != 'PASS' for x in check_rows):
        raise RuntimeError(f'{name}: incomplete/failed correctness')
    if mode == 'bench':
        aggregate(directory)
    print(f'{name}: PASS', flush=True)
    return directory


def aggregate(directory):
    plan = {j['id']: j for j in rows(directory / 'plan.csv')}
    samples = {}
    for row in rows(directory / 'samples.csv'):
        if row['id'] not in plan or row['correct'] != 'PASS':
            raise ValueError('Unrecognized/failed sample')
        samples.setdefault(row['id'], []).append(row)
    output = []
    resources = {x['id']: x for x in rows(directory / 'resources.csv')}
    for ident, j in plan.items():
        data = samples.get(ident, [])
        if len(data) != int(j['rounds']) or len({x['round'] for x in data}) != len(data):
            raise ValueError(f'Incomplete samples: {ident}')
        row = dict(j)
        for metric in ['kernel', 'e2e']:
            row.update({f'{metric}_{k}': v for k, v in summarize([float(x[f'{metric}_ms']) for x in data]).items()})
        row['effective_gbps'] = effective_gbps(int(j['n']), row['kernel_median'])
        row.update({k: v for k, v in resources[ident].items() if k != 'id'})
        row['correct'] = 'PASS'
        output.append(row)
    write_csv(directory / 'summary.csv', output)
    return output


def engine(run):
    command([sys.executable, '-B', '-m', 'unittest', 'discover', '-s', ROOT / 'tests', '-v'], run / 'statistics-tests.log')
    command([os.environ.get('COMSPEC', 'cmd.exe'), '/d', '/c', 'build.cmd', (run / 'build').relative_to(ROOT)], run / 'build.log')
    save(run / 'binary-sha256.json', {'whalegpu_bench.exe': sha(run / 'build' / 'whalegpu_bench.exe')})
    plan = [job(algorithm=a, allocation=alloc, group='engine') for a, alloc in itertools.product(
        ['phase1_shared', 'phase1_shuffle', 'shared', 'shuffle'], ['pageable_reuse', 'pinned_reuse', 'pageable_fresh'])]
    run_plan(run, 'engine', plan)


def validate(run):
    sm = read(run / 'engine' / 'hardware.json')['sm_count']
    plan = []
    for n, pattern in itertools.product([0,1,2,3,31,32,33,127,128,129,255,256,257,511,512,513,1024,65537],
                                        ['mixed_exact','negative','positive','constant','tail_extrema','fractional']):
        for c in configs(sm):
            plan.append(job(**c, n=n, pattern=pattern, rounds=1, warmups=1, repeats=1, group='eligibility'))
    directory = run_plan(run, 'validate', plan, 'validate')
    save(directory / 'eligible.json', [dict(c) for c in configs(sm)])


def matrix(run):
    plan = []
    rng = random.Random(SEED)
    for n in SIZES:
        batch = []
        for op, allocation in itertools.product(['sum','max','min'], ['pageable_reuse','pinned_reuse','pageable_fresh']):
            for algorithm in ['phase1_shared','phase1_shuffle']:
                batch.append(job(op=op,n=n,algorithm=algorithm,allocation=allocation,group='matrix'))
            for algorithm, threads in itertools.product(['shared','shuffle'],[128,256,512]):
                batch.append(job(op=op,n=n,algorithm=algorithm,threads=threads,allocation=allocation,group='matrix'))
        rng.shuffle(batch); plan += batch
    run_plan(run, 'matrix', plan)


def tune(run):
    eligible = read(run / 'validate' / 'eligible.json')
    rng = random.Random(SEED + 1)
    plan = []
    for n in TUNE_SIZES:
        batch = [job(**c, n=n, group='search') for c in eligible]
        rng.shuffle(batch); plan += batch
    directory = run_plan(run, 'tune', plan)
    summary = rows(directory / 'summary.csv')
    winners = []
    for n, op in itertools.product(TUNE_SIZES, ['sum','max','min']):
        candidates = [x for x in summary if int(x['n']) == n and x['op'] == op and x['correct'] == 'PASS']
        winners.append(min(candidates, key=lambda x: float(x['kernel_median'])))
    save(run / 'winners.json', winners)


def confirm(run):
    winners = read(run / 'winners.json')
    rng = random.Random(SEED + 2)
    plan = []
    for winner in winners:
        for pair in range(21):
            group = f"{winner['op']}-{winner['n']}-pair{pair}"
            selected = {p: int(winner[p]) if p in ['threads','items','grid'] else winner[p] for p in PARAMS}
            batch = [job(**selected,n=int(winner['n']),rounds=1,group=group)]
            batch += [job(op=winner['op'],n=int(winner['n']),algorithm=a,rounds=1,group=group)
                      for a in ['phase1_shared','phase1_shuffle']]
            rng.shuffle(batch); plan += batch
    run_plan(run, 'confirm', plan)


def profiler(run):
    info = read(run / 'profiler.json')
    if not info['path']:
        print('Profiler: NOT_FOUND; hardware counters unavailable', flush=True)
        return
    directory = run / 'profiler'
    directory.mkdir()
    plan = []
    for winner in read(run / 'winners.json'):
        if int(winner['n']) != max(TUNE_SIZES):
            continue
        selected = {p: int(winner[p]) if p in ['threads','items','grid'] else winner[p] for p in PARAMS}
        plan.append(job(**selected,n=int(winner['n']),rounds=1))
        plan += [job(op=winner['op'],n=int(winner['n']),algorithm=a,rounds=1) for a in ['phase1_shared','phase1_shuffle']]
    for i,j in enumerate(plan): j['id'] = f'profile-{i}'
    write_csv(directory / 'plan.csv',plan,FIELDS)
    sections = ['LaunchStats','Occupancy','SpeedOfLight','MemoryWorkloadAnalysis','WarpStateStats']
    args = [info['path'],'--target-processes','all','--export',directory/'kernels']
    for section in sections: args += ['--section',section]
    args += [run/'build'/'whalegpu_bench.exe','profile',directory/'plan.csv',directory]
    code = command(args,directory/'ncu.log',check=False)
    info.update(status='COLLECTED' if code==0 else 'COLLECTION_FAILED',exit_code=code,sections=sections)
    if code==0:
        command([info['path'],'--import',directory/'kernels.ncu-rep','--csv','--page','raw'],directory/'metrics.csv',check=False)
    else:
        info['missing_metrics']=['achieved_occupancy','dram_throughput','warp_stall','profiled_register_usage','profiled_shared_usage','memory_access_efficiency']
    save(run/'profiler.json',info)


def regression(run):
    ps = 'C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe'
    command([ps,'-NoProfile','-ExecutionPolicy','Bypass','-File',PHASE1/'run-tests.ps1','-Stage','all'],run/'phase1-regression.log')
    before=read(run/'phase1-before.json'); after=phase1_hashes()
    save(run/'phase1-integrity.json',dict(unchanged=before==after,before=before,after=after))
    if before!=after: raise RuntimeError('Phase 1 source changed')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['engine','validate','matrix','tune','confirm','profiler','regression','report','all'],default='all')
    parser.add_argument('--run',type=Path)
    args=parser.parse_args()
    run=args.run.resolve() if args.run else create_run()
    print(f'RUN={run}',flush=True)
    stop=threading.Event(); monitor=threading.Thread(target=telemetry,args=(run,stop),daemon=True); monitor.start()
    stages=['engine','validate','matrix','tune','confirm','profiler','regression','report'] if args.stage=='all' else [args.stage]
    try:
        for stage in stages:
            if stage=='report':
                from report import generate
                generate(run)
            else: globals()[stage](run)
            save(run/f'status-{stage}.json',dict(status='PASS',finished=stamp()))
    except Exception as exc:
        save(run/f'failure-{args.stage}-{int(time.time())}.json',dict(status='FAIL',error=str(exc),time=stamp()))
        raise
    finally:
        stop.set(); monitor.join(timeout=12)
    print('DONE',flush=True)


if __name__=='__main__':
    main()
