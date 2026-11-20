"""Phase 3 orchestration; imports Phase 2 helpers read-only."""
import argparse
import csv
import hashlib
import itertools
import json
import os
import platform
import random
import shutil
import sys
import threading
import uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PHASE1=ROOT.parent
PHASE2=PHASE1/'phase2'
sys.path.insert(0,str(PHASE2/'scripts'))
import run as phase2
from stats import summarize
save,read,rows,write_csv,sha,command=phase2.save,phase2.read,phase2.rows,phase2.write_csv,phase2.sha,phase2.command

SEED=20261117
NAIVE=['naive_col_row','naive_row_row','naive_col_bt','naive_row_bt']
TILED=[f'tiled{t}_{layout}' for layout in ['row','bt'] for t in [8,16,32]]
VARIANTS=NAIVE+TILED
BOUNDARIES=[(0,7,9),(7,0,9),(7,9,0),(1,1,1),(1,17,3),(19,1,5),(7,9,5),(8,8,8),(9,7,11),
            (15,17,13),(16,16,16),(17,15,19),(31,33,29),(32,32,32),(33,31,35),(65,47,19),
            (127,65,33),(129,97,63),(257,193,129),(513,257,65)]
SHAPES=[(n,n,n) for n in [32,64,128,256,512,1024]]+[(31,47,19),(127,65,33),(257,193,129),(513,769,257),(1025,511,257)]
SCENARIOS=[('random',s) for s in [17,2026,65537]]+[('zero',17),('identity',17),('tail',17)]
FIELDS=['id','variant','m','n','k','seed','pattern','allocation','rounds','warmups','repeats','group','role']


def flops(m,n,k):
    if min(m,n,k)<0: raise ValueError('Negative dimension')
    return 2*int(m)*int(n)*int(k)


def tflops(work,ms):
    if work<=0 or ms<=0: raise ValueError('Nonpositive work/time')
    return work/(ms*1e9)


def history_hashes():
    result={}
    for directory,subdirs,files in os.walk(PHASE1):
        subdirs[:]=[x for x in subdirs if not (Path(directory)==PHASE1 and x=='phase3')]
        for name in files:
            path=Path(directory)/name
            result[str(path.relative_to(PHASE1))]=sha(path)
    return result


def new_run():
    ident=phase2.datetime.now(phase2.TZ).strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:8]
    run=ROOT/'runs'/ident;run.mkdir(parents=True)
    save(run/'legacy-files-before.json',history_hashes())
    snapshot=run/'source'
    for owner,base in [('phase3',ROOT),('phase2',PHASE2)]:
        for folder in ['src','scripts','tests']:
            shutil.copytree(base/folder,snapshot/owner/folder,ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copy2(base/'build.cmd',snapshot/owner/'build.cmd')
    for folder in ['src','include','tests']:
        shutil.copytree(PHASE1/folder,snapshot/'phase1'/folder)
    shutil.copy2(ROOT/'DESIGN.md',snapshot/'phase3'/'DESIGN.md')
    save(run/'source-sha256.json',{str(p.relative_to(snapshot)):sha(p) for p in snapshot.rglob('*') if p.is_file()})
    save(run/'environment.json',dict(started=phase2.stamp(),os=platform.platform(),cpu=platform.processor(),
         python=sys.version,seed=SEED,warmups=10,rounds=15,repeats=10,shapes=SHAPES,
         flags='-arch=sm_120 -std=c++17 -O3 -lineinfo -Xcompiler /openmp',cache='warmed/reused; no flush',
         e2e='CPU packing for BT + A/B H2D + GEMM + C D2H + sync; reused pageable buffers; allocation excluded'))
    save(run/'profiler.json',phase2.discover_profiler())
    sanitizer=shutil.which('compute-sanitizer')
    if not sanitizer:
        candidate=Path('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/bin/compute-sanitizer.exe')
        if candidate.exists(): sanitizer=str(candidate)
    save(run/'sanitizer.json',dict(path=sanitizer,status='FOUND' if sanitizer else 'NOT_FOUND',
         racecheck='NOT_RUN',synccheck='NOT_RUN'))
    command(['nvidia-smi','-q'],run/'nvidia-smi.txt')
    return run


def job(variant,shape,pattern='random',seed=SEED,rounds=15,group='matrix',role='candidate'):
    return dict(id='',variant=variant,m=shape[0],n=shape[1],k=shape[2],seed=seed,pattern=pattern,
                allocation='pageable_reuse',rounds=rounds,warmups=10,repeats=10,group=group,role=role)


def validation_plan(variants):
    return [job(v,shape,pattern,seed,rounds=1,group='validation') for shape in BOUNDARIES
            for pattern,seed in SCENARIOS for v in variants]


def aggregate(directory):
    plan=rows(directory/'plan.csv');ids={x['id']:x for x in plan}
    if len(ids)!=len(plan): raise ValueError('Duplicate IDs')
    check=rows(directory/'correctness.csv')
    if len(check)!=len(plan) or {x['id'] for x in check}!=set(ids) or any(x['correct']!='PASS' for x in check):
        raise ValueError('Missing/failed correctness')
    grouped={ident:[] for ident in ids}
    for x in rows(directory/'samples.csv'):
        if x['id'] not in ids or x['correct']!='PASS': raise ValueError('Unknown/failed sample')
        grouped[x['id']].append(x)
    resources={x['id']:x for x in rows(directory/'resources.csv')}
    output=[]
    for ident,j in ids.items():
        samples=grouped[ident]
        if len(samples)!=int(j['rounds']) or len({x['round'] for x in samples})!=len(samples):
            raise ValueError('Missing/duplicate samples')
        r=dict(j)
        for metric in ['kernel','e2e','pack']:
            r.update({metric+'_'+k:v for k,v in summarize([float(x[metric+'_ms']) for x in samples]).items()})
        r['flops']=flops(int(j['m']),int(j['n']),int(j['k']))
        r['kernel_tflops']=tflops(r['flops'],r['kernel_median'])
        r['e2e_tflops']=tflops(r['flops'],r['e2e_median'])
        r.update({k:v for k,v in resources[ident].items() if k!='id'})
        r['max_abs_error']=max(float(x['max_abs_error']) for x in samples)
        r['max_error_ratio']=max(float(x['max_error_ratio']) for x in samples)
        r['correct']='PASS';output.append(r)
    write_csv(directory/'summary.csv',output)
    return output


def execute(run,name,plan,mode='bench'):
    directory=run/name;directory.mkdir()
    for i,j in enumerate(plan):j['id']=f'{name}-{i:05d}'
    write_csv(directory/'plan.csv',plan,FIELDS)
    print(f'{name}: {len(plan)} jobs',flush=True)
    command([run/'build'/'whalegpu_gemm.exe',mode,directory/'plan.csv',directory],directory/'execution.log')
    checks=rows(directory/'correctness.csv')
    if len(checks)!=len(plan) or any(x['correct']!='PASS' for x in checks): raise ValueError('Failed correctness stage')
    if mode=='bench':aggregate(directory)
    print(f'{name}: PASS',flush=True)
    return directory


def naive(run):
    command([sys.executable,'-B','-m','unittest','discover','-s',ROOT/'tests','-v'],run/'unit-tests.log')
    command([os.environ.get('COMSPEC','cmd.exe'),'/d','/c','..\\phase3\\build.cmd',
             str((run/'build').relative_to(ROOT))],run/'build.log')
    save(run/'binary-sha256.json',{'whalegpu_gemm.exe':sha(run/'build'/'whalegpu_gemm.exe')})
    execute(run,'naive',validation_plan(['naive_col_row']),'validate')


def memory(run):
    execute(run,'memory-validation',validation_plan(NAIVE),'validate')
    rng=random.Random(SEED)
    plan=[]
    for shape in SHAPES:
        batch=[job(v,shape,group='memory') for v in NAIVE];rng.shuffle(batch);plan+=batch
    execute(run,'memory',plan)


def tiled(run):
    execute(run,'tiled',validation_plan(TILED),'validate')


def benchmark(run):
    for name in ['naive','memory-validation','tiled']:
        checks=rows(run/name/'correctness.csv')
        if not checks or any(x['correct']!='PASS' for x in checks):raise ValueError('Missing validation')
    rng=random.Random(SEED+1);plan=[]
    for shape in SHAPES:
        batch=[job(v,shape) for v in VARIANTS];rng.shuffle(batch);plan+=batch
    directory=execute(run,'benchmark',plan)
    summary=rows(directory/'summary.csv');winners=[]
    for shape in SHAPES:
        subset=[x for x in summary if tuple(int(x[d]) for d in ['m','n','k'])==shape and x['variant'] in ['tiled8_row','tiled16_row','tiled32_row']]
        winners.append(min(subset,key=lambda x:float(x['kernel_median'])))
    save(run/'winners.json',winners)


def confirm(run):
    plan=[];rng=random.Random(SEED+2)
    for winner in read(run/'winners.json'):
        shape=tuple(int(winner[x]) for x in ['m','n','k'])
        for pair in range(21):
            group='x'.join(map(str,shape))+f'-pair{pair}'
            batch=[job('naive_col_row',shape,rounds=1,group=group,role='baseline'),
                   job(winner['variant'],shape,rounds=1,group=group,role='winner')]
            rng.shuffle(batch);plan+=batch
    execute(run,'confirm',plan)


def profiler(run):
    info=read(run/'profiler.json')
    if not info['path']:
        print('Nsight Compute: NOT_FOUND; no hardware counters collected',flush=True)
    else:
        directory=run/'profiler';directory.mkdir()
        plan=[job(v,(257,193,129),rounds=1) for v in VARIANTS]
        for i,j in enumerate(plan):j['id']=f'profile-{i}'
        write_csv(directory/'plan.csv',plan,FIELDS)
        args=[info['path'],'--target-processes','all','--export',directory/'gemm']
        for section in ['LaunchStats','Occupancy','SpeedOfLight','MemoryWorkloadAnalysis','WarpStateStats']:
            args+=['--section',section]
        args+=[run/'build'/'whalegpu_gemm.exe','profile',directory/'plan.csv',directory]
        code=command(args,directory/'ncu.log',check=False)
        info.update(status='COLLECTED' if code==0 else 'COLLECTION_FAILED',exit_code=code)
        if code==0:command([info['path'],'--import',directory/'gemm.ncu-rep','--csv','--page','raw'],directory/'metrics.csv',check=False)
        else:info['missing_metrics']=['occupancy','memory_throughput','warp_stall','memory_access_efficiency']
        save(run/'profiler.json',info)
    sanitizer=read(run/'sanitizer.json')
    if sanitizer['path']:
        for tool in ['racecheck','synccheck']:
            directory=run/tool;directory.mkdir()
            plan=[job(v,(33,31,35),rounds=1) for v in TILED]
            for i,j in enumerate(plan):j['id']=f'{tool}-{i}'
            write_csv(directory/'plan.csv',plan,FIELDS)
            code=command([sanitizer['path'],'--tool',tool,'--error-exitcode','9',run/'build'/'whalegpu_gemm.exe',
                          'validate',directory/'plan.csv',directory],directory/'tool.log',check=False)
            sanitizer[tool]='PASS' if code==0 else 'FAILED_OR_UNAVAILABLE'
        save(run/'sanitizer.json',sanitizer)


def compatibility(run):
    compat=run/'compatibility';compat.mkdir()
    p1=compat/'phase1';p1.mkdir()
    p1out=p1/'results';p1out.mkdir()
    command([os.environ.get('COMSPEC','cmd.exe'),'/d','/c','..\\build.cmd',
             str((p1/'build').relative_to(PHASE1))],p1/'build.log')
    command([p1/'build'/'whalegpu_tests.exe','all',p1out],p1/'regression.log')
    p2=compat/'phase2';p2.mkdir()
    command([sys.executable,'-B','-m','unittest','discover','-s',PHASE2/'tests','-v'],p2/'unit-tests.log')
    command([os.environ.get('COMSPEC','cmd.exe'),'/d','/c','build.cmd',
             os.path.relpath(p2/'build',PHASE2)],p2/'build.log')
    plan=[phase2.job(algorithm=a,allocation=alloc,group='compatibility') for a,alloc in itertools.product(
          ['phase1_shared','phase1_shuffle','shared','shuffle'],['pageable_reuse','pinned_reuse','pageable_fresh'])]
    phase2.run_plan(p2,'engine',plan)
    phase2.validate(p2)
    before=read(run/'legacy-files-before.json');after=history_hashes()
    changed=[k for k in before if after.get(k)!=before[k]]
    added=sorted(set(after)-set(before))
    save(run/'compatibility.json',dict(status='PASS' if not changed and not added else 'FAIL',
         phase1_cases=len(rows(p1out/'results.csv')),phase2_cases=len(rows(p2/'validate'/'correctness.csv')),
         phase2_engine_jobs=12,legacy_files_verified=len(before),changed_files=changed,added_files=added))
    if changed or added:raise ValueError('Phase 1/2 history changed')


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['naive','memory','tiled','benchmark','confirm','profiler','compatibility','report','all'],default='all');p.add_argument('--run',type=Path)
    args=p.parse_args();run=args.run.resolve() if args.run else new_run()
    print('RUN='+str(run),flush=True)
    stop=threading.Event();monitor=threading.Thread(target=phase2.telemetry,args=(run,stop),daemon=True);monitor.start()
    stages=['naive','memory','tiled','benchmark','confirm','profiler','compatibility','report'] if args.stage=='all' else [args.stage]
    try:
        for stage in stages:
            if stage=='report':
                from gemm_report import generate
                generate(run)
            else:globals()[stage](run)
            save(run/f'status-{stage}.json',dict(status='PASS',finished=phase2.stamp()))
    except Exception as exc:
        save(run/f'failure-{args.stage}-{uuid.uuid4().hex[:8]}.json',dict(error=str(exc),time=phase2.stamp()));raise
    finally:stop.set();monitor.join(timeout=12)
    print('DONE',flush=True)


if __name__=='__main__':main()
