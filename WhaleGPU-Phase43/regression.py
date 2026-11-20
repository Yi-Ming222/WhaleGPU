"""Execute original workloads only inside a fresh historical mirror."""
import csv,json,os,shutil,subprocess as sp,sys,uuid
from pathlib import Path
from history import ROOT,OLD,manifest
hist=manifest()
resume=len(sys.argv)==2
run=ROOT/'regression'/(sys.argv[1] if resume else uuid.uuid4().hex[:6])
if not resume:run.mkdir(parents=True)
print(run,flush=True)
if not resume:
    for old in OLD:shutil.copytree(old,run/old.name)
env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';env['OMP_NUM_THREADS']='8'
status=json.loads((run/'status.json').read_text()) if resume else {}
def step(name,folder,args):
    if name in status and status[name]['exit_code']==0:return
    if args[0].endswith('.exe') and not Path(args[0]).is_absolute() and '/' in args[0]:args[0]=str(run/('WhaleGPU-Phase'+folder)/args[0])
    with (run/(name+'.log')).open('wb') as f:
        code=sp.run(args,cwd=run/('WhaleGPU-Phase'+folder),stdout=f,stderr=sp.STDOUT,env=env).returncode
    status[name]=dict(exit_code=code,args=list(map(str,args)))
    (run/'status.json').write_text(json.dumps(status,indent=2))
    print(name,code,flush=True)
for phase,script in [('2','phase2/scripts/run.py'),('3','phase3/scripts/experiment.py')]:
    step('p'+phase,'1',[sys.executable,'-B',script,'--stage','all'])
step('p34','34',[sys.executable,'-B','run.py'])
for phase in ('35','36'):
    step('p'+phase,phase,['cmd','/d','/c','build.cmd','recheck'])
    if phase=='35' and status['p35']['exit_code']==0:step('p35-repeat','35',['recheck/bench.exe','recheck/repeat.csv'])
for phase,exe in [('41','softmax'),('42','layer')]:
    for stage in (1,2,3):
        step(f'p{phase}-{stage}',phase,[f'stage{stage}/{exe}.exe','validate',f'recheck{stage}.csv',phase])
    for tag in (1,2):step(f'p{phase}-perf{tag}',phase,[f'stage3/{exe}.exe','bench',f'recheck_perf{tag}.csv',str(430+tag)])
assert hist==manifest()
print('Historical integrity verified',flush=True)
sys.exit(int(any(s['exit_code'] for s in status.values())))


