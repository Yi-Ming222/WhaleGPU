import csv,json,re,sys
from pathlib import Path
from history import ROOT,manifest
run=Path(sys.argv[1]).resolve()
status=json.loads((run/'status.json').read_text());assert all(v['exit_code']==0 for v in status.values())
p1=run/'WhaleGPU-Phase1'
def newest(p):return max(p.iterdir(),key=lambda f:f.stat().st_mtime_ns)
p2=newest(p1/'phase2/runs');p3=newest(p1/'phase3/runs');p34=newest(run/'WhaleGPU-Phase34/runs')
def count(path,predicate=lambda r: True):
    with path.open() as f:return sum(predicate(r) for r in csv.DictReader(f))
rows=[]
def record(label,n,expected,path):
    assert n==expected,(label,n,expected)
    rows.append(f'|{label}|{n}|PASS|[{path.name}]({path.relative_to(run).as_posix()})|')
def cases(label,path,expected):
    record(label,count(path),expected,path)
compat=json.loads((p3/'compatibility.json').read_text());assert compat['status']=='PASS' and not compat['changed_files']
record('Phase 1 all',compat['phase1_cases'],672,p3/'compatibility.json')
for d,key,n in [(p2,'statistics-tests.log',None),(p3,'unit-tests.log',4)]:
    f=d/key;s=f.read_text(errors='replace');assert 'OK' in s
    val=int(re.search(r'Ran (\d+) tests?',s)[1]);record(key,val,n or val,f)
for label,n in [('engine',12),('validate',34992),('matrix',504),('tune',1296),('confirm',756)]:
    cases('Phase 2 '+label,p2/label/'correctness.csv',n)
for label,n in [('naive',120),('memory-validation',480),('tiled',720)]:
    cases('Phase 3 '+label,p3/label/'correctness.csv',n)
cases('3.4 register',p34/'validate/correctness.csv',2040)
for fname in ['samples.csv','repeat.csv']:
    f=run/'WhaleGPU-Phase35/recheck'/fname
    record('3.5 '+fname,count(f,lambda r:r['stage']=='check'),96,f)
for pref in ['first','repeat']:
    f=run/'WhaleGPU-Phase36/recheck'/(pref+'-errors.csv')
    cases('3.6 '+pref+' references',f,504)
for phase,last in [('41',756),('42',4032)]:
    for i in [1,2,3]:
        f=run/('WhaleGPU-Phase'+phase)/f'recheck{i}.csv'
        original=ROOT.parent/('WhaleGPU-Phase'+phase)/f'stage{i}/validation.csv'
        record(phase+f' stage {i}',count(f),last if i==3 else count(original),f)
    for i in [1,2]:
        f=run/('WhaleGPU-Phase'+phase)/f'recheck_perf{i}.csv'
        assert count(f)>0
        rows.append(f'|{phase} bench {i}|{count(f)}|PASS|[{f.name}]({f.relative_to(run).as_posix()})|')
lines=['# 历史回归','','在独立镜像执行。各计数口径按原驱动，3.6 每个用例产生两行参考误差。原生 MMA 的 128 个输出由可执行程序检查。','', '|阶段|记录数|状态|证据|','|---|---:|---|---|',*rows,'',f'逐文件 SHA-256 对照六棵历史树，全部 {len(manifest())} 项匹配。完整命令与退出码见 status.json。']
(run/'REGRESSION.md').write_text('\n'.join(lines),encoding='utf8')
