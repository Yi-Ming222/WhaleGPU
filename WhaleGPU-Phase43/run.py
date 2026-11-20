"""Run one experiment in a fresh, self-contained source snapshot."""
import csv, json, os, random, shutil, statistics as st, subprocess as sp, sys, uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from history import ROOT, manifest, sha
def replace(s,a,b):
    assert s.count(a)==1,a
    return s.replace(a,b)
def generate(snap,exp):
    p=snap/ROOT.name
    def read(folder,name):return (snap/('WhaleGPU-Phase'+folder)/name).read_text()
    if exp=='A':
        reg=read('34','register.cuh').split('// L selects')[1].split('template<int',1)[1].split('using Launch')[0]
        reg='template<int'+reg
        pipe=read('36','kernels.cuh').split('template<class T,bool Async>\n__global__ void block_gemm')[1]
        pipe='template<class T,bool Async>\n__global__ void pipeline_affine'+pipe
        s=replace(reg,'register_gemm','register_affine')+pipe
        s=s.replace('int m,int n,int k) {','int m,int n,int k,float scale,float bias) {')
        s=replace(replace(s,'=accum[i][j];','=fmaf(scale,accum[i][j],bias);'),'=out[i];','=fmaf(scale,out[i],bias);')
    elif exp=='B':
        orig=read('41','register.cuh')
        s=orig.split('__global__ void merge_tiles')[0]+orig.split('__global__ void scale_tiles')[1].split('inline void register_launch')[0]
        s=replace(s,'(float* y,size_t n,const float* work)','__global__ void scale_affine(float* y,size_t n,const float* work,float a,float b)')
        s=replace(replace(s,'void local_tile','void local_affine'),'size_t n,float* work)','size_t n,float* work,float a,float b)')
        s=replace(replace(s,'Single?float(v[j]/sum):v[j]','fmaf(a,float(v[j]/sum),b)'),'y[i]*=scale','y[i]=fmaf(a,__fmul_rn(y[i],scale),b)')
    else:
        s=read('42','register.cuh').replace('register_row','register_reload')
        s='__device__ float reload43(const float* x){ float v; asm volatile("ld.global.cg.f32 %0,[%1];":"=f"(v):"l"(x):"memory");return v;}\n'+s
        for v in ['double(local[slot])','p.get(row,j)']:
            s=replace(s,'p.store(row,j,'+v,'p.store(row,j,double(reload43(p.x+p.index(row,j)))')
    (p/(exp+'_generated.cuh')).write_text(s)
def report(run):
    rows=[]
    groups=defaultdict(list)
    def pct(x,q):
        x=sorted(x);i=(len(x)-1)*q;j=int(i);return x[j]+(i-j)*(x[min(j+1,len(x)-1)]-x[j])
    for tag in ('first','repeat'):
        for r in csv.DictReader((run/tag/'samples.csv').open()):
            for metric in ('kernel_ms','e2e_ms'):groups[tag,r['case'],r['variant'],metric].append(float(r[metric]))
    for key,x in groups.items():rows.append([*key,len(x),st.mean(x),st.median(x),st.stdev(x),pct(x,.95)])
    with (run/'summary.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow('run case variant metric n mean median sd p95'.split());w.writerows(rows)
    lines=['# '+run.name,'','Paired baseline / fused ratios. CI uses 2000 paired bootstrap medians. Times in summary.csv are milliseconds.','', '|Run|Case|Metric|Ratio|95% CI|','|---|---|---|---:|---|']
    rng=random.Random(43)
    for (tag,case,v,m),x in groups.items():
        if v!='split':continue
        y=groups[tag,case,'fused',m]; z=[a/b for a,b in zip(x,y)]
        boot=[st.median(rng.choices(z,k=len(z))) for _ in range(2000)]
        lines.append(f'|{tag}|{case}|{m}|{st.median(z):.3f}|{pct(boot,.025):.3f}–{pct(boot,.975):.3f}|')
    (run/'REPORT.md').write_text('\n'.join(lines),encoding='utf8')
def main(label):
    exp=label[0]
    assert label in ("A","B","C","A-ring","B-ring","C-ring")
    hist=manifest()
    run=ROOT/'runs'/(label+'-'+datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6]);run.mkdir(parents=True)
    print(run,flush=True)
    snap=run/'snapshot';p=snap/ROOT.name;p.mkdir(parents=True)
    for rel in hist:
        path=Path(rel)
        if path.suffix not in ('.hpp','.cuh','.inc','.cu','.h') or ('runs' in path.parts and path.name!='dispatch.inc') or any(part.startswith('stage') for part in path.parts):continue
        dest=snap/path;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT.parent/path,dest)
    for name in [exp+'.cu','common.hpp','rotation.hpp','evidence.py','run.py','history.py','DESIGN.md','PHASE43_AUDIT.md']:shutil.copyfile(ROOT/name,p/name)
    generate(snap,exp)
    def save(name,obj):(run/name).write_text(json.dumps(obj,indent=2),encoding='utf8')
    save('history.json',hist);save('source_sha256.json',{f.relative_to(snap).as_posix():sha(f) for f in snap.rglob('*') if f.is_file()})
    env=os.environ.copy();env['OMP_NUM_THREADS']='8'
    env.pop('WHALEGPU43_ROTATE',None)
    if label.endswith('-ring'):env['WHALEGPU43_ROTATE']='1'
    save('environment.json',{'OMP_NUM_THREADS':8,'WHALEGPU43_ROTATE':env.get('WHALEGPU43_ROTATE'),'validation_order_seed':43,'performance_order_seeds':[431,432]})
    def call(args,output,cwd=run):
        with (run/'commands.txt').open('a') as f:f.write(sp.list2cmdline(list(map(str,args)))+'\n')
        with (run/output).open('wb') as f:sp.run(args,cwd=cwd,env=env,stdout=f,stderr=sp.STDOUT,check=True)
    call(['nvidia-smi','-q'],'gpu.txt');call(['nvcc','--version'],'nvcc.txt')
    save('tools.json',{x:shutil.which(x) or 'NOT_FOUND' for x in ('ncu','compute-sanitizer')})
    inc=next(snap.glob('WhaleGPU-Phase34/runs/*/source/dispatch.inc')).parent
    cmd=f'nvcc -std=c++17 -O3 -arch=sm_120 -gencode arch=compute_120,code=compute_120 -lineinfo -Xcompiler /openmp -Xptxas=-v -I"{os.path.relpath(inc,p)}" -I"../WhaleGPU-Phase1/include" {exp}.cu -o "..\\..\\{exp}.exe"'
    (run/'build.cmd').write_text('@echo off\ncall "C:\\BuildTools2022\\VC\\Auxiliary\\Build\\vcvars64.bat" -vcvars_ver=14.39\nif errorlevel 1 exit /b 1\n'+cmd+'\nexit /b %errorlevel%\n')
    call(['cmd','/d','/c','..\\..\\build.cmd'],'build.log',p)
    exe=run/(exp+'.exe');save('binary.json',{exe.name:sha(exe)})
    env['PATH']=str(ROOT.parent/'WhaleGPU-Phase35/tools')+os.pathsep+env['PATH']
    for flag in ('sass','ptx'):call([ROOT.parent/'WhaleGPU-Phase35/tools/cuobjdump.exe','-'+flag,exe],flag+'.txt')
    for tag,mode,seed in [('validate','validate',43),('first','bench',431),('repeat','bench',432)]:
        (run/tag).mkdir();call([exe,mode,tag,str(seed)],tag+'.log')
    report(run)
    call([sys.executable,'-B',p/'evidence.py',run],'report.log')
    assert hist==manifest()
if __name__=='__main__':main(sys.argv[1])

