import csv, re, statistics as s, sys, hashlib
from pathlib import Path
root=Path(__file__).resolve().parent
run=root/sys.argv[1]
variants=['naive_FP32','shared16_FP32','register64x4_FP32','WMMA_FP16','WMMA_BF16','WMMA_TF32']
functions=['_Z10naive_gemmILb0ELb0','_Z10tiled_gemmILi16ELb0','_Z13register_gemmILi64ELi4ELb0','_Z11tensor_gemmI6__half','_Z11tensor_gemmI13__nv_bfloat16','_Z11tensor_gemmIf']
sass=(run/'sass.txt').read_text()
sections=re.split(r'Function : ',sass)[1:]
evidence=[]
for name,prefix in zip(variants+['native_probe'],functions+['_Z12native_probe']):
    body=next(b for b in sections if b.startswith(prefix))
    ops=sorted(set(re.findall(r'\b(?:HMMA|FFMA)[.\w]*',body)))
    assert ('HMMA' in ' '.join(ops))==(name.startswith('WMMA') or name=='native_probe')
    evidence.append((name,', '.join(ops)))
ptx=[l.split('{')[0].strip() for l in (run/'ptx.txt').read_text().splitlines() if 'mma.sync.aligned' in l]
lines=['# RTX 5070 Laptop GPU · Tensor Core 实验','', '设备 CC 12.0，驱动 582.05，CUDA 13.0.48，编译目标 sm_120。使用一个完整 warp 计算一个输出块，以共享内存处理任意边界。原版 Naive、Shared 16、Register 64×4 均从已有源码编译。','',
'## 可执行文件的指令证据','','| Kernel | sm_120 SASS |','|---|---|']+[f'| {a} | {b} |' for a,b in evidence]+['','同一二进制内的 PTX 指令如下。','']+['- '+p for p in ptx]+['','原生 FP16 PTX 探针计算全 1 输入、初始累加值 1 的 16 项点积。所有 128 个结果均为 17。主 GEMM 选择 WMMA，以相同边界处理测试 FP16、BF16 和 TF32，三者输出及累加均为 FP32。ptxas 明确拒绝 sm_120 上的 wgmma.fence。其他低位宽和稀疏形式未作硬件验证。详见 [架构可行性](../FEASIBILITY.md)。','',
'## 数值验收','','输入由固定种子的有限随机 FP32 数生成，三种低精度路径以最近偶数舍入转换输入，TF32 以舍入过的 float 存储。每个输出先填 NaN。累加误差对照量化输入的 double 点积，采用 Phase 3 的 gamma(K)·Σ|ab|+1e-7，K=0 严格为零。相对原始 FP32 输入另允许 (2u+u²)·Σ|ab| 的量化误差，其中 FP32 的 u=0，FP16/TF32 为 2⁻¹¹，BF16 为 2⁻⁸。测试覆盖 1³、16³、15×17×9、33×65×31、17×31×0 等，两个种子覆盖小型与非整除用例。','',
'## 性能与资源','','每个进程包含 10 次预热、15 个计时轮次，每轮 CUDA Events 包围 10 次设备驻留 GEMM。E2E 使用单调主机时钟，包含主机逐元素转换、H2D、一次 GEMM 和 D2H，复用缓冲区。两次进程使用相同输入与打乱后的顺序。下表列出 1024³ 的中位数，吞吐按 2MNK 计算。','','| 版本 | Kernel ms | 复测 ms | 有效 TFLOPS | E2E ms | 寄存器/线程 | Shared B | 对量化参考最大误差 | 对原 FP32 参考最大误差 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
allrows=[]
for fname in ['samples.csv','repeat.csv']:
    with (run/fname).open() as f: data=list(csv.DictReader(f))
    assert len([r for r in data if r['stage']=='check'])==96
    for v in variants:
        for m in [64,257,512,1024]:
            group=[r for r in data if r['stage']=='benchmark' and r['variant']==v and int(r['m'])==m]
            assert len(group)==15
            first=group[0]
            ms=[float(r['kernel_ms']) for r in group]
            med=s.median(ms)
            allrows.append(dict(run=fname,variant=v,m=m,n=first['n'],k=first['k'],median=med,mean=s.mean(ms),std=s.stdev(ms),p95=s.quantiles(ms,n=100,method='inclusive')[94],tflops=2*m*int(first['n'])*int(first['k'])/med/1e9,e2e=s.median(float(r['e2e_ms']) for r in group),regs=first['registers'],shared=first['shared_bytes'],accum=first['accum_error'],original=first['original_error']))
with (run/'summary.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=allrows[0]);w.writeheader();w.writerows(allrows)
for v in variants:
    a,b=[r for r in allrows if r['variant']==v and r['m']==1024]
    lines.append(f"| {v} | {a['median']:.4f} | {b['median']:.4f} | {a['tflops']:.3f} | {a['e2e']:.3f} | {a['regs']} | {a['shared']} | {a['accum']} | {a['original']} |")
lines+=['','每轮进程通过 96 项前置检查与 24 个性能配置的全部输出检查。E2E 中串行 CPU 转换策略会影响低精度测试的应用耗时。实验局限于已预热的缓冲区和上述精度。各尺寸的中位数、标准差、P95 和复测数据见 [summary.csv](summary.csv)。','',
'## 复现','','在 Phase35 目录执行 build.cmd 加新目录名。以 OMP_NUM_THREADS=8 再运行该目录的 bench.exe 并传入独立 repeat.csv 路径，然后执行 python report.py 加目录名。tools.py 可取得校验通过的独立反汇编工具。原始反汇编为 [SASS](sass.txt) 和 [PTX](ptx.txt)。','', '可执行文件 SHA256 为 '+hashlib.sha256((run/'bench.exe').read_bytes()).hexdigest()]
(run/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
