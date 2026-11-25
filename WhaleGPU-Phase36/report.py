"""Generate a report from two completed benchmark processes."""
import csv
import hashlib
import json
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

run=Path(sys.argv[1]).resolve()
root=Path(__file__).resolve().parent
def read(name):
    with (run/name).open() as f: return list(csv.DictReader(f))
groups=defaultdict(list)
dims=['dtype','kernel','flow','m','n','k']
for rep in ['first','repeat']:
    assert (run/(rep+'.log')).read_text().startswith('PASS')
    for row in read(rep+'-samples.csv'):
        groups[(rep,*(row[d] for d in dims))].append(row)
    errors=read(rep+'-errors.csv')
    assert len(errors)==504
    for e in errors:
        assert sum(int(e['bin'+str(i)]) for i in range(5))==int(e['count'])
assert len(groups)==144
summary={}
for key,rows in groups.items():
    assert {int(r['round']) for r in rows}==set(range(15)) and len(rows)==15
    result=dict(zip(['run']+dims,key))
    for timer in ['kernel_ms','e2e_ms']:
        values=[float(r[timer]) for r in rows]
        for label,value in [('median',st.median(values)),('mean',st.mean(values)),('std',st.stdev(values)),('p95',st.quantiles(values,n=100,method='inclusive')[94])]:
            result[timer+'_'+label]=value
    result['TFLOPS']=2*int(key[-3])*int(key[-2])*int(key[-1])/(result['kernel_ms_median']*1e9)
    summary[key]=result
with (run/'summary.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=next(iter(summary.values())).keys())
    writer.writeheader();writer.writerows(summary.values())
sass=(run/'sass.txt').read_text()
ptx=(run/'ptx.txt').read_text()
assert all(s in ptx for s in ['wmma.mma.sync','cp.async','cvt.rn.bf16.f32','cvt.rn.f16.f32'])
ops=[]
for section in sass.split('Function : ')[1:]:
    symbol=section.splitlines()[0].strip()
    if 'block_gemm' in symbol or 'tensor_gemm' in symbol:
        instructions=sorted(set(re.findall(r'\b(?:HMMA|LDGSTS)[.\w]*',section)))
        assert any(x.startswith('HMMA') for x in instructions)
        assert ('Lb1E' in symbol)==any(x.startswith('LDGSTS') for x in instructions)
        ops.append(symbol+' → '+', '.join(instructions))
assert len(ops)==6
lines=['# Phase 3.6 · Tensor Core 性能实验','',
'RTX 5070 Laptop GPU，CC 12.0，驱动 582.05，CUDA 13.0.48。复用 Phase 3.5 的 warp16，每个 Warp 计算 16×16 输出。block32 让四个 Warp 共享一个 32×32 输出块的输入。pipeline32 在同一块形状上加入两级缓冲和对齐的 16 字节异步加载；非整除边界使用补零和标量加载。所有配置使用 FP32 累加。','',
'## 时间与流程','',
'CUDA Events 测驻留设备的 GEMM。独立的 E2E 样本由 CPU 时钟测量，起点为主机输入，终点为输出回到主机。CPU 流程现场转换后上传 16 位数据；GPU 流程上传 FP32，再启动两个转换 Kernel；预转换流程上传已准备好的主机 16 位数据。三者均含传输与同步，分配排除。预转换的初始准备时间排除，因此适用于输入可以预先准备的场景。两个新进程采用相同输入、不同配置顺序种子，每配置预热 10 次、测 15 轮，每轮 Kernel 时间均摊 10 次。','',
'1024³ 中位时间。每格为首轮 / 复测，单位 ms。Kernel 列固定预转换输入流程，E2E 列在同一行保持 Kernel 固定。','',
'| 输入 | Kernel | Kernel-only | E2E CPU | E2E GPU | E2E 预转换 |','|---|---|---:|---:|---:|---:|']
def pair(dtype,kernel,flow,timer):
    return ' / '.join(f"{summary[(rep,dtype,kernel,flow,'1024','1024','1024')][timer+'_ms_median']:.3f}" for rep in ['first','repeat'])
for dtype in ['FP16','BF16']:
    for kernel in ['warp16','block32','pipeline32']:
        cells=[pair(dtype,kernel,'prepacked','kernel')]+[pair(dtype,kernel,f,'e2e') for f in ['cpu','gpu','prepacked']]
        lines.append('| '+' | '.join([dtype,kernel]+cells)+' |')
lines+=['','每次进程覆盖 72 个性能配置，包括 64³、257×193×129、512³ 和 1024³。完整统计含样本标准差、P95 与有效 TFLOPS。[全部尺寸](summary.csv)','',
'## 数值验证','',
'每进程 252 项形状、种子、类型、Kernel 和流程组合；全部输出逐元素检查。首个写回前用 NaN 填充，含 K=0、非整除 Tile、单元素、两个随机种子和跨多个流水线阶段的尾块。转换独立与 CPU 最近偶数舍入位模式比对。累计误差使用量化输入的 double 参考；总误差使用原 FP32 输入的 double 参考。沿用 Phase 3.5 的 gamma(K)·Σ|ab|+1e-7 累加界，以及 (2u+u²)·Σ|ab| 的输入量化项。','',
'逐配置保存最大绝对误差、max(|误差|/max(|参考值|,1e-7))、RMSE 与绝对误差直方图。分箱右端点依次为 1e-6、1e-4、1e-3、1e-2、+∞。下表为复测 1024³、pipeline32、GPU 转换的完整输出。','',
'| 输入 | 参考 | 最大绝对 | 最大相对 | RMSE | 各箱元素数（从小至大） |','|---|---|---:|---:|---:|---|']
for e in errors:
    if e['m']=='1024' and e['kernel']=='pipeline32' and e['flow']=='gpu':
        lines.append('| '+' | '.join([e['dtype'],e['reference'],e['max_abs'],e['max_rel'],e['rmse'],', '.join(e['bin'+str(i)] for i in range(5))])+' |')
lines+=['','最大相对误差对接近零的参考值敏感，应结合 RMSE 和分箱判断。原始记录见 [误差分布](repeat-errors.csv)。','',
'## 指令与工具','',
'PTX 包含两种 cvt.rn 转换、wmma.mma.sync 与 cp.async。从实际运行的 sm_120 二进制按函数检查得到：','']+['- '+op for op in ops]+['',
'当前环境 PATH、CUDA 与 NVIDIA 标准安装目录均未发现 ncu.exe 或 compute-sanitizer.exe。运行记录包含静态指令与实际延迟；异步拷贝的动态重叠程度、Achieved Occupancy、racecheck 和 synccheck 尚无测量。详见 [环境](../environment.json)、[PTX](ptx.txt)、[SASS](sass.txt)。','',
'复现时在 Phase36 目录运行 build.cmd 加一个新目录名，再运行 python report.py 加该目录名。二进制 SHA256 为 '+hashlib.sha256((run/'bench.exe').read_bytes()).hexdigest()]
(run/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
