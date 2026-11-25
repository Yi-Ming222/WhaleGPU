import csv
import hashlib
import sys
from pathlib import Path
from collections import defaultdict
root=Path(__file__).resolve().parent
run=root/'stage3'
sys.dont_write_bytecode=True
sys.path.insert(0,str(root.parent/'WhaleGPU-Phase1/phase2/scripts'))
from stats import summarize

def read(name):
    with (run/name).open() as f:return list(csv.DictReader(f))

valid=read('validation.csv')
fields=['variant','rows','h','eps','affine','column_major']
assert len({tuple(r[f] for f in fields+['pattern']) for r in valid})==len(valid)==4032
groups=defaultdict(list)
for rep in ['first','repeat']:
    for r in read(rep+'.csv'):groups[(rep,*(r[f] for f in fields))].append(r)
assert len(groups)==448
summary={}
for key,data in groups.items():
    assert len(data)==11 and {int(r['round']) for r in data}==set(range(11))
    row=dict(zip(['run']+fields,key))
    for t in ['kernel_ms','e2e_ms']:
        row.update({t+'_'+name:v for name,v in summarize([float(r[t]) for r in data]).items()})
    logical=8*int(row['rows'])*int(row['h'])+int(row['affine'])*8*int(row['h'])
    row['effective_GBs']=logical/(row['kernel_ms_median']*1e6)
    for f in ['stats_regs','stats_shared','apply_regs','apply_shared']:row[f]=data[0][f]
    summary[key]=row
with (run/'summary.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=next(iter(summary.values())).keys())
    writer.writeheader();writer.writerows(summary.values())
variants=['naive','block','warp','online','merge','register4','fused4']
eps=max((r['eps'] for r in valid),key=float)
def result(rep,v,shape=('128','1024'),col='0',affine='1',epsilon=eps):
    return summary[rep,v,*shape,epsilon,affine,col]
lines=['# WhaleGPU Phase 4.2 · LayerNorm 实验报告','',
'设备为 RTX 5070 Laptop GPU，CC 12.0，驱动 582.05，nvcc 13.0.48。输入、gamma/beta、输出为 FP32，所有 GPU 实现以 FP64 归约。工程独立于历史目录，仅编译时读取 Phase 1 CUDA 工具头文件，报告复用 Phase 2 统计定义。','',
'## 数学定义和实现','','第 r 行统计全 H 个值，μ=Σx/H，v=Σ(x−μ)²/H，输出 (x−μ)/√(v+ε)，仿射开启时再逐特征乘 gamma 加 beta。输入合约为有限 FP32 和有限正 epsilon；仿射结果需能以 FP32 表示。零行或零特征输出为空，空行无统计量。布局实验保持逻辑矩阵与参考相同。','',
'| 实现 | 行内统计方式 | 写回 |','|---|---|---|',
'| naive | 每线程一行，两遍式 | 独立 apply |','| block | 每 Block 一行，两遍式，Warp shuffle 和共享内存合并 | 独立 apply |',
'| warp | 每 Warp 一行；每 Block 同时处理最多 8 行 | 独立 apply |',
'| online | 单线程按输入顺序更新 count、mean、M2 | 独立 apply |',
'| merge | 256 个局部 Welford 状态按共享内存树合并 | 独立 apply |',
'| register4 | 每线程缓存前四个条带元素，两遍式归约；H>1024 时跨步重读尾部 | 独立 apply |',
'| fused4 | 与 register4 相同的统计和映射 | 同 Kernel 直接仿射写回；继续输出均值和方差用于审计 |','',
'Welford 单样本更新为 M2 += (x−旧均值)(x−新均值)。两组统计量合并时，M2 还要加上两均值差的平方乘 n₁n₂/(n₁+n₂)。空局部状态跳过合并。','',
'## 数值稳定性','','三个阶段分别通过 576、2880、4032 个组合，最终矩阵覆盖 12 个形状 × 6 个数据模式 × 2 个 epsilon × 2 个布局 × 2 个仿射设置 × 7 个实现。数据模式 0–5 依次为有正负的随机数、常数、1e8 附近 8 的整数倍扰动、1e−12 量级、接近 ±3.2e38 的极值、仅行末一个值比 1e8 高一 ULP。输出和非空行统计均预填 NaN。非空均值与方差逐行对照 CPU double 两遍参考，输出完整逐元素核对。','',
'输出容差为 3e−6 + 2e−5·|参考值|。均值容差为 1e−14 + 2e−12·max(|μ|,√v)，方差容差为 1e−30 + 1e−5·v，并要求方差非负。极值平方由 FP64 承载。以下三列为各模式所有形状和设置中的最大**输出绝对误差**，末列为 H=4099、仅行末一 ULP 扰动的最大**方差绝对误差**。该行总体方差约为 0.01561。','',
'| 实现 | 大偏移 | 极小方差 | 极端幅值 | 稀疏扰动方差误差 |','|---|---:|---:|---:|---:|']
for v in variants:
    data=[r for r in valid if r['variant']==v]
    vals=[max(float(r['output_error']) for r in data if r['pattern']==p) for p in ['2','3','4']]
    vals.append(max(float(r['var_error']) for r in data if r['pattern']=='5' and r['h']=='4099'))
    lines.append('| '+v+' | '+' | '.join(f'{x:.4g}' for x in vals)+' |')
lines+=['','这些误差针对存储后的 FP32 输入。GPU 与参考两遍式在串行归约顺序相同时可能精确一致；其他顺序可产生舍入差异。完整统计量误差按行取最坏值，记录于 [validation.csv](validation.csv)。','',
'## 性能设计与复测','','四种形状为 1×33、17×257、128×1024、65×4099。每个形状交叉测试行/列主序、ε=1e−5/1e−12、仿射开启/关闭。每个进程 224 配置，预热 5 次，11 个独立计时区间，每区间的 CUDA Events 均摊 5 次完整算子。E2E 用 CPU 时钟测一次 pageable H2D + 完整算子 + 阻塞 D2H，含开启时参数上传，复用分配。每个计时区间后及每次 E2E 后均验证完整输出。使用相同数据、不同顺序种子 42 和 420 独立复测。','',
'有效带宽折算逻辑字节数 8RH + (仿射开启时 8H)，中间统计量写入及重读未计入。缓冲区反复使用。下表固定 128×1024、行主序、ε=1e−5、仿射开启，时间单位 ms。资源列为每线程寄存器 / 每 Block 静态共享内存字节，来自 cudaFuncGetAttributes。','',
'| 实现 | 首轮 Kernel | 复测 Kernel | 复测 std | 复测 P95 | 复测 E2E | 有效 GB/s | 统计或融合资源 | apply 资源 |','|---|---:|---:|---:|---:|---:|---:|---|---|']
for v in variants:
    a=result('first',v);b=result('repeat',v)
    lines.append(f"| {v} | {a['kernel_ms_median']:.5f} | {b['kernel_ms_median']:.5f} | {b['kernel_ms_stddev']:.5f} | {b['kernel_ms_p95']:.5f} | {b['e2e_ms_median']:.5f} | {b['effective_GBs']:.2f} | {b['stats_regs']} / {b['stats_shared']} | "+('融合' if v=='fused4' else f"{b['apply_regs']} / {b['apply_shared']}")+' |')
lines+=['','以下固定 ε=1e−5、开启仿射，按该工作负载的中位 Kernel 时间列出各轮最快实现。','',
'| 形状 | 布局 | 首轮（ms） | 复测（ms） |','|---|---|---|---|']
for shape in [('1','33'),('17','257'),('128','1024'),('65','4099')]:
    for col in ['0','1']:
        cells=[]
        for rep in ['first','repeat']:
            best=min((result(rep,v,shape,col) for v in variants),key=lambda r:r['kernel_ms_median'])
            cells.append(f"{best['variant']} ({best['kernel_ms_median']:.5f})")
        lines.append('| '+'×'.join(shape)+' | '+('列' if col=='1' else '行')+' | '+' | '.join(cells)+' |')
lines+=['','完整 448 行配置统计包含 epsilon 与仿射的所有交叉组合。[summary.csv](summary.csv) · [首轮](first.csv) · [复测](repeat.csv)','',
'## 同步、工具与复现','','源代码中两遍式 Block 归约使用完整 warp shuffle，跨 warp 结果依赖块屏障发布。合并 Welford 的各层先写独立槽位，再执行全块屏障。warp 版本的行边界对同一 warp 一致。拆分算子使用同一默认流排序。上述为源码审查，实机正确性测试仅检查本次输出。PATH 与 CUDA/NVIDIA 标准目录未找到 Nsight Compute、Compute Sanitizer；未采集硬件计数器，也未运行 racecheck 或 synccheck。[工具探测](environment.json)','',
'在 Phase42 目录运行 build.cmd 加新目录名，随后执行该目录的 layer.exe bench 加输出 CSV 路径和顺序种子即可重测。验证 CSV 的计时字段 0 表示未测量。运行二进制 SHA256 为 '+hashlib.sha256((run/'layer.exe').read_bytes()).hexdigest()]
(run/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
