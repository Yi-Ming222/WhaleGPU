"""Build cross-run analysis from completed, named experiment directories."""
import csv,json,re,statistics as st,sys
from pathlib import Path
from history import ROOT,manifest
def csvrows(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))
def ratio(raw,a,b,metric):
    x={r['pair']:float(r[metric]) for r in raw if r['variant']==a}
    y={r['pair']:float(r[metric]) for r in raw if r['variant']==b}
    assert len(x)==len(y)==21
    return st.median(x[k]/y[k] for k in x)
def results(path):
    exp=path.name[0]
    main={'A':'512x512x512','B':'262147_','C':'128x1024_0_17_eps5_row_aff'}[exp]
    rows=csvrows(path/'summary.csv')
    cases=[r['case'] for r in rows if main in r['case'] and r['run']=='first' and r['variant']=='split' and r['metric']=='kernel_ms']
    assert cases
    lines=['## '+path.name,'',f'[实验报告及曲线](runs/{path.name}/REPORT.md) · [原始 CUDA Events 和 E2E 样本](runs/{path.name}/repeat/samples.csv) · [资源](runs/{path.name}/repeat/resources.csv) · [spill](runs/{path.name}/spills.csv)','',
           '|主配置|进程|split / fused Kernel µs（中位数）|配对比|split / fused E2E µs（中位数）|配对比|','|---|---|---:|---:|---:|---:|']
    for case in cases:
        for tag in ['first','repeat']:
            raw=[r for r in csvrows(path/tag/'samples.csv') if r['case']==case]
            def val(v,m):return st.median(float(r[m])*1000 for r in raw if r['variant']==v)
            fields=[case,tag]
            for metric in ['kernel_ms','e2e_ms']:fields.extend([f'{val("split",metric):.3f} / {val("fused",metric):.3f}',f'{ratio(raw,"split","fused",metric):.3f}'])
            lines.append('|'+'|'.join(fields)+'|')
    checks=csvrows(path/'validate/checks.csv')
    n=len(checks)
    lines+=['',f'正确性文件 {n} 行。各行给出元素数、最大绝对误差与 RMSE。']
    if exp=='C':
        for suffix in ['_mean','_variance','output']:
            vals=[float(r['max_abs']) for r in checks if (not r['variant'].endswith(('_mean','_variance')) if suffix=='output' else r['variant'].endswith(suffix))]
            if vals:lines.append(f'- {suffix} 最大绝对误差 {max(vals):.7g}。')
    if (path/'resources_verified.csv').exists():
        res=csvrows(path/'resources_verified.csv')
        lines+=['','|主配置|版本及内核|Grid X×Y|线程/Block|寄存器/线程|Shared B|spill 写/读 B|理论 Block/SM|理论 Occupancy|','|---|---|---:|---:|---:|---:|---:|---:|---:|']
        for r in res:
            if r['process']!='repeat' or r['case'] not in cases:continue
            lines.append('|{}|{} · {}|{}×{}|{}|{}|{}|{}/{}|{}|{:.1%}|'.format(r['case'].split('_')[0],r['variant'],r['kernel'],r['grid_x'],r['grid_y'],r['threads'],r['regs'],int(r['static_shared'])+int(r['dynamic_shared']),r['spill_store_bytes'],r['spill_load_bytes'],r['blocks_per_sm'],float(r['theoretical_occupancy'])))
        verdict=csvrows(path/'paired_verdicts.csv')
        lines+=['',f'[所有尺寸的配对结论](runs/{path.name}/paired_verdicts.csv)。两个进程的区间均高于 1 才标为 gain，均低于 1 标为 regression。']
        for metric in ['kernel_ms','e2e_ms']:
            counts={v:sum(r['metric']==metric and r['verdict']==v for r in verdict) for v in ('gain','regression','uncertain')}
            lines.append(f'- {metric} 按配置计数：收益 {counts["gain"]}，回退 {counts["regression"]}，不确定 {counts["uncertain"]}。')
    instr=csvrows(path/'instructions.csv')
    if exp=='A':
        lines+=['','FP32 核保留相同的 64×4 寄存器 GEMM，FP16/BF16 核保留 pipeline32 和 FP32 累加。运行时 fmaf 只作用在 G32 的最终存储处。分离路径多一次启动和 8MN 个逻辑字节。以下为融合符号内的静态指令。']
        for r in instr:
            if 'affineI' in r['symbol'] and r['opcode'].startswith(('HMMA','LDGSTS','FFMA')):lines.append(f'- {r["symbol"]} · {r["opcode"]} × {r["static_count"]}')
    if exp=='B':
        lines+=['','register4 ≤1024 时由两次启动变一次。更长输入由四次变三次，保留局部结果和 merge，仅将仿射放在最后一次重标定中，保留乘法的 FP32 舍入。两组都移除一次概率数组的逻辑读写，合计 8N bytes。实验域到 4,194,305，未测超大 grid 回退。']
    if exp=='C' and 'ring' not in path.name:
        s=(path/'sass.txt').read_text(errors='replace').split('Function : _Z15register_reload')[1].split('Function :')[0]
        addresses=['7620','7700','7710','7730','7750','77b0','7800','7950']
        ev=[next((line.strip().split(';')[0] for line in s.splitlines() if '/*'+pc+'*/' in line),'') for pc in addresses]
        assert all(ev) and 'STRONG.GPU' in ev[3] and 'STG' in ev[-1]
        lines+=['','C3 机器码核对。Params 的首个指针 X 位于参数基址 0x380，Y 指针在 +24 即 0x398。输出段先形成 X 地址，读取 FP32 并转 double、减均值，随后经归一化与可选仿射写 Y。','', '```text',*ev,'```','','C2/C3 比较同时改变缓存提示（C3 使用 .cg）、数据依赖、指令调度及寄存器活跃区间。静态寄存器数相同也不能隔离这些因素。']
        for tag in ('first','repeat'):
            raw=[r for r in csvrows(path/tag/'samples.csv') if r['case']==cases[0]]
            lines.append(f'- {tag} 配对 Kernel 比。split/fused {ratio(raw,"split","fused","kernel_ms"):.3f}，fused_empty/fused {ratio(raw,"fused_empty","fused","kernel_ms"):.3f}，reload/fused {ratio(raw,"reload","fused","kernel_ms"):.3f}，block_split/split {ratio(raw,"block_split","split","kernel_ms"):.3f}。')
    return lines
runs=[Path(s).resolve() for s in sys.argv[1:]]
lines=['# WhaleGPU Phase 4.3 实验汇总','','各主配置在 DESIGN.md 预登记。各次运行采用同输入、同精度、独立进程，表中比值为 21 个成对比值的中位数。区间、均值、标准差和 P95 见各实验报告及 summary.csv。各形状包括正负结果。','']
if runs:
    device=csvrows(runs[0]/'first/device.csv')[0]
    gpu=(runs[0]/'gpu.txt').read_text(errors='replace')
    driver=re.search(r'Driver Version\s*:\s*(\S+)',gpu)[1]
    lines+=['## 测试环境与文件','',f'{device["gpu"]}，CC {device["major"]}.{device["minor"]}，{device["sm"]} SM，L2 {int(device["l2_bytes"])//1048576} MiB，驱动 {driver}。nvcc 13.0.48，sm_120、C++17、-O3；完整构建命令见各 run 的 build.cmd。','',
            '[审计协议](PHASE43_AUDIT.md) · [设计](DESIGN.md) · [复现指南](REPRODUCE.md)。A.cu、B.cu、C.cu 分别为三个实验入口。common.hpp、rotation.hpp 提供计时与缓冲区轮换。run.py 生成并冻结 CUDA 头文件、编译并验证。history.py、audit_results.py、evidence.py、finalize.py、regression.py、regression_report.py、rebuild.py 分别负责指纹、证据核验、报告及复现。','']
    for path in sorted((ROOT/'regression').glob('*/REGRESSION.md')):lines+=['[原版阶段回归]('+path.relative_to(ROOT).as_posix()+')','']
    for p in sorted((ROOT/'rebuild').glob('*/*/verified.json')):
        if p.parent.name in [r.name for r in runs]:lines+=['[独立重建的校验 '+p.parent.name+']('+p.relative_to(ROOT).as_posix()+')','']
for path in runs:lines+=results(path)+['']
lines+=['## 证据边界','','启动数与逻辑流量来自代码。Kernel 延迟与 E2E 延迟分别实测；分离、融合以及额外空启动的差值不具有可加性。GPU 事件覆盖完整调用链，可能包含设备等待主机提交的空档。每个变体按相同规则预热，不删极值。理论驻留与理论 occupancy 来自 CUDA API；未采集 achieved occupancy、L2 命中率和实际 DRAM 流量。未执行 racecheck 或 synccheck。',f'历史树逐文件核对通过 {len(manifest())} 项。']
(ROOT/'REPORT.md').write_text('\n'.join(lines),encoding='utf8')
