"""Generate a report only from persisted measurements, never synthetic results."""
import math
import random
import shutil
import statistics
from pathlib import Path
from stats import summarize, percentile


def paired_interval(values, seed=20261117):
    rng = random.Random(seed)
    boot = [statistics.median(rng.choices(values, k=len(values))) for _ in range(2000)]
    return percentile(boot, .025), percentile(boot, .975)


def generate(run):
    from run import rows, read, save, write_csv, command, ROOT, sha
    from audit import audit
    audit(run)
    matrix=rows(run/'matrix'/'summary.csv')
    search=rows(run/'tune'/'summary.csv')
    confirm=rows(run/'confirm'/'summary.csv')
    winners=read(run/'winners.json')
    hw=read(run/'engine'/'hardware.json')
    env=read(run/'environment.json')
    profiler=read(run/'profiler.json')
    eligible=rows(run/'validate'/'correctness.csv')
    integrity=read(run/'phase1-integrity.json')
    if not integrity['unchanged'] or any(x['correct']!='PASS' for x in eligible):
        raise ValueError('Integrity/correctness failure prevents report')
    out=run/'report'
    out.mkdir()
    charts=out/'charts'; charts.mkdir()
    comparisons=[]
    confirmed_summary=[]
    for winner in winners:
        subset=[x for x in confirm if x['n']==winner['n'] and x['op']==winner['op']]
        for label in ['winner','phase1_shared','phase1_shuffle']:
            data=[x for x in subset if (not x['algorithm'].startswith('phase1_') if label=='winner' else x['algorithm']==label)]
            if len(data)!=21: raise ValueError('Incomplete independent confirmation')
            entry=dict(op=winner['op'],n=int(winner['n']),label=label)
            for metric in ['kernel','e2e']:
                entry.update({f'{metric}_{k}':v for k,v in summarize([float(x[f'{metric}_median']) for x in data]).items()})
            confirmed_summary.append(entry)
        for baseline in ['phase1_shared','phase1_shuffle']:
            ratios=[]
            for group in sorted({x['group'] for x in subset}):
                pair=[x for x in subset if x['group']==group]
                base=next(x for x in pair if x['algorithm']==baseline)
                opt=next(x for x in pair if not x['algorithm'].startswith('phase1_'))
                ratios.append(float(base['kernel_median'])/float(opt['kernel_median']))
            lo,hi=paired_interval(ratios)
            comparisons.append(dict(op=winner['op'],n=int(winner['n']),baseline=baseline,
                                    paired_median_speedup=statistics.median(ratios),bootstrap95_low=lo,
                                    bootstrap95_high=hi,pairs=len(ratios),
                                    conclusion='faster_in_this_run' if lo>1 else 'slower_in_this_run' if hi<1 else 'inconclusive'))
    write_csv(out/'confirmed-summary.csv',confirmed_summary)
    write_csv(out/'paired-comparison.csv',comparisons)
    specs=[]
    def add_chart(filename,title,ylabel,series,logy=False):
        specs.append(dict(title=title,ylabel=ylabel,series=series,logy=logy,output=str(charts/filename)))
    for op in ['sum','max','min']:
        for metric,ylabel,logy in [('kernel_median','Median complete reduction (ms)',True),('effective_gbps','Effective logical bandwidth (GB/s)',False)]:
            series=[]
            for algorithm in ['phase1_shared','phase1_shuffle','shared','shuffle']:
                data=sorted([x for x in matrix if x['op']==op and x['algorithm']==algorithm and x['threads']=='256' and x['allocation']=='pageable_reuse'],key=lambda x:int(x['n']))
                series.append(dict(name=algorithm+' / 256 threads',points=[[int(x['n']),float(x[metric])] for x in data]))
            add_chart(f'{op}-{metric}.png',f'{op.upper()} | same input, items=1, natural grid, striped',ylabel,series,logy)
        series=[]
        for allocation in ['pageable_reuse','pinned_reuse','pageable_fresh']:
            data=sorted([x for x in matrix if x['op']==op and x['algorithm']=='phase1_shuffle' and x['allocation']==allocation],key=lambda x:int(x['n']))
            series.append(dict(name=allocation,points=[[int(x['n']),float(x['e2e_median'])] for x in data]))
        add_chart(f'{op}-allocation.png',f'{op.upper()} | original Phase 1 shuffle | allocation strategy','Median end-to-end (ms)',series,True)
    series=[]
    for algorithm in ['shared','shuffle']:
        for threads in ['128','256','512']:
            data=sorted([x for x in matrix if x['op']=='sum' and x['algorithm']==algorithm and x['threads']==threads and x['allocation']=='pageable_reuse'],key=lambda x:int(x['n']))
            series.append(dict(name=f'{algorithm} / {threads}',points=[[int(x['n']),float(x['kernel_median'])] for x in data]))
    add_chart('sum-blocks.png','SUM | block size | items=1, natural grid, striped','Median complete reduction (ms)',series,True)
    series=[]
    for algorithm in ['shared','shuffle']:
        for layout in ['striped','contiguous']:
            data=sorted([x for x in search if x['op']=='sum' and x['algorithm']==algorithm and x['threads']=='256' and x['items']=='4' and x['grid']==str(hw['sm_count']*2) and x['layout']==layout],key=lambda x:int(x['n']))
            series.append(dict(name=f'{algorithm} / {layout}',points=[[int(x['n']),float(x['kernel_median'])] for x in data]))
    add_chart('sum-layout.png','SUM | 256 threads, 4 items/iteration, grid cap = 2 * SM','Median complete reduction (ms)',series,True)
    save(out/'plot-spec.json',specs)
    command(['C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',ROOT/'scripts'/'plot.ps1','-Spec',out/'plot-spec.json'],out/'plot.log')
    # Capture reporting tools added after initial engine build without replacing its snapshot.
    shutil.copytree(ROOT/'scripts',out/'reporting-source',ignore=shutil.ignore_patterns('__pycache__'))
    save(out/'reporting-source-sha256.json',{str(p.relative_to(out/'reporting-source')):sha(p) for p in (out/'reporting-source').rglob('*') if p.is_file()})
    lines=['# WhaleGPU Phase 2 性能实验报告','',f"实验 ID：`{run.name}`；开始时间：{env['started']}。",'',
           '## 环境与可追溯性','',f"- GPU：{hw['gpu']}，CC {hw['cc']}，{hw['sm_count']} SM，显存 {hw['memory_bytes']} bytes，L2 {hw['l2_bytes']} bytes。",
           f"- CUDA / 编译器：Runtime {hw['runtime']}，nvcc {hw['nvcc']}，MSVC {hw['msvc']}；`{env['compiler_flags']}`。",
           f"- OS：{env['os']}；CPU：{env['cpu']}。驱动发行版本见 [nvidia-smi](../nvidia-smi.txt)。",
           f"- 此进程 CUDA 初始化主机耗时：{hw['cuda_init_host_ms']:.3f} ms，未混入稳定样本。各配置首次调用另见 first_calls.csv。",
           '- 原始配置、样本、正确性、静态资源、编译命令、源码/二进制 SHA-256 均保留。GPU 时钟、温度、功耗采样见 [telemetry.csv](../telemetry.csv)。',
           f"- 参数正确性矩阵：{len(eligible):,}/{len(eligible):,} PASS；Phase 1 672 项回归通过，源码 SHA-256 前后一致。",'',
           '## 测试方法','',
           '- 固定种子打乱相同规模内的配置顺序。默认每配置预热 10 次；15 轮，每轮 CUDA Events 包围 10 次完整归约，输出每次归约均摊 ms。',
           '- Kernel 表和曲线使用完整多轮归约时间；它包含提交间隙，不是各独立 Kernel 时间之和的 profiler 测量。',
           '- 端到端时间来自 CPU steady_clock，包含 H2D、完整归约、D2H 和同步。pageable_fresh 还包含设备分配/释放；reuse 排除分配。主机数据生成和 host allocation 均排除。',
           '- 均值/中位数/样本标准差/P95 全部记录，P95 线性插值，不删除慢样本。单样本代表 10 次归约的均摊，不能解释为单次 launch 的 P95。E2E 每轮单次。',
           '- 所有版本使用相同输入和 warmed/reused 缓存策略。最大 33,554,435 元素约 128 MiB，超过 32 MiB L2；较小规模可能主要命中缓存。',
           '- 有效带宽 `(4*n+4)/(median_ms*1e6)` GB/s 为逻辑输入/最终输出吞吐；中间轮次流量另记，不能等同于 DRAM 控制器实测带宽。',
           '- Auto-Tuner 搜索 128/256/512 threads × 1/4/8 items × natural/2SM/8SM grid × shared/shuffle × striped/contiguous；每个算子/规模 108 候选。',
           '- 搜索使用 pageable_reuse；获胜配置在新轮次与两个原版基线做 21 组随机顺序配对复测。报告独立复测，不把搜索最低样本直接当作性能提升。',
           '- 配对区间为 2,000 次 bootstrap 的中位数 95% 区间；桌面共享 GPU 与时间相关性未消除，区间仅描述本次实验，不是跨设备/跨日保证。','',
           '## 性能数据','',f"固定矩阵 {len(matrix)} 组；搜索 {len(search)} 个候选；独立验证 {len(confirm)} 组。完整统计：[矩阵](../matrix/summary.csv)、[搜索](../tune/summary.csv)、[独立验证](confirmed-summary.csv)。",'',
           '以下为最大输入、pageable_reuse 的统计，单位 ms。', '',
           '| Op | 实现 | Threads | Kernel mean | median | stddev | P95 | E2E median | 有效 GB/s |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for x in matrix:
        if int(x['n'])==max(env['sizes']) and x['allocation']=='pageable_reuse':
            lines.append(f"| {x['op']} | {x['algorithm']} | {x['threads']} | {float(x['kernel_mean']):.6f} | {float(x['kernel_median']):.6f} | {float(x['kernel_stddev']):.6f} | {float(x['kernel_p95']):.6f} | {float(x['e2e_median']):.6f} | {float(x['effective_gbps']):.2f} |")
    lines+=['','## 自动调参获胜参数','',
            '以下是有限搜索空间中的获胜配置，不代表全局最优；grid=0 表示 natural，非零为 Block 数量上限。items 是每次 grid-stride 迭代的每线程元素数。', '',
            '| Op | n | 算法 | Threads | Items | Grid cap | Layout | 实际首轮 Blocks | 搜索 median ms | 独立 median ms |',
            '|---|---:|---|---:|---:|---:|---|---:|---:|---:|']
    for x in winners:
        actual=next(r for r in confirmed_summary if r['op']==x['op'] and r['n']==int(x['n']) and r['label']=='winner')
        lines.append(f"| {x['op']} | {x['n']} | {x['algorithm']} | {x['threads']} | {x['items']} | {x['grid']} | {x['layout']} | {x['first_grid']} | {float(x['kernel_median']):.6f} | {actual['kernel_median']:.6f} |")
    lines+=['','## 优化前后独立对比','',
            '比值 = 原版时间 / 获胜配置时间，大于 1 表示本次复测更快。每项使用相同输入、分配策略、预热和重复次数。','',
            '| Op | n | 原版基线 | 配对中位加速比 | bootstrap 95% | 结果 |','|---|---:|---|---:|---|---|']
    for x in comparisons:
        lines.append(f"| {x['op']} | {x['n']} | {x['baseline']} | {x['paired_median_speedup']:.3f}x | [{x['bootstrap95_low']:.3f}, {x['bootstrap95_high']:.3f}] | {x['conclusion']} |")
    lines+=['','## 性能曲线','']
    for op in ['sum','max','min']:
        for name in ['kernel_median','effective_gbps','allocation']:
            lines += [f'![{op} {name}](charts/{op}-{name}.png)','']
    lines+=['![Block size](charts/sum-blocks.png)','','![Data mapping](charts/sum-layout.png)','',
            '## 资源与 Profiler','',f"Nsight Compute 状态：**{profiler['status']}**。探测路径、缺失项和错误见 [profiler.json](../profiler.json)。",'',
            '| 指标 | 本次来源/状态 |','|---|---|',
            '| Achieved Occupancy | '+('见独立 ncu 记录' if profiler['status']=='COLLECTED' else '缺失；没有硬件实测')+' |',
            '| DRAM throughput / Warp stall / Memory access efficiency | '+('见独立 ncu 记录' if profiler['status']=='COLLECTED' else '缺失；不由 Events 推断为已采集')+' |',
            '| Registers / shared memory | cudaFuncGetAttributes + launch shared bytes，静态资源，非 profiler 实测 |',
            '| Theoretical occupancy | CUDA occupancy API 的驻留上界，非 achieved occupancy |','',
            '搜索获胜配置的静态资源（原版资源列为 NA，未改写原版源文件以暴露内部 Kernel）：','',
            '| Op | n | Registers/thread | Shared static+dynamic bytes | 理论 occupancy | 归约轮数 |','|---|---:|---:|---:|---:|---:|']
    for x in winners:
        lines.append(f"| {x['op']} | {x['n']} | {x['registers_per_thread']} | {int(x['static_shared_bytes'])+int(x['dynamic_shared_bytes'])} | {float(x['theoretical_occupancy']):.1%} | {x['passes']} |")
    faster=sum(x['conclusion']=='faster_in_this_run' for x in comparisons)
    uncertain=sum(x['conclusion']=='inconclusive' for x in comparisons)
    lines+=['','## 瓶颈分析与证据边界','',
            f'- 独立配对对比中，{faster}/{len(comparisons)} 项区间完全高于 1，{uncertain} 项区间跨越 1。每项详见上表；不作所有规模/所有算子都提升的结论。',
            '- Kernel 延迟随输入规模、归约轮数及网格配置变化，曲线和 resources.csv 可定位相关性。较小输入受提交/调度等固定成本影响的判断是机制解释，不是 Warp Stall 计数器结论。',
            '- 每线程多元素与网格限制会同时改变寄存器内串行工作、全局中间结果和 Kernel 轮数。归约轮数降低可以解释部分时间差，但不能仅凭它确定唯一瓶颈。',
            '- striped/contiguous 的固定参数曲线单独展示数据分工影响；由于没有访存效率计数器，不能把时间差精确归因于某个事务合并比例。',
            '- allocation 曲线使用同一个原版 shuffle Kernel；E2E 差异包含传输、同步和 fresh allocation。不能把 pinned/fresh 的 E2E 变化算成 Kernel 算法收益。',
            '- 未锁定 GPU 时钟、未控制其他桌面任务、未做冷缓存 flush；保留全部慢样本和遥测。当前结论是本机本次受控软件条件下的实验结果。',
            '- 后续有 Nsight Compute 后，应优先对大输入的基线和获胜配置采集 MemoryWorkloadAnalysis、Occupancy、WarpStateStats，验证带宽/并行度/访存事务的瓶颈假设。', '',
            '## 复现与原始记录','',
            '`python -B scripts/run.py --stage all` 创建新 run，从构建、正确性到报告完整执行。中断后可对已有 run 使用 `--run <path> --stage <下一阶段>`；已存在的阶段目录拒绝覆盖。',
            '', '参考：[CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)、[Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)。', '']
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    save(out/'audit.json',dict(matrix_jobs=len(matrix),search_jobs=len(search),confirmation_jobs=len(confirm),
         validation_cases=len(eligible),phase1_source_unchanged=integrity['unchanged'],plots=len(specs),status='PASS'))
    print(f'Report: {out / "REPORT.md"}',flush=True)
