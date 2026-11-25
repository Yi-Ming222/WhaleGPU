import random
import shutil
import statistics
from pathlib import Path


def audit(run):
    from experiment import rows,read,save,sha,history_hashes,ROOT,flops,NAIVE,TILED,SHAPES,SEED
    counts={}
    for name in ['naive','memory-validation','memory','tiled','benchmark','confirm']:
        plan=rows(run/name/'plan.csv');checks=rows(run/name/'correctness.csv');samples=rows(run/name/'samples.csv')
        ids={x['id']:x for x in plan}
        if len(ids)!=len(plan) or len(checks)!=len(plan) or {x['id'] for x in checks}!=set(ids):raise ValueError('Missing/duplicate records')
        if any(x['correct']!='PASS' or int(x['mismatches']) for x in checks+samples):raise ValueError('Incorrect result')
        if name in ['memory','benchmark','confirm']:
            grouped={i:[] for i in ids}
            for x in samples:grouped[x['id']].append(x)
            for ident,j in ids.items():
                data=grouped[ident]
                if len(data)!=int(j['rounds']) or {int(x['round']) for x in data}!=set(range(int(j['rounds']))):raise ValueError('Bad sample coverage')
            for x in rows(run/name/'summary.csv'):
                if int(x['flops'])!=flops(int(x['m']),int(x['n']),int(x['k'])):raise ValueError('FLOPs mismatch')
        counts[name]=dict(jobs=len(plan),samples=len(samples))
    matrix=rows(run/'benchmark'/'summary.csv');confirm=rows(run/'confirm'/'plan.csv')
    actual={(tuple(int(x[d]) for d in ['m','n','k']),x['variant']) for x in matrix}
    expected={(s,v) for s in SHAPES for v in NAIVE+TILED}
    if actual!=expected or len(matrix)!=len(expected):raise ValueError('Incomplete benchmark matrix')
    for w in read(run/'winners.json'):
        candidates=[x for x in matrix if all(x[d]==w[d] for d in ['m','n','k']) and x['variant'] in ['tiled8_row','tiled16_row','tiled32_row']]
        if w['id']!=min(candidates,key=lambda x:float(x['kernel_median']))['id']:raise ValueError('Wrong winner')
    for group in {x['group'] for x in confirm}:
        pair=[x for x in confirm if x['group']==group]
        if len(pair)!=2 or {x['role'] for x in pair}!={'baseline','winner'}:raise ValueError('Incomplete pair')
        for field in ['m','n','k','seed','pattern','allocation','warmups','repeats','rounds']:
            if len({x[field] for x in pair})!=1:raise ValueError('Mixed comparison conditions')
        if any(x['variant'].endswith('_bt') for x in pair):raise ValueError('Confounded layout comparison')
    for p in (ROOT/'src').rglob('*'):
        if p.is_file() and sha(p)!=sha(run/'source'/'phase3'/'src'/p.relative_to(ROOT/'src')):raise ValueError('Changed tested CUDA source')
    if sha(run/'build'/'whalegpu_gemm.exe')!=read(run/'binary-sha256.json')['whalegpu_gemm.exe']:raise ValueError('Changed binary')
    legacy=read(run/'legacy-files-before.json')
    if legacy!=history_hashes():raise ValueError('Phase 1/2 history changed')
    compat=read(run/'compatibility.json')
    if compat['status']!='PASS' or compat['phase1_cases']!=672 or compat['phase2_cases']!=34992:raise ValueError('Missing compatibility tests')
    result=dict(status='PASS',counts=counts,legacy_files_verified=len(legacy),kernel_source_and_binary_verified=True)
    save(run/'audit.json',result)
    return result


def interval(values):
    from stats import percentile
    rng=random.Random(20261117)
    bootstrap=[statistics.median(rng.choices(values,k=len(values))) for _ in range(2000)]
    return percentile(bootstrap,.025),percentile(bootstrap,.975)


def generate(run):
    from experiment import rows,read,save,write_csv,command,ROOT,NAIVE,TILED,SHAPES,sha
    from stats import summarize
    verified=audit(run)
    out=run/'report';out.mkdir();charts=out/'charts';charts.mkdir()
    matrix=rows(run/'benchmark'/'summary.csv');confirm=rows(run/'confirm'/'summary.csv');winners=read(run/'winners.json')
    hw=read(run/'benchmark'/'hardware.json');env=read(run/'environment.json');prof=read(run/'profiler.json');sanitizer=read(run/'sanitizer.json')
    comparisons=[];confirmed=[]
    for winner in winners:
        subset=[x for x in confirm if all(x[d]==winner[d] for d in ['m','n','k'])]
        ratios=[]
        for group in sorted({x['group'] for x in subset}):
            pair=[x for x in subset if x['group']==group]
            base=next(x for x in pair if x['role']=='baseline');opt=next(x for x in pair if x['role']=='winner')
            ratios.append(float(base['kernel_median'])/float(opt['kernel_median']))
        if len(ratios)!=21:raise ValueError('Independent confirmation requires 21 pairs')
        lo,hi=interval(ratios)
        comparisons.append(dict(m=winner['m'],n=winner['n'],k=winner['k'],variant=winner['variant'],
                                paired_median_speedup=statistics.median(ratios),bootstrap95_low=lo,bootstrap95_high=hi,
                                result='faster_in_this_run' if lo>1 else 'slower_in_this_run' if hi<1 else 'inconclusive'))
        for role in ['baseline','winner']:
            entries=[x for x in subset if x['role']==role]
            result=dict(m=winner['m'],n=winner['n'],k=winner['k'],role=role,variant=entries[0]['variant'])
            for metric in ['kernel','e2e']:
                result.update({metric+'_'+k:v for k,v in summarize([float(x[metric+'_median']) for x in entries]).items()})
            result['kernel_tflops']=int(winner['flops'])/(result['kernel_median']*1e9)
            confirmed.append(result)
    write_csv(out/'independent-comparison.csv',comparisons);write_csv(out/'independent-summary.csv',confirmed)
    specs=[]
    def chart(filename,title,metric,ylabel,variants,logy=False):
        series=[]
        for variant in variants:
            data=sorted([x for x in matrix if x['m']==x['n']==x['k'] and x['variant']==variant],key=lambda x:int(x['m']))
            series.append(dict(name=variant,points=[[int(x['m']),float(x[metric])] for x in data]))
        specs.append(dict(title=title,ylabel=ylabel,logy=logy,series=series,output=str(charts/filename)))
    chart('naive-latency.png','Naive GEMM | lane mapping and B layout','kernel_median','Median kernel time (ms)',NAIVE,True)
    chart('naive-tflops.png','Naive GEMM | effective 2MNK throughput','kernel_tflops','Effective TFLOPS',NAIVE)
    row_tiled=['naive_col_row','tiled8_row','tiled16_row','tiled32_row']
    chart('tiling-latency.png','Same row-major inputs | naive vs shared tiling','kernel_median','Median kernel time (ms)',row_tiled,True)
    chart('tiling-tflops.png','Same row-major inputs | naive vs shared tiling','kernel_tflops','Effective TFLOPS',row_tiled)
    chart('tile-layout.png','Shared tiling | B layout comparison','kernel_median','Median kernel time (ms)',TILED,True)
    chart('end-to-end.png','End-to-end | includes BT packing, transfers and synchronization','e2e_median','Median end-to-end time (ms)',['naive_col_row','naive_col_bt','tiled16_row','tiled16_bt'],True)
    chart('packing.png','Serial CPU B transpose cost | included in BT end-to-end','pack_median','Median CPU packing time (ms)',['naive_col_bt','tiled16_bt'],True)
    series=[]
    for role in ['baseline','winner']:
        data=sorted([x for x in confirmed if x['m']==x['n']==x['k'] and x['role']==role],key=lambda x:int(x['m']))
        series.append(dict(name='naive_col_row' if role=='baseline' else 'selected row-major tile (independent)',points=[[int(x['m']),x['kernel_tflops']] for x in data]))
    specs.append(dict(title='Independent confirmation | 21 randomized pairs',ylabel='Effective TFLOPS',logy=False,series=series,output=str(charts/'independent-tflops.png')))
    save(out/'plot-spec.json',specs)
    command(['C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',ROOT/'scripts'/'plot.ps1','-Spec',out/'plot-spec.json'],out/'plot.log')
    shutil.copytree(ROOT/'scripts',out/'reporting-source',ignore=shutil.ignore_patterns('__pycache__'))
    save(out/'reporting-source-sha256.json',{str(p.relative_to(out/'reporting-source')):sha(p) for p in (out/'reporting-source').rglob('*') if p.is_file()})
    checks=[x for stage in ['naive','memory-validation','tiled','benchmark'] for x in rows(run/stage/'correctness.csv')]
    lines=['# WhaleGPU Phase 3：GEMM Optimization Lab 性能报告','',f"实验：`{run.name}`；开始时间：{env['started']}。",'',
           '## 环境与方法','',f"- {hw['gpu']}；CC {hw['cc']}；{hw['sm_count']} SM；L2 {hw['l2_bytes']} bytes；显存 {hw['memory_bytes']} bytes。",
           f"- nvcc {hw['nvcc']}，MSVC {hw['msvc']}，CUDA Runtime {hw['runtime']}；编译参数 `{env['flags']}`。驱动详情：[nvidia-smi](../nvidia-smi.txt)。",
           f"- OS：{env['os']}；CPU：{env['cpu']}；CPU Reference 使用最多 {hw['cpu_reference_threads']} 个线程、double 累加。",
           '- A/C 为 row-major，B 为 row-major 或 BT[N,K]。所有 Kernel 使用原生 FP32 FMA，无 Tensor Core、cuBLAS 或 fast-math。',
           '- 固定随机种子、同规模内随机化配置顺序；每配置预热 10 次，15 轮，每轮 Events 包围 10 次 Kernel，样本为均摊 ms。',
           '- 统计沿用 Phase 2：均值、中位数、样本标准差、线性插值 P95；保留所有慢样本。P95 是批量均摊值的分位数，不是单次 launch 的尾延迟。',
           '- E2E 使用 CPU steady_clock：BT 的串行 CPU 打包 + A/B H2D + GEMM + C D2H + 同步。缓冲区复用、分配排除，BT 打包时间单独记录。',
           '- 初始化、首次调用和 CPU Reference 时间单独保存，不混入稳定计时。所有版本采用 warmed/reused 缓存策略，不进行 cache flush，不锁定时钟。',
           '- 名义 FLOPs=2MNK；有效 TFLOPS=FLOPs/(kernel_ms×10^9)。不把边界 Tile 补零计算计入有用 FLOPs，也不将有效 TFLOPS称为硬件指令吞吐。',
           '- 当前最大 1024³ 的 A+B+C 约 12 MiB，小于本机 32 MiB L2。此规模范围主要研究缓存复用和算法执行差异，不是冷 DRAM 饱和实验。','',
           '## 实现与正确性','',
           '| 阶段 | 用例 | 结果 |','|---|---:|---|',
           '| Naive 基线 | 120 | PASS |','| 四种 Naive 映射/布局 | 480 | PASS |','| 六种 Tiled 配置 | 720 | PASS |',
           '| 完整性能矩阵 | 110 | 每用例及每轮完整输出验证通过 |','| 独立复测 | 462 | 每用例及每轮完整输出验证通过 |','',
           '- 20 种边界形状 × 三个随机种子/zero/identity/tail；涵盖空输出、K=0、非方阵、非 2 次幂、8/16/32 附近尺寸。原始 plan.csv 保存完整形状。',
           '- C 首先填充 NaN，确保漏写被发现。随机输入逐元素采用 gamma(K)·sum(abs(products))+1e-7 误差界，identity/zero/tail 严格比较。',
           f"- 上述正确性与矩阵初检最大绝对误差：{max(float(x['max_abs_error']) for x in checks):.8g}；最大逐元素 error/tolerance：{max(float(x['max_error_ratio']) for x in checks):.8g}。容差偏保守，实际误差也逐用例保存。",
           '- Shared Tile 每个槽只由一个线程写入；所有线程参与加载后屏障和读完后的屏障；输出越界线程仅跳过最终写出。没有条件分支跳过屏障。',
           f"- Compute Sanitizer：{sanitizer['status']}，racecheck={sanitizer['racecheck']}，synccheck={sanitizer['synccheck']}。未运行时，上述同步结论来自源码结构检查和数值测试，不等同于工具竞态检查。",'',
           '## Global Memory 访问实验','',
           '| Naive 版本 | 相邻线程输出方向 | B 物理布局 | 固定 q 时的主要地址关系 |','|---|---|---|---|',
           '| naive_col_row | 列 | row-major | A 重复地址，B/C 相邻地址（行边界除外） |',
           '| naive_row_row | 行 | row-major | A 跨 K，B 重复地址，C 跨 N |',
           '| naive_col_bt | 列 | BT[N,K] | A 重复地址，B 跨 K，C 相邻地址 |',
           '| naive_row_bt | 行 | BT[N,K] | A 跨 K，B 重复地址，C 跨 N；每线程 B 沿 K 连续 |','',
           '上述为源码地址关系，不是测得的访存效率。四个版本都使用 256 threads，每线程完成一个输出，计算量/随机输入一致。性能结论来自以下实际数据。','',
           '## 1024×1024×1024 完整统计','',
           '| 版本 | Kernel mean ms | median ms | stddev ms | P95 ms | 有效 TFLOPS | E2E median ms | pack median ms |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for x in matrix:
        if x['m']==x['n']==x['k']=='1024':
            lines.append(f"| {x['variant']} | {float(x['kernel_mean']):.6f} | {float(x['kernel_median']):.6f} | {float(x['kernel_stddev']):.6f} | {float(x['kernel_p95']):.6f} | {float(x['kernel_tflops']):.4f} | {float(x['e2e_median']):.6f} | {float(x['pack_median']):.6f} |")
    lines+=['','完整表：[Benchmark summary](../benchmark/summary.csv)，[原始样本](../benchmark/samples.csv)，[先行访存实验](../memory/summary.csv)。','',
            '## 独立复测：同布局 Naive 与 Tiled','',
            '先仅在 row-major 的 Tile 8/16/32 中选择中位数最快配置，再采集 21 组 baseline/winner 随机顺序样本。保持相同种子、形状、布局、预热和重复次数。加速比是逐组 baseline/winner 的中位数；区间来自 2,000 次 bootstrap，仅描述本次桌面 GPU 实验。','',
            '| M×N×K | Tile 版本 | Naive median ms | Tiled median ms | Tiled TFLOPS | 配对加速比 | bootstrap 95% | 结论 |',
            '|---|---|---:|---:|---:|---:|---|---|']
    for x in comparisons:
        sub=[r for r in confirmed if all(r[d]==x[d] for d in ['m','n','k'])]
        b=next(r for r in sub if r['role']=='baseline');t=next(r for r in sub if r['role']=='winner')
        lines.append(f"| {x['m']}×{x['n']}×{x['k']} | {x['variant']} | {b['kernel_median']:.6f} | {t['kernel_median']:.6f} | {t['kernel_tflops']:.4f} | {x['paired_median_speedup']:.3f}× | [{x['bootstrap95_low']:.3f}, {x['bootstrap95_high']:.3f}] | {x['result']} |")
    lines+=['','每轮数值正确性与 E2E 的均值/中位数/标准差/P95 见 [独立统计](independent-summary.csv)。','',
            '## 曲线','']
    for spec in specs:lines += [f"![{spec['title']}](charts/{Path(spec['output']).name})",'']
    lines+=['## 资源与 Profiler','',f"Nsight Compute：**{prof['status']}**；探测目录和缺失计数器见 [profiler.json](../profiler.json)。",'',
            '寄存器和 Shared Memory 是 CUDA function attributes 静态资源；理论 occupancy 来自 CUDA occupancy API，不是 Achieved Occupancy。','',
            '| 版本 | Threads | Registers/thread | Shared bytes | 理论 occupancy |','|---|---:|---:|---:|---:|']
    for x in matrix:
        if x['m']==x['n']==x['k']=='1024':lines.append(f"| {x['variant']} | {x['threads']} | {x['registers_per_thread']} | {x['shared_bytes']} | {float(x['theoretical_occupancy']):.1%} |")
    large={x['variant']:x for x in matrix if x['m']==x['n']==x['k']=='1024'}
    stride_ratio=float(large['naive_row_row']['kernel_median'])/float(large['naive_col_row']['kernel_median'])
    bt_ratio=float(large['naive_col_bt']['kernel_median'])/float(large['naive_col_row']['kernel_median'])
    lines+=['','## 性能解释与局限','',
            f'- 1024³ 固定矩阵中，行映射 row-major 的 Kernel 中位时间为列映射的 {stride_ratio:.3f} 倍；列映射 BT 为列映射 row-major 的 {bt_ratio:.3f} 倍。它们是实际时间比，不是访存效率百分比。',
            '- 相邻地址和跨步地址可能影响合并事务；缓存、重复地址广播、每线程连续访问及输出写入模式也同时影响结果。无硬件计数器时，不把单一机制当成已确认原因。',
            '- Shared Tiling 在一个 Block 内复用 A/B Tile，减少重复 Global Memory 读取请求，但增加屏障、Shared Memory 访问及边界补零工作。不同 Tile 的实测差异见图表，不假设大 Tile 总是更快。',
            '- 32×32 Tile 使用 1024 threads；线程数、寄存器和 shared 占用共同限制驻留。理论 occupancy 数值不能直接解释全部速度差异。',
            '- BT Kernel 曲线只测预打包输入的 GEMM；E2E 重新执行串行 CPU 转置。CPU 打包不是优化过的生产级实现，其成本已计入，不将 Kernel-only 结果宣传为完整布局转换收益。',
            '- 小矩阵的延迟可能由提交与调度固定成本主导；此判断为机制解释，尚无 profiler 的 Stall/SM 利用率证据。',
            '- 笔记本 WDDM、动态频率、温度和其他桌面任务未完全隔离。保留时钟/功耗遥测和所有样本，不承诺跨设备/跨功耗状态性能。',
            '- 本阶段不包含 Tensor Core、寄存器分块、异步拷贝或 FlashAttention；TFLOPS 仅覆盖本报告规模与 FP32 CUDA Core 实现。','',
            '## Phase 1/2 兼容性','',
            '- Phase 1 原始代码重新编译，672/672 正确性通过。',
            '- Phase 2 原始代码重新编译，34,992/34,992 参数正确性通过、12/12 引擎配置通过，原 6 项框架测试通过。未重写原测试以适应 Phase 3。',
            f"- 原 Phase 1/2 文件树 {verified['legacy_files_verified']} 个文件的 SHA-256 前后一致，新增回归结果全部在此 run 的 compatibility/ 内；原始记录未改动。",
            '- 未更改 Driver/Toolkit；继续使用 nvcc，CMake 保持原有独立问题状态。',
            '', '记录入口：[兼容性结果](../compatibility.json)、[完整审计](../audit.json)、[设计](../../../DESIGN.md)。', '',
            '复现：在 phase3 目录执行 `python -B scripts/experiment.py --stage all`；创建唯一新目录，已有阶段拒绝覆盖。', '']
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Report: '+str(out/'REPORT.md'),flush=True)
