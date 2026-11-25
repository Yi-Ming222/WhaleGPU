import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

root=Path(__file__).resolve().parent
run=root/'stage3'
sys.dont_write_bytecode=True
sys.path.insert(0,str(root.parent/'WhaleGPU-Phase1/phase2/scripts'))
from stats import summarize

def read(path):
    with path.open() as f: return list(csv.DictReader(f))

valid=read(run/'validation.csv')
assert len({(r['variant'],r['n'],r['pattern'],r['seed']) for r in valid})==len(valid)==756
groups=defaultdict(list)
for repeat in ['first','repeat']:
    for r in read(run/(repeat+'.csv')):
        groups[repeat,r['variant'],int(r['n'])].append(r)
assert len(groups)==60
records={}
for key,rs in groups.items():
    assert len(rs)==11 and {int(r['round']) for r in rs}==set(range(11))
    record=dict(run=key[0],variant=key[1],n=key[2])
    for timer in ['kernel_ms','e2e_ms']:
        record.update({timer+'_'+k:v for k,v in summarize([float(r[timer]) for r in rs]).items()})
    record['effective_GBs']=8*key[2]/(record['kernel_ms_median']*1e6)
    records[key]=record
with (run/'summary.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=next(iter(records.values())).keys());w.writeheader();w.writerows(records.values())
mass=max(abs(float(r['mass'])-1) for r in valid if int(r['n']))
error=max(float(r['max_abs']) for r in valid)
lines=['# Phase 4.1 · Softmax Optimization Lab','',
'RTX 5070 Laptop GPU，sm_120，驱动 582.05，nvcc 13.0.48。单个连续 FP32 向量沿全长归一化，使用稳定指数 exp(x−max)。多个 +∞ 平分概率，全 −∞ 输出均匀分布，空输入输出为空。NaN 输入不在本实验定义域内。','',
'## 实现与归约','',
'| 版本 | 执行方式 | 同步与复用 |','|---|---|---|',
'| naive | 单线程扫描最大值、指数和、输出 | double 累加分母 |',
'| warp | 单 Warp 跨步处理任意长度 | shuffle 归约、广播 max/sum |',
'| block | 单个 256 线程 Block | 每 Warp 归约，首 Warp 合并，块屏障发布结果 |',
'| shared_tree | 单个 256 线程 Block | 共享内存树逐层合并，所有线程参加屏障 |',
'| multistage | 最多 512 个 Block，五次 launch | 局部最大值、全局最大值、局部指数和、总和、归一化；默认流保持轮次顺序 |',
'| register4 | 每线程四个跨步元素，1024 元素/块 | 短输入寄存器暂存指数，一次 launch；长输入三次 launch，第一轮写局部指数，第二轮合并局部统计，第三轮原位缩放 |','',
'寄存器分块的全局分母为 Σₜ sₜ exp(mₜ−m)，其中 mₜ、sₜ 是块内最大值和局部稳定指数和。该重标定允许局部指数先写入输出缓冲区。特殊无穷值使用显式权重规则。长输入的局部统计暂存为 FP32，求和使用 double。','',
'## 正确性与独立测量','',
f'分阶段矩阵分别通过 126、630、756 项，前两阶段与最终矩阵有重复。最终矩阵包含 14 种规模 × 9 个数据/种子组合 × 6 种实现。初次 launch 前用 NaN 填充非空输出。CPU 以 double 独立计算所有概率。逐元素要求有限且属于 [0,1]，绝对误差 ≤ 2e−8 + 2e−5·参考概率，同时以 double 求和检查 |Σp−1| ≤ 2e−6。最终矩阵最大逐元素绝对误差 {error:.3g}，最大非空概率和偏差 {mass:.3g}。',
'','两次进程使用相同数据种子 17 和不同配置顺序种子 41、410。每配置预热 5 次，11 个样本，每个 CUDA Events 区间均摊 5 次完整算子；每个 E2E 样本测一次 pageable H2D、完整算子和阻塞 D2H，复用设备缓冲区。每个样本同时校验 Kernel 与 E2E 的完整输出。阶段验证 CSV 中的零时间表示未计时。完整统计见 [summary.csv](summary.csv)，含均值、中位数、样本标准差和线性 P95。','',
'## 实测结果','',
'下表为 N=262147，时间单位 ms。有效带宽以复测 Kernel 中位数折算 8N 字节，即逻辑输入和最终输出。实际中间流量、缓存命中与 DRAM 吞吐未采集。','',
'| 版本 | 首轮 Kernel | 复测 Kernel | 复测 std | 复测 P95 | 复测 E2E | 有效 GB/s |','|---|---:|---:|---:|---:|---:|---:|']
for v in ['naive','warp','block','shared_tree','multistage','register4']:
    a=records['first',v,262147];b=records['repeat',v,262147]
    lines.append(f"| {v} | {a['kernel_ms_median']:.5f} | {b['kernel_ms_median']:.5f} | {b['kernel_ms_stddev']:.5f} | {b['kernel_ms_p95']:.5f} | {b['e2e_ms_median']:.5f} | {b['effective_GBs']:.2f} |")
lines+=['','| N | 首轮最快（ms） | 独立复测最快（ms） |','|---:|---|---|']
for n in [33,257,1025,16387,262147]:
    best=[min((r for (runid,v,size),r in records.items() if runid==rep and size==n),key=lambda r:r['kernel_ms_median']) for rep in ['first','repeat']]
    lines.append('| '+str(n)+' | '+' | '.join(f"{r['variant']} ({r['kernel_ms_median']:.5f})" for r in best)+' |')
lines+=['','最快值按每轮观测中位数选择。单 Warp 或单 Block 限制了长向量可利用的并行度。多阶段增加 launch 与中间缓冲读写；register4 在长输入时用局部统计换取跨 Block 并行，并将最终缩放压缩到第三次 launch。这些是算法结构解释，独立两轮的时差仍含桌面调度与缓存影响。','',
'## 工具与复现','',
'PATH、CUDA 和 NVIDIA 标准安装目录均未找到 Nsight Compute 或 Compute Sanitizer，未取得计数器、竞态或同步工具结果。见 [设备与探测记录](environment.json)。',
'','在本工程目录用 build.cmd 加一个未使用过的目录名重新构建，构建脚本自动运行完整正确性矩阵。之后用该目录的 softmax.exe bench <输出CSV> <顺序种子> 运行两个独立进程。原始样本见 [首轮](first.csv)、[复测](repeat.csv)，逐用例误差见 [验证](validation.csv)。',
'','实测二进制 SHA256：'+hashlib.sha256((run/'softmax.exe').read_bytes()).hexdigest()]
(run/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
