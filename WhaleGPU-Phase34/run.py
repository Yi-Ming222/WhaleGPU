"""Run with python -B run.py from this directory."""
import csv
import itertools
import os
import random
import shutil
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

sys.dont_write_bytecode = True
root = Path(__file__).resolve().parent
legacy = root.parent / "WhaleGPU-Phase1"
p3 = legacy / "phase3"
sys.path.insert(0, str(p3 / "scripts"))
import experiment as lab

run = root / "runs" / (datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:6])
src = run / "source"
src.mkdir(parents=True)
print(run, flush=True)
before = {str(p.relative_to(legacy)): lab.sha(p) for p in legacy.rglob("*") if p.is_file()}
lab.save(run/"legacy-before.json", before)
configs = []
filters = []
dispatch = []
for tile, reg, threads, layout in itertools.product([16,32,64],[1,2,4],[16,64,256,1024],["row","column"]):
    required = (tile//reg)**2
    name = f"T{tile}_R{reg}_{layout}"
    reason = "exceeds_1024" if required>1024 else "thread_count_mismatch" if required!=threads else "eligible"
    filters.append(dict(name=name,tile=tile,reg=reg,threads=threads,layout=layout,status=reason))
    if reason=="eligible":
        configs.append(name)
        params = f"{tile},{reg},{str(layout=='column').lower()}"
        dispatch.append(f'if(name=="{name}") return {{false,launch_register<{params}>,inspect_register<{params}>}};')
lab.write_csv(run/"configuration-filter.csv",filters)
(src/"dispatch.inc").write_text("\n".join(dispatch))
for name in ["run.py","register.cuh","DESIGN.md"]:
    shutil.copy2(root/name,src/name)
# Preserve the reference, timing code and baseline in the compilation snapshot.
for path in [p3/"src/gemm.cuh",p3/"src/reference.hpp",legacy/"include/common.cuh"]:
    shutil.copy2(path,src/path.name)
header=(src/"register.cuh").read_text().replace("../WhaleGPU-Phase1/phase3/src/gemm.cuh","gemm.cuh")
(src/"register.cuh").write_text(header)
driver=(p3/"src/bench.cu").read_text().replace('"gemm.cuh"','"register.cuh"').replace("Variant v;","RegisterVariant v;").replace("variant(x[1])","select_register(x[1])")
(src/"bench.cu").write_text(driver)
build=(p3/"build.cmd").read_text().replace("-Isrc -I..\\include src\\bench.cu","-I. -Xptxas=-v bench.cu").replace('"%~1\\whalegpu_gemm.exe"','"whalegpu_gemm.exe"')
(src/"build.cmd").write_text(build)
with (run/"build.log").open("w") as log:
    subprocess.run(["cmd","/d","/c","build.cmd","."],cwd=src,stdout=log,stderr=subprocess.STDOUT,check=True)
(run/"build").mkdir()
shutil.move(src/"whalegpu_gemm.exe",run/"build/whalegpu_gemm.exe")
lab.save(run/"hashes.json",{str(p.relative_to(run)):lab.sha(p) for p in run.rglob("*") if p.is_file()})
lab.command(["nvidia-smi","-q"],run/"nvidia-smi.txt")
variants=["tiled16_row"]+configs
lab.execute(run,"validate",lab.validation_plan(variants),"validate")
shapes=[(64,64,64),(257,193,129),(513,769,257),(512,512,512),(1024,1024,1024)]
rng=random.Random(3400)
plan=[]
for shape in shapes:
    batch=[lab.job(v,shape) for v in variants]
    rng.shuffle(batch)
    plan.extend(batch)
lab.execute(run,"search",plan)
matrix=lab.rows(run/"search/summary.csv")
winners=[]
plan=[]
for shape in shapes:
    candidates=[r for r in matrix if tuple(int(r[d]) for d in "mnk")==shape and r["variant"] in configs]
    winner=min(candidates,key=lambda r:float(r["kernel_median"]))
    winners.append(winner)
    for pair in range(21):
        group=f"{shape[0]}-{pair}"
        batch=[lab.job(v,shape,rounds=1,group=group,role=role) for v,role in [(winner["variant"],"winner"),("tiled16_row","baseline")]]
        rng.shuffle(batch)
        plan.extend(batch)
lab.save(run/"winners.json",winners)
lab.execute(run,"confirm",plan)
confirmed=lab.rows(run/"confirm/summary.csv")
after={str(p.relative_to(legacy)): lab.sha(p) for p in legacy.rglob("*") if p.is_file()}
assert before==after,"Legacy tree changed"
checks=lab.rows(run/"validate/correctness.csv")
lines=["# Register Tiling 实机报告","",f"实验 {run.name}。硬件与工具链详见 [hardware.json](search/hardware.json) 和 [nvidia-smi](nvidia-smi.txt)。",
       "",f"合法参数 {len(configs)} 组，排除 {len(filters)-len(configs)} 个组合。原版基线和全部候选通过 {len(checks)} 项完整输出检查。覆盖 Phase 3 的 20 种尺寸和六种输入场景，含 K=0、非整除矩形、不同随机种子及 NaN 写前哨兵。最大绝对误差为 {max(float(x['max_abs_error']) for x in checks):.6g}。","",
       "搜索对每种配置预热 10 次，采集 15 个样本，每个 CUDA Events 样本均摊 10 次 GEMM。独立进程中按尺寸执行 21 组随机顺序配对。下表为独立复测中位数，加速比取配对时间比中位数。有效吞吐按 2MNK 计算。E2E 包含同步及 H2D/D2H。","",
       "| M×N×K | 候选 | 基线 ms | 候选 ms | TFLOPS | 候选 E2E ms | 配对倍数 [bootstrap 95%] | 寄存器/线程 | Shared B | 理论 occupancy |","|---|---|---:|---:|---:|---:|---|---:|---:|---:|"]
from gemm_report import interval
for shape,w in zip(shapes,winners):
    data=[r for r in confirmed if int(r["m"])==shape[0]]
    def values(role,metric):
        return {r["group"]:float(r[metric]) for r in data if r["role"]==role}
    base=values("baseline","kernel_median")
    opt=values("winner","kernel_median")
    ratios=[base[key]/opt[key] for key in base]
    lo,hi=interval(ratios)
    med=statistics.median(opt.values())
    lines.append(f"| {'×'.join(map(str,shape))} | {w['variant']} | {statistics.median(base.values()):.5f} | {med:.5f} | {int(w['flops'])/med/1e9:.3f} | {statistics.median(values('winner','e2e_median').values()):.4f} | {statistics.median(ratios):.3f} [{lo:.3f}, {hi:.3f}] | {w['registers_per_thread']} | {w['shared_bytes']} | {float(w['theoretical_occupancy']):.1%} |")
lines+=["","区间跨越 1 的比较应视为本轮证据不足。row / column 指协作加载的枚举顺序，两者使用同一物理布局。完整均值、中位数、标准差、P95、寄存器、共享内存及理论 occupancy 见 [搜索数据](search/summary.csv)。原始样本见 [独立复测](confirm/samples.csv)。理论 occupancy 来自 CUDA occupancy API。","","## 源码审计","",f"原工程 {len(before)} 个文件的 SHA256 前后一致。原版 tiled16_row 与 CPU Reference 使用原文件副本。驱动仅替换头文件、变体类型和选择器。可审查 [编译快照](source/bench.cu)、[指纹](hashes.json) 和 [ptxas 资源与 spill 日志](build.log)。",
"","线程以互不重叠的 R×R 区域拥有输出。局部累加器使用编译期常量下标和完全展开循环。每个共享槽由唯一线程写入，越界加载补零，加载后及覆盖前都有全 Block 屏障。边界仅在最终写回时按行列判断。K=0 跳过乘加并写零。工具状态与安装步骤见 [Windows 工具检查](../../TOOLS.md)。","",
"最大方阵输入输出共约 12 MiB，本次使用预热后的设备缓冲区。结果适用于此设备、本次运行及所列矩阵尺寸。"]
(run/"REPORT.md").write_text("\n".join(lines),encoding="utf-8")
print(run/"REPORT.md")
