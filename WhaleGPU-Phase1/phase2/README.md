# WhaleGPU Phase 2：GPU Performance Engineering

在 Phase 1 基础上新增的性能实验系统。原版 CUDA 源码、CPU Reference、672 项测试和构建入口均保留。这里直接链接原版 Reduction 作为对照，参数化算法位于 `src/kernels.cuh`。

已完成两次实机实验：[验收与重复实验说明](ACCEPTANCE.md)、[最终性能报告](runs/20261117-230751-44663126/report/REPORT.md)。

## 使用

在本目录执行，依赖本机已有 Python、PowerShell/.NET、MSVC 14.39、CUDA 13.0：

```powershell
python -B scripts/run.py --stage all
```

每次创建 `runs/<上海时间-随机ID>/`，不会覆盖旧实验。运行可能持续数分钟，包含 34,992 个正确性用例、504 个固定矩阵配置、1,296 个调参候选、756 个独立复测配置及 Phase 1 回归。性能测试期间避免启动其他 GPU 重负载；脚本不改变时钟、电源设置或驱动。

分阶段运行：先执行 `--stage engine` 创建 run，再使用该路径执行后续阶段：

```powershell
python -B scripts/run.py --stage engine
python -B scripts/run.py --run runs/<run-id> --stage validate
python -B scripts/run.py --run runs/<run-id> --stage matrix
python -B scripts/run.py --run runs/<run-id> --stage tune
python -B scripts/run.py --run runs/<run-id> --stage confirm
python -B scripts/run.py --run runs/<run-id> --stage profiler
python -B scripts/run.py --run runs/<run-id> --stage regression
python -B scripts/run.py --run runs/<run-id> --stage report
```

阶段目录已存在时拒绝覆盖。失败实验保留日志；重新开始使用新 run。统计/记录验证单元测试：

```powershell
python -B -m unittest discover -s tests -v
```

## 实验记录

- `environment.json`、`nvidia-smi.txt`、`telemetry.csv`：硬件、软件、固定种子、功耗/温度/时钟。
- `source/`、`source-sha256.json`、`binary-sha256.json`：实际编译源码和二进制指纹。
- 每阶段 `plan.csv`：完整任务参数，`hardware.json`：实际设备、编译器和 CUDA 初始化时间。
- `first_calls.csv`：每配置第一次调用主机时间，不进入稳定统计。
- `correctness.csv`：CPU double 参考、GPU 结果、误差、容差；错误候选不参与最优选择。
- `samples.csv`：每轮 Events 均摊 Kernel 时间、CPU E2E 时间和正确性。
- `summary.csv`：均值、中位数、样本标准差、P95、有效带宽、网格/轮数、静态资源。
- `winners.json`：每算子/规模搜索获胜参数；`confirm/`：独立随机顺序配对复测。
- `profiler.json`：Nsight Compute 探测结果、缺失项；有工具时额外保留独立 ncu 报告。
- `phase1-regression.log`、`phase1-integrity.json`：原测试回归和源码保护。
- `experiment-audit.json`：跨文件完整性、公平比较条件、资格与获胜配置核对。
- `report/REPORT.md`、`report/charts/*.png`：报告和可独立使用的性能曲线。

## 参数与计时语义

搜索 128/256/512 threads、1/4/8 items per loop、natural/2SM/8SM grid cap、shared/shuffle、striped/contiguous。部分小规模配置具有相同实际网格或布局行为；108 候选是参数组合数，不保证 108 种不同执行路径。Block 数量实际值另存 first_grid。

`pageable_reuse` / `pinned_reuse` 的 E2E 不含分配；`pageable_fresh` 包含设备分配/释放。输入 host allocation 都排除。所有 Kernel Events 排除内存分配和传输，包含全部归约轮次和可能的提交空隙。默认预热 10 次，15 轮，每轮 10 次均摊；独立验证 21 组配对。未删除异常慢样本。

有效带宽是逻辑 `(input bytes + final output bytes) / median time`，不是 DRAM 硬件计数器。理论 occupancy、寄存器和 shared memory 静态资源不能替代 achieved occupancy、warp stall 或 memory efficiency。Nsight Compute 不存在时明确报告缺失，不安装或改变环境。

CUDA 支持本实验的有限 float32 输入、固定参数集合，尚不是通用张量库。报告会区分实测相关性与待 profiler 验证的瓶颈假设。详细约定见 [DESIGN.md](DESIGN.md)。
