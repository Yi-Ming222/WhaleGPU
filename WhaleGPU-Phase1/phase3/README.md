# WhaleGPU Phase 3：GEMM Optimization Lab

独立的原生 CUDA C++ GEMM 实验工程，支持 `C[M,N]=A[M,K]×B[K,N]`、非方阵和非整除尺寸。包含 4 个 Naive 访问模式及 6 个 Shared Memory Tiled 版本，不使用 Tensor Core 或 cuBLAS。

本轮已实机验收：[分阶段验收与结论](ACCEPTANCE.md)。

## 运行

在本目录执行：

```powershell
python -B scripts/experiment.py --stage all
```

沿用已验证的 nvcc 13.0.48、MSVC 14.39、sm_120 编译链路。Python、PowerShell/.NET 和 CUDA 均使用现有安装，不需要额外 Python 包。CPU Reference 使用 OpenMP，最多 8 线程。脚本只设置编译及 PowerShell 子进程环境，不修改系统策略、驱动或 Toolkit。

分阶段：

```powershell
python -B scripts/experiment.py --stage naive
python -B scripts/experiment.py --stage memory --run runs/<run-id>
python -B scripts/experiment.py --stage tiled --run runs/<run-id>
python -B scripts/experiment.py --stage benchmark --run runs/<run-id>
python -B scripts/experiment.py --stage confirm --run runs/<run-id>
python -B scripts/experiment.py --stage profiler --run runs/<run-id>
python -B scripts/experiment.py --stage compatibility --run runs/<run-id>
python -B scripts/experiment.py --stage report --run runs/<run-id>
```

每次创建唯一 run；已有阶段目录拒绝覆盖。失败日志保留。框架测试：`python -B -m unittest discover -s tests -v`。

## 文件与记录

```text
DESIGN.md                  实施前设计与测量合约
build.cmd                  独立 nvcc 入口
src/gemm.cuh               4 个 Naive / 6 个 Tiled Kernel 与资源查询
src/reference.hpp          确定性随机输入、CPU double Reference、逐元素比较
src/bench.cu               正确性/Events/E2E 驱动
scripts/experiment.py      复用 Phase 2 统计与工具的分阶段调度
scripts/gemm_report.py     记录审计、独立配对对比、报告
scripts/plot.ps1           PNG 性能曲线
tests/test_framework.py    FLOPs、单位、统计、覆盖与异常记录检查
runs/<id>/                 原始样本、源码快照、硬件、兼容回归、报告
```

run 内每个实验阶段包含 plan.csv、correctness.csv、samples.csv、summary.csv（计时阶段）、resources.csv、references.csv、first_calls.csv 和 hardware.json。记录均按 id 连接。源码快照、SHA256、二进制指纹、完整命令和 GPU 遥测保留在 run 根目录。

`compatibility/` 重新编译并运行未经修改的 Phase 1/2 源码；所有新回归结果也写入本目录。`legacy-files-before.json` 和兼容性结果保护原目录的代码、可执行文件及全部历史记录。

## 实验含义

- Naive 固定 256 threads，分别沿输出列/行映射线程；B 使用 row-major 或 BT[N,K] 布局，C 始终 row-major。
- Tile 为 8/16/32，Shared Memory 缓存 A/B，越界补零；所有线程参与加载后与复用前的同步屏障。
- 正确性包含 20 个边界形状、3 个随机种子、zero/identity/tail；M/N=0 输出为空，K=0 输出零。数值合约为有限 float32；尺寸验证入口目前限制各维不超过 2048。
- 默认预热 10 次、15 轮、每轮 10 次均摊。中位数/样本标准差/P95 直接复用 Phase 2 统计模块，不删除慢样本。
- Kernel 测设备驻留输入；E2E 包括 BT CPU 打包、H2D、GEMM、D2H、同步，分配排除。CPU 打包时间也单独保存。
- FLOPs=2MNK，有效 TFLOPS=FLOPs/(ms×10^9)。它是有用工作吞吐，不是 profiler 硬件计数器。
- 独立复测只对相同 row-major 布局下的最佳 Tile 和 Naive 做比较，21 组随机顺序配对，避免混入布局转换收益。
- Nsight Compute/Compute Sanitizer 重新探测；无工具时明确标记 NOT_FOUND/NOT_RUN。静态资源和理论 occupancy 不等于硬件实测。

实机记录：[分阶段实验报告](runs/20261117-234207-d988b3a2/report/REPORT.md)。该报告只反映本机本次输入与缓存/功耗条件；最大 Benchmark 为 1024³，A+B+C 约 12 MiB，小于本机 L2。
