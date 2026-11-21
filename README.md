# WhaleGPU

WhaleGPU 是一个 CUDA 学习与 GPU Kernel 优化实验仓库。从 Vector Add 和归约入手，逐步研究矩阵乘法（GEMM）、Tensor Core、Softmax、LayerNorm 和算子融合。各阶段的源码、测试和实验脚本可用于追踪实现方式、数值正确性与性能测量口径。

## 快速开始

Phase 1 的构建脚本面向 Windows，使用 CUDA Toolkit 13.0、MSVC Build Tools 2022（工具集 14.39），并编译为 `sm_120`。运行前确认 GPU 支持该目标，驱动可用，且本机的 Toolkit 和 MSVC 安装路径与 [Phase 1 构建脚本](WhaleGPU-Phase1/build.cmd) 一致。Python 实验入口和学习平台还需要 Python 3，图形学习平台使用 Tkinter。

在仓库根目录打开 PowerShell。先用第一条命令编译并执行最小 GPU 测试，再按需用第二条命令运行 Phase 1 的完整用例。

```powershell
.\WhaleGPU-Phase1\run-tests.ps1 -Stage smoke
.\WhaleGPU-Phase1\run-tests.ps1 -Stage all
```

`smoke` 检查 CUDA 编译、设备内存分配、Kernel 启动和结果回传。`all` 运行 Vector Add 及 Shared Memory、Warp Shuffle 两种归约实现的正确性与计时用例。每次运行都会创建独立目录，二进制位于 `WhaleGPU-Phase1/build/<run-id>/`，日志和运行元数据位于 `WhaleGPU-Phase1/results/<run-id>/`。完整用例的 `results.csv` 和 `timings.csv` 分别记录用例结果和计时样本。详见 [测试入口](WhaleGPU-Phase1/run-tests.ps1) 和 [Phase 1 工程设计](WhaleGPU-Phase1/docs/DESIGN.md)。

## 实验路线

| 阶段 | 实验内容 | 从这里读起 |
| --- | --- | --- |
| 1 | Vector Add，Shared Memory 与 Warp Shuffle 版 SUM、MAX、MIN 归约 | [工程设计](WhaleGPU-Phase1/docs/DESIGN.md) |
| 2 | 参数化归约，候选配置搜索，与原始归约基线配对复测 | [技术设计与测量合约](WhaleGPU-Phase1/phase2/DESIGN.md) |
| 3 | Naive 与 Tiled GEMM，CPU 参考和实验驱动 | [GEMM 源码](WhaleGPU-Phase1/phase3/src/gemm.cuh) · [实验脚本](WhaleGPU-Phase1/phase3/scripts/experiment.py) |
| 3.4 | 寄存器分块 GEMM，对比输出 Tile、每线程分工和协作加载方式 | [实验设计](WhaleGPU-Phase34/DESIGN.md) |
| 3.5 | Tensor Core 实验，对比 FP32 基线与 FP16、BF16、TF32 的 WMMA 路径 | [Kernel](WhaleGPU-Phase35/tensor.cuh) · [基准程序](WhaleGPU-Phase35/bench.cu) |
| 3.6 | 对比低精度输入转换的位置，以及共享加载、双缓冲与异步预取 | [实验设计](WhaleGPU-Phase36/DESIGN.md) |
| 4.1 | FP32 Softmax，对比 Warp、Block、多阶段归约与融合实现 | [实验设计](WhaleGPU-Phase41/DESIGN.md) |
| 4.2 | LayerNorm，对比两遍统计、Welford 归约与融合写回 | [实验设计](WhaleGPU-Phase42/DESIGN.md) |
| 4.3 | 对比 GEMM 后处理、Softmax 后处理及 LayerNorm 的分步和融合路径 | [实施协议](WhaleGPU-Phase43/DESIGN.md) · [复现说明](WhaleGPU-Phase43/REPRODUCE.md) |

CUDA 环境检查可从 [CMake smoke test](cuda-setup/smoke-test/CMakeLists.txt) 入手。学习材料包括 [Vector Add 示例](GPU-Learning/01-vector-add/main.cu)、[图形学习平台](GPU-Learning/launch-platform.cmd) 和 [GPU 练习目录](gpu-lab/)。

## 阅读计时结果与复现实验

Phase 2 分别记录 CUDA Events 测得的完整设备归约时间，以及包含 H2D、归约、D2H 和同步的端到端时间。比较配置时应使用相同的输入、算子、分配策略和采样设置。按逻辑字节数折算的有效带宽与硬件计数器采集的 DRAM 吞吐采用不同口径。详见 [Phase 2 测量合约](WhaleGPU-Phase1/phase2/DESIGN.md)。

Phase 4.3 的完整历史复现需要按 [复现说明](WhaleGPU-Phase43/REPRODUCE.md) 从经校验的备份恢复审计材料、实验快照和报告脚本，并准备 `cuobjdump.exe` 与 `nvdisasm.exe`。这些材料和工具二进制需要单独准备。复现说明列出了实验运行、回归、统计核验与独立重建的入口。
