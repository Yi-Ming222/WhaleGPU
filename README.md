# WhaleGPU

WhaleGPU 是一个 CUDA 学习与 GPU Kernel 优化实验仓库。从 Vector Add 和归约入手，逐步研究矩阵乘法（GEMM）、Tensor Core、Softmax、LayerNorm 和算子融合。各阶段的源码、测试和实验脚本可用于追踪实现方式、数值正确性与性能测量口径。

## 选择入口

Level 3 已验收的核心能力包括四类算子的 CUDA 实现与数值校验、不同优化策略的基准比较，以及从执行计划、GPU 回执到性能报告的追溯。可从仓库根目录按目的选择入口：

| 目的 | 操作与运行边界 |
|---|---|
| 只读展示已有结果 | 取得 Phase C 冻结附件，核对 SHA256 后直接读包内历史报告。无需编译或启动 GPU。见 [Paired Runner：冻结附件](paired-runner/README.md#冻结附件与只读查看)。 |
| 演练计划与错误恢复 | Phase A 的 `plan` 和 `dry-run` 使用 Python 并写入新归档；`inspect` 只读核对该归档。操作命令见 [Paired Runner 手册](paired-runner/README.md)。 |
| 在本机执行 CUDA | 可从基础 smoke 测试开始。Paired Runner Phase B 可独立构建和验证一个 Unit，Phase C 会构建并执行多 Unit 配对测量。两者都会创建新的运行目录。 |

## 已验证环境与兼容性

Level 3 的实机验证主要在一台 Windows 11 笔记本上完成。以下版本来自历史设备回执和构建记录。

| 项目 | 实际验证环境 |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Laptop GPU，Compute Capability 12.0，CUDA 编译目标 `sm_120` |
| 操作系统与驱动 | Windows 11 25H2（build 26200）；历史环境审计记录的 NVIDIA 驱动版本为 582.05 |
| CUDA | CUDA Toolkit 13.0，NVCC 13.0.48 |
| 主机编译器 | Visual Studio Build Tools 2022，MSVC x64 工具集 14.39.33519，编译器版本 19.39.33523 |

其他 GPU 架构、操作系统及 CUDA/MSVC 版本组合尚未经过系统性验证。迁移时可能需要调整工具链路径、编译目标或依赖特定架构的实现。Paired Runner B/C 还会核对设备 0 的型号，并依赖 Windows SDK 的 `bcrypt.lib`。具体条件见 [Paired Runner 环境说明](paired-runner/README.md#真实-cuda-执行所需环境)。

## 快速开始：基础 GPU 测试

运行前确认 GPU 支持脚本的编译目标、驱动可用，且本机的 Toolkit 和 MSVC 安装路径与 [Phase 1 构建脚本](WhaleGPU-Phase1/build.cmd) 一致。Python 实验入口和学习平台还需要 Python 3，图形学习平台使用 Tkinter。

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
| Paired Runner A/B/C | 以寄存器 GEMM 为例，依次学习平衡随机化、真实设备回执和多 Unit 比较 | [命令、环境与归档](paired-runner/README.md) |

Phase 2 和 Phase 3 分别位于 `WhaleGPU-Phase1/phase2` 和 `WhaleGPU-Phase1/phase3`。3.4 起的工程位于同级目录，使用跨阶段头文件；复现时应保留目录相对位置。CUDA 环境检查另有 [CMake smoke test](cuda-setup/smoke-test/CMakeLists.txt)。

教学演示可从 [Vector Add 示例](GPU-Learning/01-vector-add/main.cu)、[图形学习平台](GPU-Learning/launch-platform.cmd) 和[已实现的归约参考模型](GPU-Learning/platform/reduction_model.py)入手。[学生练习说明](gpu-lab/README.md)区分了真实 GPU Map、CPU 逐步演算和待填关卡。`gpu-lab/puzzles.py` 中 Zip、Grid-Stride、Shared、Reduction 四题留给学生完成；这些关卡会在填好前报告 `NotImplementedError`。

## 已知局限与复现实验

编译成功只验证构建链路。复现时还需在目标设备上执行相同用例，按原有 CPU 参考和误差界检查数值结果，再比较相同输入、精度与计时区间下的性能。GPU 架构、驱动、工具链及运行时的温度、时钟和功耗状态均可能影响观测结果。

Phase C 的冻结基准来自 RTX 5070 Laptop GPU 上一次 FP32、`513×769×257` 的运行：8 个 Unit、16 个有效配对。A（T32）和 B（T64）的中位批次平均设备时间分别为 **41.875200** 和 **40.940799 µs/GEMM**，配对差 A−B 的均值为 **+1.368400 µs/GEMM**。这些数值描述该次单进程观测；跨进程、跨会话的稳定优势尚未验证。[原始报告的只读查看方法](paired-runner/README.md#冻结附件与只读查看)

Phase 2 分别记录 CUDA Events 测得的完整设备归约时间，以及包含 H2D、归约、D2H 和同步的端到端时间。比较配置时应使用相同的输入、算子、分配策略和采样设置。按逻辑字节数折算的有效带宽与硬件计数器采集的 DRAM 吞吐采用不同口径。详见 [Phase 2 测量合约](WhaleGPU-Phase1/phase2/DESIGN.md)。

Phase 4.3 的完整历史复现需要按 [复现说明](WhaleGPU-Phase43/REPRODUCE.md) 从经校验的备份恢复审计材料、实验快照和报告脚本，并准备 `cuobjdump.exe` 与 `nvdisasm.exe`。这些材料和工具二进制需要单独准备。复现说明列出了实验运行、回归、统计核验与独立重建的入口。

## 欢迎复现反馈与贡献

欢迎分享不同 GPU 架构上的复现报告、其他操作系统或工具链的兼容性反馈、可复现的 Bug，以及对 kernel 实现和优化方法的技术建议。仓库目前已归档，GitHub 上的 Issue 和 Pull Request 为只读。对于兼容性 Pull Request，最有帮助的证据是修改涉及的配置、原问题的复现步骤，以及同一用例修改前后的验证结果。

建议在反馈中列出 GPU 型号与计算能力、操作系统、驱动、CUDA 和主机编译器版本，附上提交 SHA、实际构建命令、输入尺寸及测试结果。报告 Bug 时请说明预期行为与实际现象，并附上能定位问题的必要日志片段，隐去凭据和私人路径。

感谢每一位阅读、复现、审查和改进 WhaleGPU 的开发者。
