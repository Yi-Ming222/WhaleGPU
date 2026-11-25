# WhaleGPU Phase 1：基础 Kernel 引擎

独立的原生 CUDA C++ 工程，包含 Vector Add，以及 SUM / MAX / MIN 的 Shared Memory 和 Warp Shuffle 两种实现。2026-11-17 在 RTX 5070 Laptop GPU 上完成 672 个用例全量回归，全部通过。

## 运行

在本目录的 PowerShell 中执行：

```powershell
.\run-tests.ps1 -Stage all
```

分阶段执行：

```powershell
.\run-tests.ps1 -Stage smoke
.\run-tests.ps1 -Stage vector
.\run-tests.ps1 -Stage shared
.\run-tests.ps1 -Stage shuffle
```

每次先编译，失败不运行旧二进制；构建、测试失败返回非零退出码。脚本只在构建子进程中加载 MSVC 14.39 和 CUDA 13.0，使用 `nvcc -arch=sm_120 -std=c++17 -O3 -lineinfo`，不修改系统环境、不使用 CMake、不跳过编译器兼容检查。路径沿用本机已验证配置：`C:\BuildTools2022`、CUDA 默认 v13.0 安装目录。

## 文件

```text
WhaleGPU-Phase1/
  build.cmd                  nvcc 构建入口，参数为输出目录
  run-tests.ps1              唯一运行目录、环境、快照、日志
  include/common.cuh         CUDA 错误检查、资源管理、算子接口
  include/reference.hpp      独立 CPU Reference
  src/vector_add.cu          Vector Add
  src/reduction.cu           Shared / Shuffle，三个归约算子
  tests/main.cu              数据生成、正确性校验、CUDA Events
  docs/DESIGN.md             工程设计与测试约定
  docs/CMAKE-ISSUE.md         独立 CMake 问题
  docs/PHASE1-REPORT.md       分阶段实机结果
  build/<run-id>/            每次构建的可执行文件
  results/<run-id>/          每次运行的不可覆盖式历史记录
```

运行 ID 包含本地时间、阶段名和随机标识。每次保存 `run.json`、`hardware.txt`、`nvidia-smi.csv`、源码快照与 SHA-256、二进制 SHA-256、编译和测试日志。算子测试另保存 `results.csv`（每个用例的统计与正确性）和 `timings.csv`（各采样均摊时间）。Smoke 是编译/设备连通性验收，仅记录日志与硬件，不作为性能测试。

详见 [设计](docs/DESIGN.md)、[测试报告](docs/PHASE1-REPORT.md)、[CMake 问题](docs/CMAKE-ISSUE.md)。现有 GPU-Learning、cuda-setup、gpu-lab 目录不在本工程写入范围内。
