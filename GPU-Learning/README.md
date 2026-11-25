# GPU-Learning

原生 Windows CUDA 学习项目。2026-11-14 实测 Vector Addition 通过：1,048,579 个元素，0 个不一致。

## 运行

在本目录 PowerShell 执行 `./build-and-run.cmd`。

脚本在当前进程选择 MSVC 19.39，调用 CUDA 13.0 nvcc，以 sm_120 编译并在 GPU 上执行；失败返回非零退出码。不会修改系统环境变量。

## 已验证环境

- NVIDIA GeForce RTX 5070 Laptop GPU，计算能力 12.0
- NVIDIA Driver 582.05，保持原有驱动
- CUDA Toolkit 13.0，nvcc 13.0.48
- Visual Studio 2022 Build Tools 17.14.45
- MSVC 19.39.33523；工具目录 14.39.33519
- Windows SDK 10.0.22621.0
- CUDA_PATH 和 CUDA bin PATH 由官方安装器配置

CMake 3.31.6-msvc6 可查询版本，但配置项目时异常退出（-1073740791 / 0xC0000409）。英文临时路径和精简进程 PATH 下仍可复现，原因未确定。当前使用已实测的 nvcc 直接构建入口，不依赖 CMake；未绕过编译器兼容检查。

## 学习目录

1. 01-vector-add：已实现并通过 GPU / CPU 对比
2. 02-basic-reduction：待实现
3. 03-shared-memory-reduction：待实现
4. 04-warp-shuffle-reduction：待实现
5. 05-two-stage-reduction：待实现
6. 06-benchmark：待实现
7. 07-optimization：待实现

后续可使用 CUDA Events 测量 GPU 执行时间，正式优化前先做正确性检查。

## 图形化平台（第一阶段）

运行 `launch-platform.cmd`，或执行 `python -B platform/app.py`。依赖仅为现有 Python 标准库和 Tkinter。

- Learning Center：7 个主题与本地学习进度。
- Kernel Laboratory：原始 CUDA 源码、编译、编译并运行、实际输出；可读取最近记录。
- Benchmark：固定规模 Vector Addition 的 1–20 次独立进程测试，均值、最小值、最大值与逐次延迟柱状图。
- Performance Analysis：明确列出未采集指标和后续 Nsight 导入接口方向。
- GPU Visualization：由 CPU 求和逻辑生成的 Reduction 步骤，支持播放、暂停、单步和重置；不是 GPU 硬件 trace。

编译总是先执行，失败时不会运行旧二进制。编译超时 120 秒，执行超时 30 秒，超时时终止进程树。界面只允许一个后台任务，不接受任意 Shell 命令。实验记录保存在 records/*.json，包含源码快照、退出码、stdout、stderr、逐次时间。学习进度保存在 records/learning.json。

Benchmark 数值是 CPU 侧进程墙钟耗时，不含编译，包含进程启动、CUDA 初始化、内存分配、H2D、launch、Kernel、D2H、CPU 验证和退出。无预热。未采集 Kernel-only CUDA Events 时间、Occupancy、GPU 吞吐或 profiler 指标。第一阶段尚不支持编辑源码、多版本/多规模对比和 Nsight 报告导入。

模块：platform/app.py（桌面页面和调度）、runner.py（固定参数进程调用和实验存储）、lessons.py（课程与教学计算模型）。后续新增实验需注册到 runner.EXPERIMENTS 并实现结构化结果协议。

运行页面/白名单/教学模型测试：`python -B platform/test_platform.py`。

## Phase 2 可视化
GPU Visualization 已升级为线程网格和 Shared Memory 双视图。支持 Shared Reduction、Warp Shuffle、多 Warp 和 Two-stage 教学模拟，含前进/返回/播放/暂停/重置/速度控制。详细测试与边界见 PHASE2-REPORT.md。
运行测试：python -B platform/test_reduction.py。
