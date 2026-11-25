# Phase 3：GEMM Optimization Lab 技术设计

实施前提交，2026-11-17，Asia/Shanghai。所有新文件放入独立 phase3 目录；不修改 Phase 1/2 源码、构建入口和原始实验记录。保留 CUDA 13.0、nvcc 13.0.48、sm_120、MSVC 1939 和现有驱动。

## 阶段与交付

1. Naive GEMM、独立 double CPU Reference、随机数据及边界正确性验收。
2. Memory Coalescing 实验：输出列/行线程映射 × B row-major/预转置布局，固定 256 threads，4 个 Naive 版本。完成正确性与初始 Benchmark。
3. Shared Memory Tiling：8/16/32 Tile × 两种 B 布局，6 个版本。边界 Tile、共享内存同步与非整除输入验收。
4. 10 版本完整性能矩阵、同布局最佳 Tile 选择、独立 21 组配对复测、Markdown 报告与 PNG 曲线。
5. 原 Phase 1 672 项测试、原 Phase 2 34,992 项参数正确性矩阵/12 组引擎检查/6 项框架测试，全部使用原始源码，在 Phase 3 实验目录内编译并保存新回归数据。对 Phase 1/2 整棵历史文件树做前后 SHA256 核对。

## 数学与内存合约

`C[M,N] = A[M,K] * B[K,N]`，alpha=1、beta=0，float32 输入输出，float32 FMA 累加。CPU Reference 按逻辑 row-major 输入执行独立 double 累加，不重用 GPU 索引函数。有限浮点语义，不定义 NaN/无穷/溢出。M/N/K 可不相等、可非 2 次幂；M 或 N 为 0 时输出为空，不计时；K=0 且输出非空时输出 0。

Naive 每线程计算一个输出。列映射：linear id -> row=id/N、col=id%N；行映射：row=id%M、col=id/M。两者输出均为 row-major。B 可按 row-major 或 BT[N,K] 物理布局存储。访问模式是源码可检查的事实；访存事务数/缓存命中率/效率须有 Nsight 计数器才作为实测指标。

Tiled 每个 Block 计算 T×T 输出，T=8/16/32。A/B Tile 越界处补零，全部线程都参加两次屏障：加载完成后再读、计算完成后再覆盖。输出越界线程不写 C，但不会提前退出屏障。BT 版协作读取连续 K 数据并在 Shared Memory 中转置，之后使用相同数学乘法顺序。无 Tensor Core、无 cuBLAS、无 fast-math。

## 正确性

随机输入使用固定种子的 32-bit LCG，可复现。覆盖三个随机种子、全零、矩形 identity A、尾部极值，共六种场景。测试含 0/1、8/16/32 边界、长条/非方阵/非整除尺寸。每次比较整个 C 矩阵，保存不一致数量、最大绝对误差、归一化误差、容差。

随机/identity 的每元素容差为 `gamma(K) * sum(abs(Aik*Bkj)) + 1e-7`，gamma(K)=K*float_epsilon/(1-K*float_epsilon)；CPU 用 double。零和尾部精确模式严格比较。输入与参考准备在所有计时区间之外，Reference 时间单独记录。输出在正确性阶段预置 NaN，检查漏写；实测每个 sample 后也验证全部输出。

## 公平计时

复用 Phase 2 的 summarize/percentile 定义：均值、中位数、样本标准差、线性插值 P95，不删除慢样本。预热 10 次；15 轮，每轮 CUDA Events 包围 10 次 Kernel，输出均摊 ms；独立复测 21 组，每组 baseline 与 winner 顺序由固定种子随机化。

CUDA 初始化、第一次调用和 CPU Reference 不进入稳定样本。Kernel 计时只包含设备驻留输入的 GEMM，所有版本统一 warmed/reused 缓存策略，不包含数据上传、下载和分配。CPU steady_clock E2E 包含 B 打包（BT 版）、A/B H2D、GEMM、C D2H 和同步；复用主机/设备缓冲区，排除分配。row-major 的打包时间按约定记为 0，BT 每次重新打包并保存 pack_ms。不能把预转置 Kernel 时间称为完整转置方案端到端收益。

名义 FLOPs=`2*M*N*K`（FMA 计 2 FLOPs）；有效 TFLOPS=`FLOPs/(kernel_ms*1e9)`。这是按有用矩阵尺寸计算的算法吞吐，不是硬件执行指令数；边界 Tile 的补零工作不增加名义 FLOPs。E2E 同样给出单独 effective TFLOPS。空输入和 K=0 只做正确性，不用于吞吐排名。

独立对比只在相同尺寸、种子、输入布局、计时/预热/重复条件下进行；选择 row-major Tiled 三种 Tile 中的最佳候选，再用新样本与 naive_col_row 对比。其他 Naive 和 BT 布局单独出表分析，避免混淆线程分工、数据打包与 Tiling 收益。

## 可追溯与 Profiler

每次 run 保存配置、种子、源码快照与 SHA256、二进制指纹、硬件/编译器/OS、功耗温度时钟遥测、逐用例 CPU/GPU 误差、原始计时、统计、独立复测、兼容性回归、报告和曲线。

重新探测 Nsight Compute，若可用单独采集 LaunchStats/Occupancy/MemoryWorkloadAnalysis/WarpStateStats/SpeedOfLight；采集时间不进入 benchmark。缺工具或缺计数器权限时记录缺失。可用时也可独立运行 Compute Sanitizer racecheck/synccheck；未运行则只报告源码屏障检查和数值正确性，不宣称已通过竞态检测。

参考：沿用 Phase 2 已保存的计时定义及 NVIDIA CUDA Best Practices / Nsight Compute Profiling Guide。无 profiler 时仅用实测数据支持性能差异，瓶颈因果解释注明为待验证假设。
