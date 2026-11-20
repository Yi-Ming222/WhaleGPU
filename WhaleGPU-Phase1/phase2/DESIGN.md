# Phase 2 技术设计与实施计划

实施前提交，2026-11-17。Phase 1 源文件、接口、测试和构建脚本保持不变；Phase 2 只新增本目录，直接链接原版 reduction.cu 作为基线。

## 阶段

1. Benchmark Engine：进程内 CUDA 初始化；独立首次调用记录；配置预热；CUDA Events 完整归约链时间；steady_clock 端到端时间；原始样本与统计单元测试。
2. 参数化归约：128/256/512 threads，1/4/8 elements per thread（每次循环的处理粒度），natural/SM*2/SM*8 网格限制，shared/shuffle，striped/contiguous 数据分工。独立 CPU double 参考、空输入、尾部和多轮归约验证。
3. 固定矩阵与 Auto-Tuner：同输入、同预热、同采样配置，固定种子打乱顺序；每个候选先通过正确性，再参与按 kernel 中位数的搜索；所有失败也落盘。
4. 独立验证：获胜参数与原始 Phase 1 shared/shuffle 基线在新轮次交替复测，报告配对性能比及波动。基线使用同一编译选项和计时器。
5. Profiler 探测与 Markdown 报告、PNG 性能曲线（本机 .NET Chart）；重新运行未改动的 Phase 1 全部测试，并校验历史源码指纹。

## 测量合约

- 默认 warmups=10、rounds=15、repeats=10。每个 GPU 样本为 Events 包围 10 次完整归约的均摊时间；独立验证使用 21 轮、相同 repeats。P95 使用线性插值，标准差为样本标准差，单位 ms。
- 初始化与每个配置首次调用单列保存，不进入稳定样本。首次配置调用不是每个 Kernel 的冷进程启动测量。
- Kernel 时间排除分配、传输、CPU 验证；包含多轮归约与提交间隙。端到端样本每次执行一次 H2D→归约→D2H→同步，使用 CPU steady_clock。
- allocation 策略为 pageable_reuse、pinned_reuse、pageable_fresh。前两者分配在计时区间外，fresh 的端到端时间包含 cudaMalloc/cudaFree；三者 Kernel 区间均排除分配。主机输入分配和输入生成都不计入 E2E。
- Kernel 数据常驻设备，采用 warmed/reused 缓存策略，不声称是冷 DRAM 测试；E2E 紧随其后独立测量，不能用两个不同计时域相减作为精确传输时间。
- 有效带宽=(4*n+4)/kernel_seconds/1e9 GB/s；另记录各轮实际逻辑读写估算。不是 DRAM 实测吞吐。空输入只做正确性，不用于带宽排名。
- 对比只在相同规模、算子、allocation、数据模式、计时设置下进行。基准曲线包含超过本机 L2 的输入规模，实际 L2 大小由 runtime 记录。
- 不锁定时钟，不修改 Driver/Toolkit，不安装工具；保留 GPU 温度、时钟、功耗等遥测并标注 WDDM/共享桌面负载限制。

## 正确性与可追溯

输入 float32，输出 float32，有限值语义。CPU SUM 用 double，精确整数/二进制模式严格相等；非精确小数使用根据串行累加长度和树深度推导的保守误差界。MAX/MIN 严格相等。全部候选需通过边界矩阵并验证本次规模的输出。

每个 run 目录包含配置、种子、工具命令、源码快照与 SHA256、二进制 SHA256、GPU/CPU/OS/编译器版本、原始样本、逐候选正确性、摘要、最佳参数和独立验证。Phase 1 源码在运行前后对照 SHA256。失败状态明确保存且不作为最优候选。

## Profiler

探测 PATH、NVIDIA/CUDA 标准安装目录及 Nsight 目录。可用时单独运行 ncu，收集 LaunchStats、Occupancy、SpeedOfLight、MemoryWorkloadAnalysis、WarpStateStats；保留原始报告与错误。无工具或无计数器权限时明确缺失。

cudaFuncGetAttributes 提供寄存器、静态/动态 shared memory；cudaOccupancyMaxActiveBlocksPerMultiprocessor 提供理论驻留上界。这些不是实测 Achieved Occupancy，不能替代 Warp Stall、DRAM Throughput 或访存效率计数器。

方法参考：[CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)、[Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)。
