# 工程设计

## 实施与验收阶段

1. 独立目录、nvcc 构建、错误处理、日志基础设施；设备探测和 2+3 GPU smoke。
2. Vector Add、CPU 逐元素参考、多规模正确性和 Events 计时。
3. Shared Memory SUM/MAX/MIN，单 Block 到多轮跨 Block 归约。
4. Warp Shuffle SUM/MAX/MIN，并重新构建后全量回归。

## 接口与算法

输入、输出为 float32；输入长度为 size_t，固定 256 线程 Block，默认 stream，测试设备为 device 0。输入缓冲区只读。Vector Add 每个线程处理一个元素，`i < n` 保证尾部安全。

Reduction 每个线程装载一个输入或单位元：SUM=0、MAX=-inf、MIN=+inf。Shared 版用 256 个共享内存槽，每层做二分合并和全 Block 同步，线程 0 写出一个 Block 结果。Shuffle 版先对每个完整 Warp 执行 full-mask shuffle，再由第一个 Warp 合并 8 个 Warp 的共享内存结果；其余 24 Lane 使用单位元。所有参与 shuffle 的 Warp 都有完整 32 Lane，越界输入线程仍执行同步和 shuffle。

跨 Block 通过同一 stream 上后续 Kernel 合并，无跨 Block 屏障、无全局原子操作。两块临时缓冲区轮换读写，每块容量至少 `max(1, ceil(n/256))` 个 float；输入、两块临时缓冲区不得别名。每次重入从原始输入开始，不依赖上次结果。返回指向最终临时结果的设备指针，调用者保持其生命周期。测试最大规模需要三轮归约。

空 Vector Add 无 Kernel，耗时记为约定值 0，`kernel_passes=0`；空 Reduction 启动一个 Block，输出单位元。单元素 Reduction 仍启动一轮。空 MAX/MIN 的 ±inf 为工程定义的返回语义。

当前合约覆盖有限 float32 输入；未定义 NaN、输入无穷、溢出和有符号零保留语义。不是支持任意类型和任意 GPU 的通用库。未做 Nsight 分析或 Compute Sanitizer 验证。

## 正确性测试

16 个规模：0、1、2、3、31、32、33、255、256、257、513、1024、65537、1048576、1048579、2097155。覆盖空输入、单元素、Warp/Block 边界、二次幂、非二次幂及多轮归约。

每个规模 6 种可复现模式：

- mixed_exact：正负混合的二进制精确分数。
- negative / positive：全负/全正小整数，检查 MAX/MIN 的单位元错误。
- constant：全部 0.25，检查重复值及累加。
- tail_extrema：末尾为 +32768，倒数第二个为 -32768，检查尾部被遗漏的问题。
- fractional：按固定整数公式生成的非精确十进制小数。

CPU Reference 采用独立串行实现，SUM 使用 double 累加。Vector Add 全部逐元素严格相等；MAX/MIN 严格相等；前五种数据模式的 SUM 严格相等。fractional SUM 使用 `gamma(k) * sum(abs(x))` 绝对误差上界，`k=8*归约轮数`、`gamma(k)=k*float_epsilon/(1-k*float_epsilon)`。该界偏保守，CSV 同时记录实际误差和容差，便于审查；精确数据模式避免仅依靠宽松容差验证。

Vector Add 的 expected/actual 标量列为 0 占位，使用 mismatches 和 max_abs_error 检查整个向量；Reduction 两列为 CPU/GPU 标量。每个用例在完整计时执行后回传最终结果校验，不在计时区间执行 CPU 参考。CUDA API、Kernel 提交和完成阶段的错误都导致非零退出码。

## 计时与记录

每个非空 Vector Add 和每个 Reduction 用例预热 2 次，随后 5 个独立 Events 区间，每个区间重复 10 次操作，记录区间时间除以 10 的均摊毫秒值。results.csv 记录五个均摊值的均值、最小值、最大值；timings.csv 保存每个均摊值，不是单次 launch 的延迟分布。

Events 区间包含完整算子：Vector Add 为一轮，Reduction 为全部轮次。排除内存分配、H2D/D2H、输入生成和 CPU 校验。区间仍可能包含 Kernel 提交之间的 GPU 空隙及 WDDM 调度影响，因此不是隔离的硬件指令执行时间。未锁定频率、未清空缓存、未控制其他 GPU 负载；当前数值作为工程基线，不能作为稳定性能排名。

results.csv 每行包含算子、版本、规模、模式、正确性、误差、阈值、轮数、耗时以及 GPU/CC/显存/驱动 API/Runtime/Block/采样配置。目录级元数据额外包含实际驱动发行版本、设备 UUID、PCI 地址、SM 数、nvcc/MSVC 版本、OS、CPU、起止时间、源码和二进制指纹。
