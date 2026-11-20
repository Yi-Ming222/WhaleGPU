"""Explicit CPU teaching model; never presented as GPU execution traces."""
LESSONS = {
'Thread / Block / Grid': '一个线程计算一个元素。全局索引 i = blockIdx.x * blockDim.x + threadIdx.x。\n\n本实验 N=1,048,579，每个 block 256 个线程。最后一个 block 的无效线程由 i < n 保护。\n练习：计算需要多少个 block，最后一个 block 有多少有效线程。',
'Warp / Lane': '一个 warp 包含 32 个线程。对于一维 block，lane = threadIdx.x % 32。\n\n同一个 warp 中分支路径不同可能降低执行效率。不要用 warp 的调度行为代替需要的同步。',
'Global Memory': 'cudaMalloc 分配 GPU 内存，cudaMemcpy 在主机与设备之间复制数据。\n\nVector Addition 中相邻线程访问相邻 float，适合合并访问。每个元素读取两个 float、写入一个 float。',
'Shared Memory': 'Shared Memory 属于一个 block，block 内线程共同访问。\n\nReduction 可先把输入载入共享内存，再在 block 内逐轮合并。它的容量有限，并可能发生 bank conflict。',
'Synchronization': '__syncthreads() 是 block 范围的屏障。所有参与的线程必须按一致的控制流到达屏障。\n\n尾部无效线程可写入归约单位元 0，然后继续参与同步；不能在屏障前直接退出。',
'Warp Shuffle': '__shfl_down_sync(mask, value, offset) 在参与的 warp 线程之间交换寄存器值。\n\nmask 必须准确描述参与线程。典型归约 offset 为 16、8、4、2、1。读取不参与的源 lane 结果不可依赖。',
'Reduction': '求和归约将多个输入逐步合并。共享内存二叉归约每轮将活跃数量减半。\n\n跨 block 无法直接使用 __syncthreads()：两阶段归约先输出 block 小计，再由第二个 Kernel 合并。\n\n可视化页面展示 CPU 计算的教学步骤，不是 GPU 硬件执行轨迹。'}


def reduction_steps(values):
    states = [list(values)]
    while len(values) > 1:
        values = [sum(values[i:i+2]) for i in range(0, len(values), 2)]
        states.append(values)
    return states
