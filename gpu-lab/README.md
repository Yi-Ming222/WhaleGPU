# GPU 小实验室

从 Map、Zip 和线程索引开始，最后做每个 block 的归约。只用 Python 标准库。

## 先运行，再改代码

在这个目录打开 PowerShell，执行下面三条命令。也可以直接让当前 Codex 对话帮你运行其中一条。

```powershell
python -B gpu_map.py
python -B check.py Map
python -B trace.py
```

- `gpu_map.py` **在 NVIDIA GPU 上执行**。程序把五个整数从 CPU 复制到显存，启动 1 个 block、8 个线程。线程 0 到 4 各处理一个元素，其余线程跳过。程序等 GPU 完成后读回结果，与 CPU 逐项比较。预期输出为 `[-6, 0, 8, 14, 20]`。
- `check.py Map` 检查 `puzzles.py` 里的 CPU 热身示例。把 `2 * x` 改成 `3 * x`，观察测试报出的差异，再恢复。
- `trace.py` **只在 CPU 上逐阶段演算**。预期块小计为 `[52, 39]`，总和为 `91`。Python 列表扮演共享内存，每个阶段串行完成，因此无法检测 GPU 数据竞争或漏写的同步屏障。

## 逐关做题

每次只改 `puzzles.py` 中当前关的函数，再运行对应命令。空白关卡会报 `NotImplementedError`。完成全部关卡后，用 `python -B check.py` 一次检查所有题。

| 关卡 | 要做什么 | 检查 |
| --- | --- | --- |
| 1. Map | 看懂热身示例。每个下标独立把元素乘以 2。 | `python -B check.py Map` |
| 2. Zip | 相同下标的两个元素相加。两组长度不同时抛出 `ValueError`。 | `python -B check.py Zip` |
| 3. Grid-Stride Loop | 返回每个线程访问的下标列表。线程从自己的全局编号起步，每次跨过整个 grid 的线程数。没有元素可读的线程也保留空列表。 | `python -B check.py Stride` |
| 4. Shared Memory 和同步 | 每个 block 先把一块数据写入临时列表，缺位补零。等整块写完，再让每个位置读取自己和右邻居。块尾的右邻居视作零。想想把两阶段合并会读到什么。 | `python -B check.py Shared` |
| 5. Reduction | 每个线程先把自己访问的元素累加，再把线程小计放进块内临时列表。每轮把右半区加到左半区，轮次之间分阶段执行。返回每个 block 的小计。 | `python -B check.py Reduction` |

例如 2 个 block、每块 3 个线程时，全局线程 0 读下标 0、6、12。最后一关固定使用 4 个 block、每块 64 个线程。在这道题的分配规则下，1024 个元素正好由 256 个线程各读 4 个。每个 block 有 64 份线程小计，经 6 轮两两合并得到 1 份块小计。4 个块最终写出 4 份小计，再由 CPU 求总和。换成 1031 个元素试试，观察哪几个线程多读了一次。

迁移到 CUDA 时，写完 shared memory 后需要块内屏障，归约轮次之间也需要屏障。让同一 block 的所有线程走到同一处 `__syncthreads()`。尾部超界的线程可以贡献零，并继续参与块内同步。跨 block 汇总可以先交给 CPU，后续再练第二次 kernel 启动。

## 本机环境和后续路线

这台机器用 `nvidia-smi` 识别到了 RTX 5070 Laptop GPU，驱动版本 582.05，计算能力 12.0。输出里的 CUDA 13.0 表示驱动报告的兼容版本。当前 PowerShell 和 CUDA 默认目录均未找到 `nvcc`，当前终端也未找到 `cl`。WSL 列表检查返回访问被拒绝，发行版状态未能确认。

本机的 GPU 示例用 Python 调用 Windows 上的 NVIDIA 驱动，由驱动加载并即时编译 `map.ptx`。它针对本机的 `sm_120` 和支持 PTX 9.0 的驱动。做 CPU 关卡时可以跳过这两个 GPU 示例文件。想看 GPU 上的边界判断，可以在 `map.ptx` 里找 `setp.ge` 和紧随其后的跳转。

以后想自己编译 CUDA C++ 时，可在原生 Windows 上使用兼容的 CUDA Toolkit 和 MSVC，也可以在已有 CUDA 开发工具的 Linux GPU 环境里做同一套题。优先把 Map 移植成 `.cu`，再将每关 CPU 测试里的输入和预期值用来核对 GPU 输出。如果选择 WSL 2，先确认发行版可用，再按 NVIDIA 的 WSL 指南核对驱动和开发工具要求。链接见下方。

- [NVIDIA Windows CUDA 安装指南](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/)
- [NVIDIA CUDA on WSL 指南](https://docs.nvidia.com/cuda/wsl-user-guide/)

可以对 Codex 说：“陪我做第 3 关。先让我猜 2 个 block、每块 3 个线程时，线程 1 读哪些下标。等我改好代码，再运行测试并解释第一处差异。”

## 浏览器版 GPU Playground

在包含本 README 的目录启动 Python 服务，服务仅绑定到本机回环地址：

```powershell
python -B playground.py
```

打开 <http://127.0.0.1:8765/>。无需安装前端依赖或 CUDA Toolkit。端口被占用时可加 `--port 8766`。用 Ctrl+C 停止服务。

| 实验 | 可以尝试 |
| --- | --- |
| 归约 | 默认 13 个元素、2 × 4 线程。观察 Block 0 的 `[10, 12, 14, 16] → [24, 28] → 52`，再看第二阶段把 `[52, 39]` 合为 91。要复现相邻配对例子，用 1 个 Block、4 个线程、输入 `2, 4, 6, 8`，选择「相邻子树」。 |
| 线程拓扑 | 修改 Block 与线程数，点 Block 查看成员，点线程查看全局编号和访问下标。 |
| 网格步长 | 默认 16 个元素和 4 个线程。比较 `i += stride` 和 `i += 1`，查看高亮连线和最终访问次数。 |
| 共享内存与同步 | 观察每个 Block 单独统计抵达数。Block 0 可以在 Block 1 等待时读取右邻居。 |
| 数据竞争 | 用两个线程分别加 1 和 2。「先全部读取，再写回」的非原子结果是 2，原子结果是 3。切换「逐线程完成更新」看另一种合法交错。 |

可使用前后单步、拖动进度条、自动播放和调速，点「重置」恢复当前实验的默认参数。进阶模式显示线程编号公式或屏障作用域。所有教学轨迹由 Python 在 CPU 上生成。竞争调度用于演示可能的交错，浏览器播放顺序不代表实测 GPU 调度。

右上角「运行真实 GPU Map」调用已有的 `gpu_map.py`，只有程序成功执行并通过自带校验时才标记为 GPU 结果。驱动调用失败时会显示失败原因和 CPU 参考值。

`playground.py` 提供同源 HTTP 界面与 JSON API。`POST /api/experiment` 接收实验模式与参数，返回索引分配和逐步快照。`POST /api/map` 执行原有 GPU 示例。网页归约与终端 `trace.py` 共用索引和归约逻辑；`puzzles.py` 仍用于手写练习。
