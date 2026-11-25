# WhaleGPU Paired Runner

本手册中的 PowerShell 命令均从**仓库根目录**执行，使用 Python 3.10+。A/B/C 表示三个工程阶段；测量记录中的 A、B 分别表示两个 kernel：`T32_R4_row` 和 `T64_R4_row`。

| 目的 | 入口 | GPU 与磁盘行为 |
|---|---|---|
| 查看历史证据 | [冻结附件与只读查看](#冻结附件与只读查看) | 只读既有 ZIP 或报告，无需 CUDA |
| Phase A：演练计划 | `run_paired.py plan / dry-run / inspect` | plan 和 dry-run 写入 CPU 演练归档；inspect 只读 |
| Phase B：验证真实链路 | `run_cuda.py build`，再 `verify` | build 编译新 EXE；verify 执行一个真实 GPU Unit，写入设备回执 |
| Phase C：比较多个 Unit | `phase_c.py` | 一条命令创建新计划、编译、执行、采集遥测并生成报告 |

## Phase A：CPU 演练与只读检查

此阶段的工作进程是同一 Python 解释器运行的内置回执器，无需 NVCC、GPU 或历史 EXE。`PROCESS_ID` 是占位符，使用本阶段 plan 输出的实际值。

1. `python -B paired-runner/run_paired.py plan --units 4 --seed 431` 封存一份计划并输出 process_id。
2. `python -B paired-runner/run_paired.py dry-run PROCESS_ID` 为该计划独占一次尝试，启动工作进程并保存实际回执。
3. `python -B paired-runner/run_paired.py inspect PROCESS_ID` 只读核对计划和归档，输出各 unit、block 的完整性、缺失 batch、异常及实际记录。

每个 unit 含一个 AB 和一个 BA 区块，每个区块有两条 batch 记录。偶数个 unit 的两种起始方向等量，奇数时相差一。固定 units、seed 和相同 Python 随机算法环境可复现顺序。新进程有新身份；重放身份、参数相同时可复算完整计划。可通过 `plan --session-id UUID --experiment-id UUID` 沿用父身份，UUID 使用 32 位小写十六进制。各级子 ID 含进程前缀。process_id 是应用级 UUID，os_pid 是实际子进程 PID。

固定输出根为脚本同目录的 `artifacts/PROCESS_ID/`，与调用者工作目录无关。独占创建目录及文件，拒绝已有尝试和归档路径中的链接或重解析点。此本地目录应仅由当前用户管理。

| 文件 | 含义 |
|---|---|
| manifest.json | 初始只写一次。含 schema_version、身份、UTC、完整计划与 SHA256、配置、种子和 planned 状态 |
| attempt/start.json | 控制进程身份及所认领 Manifest 的规范 JSON SHA256 |
| attempt/batches.jsonl | 按提交顺序逐条写入并同步；每条含实际工作进程 PID、时间、顺序、kernel、unit/block/batch ID |
| attempt/terminal.json | 正常完成、失败或可捕获中断的终态；含有效回执数及其 JSONL 字节 SHA256。异常另含类型、消息和下一个预期 batch |

SHA256 对象摘要使用 UTF-8 的键排序紧凑 JSON，无尾换行。input_hash 的范围是输入配方，另有 input_hash_scope；矩阵字节哈希和 CUDA EXE 哈希为空。每条实际回执的 actual_launch_count 为 0；measurement_config 中的 10 是未来测量的计划值。预热关闭，不产生延迟、配对效应或有效性能统计样本。

inspect 从不可变计划、有效回执前缀和终态合成 execution_status。有尝试但无可信终态时报告 incomplete；这也可能表示仍在执行，单凭磁盘状态不能判断进程存活。半条记录或身份/计划冲突会留下诊断。失败与中断保留既有回执；后续使用新 process_id 从头开始，不能跨进程拼对。

## 真实 CUDA 执行所需环境

Phase B/C 的现有执行器选择设备 0，并检查设备名为 `NVIDIA GeForce RTX 5070 Laptop GPU`。构建器面向 Windows x64，使用 CUDA Toolkit 13.0、兼容驱动、MSVC Build Tools 2022 的 14.39 工具集及 Windows SDK（链接 `bcrypt.lib`）。本机需提供以下外部依赖：

| 依赖 | 当前实现使用的位置或参数 |
|---|---|
| CUDA Toolkit | `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0` 下的 `bin\nvcc.exe`、头文件和库 |
| MSVC 环境 | `C:\BuildTools2022\VC\Auxiliary\Build\vcvars64.bat`，使用 `-vcvars_ver=14.39` |
| 编译口径 | `-arch=sm_120 -std=c++17 -O3 -lineinfo -Xcompiler /openmp -Xptxas=-v`；完整参数、include 路径和工作目录随每次构建记录 |
| GEMM 依赖 | 使用 [P34 寄存器 kernel](../WhaleGPU-Phase34/register.cuh)、[固定派发表](../WhaleGPU-Phase34/runs/20261118-092322-b6e99f/source/dispatch.inc) 和 [P3 CPU 参考](../WhaleGPU-Phase1/phase3/src/reference.hpp)，须保留相对目录 |
| Phase C 遥测 | 当前驱动提供的 `nvidia-smi` 须能从 PATH 调用才能取得 GPU 读数。缺失字段按未知处理 |

Phase B 历史成功构建记录的版本为 NVCC 13.0.48、MSVC x64 19.39.33523。新机器的实际命令和结果以本机生成的构建记录为准。[构建器源码](run_cuda.py)

## Phase B：独立构建与单 Unit 验证

```powershell
python -B paired-runner/run_cuda.py build --seed 431
```

构建成功后，将下面的 `PROCESS_ID` 替换为**这次 build 输出的 ID**。A 阶段演练 ID 和已完成的历史运行 ID 不适用于这次验证。

```powershell
python -B paired-runner/run_cuda.py verify PROCESS_ID
```

输入为 `513×769×257` 的 row-major FP32 GEMM。矩阵生成种子为 `20261117`，命令中的 `--seed` 控制配对顺序。执行器核对实际输入字节 SHA256 与上传读回结果，两配置共用驻留输入。计时前后分别校验两侧：每次先重置输出 C，再与 CPU 参考逐元素比较。

一个 Unit 有两个区块（AB 和 BA），每个区块有两个批次。每批计划连续提交 10 次同配置 GEMM。start/stop CUDA Event 与 kernel 使用同一默认 stream，结束事件同步后才记录确认完成数。请分别读取 planned、submitted 和 completed；异步错误或进程中断可使完成数未知。每批 Event 总 ms 乘 1000 再除以确认完成数，得到 µs/GEMM，可能包含设备队列等待。下载、CPU 校验与首次正确性调用位于计时区间外。[执行器源码](cuda_executor.cu)

## Phase C：多 Unit 配对测量

```powershell
python -B paired-runner/phase_c.py --units 8 --seed 431
```

该入口接收 2–4096 个 Unit；运行时会重新构建并启动真实 GPU 测量。输入、每批 launch 数与 Phase B 相同。每个 Unit 内各执行一个 AB 和 BA 区块，随机化哪一方向在前，同时平衡两种首发方向的数量。Unit 墙钟时间由主机单调时钟测得，包含同步与回执输出。[Phase C 入口及报告逻辑](phase_c.py)

### 如何读 B/C 归档

输出根为 `paired-runner/cuda-artifacts/PROCESS_ID/`。以下为运行后生成的相对路径：

| 产物 | 用途 |
|---|---|
| `manifest.json` | 封存输入规格、矩阵哈希、完整顺序和计划摘要 |
| `build/command.json`、`build/compiler.log`、`build.json` | 核对编译参数、工具输出、头文件身份及 EXE SHA256；构建失败可查看 `build-error.json` |
| `attempt/events.jsonl`、`attempt/stderr.log` | 检查实际顺序、设备、正确性、提交/完成数、同步结果和错误信息 |
| `attempt/terminal.json` | 检查退出码、执行状态和成功回执哈希；无可信终态时保留为不完整运行 |
| `telemetry.jsonl`（C） | 查询窗口、温度、Graphics Clock、功耗及采集错误。一次读取可能覆盖多个 Unit，不能据此还原单批瞬时状态 |
| `pairs.json`、`REPORT.md`（C） | 各配对有效性、时间差与比值、异常标记、Unit 墙钟与汇总。有效样本中的统计异常按 1.5×IQR 标记并保留；失败或不完整的运行不进入有效比较 |

## 冻结附件与只读查看

两个 ZIP 作为独立交付附件提供。完整本地工作区中位于 `paired-runner/`；只有源码检出的接收者，请通过交付负责人确认的渠道取得同名附件。将附件放在下列命令所示位置，或将命令中的路径替换为实际位置。先比对整个 ZIP 的 SHA256，包内 `SHA256.json` 还可用于逐成员核验。

| 文件名 | 期望 SHA256 |
|---|---|
| `phase-b-frozen.zip` | `2703b474fcde91aa92fc99926d79d2286f593eabd0e71d6d0f841604d4756046` |
| `phase-c-baseline-db96eeca.zip` | `b0378a884ba57bc62c1e2f22ae304af18d5baac157b5e2a716b378d491bef280` |

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath '.\paired-runner\phase-b-frozen.zip'
Get-FileHash -Algorithm SHA256 -LiteralPath '.\paired-runner\phase-c-baseline-db96eeca.zip'
```

Phase C 冻结运行 ID 为 `db96eeca58bf4dd2bfbffcf6f74c740e`。在完整本地工作区，可直接打开当时生成的报告：

```powershell
Get-Content -Raw -Encoding UTF8 -LiteralPath '.\paired-runner\cuda-artifacts\db96eeca58bf4dd2bfbffcf6f74c740e\REPORT.md'
```

只有 ZIP 时，以下命令直接读取包内同一份报告，不解压、不构建、不启动 GPU：

```powershell
python -B -c "import zipfile; z=zipfile.ZipFile('paired-runner/phase-c-baseline-db96eeca.zip'); print(z.read('paired-runner/cuda-artifacts/db96eeca58bf4dd2bfbffcf6f74c740e/REPORT.md').decode('utf-8')); z.close()"
```

该历史基准有 8 个 Unit、16 个有效配对、32 个计时批次，5 个配对带统计异常标记且全部保留。报告描述单次进程的观测；跨会话可重复的性能优势尚未验证。

## 自动化测试

从仓库根目录运行三个阶段的 Python 测试：

```powershell
python -B -m unittest discover -s paired-runner -p "test*.py" -v
```

仅检查 Phase A 时可使用 `python -B -m unittest discover -s paired-runner -p test_paired.py -v`。测试使用临时输出；Phase A 的冲突和中断测试会打印保留的临时归档位置。B/C 测试含 CPU 侧故障注入。历史真实 GPU 执行应查阅冻结回执；需要本机的新回执时，再选择上面的 B 或 C 命令。
