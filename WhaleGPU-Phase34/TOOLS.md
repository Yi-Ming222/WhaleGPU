# Phase 3.4 工具探测与 Windows 配置

2026-11-18 实机探测。设备 RTX 5070 Laptop GPU，驱动 582.05。CUDA_PATH 指向 `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0`，卸载登记显示 NVIDIA CUDA Toolkit 13.0。

| 工具 | 状态 | 本次实验影响 |
|---|---|---|
| Nsight Compute CLI | NOT_FOUND | Achieved Occupancy、访存事务和 Warp Stall 缺少硬件计数器数据 |
| Compute Sanitizer CLI | NOT_FOUND | memcheck、racecheck、synccheck 均未执行 |

探测包括 PowerShell 的 PATH 命令解析、64 位和 32 位 NVIDIA Corporation 安装目录、CUDA 安装目录的递归文件搜索，以及 HKLM（含 WOW6432Node）和 HKCU 卸载登记。自定义路径下未经登记的便携副本可能超出探测范围。本次只读检查，未执行安装或更改系统配置。

## Windows 配置方案

1. **Nsight Compute。** 选 NVIDIA 的 2025.3 Windows x86_64 独立安装包。该系列支持 CUDA 13.0 和 Blackwell，推荐 Windows 驱动 581.15 或以上，本机 582.05 满足此条件。安装工具后用实际安装路径运行 `ncu.exe --version`。[NVIDIA 下载与兼容性说明](https://developer.nvidia.com/tools-overview/nsight-compute/get-started-2025_3)
2. **Compute Sanitizer。** 使用与本机匹配的 CUDA 13.0.0 Windows 本地安装介质，自定义安装时仅勾选 Compute Sanitizer，并核对安装摘要。官方组件名为 `sanitizer_13.0`。也可由管理员在已核对的 13.0 安装程序后加参数 `-s sanitizer_13.0`。安装后在 CUDA 13.0 目录下查找 `compute-sanitizer.exe` 并执行 `--version`。[CUDA 13.0 Windows 子组件说明](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-installation-guide-microsoft-windows/index.html)
3. **命令发现。** 两个工具都可用绝对路径调用。若需要让调度脚本发现工具，仅将实际包含对应 exe 的目录临时加入当前 PowerShell 的 PATH，再执行 `Get-Command ncu,compute-sanitizer`。记录工具路径、版本、`nvcc --version` 和 `nvidia-smi` 输出。
4. **性能计数器。** 若独立采集返回 `ERR_NVGPUCTRPERM`，可由获授权的管理员在提升权限的终端运行这次采集。保存原始 ncu-rep 和错误日志。[NVIDIA 性能计数器权限说明](https://developer.nvidia.com/nvidia-development-tools-solutions-err-nvgpuctrperm-nsightcompute)
5. **验收。** 用同一组非整除 M/N/K 输入在独立进程对基线与 Register Tile 运行 ncu，记录 LaunchStats、Occupancy 和 MemoryWorkloadAnalysis。对验证入口分别运行 `compute-sanitizer --tool memcheck --error-exitcode 9`、`--tool racecheck` 和 `--tool synccheck`，在每条命令后追加可执行文件及其验证参数。保存退出码和诊断日志。正式 CUDA Events 数据使用常规实验进程采集。[Compute Sanitizer 手册](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)
