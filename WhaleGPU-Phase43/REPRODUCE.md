# Phase 4.3 复现

在包含 WhaleGPU-Phase1、34、35、36、41、42 和 43 的工程根目录使用 Windows PowerShell。复现实验需要 CUDA Toolkit 13.0 和 MSVC 14.39。构建脚本仅在子进程中调用 `vcvars64.bat` 加载 MSVC 编译环境。开始前确认仓库相对目录 `WhaleGPU-Phase35/tools/` 中具备反汇编工具 `cuobjdump.exe` 和 `nvdisasm.exe`。

完整历史复现还需从经校验的完整备份恢复历史审计材料（包括 `WhaleGPU-Phase43/PHASE43_AUDIT.md`）、各阶段实验快照、报告生成脚本 `WhaleGPU-Phase43/finalize.py` 及上述工具。准备好完整目录树后，先运行 `python -B WhaleGPU-Phase43/history.py` 校验审计锚点，再执行下列步骤。

逐条执行以下命令并检查退出码。每次分配新目录，完整正确性验证失败即终止该实验。GPU 性能任务应串行运行。输入生成种子包含在 case 字段。不同精度单独分析。

~~~powershell
python -B WhaleGPU-Phase43/run.py A
python -B WhaleGPU-Phase43/run.py B
python -B WhaleGPU-Phase43/run.py C
python -B WhaleGPU-Phase43/regression.py
python -B WhaleGPU-Phase43/run.py A-ring
python -B WhaleGPU-Phase43/run.py B-ring
python -B WhaleGPU-Phase43/run.py C-ring
~~~

在每个 run 的 snapshot/WhaleGPU-Phase43 目录，从普通命令提示符执行 ..\..\build.cmd 可重建相应二进制。精确复演旧记录时，先拷贝整个 run 至独立临时目录，再从该副本编译，避免更新被审计的二进制。已生成的 A/B/C_generated.cuh 就在对应快照内。commands.txt 给出原命令及种子；若直接调用 exe，须提前创建输出目录，设置 OMP_NUM_THREADS=8；ring 运行另设 WHALEGPU43_ROTATE=1，复用单缓冲区运行需清除此变量。

每个实验保存 validate/checks.csv、first 与 repeat 的原始 samples.csv、resources.csv 和 allocation.csv，以及 ptx.txt、sass.txt、build.log、source_sha256.json、binary.json。summary.csv 的标准差为样本标准差，P95 为线性插值。REPORT.md 按轮次配对，对比中位时间比。kernel_ms.svg、e2e_ms.svg 的横轴索引对应报告末尾配置表。

主流程包括 split、fused；诊断用空启动、原生产者、固定输入后处理或 copy。C 另比较 block_split 和输出阶段重读 X 的 reload。C 的 E2E 包含 Y 与两组 FP64 统计量回传。一般诊断项的独立 E2E 有不同输入输出，不适合作为主基线。allocation.csv 为单个固定大小测试分配的主机耗时；它是分配微基准。

ring 实验单独编译，在主形状的 split、fused 上复核缓存敏感性。按硬件 L2 容量决定缓冲区数量，两个版本使用大小相等的独立循环地址区，每区的名义工作集超过 L2，输入值完全相同。记录所有参与计时的槽位输出。此测试改变设备地址复用距离，实际缓存命中取决于硬件和调度。方法不提供 DRAM 字节数。

在工程根目录调用 python -B WhaleGPU-Phase43/finalize.py 后跟每个成功 run 的路径，可重建跨实验 REPORT.md。回归镜像单独保存在 regression 下，各阶段原生脚本的日志、状态与生成物均在镜像内。不要将镜像用作历史完整性基准。

核验统计与独立重建时，用本轮实际目录替换 RUN 参数。

~~~text
python -B WhaleGPU-Phase43/audit_results.py RUN_A RUN_A_RING RUN_B RUN_B_RING RUN_C RUN_C_RING
python -B WhaleGPU-Phase43/regression_report.py REGRESSION_MIRROR
python -B WhaleGPU-Phase43/rebuild.py RUN_A RUN_B RUN_C
python -B WhaleGPU-Phase43/finalize.py RUN_A RUN_A_RING RUN_B RUN_B_RING RUN_C RUN_C_RING
~~~

resources_verified.csv 按实际执行版本将 CUDA 资源、Grid、理论驻留和对应的 ptxas spill 符号合并。paired_verdicts.csv 保留两轮区间和负面结果。首次实施时的独立运行和构建失败日志也保留在 runs 下。汇总报告明确列出最终使用的运行编号。旧版资源文件的 empty43 理论 occupancy 按线程计，最终版按整 warp 计。
