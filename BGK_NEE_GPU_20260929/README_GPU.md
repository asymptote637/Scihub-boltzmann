# LBM 二维 / 三维与热流 GPU 工作台

2026-09-30 第二轮扩展：开放入口/出口、障碍尾流、周期体力、振荡壁面、Rayleigh–Bénard 与混合对流；新增阵列、探针/FFT、快照回放、双参数矩阵和配置 JSON。详见 [ADVANCED_EXPERIMENTS.md](ADVANCED_EXPERIMENTS.md)。

2026-09-30：新增 D2Q25、D3Q19、D3Q27、TRT、通道与周期障碍物、独立温度分布函数和自然对流，并扩展原生控制台的预设、参数扫描、三维截面、诊断与检查点续算。入口不变。最新说明和验证边界见 [RESEARCH_EXPANSION.md](RESEARCH_EXPANSION.md)。下面的历史基准和命令行说明仍针对原方腔实现；`run_gpu.py` 不作为新扩展参数入口。

2026-09-29 模型扩展：现在支持 BGK/MRT 与非平衡外推/半格点反弹独立组合，CPU、CuPy 和 CUDA 均可运行。默认保持 BGK＋NEE。操作、物理定义、验证命令见 [MODEL_SWITCHING.md](MODEL_SWITCHING.md)。下列历史测速和零差异结果只针对原 BGK＋NEE，不能套用于新组合。

## 新版统一工作台

现已增加原生 Qt **CPU / GPU 仿真工作台**，将参数输入、后端选择、串行队列、数据归档和对比分析集成在同一窗口。双击 `open_dashboard_windows.bat`，或无参数双击原 `run_gpu_windows.bat`，均可打开界面。完整说明见 [README_WORKBENCH.md](README_WORKBENCH.md)。新工作台数据位于 `workspace/`，原命令行结果仍位于 `results/`，两者可统一登记和分析。

本目录是独立实验分支，不覆盖原 CPU 目录，不接管原搜索队列或仪表盘。开发顺序为：冻结源文件核验、逐数组移植对照、融合内核优化、再次完整对照、同条件测速、命令行和结果文件检查。

## 当前结论

- 两种 GPU 后端均已通过 CPU 对照，计算全程使用 `float64`。
- `64 x 64, Re=100`：CPU、逐数组 GPU、融合 GPU 均在第 **16148** 步收敛。
- 所有已检查时刻的 `f / rho / ux / uy` 最大绝对差均为 **0**。GPU 并行归约得到的残差存在末位差异，融合版所有对照点的残差最大绝对差为 `2.08166817117217e-17`。
- `256 x 256, Re=5468` 三轮同条件测速：融合版中位数 **0.4631 ms/步**，相对 CPU **73.73 倍**，相对逐数组 GPU **33.70 倍**。
- 尚未在 GPU 上完成高 Re 的百万步搜索、临界 Re 搜索或网格无关性研究。移植对照通过不等于这些物理结论已经验证。

详细数据见 [VALIDATION_REPORT.md](VALIDATION_REPORT.md)，机器可读证据保存在 `validation/`。

## 文件与环境

| 文件 | 用途 |
|---|---|
| `reference_cpu/` | 按原清单复制并核验的 CPU 参照、历史低 Re 结果 |
| `reference.py` | 核验 18 个冻结源文件并加载原 CPU 求解器 |
| `gpu_solver.py` | 可读性优先的 CuPy 逐数组版，以及共用停止、保护和输出逻辑 |
| `gpu_fused.py`、`kernels.cu` | 融合版：pull/BGK、NEE 壁面、分块诊断归约 |
| `run_gpu.py` | 单算例命令行入口，默认使用融合版 |
| `validate_gpu.py` | 轨迹、壁面、角点、异常保护、停止规则和全程收敛对照 |
| `benchmark_gpu.py` | CPU、逐数组 GPU、融合 GPU 的重复测速 |
| `verify_delivery.py` | 命令行、导出文件、验证门禁测试及原 CPU 只读审计 |
| `results/` | GPU 独立结果，按后端、网格、Re 和时间分目录 |

已验证环境：Windows 11、RTX 4060 Laptop GPU 8 GB、驱动 616.56、Python 3.12.14、NumPy 1.26.4、CuPy 13.6.0、CUDA Runtime / NVRTC 12.9。依赖版本固定在 `requirements.txt`。

Python 环境位置：

```text
D:\LBM\envs\bgk-nee-gpu-20260929\Scripts\python.exe
```

将环境放在纯 ASCII 路径，是为了避开本机 NVRTC 在中文环境路径下找不到 CuPy 头文件的问题；项目源文件和结果仍保留在当前科研目录。CUDA DLL 和编译缓存由 `gpu_environment.py` 在当前进程内配置，没有修改系统环境变量、驱动或原 CPU 虚拟环境。没有安装完整 CUDA Toolkit，也不需要本地 `nvcc`。本项目不调用 cuRAND / cuSOLVER，环境报告中这两项可选库缺失不影响已验证流程。

安装时留下的 `.venv_unicode_failed/` 是未使用的失败临时环境，所有入口均不引用它。2026-09-29 已移动到 `../99_旧版本与历史文件归档/2026-09-29/02_废弃安装环境/.venv_unicode_failed/`；当前GPU环境仍使用上文的 `D:\LBM\envs\bgk-nee-gpu-20260929`。

## 运行

双击 `check_gpu_windows.bat` 检查环境；无参数双击 `run_gpu_windows.bat` 现在打开统一工作台，不会直接开始计算。向该脚本传递参数时仍走原 GPU 命令行入口。直接执行 `run_gpu.py` 的默认参数保持 `N=256, Re=5468, U=0.04, max_iter=1000000`，每次从静止初态独立开始。

更建议先用 PowerShell 明确参数，执行短算例：

```powershell
Set-Location -LiteralPath 'D:\BaiduSyncdisk\我的文件\Leon\科研\BGK_NEE_GPU_20260929'
$python = 'D:\LBM\envs\bgk-nee-gpu-20260929\Scripts\python.exe'
& $python -X utf8 run_gpu.py --dry-run
& $python -X utf8 -u run_gpu.py --backend fused --grid 256 --re 5468 --max-iter 20000
```

单算例的百万步预算命令如下；本次交付没有自动启动此长任务：

```powershell
& $python -X utf8 -u run_gpu.py --backend fused --grid 256 --re 5468 --max-iter 1000000
```

切换到逐数组参照版使用 `--backend array`。其他参数可用 `--help` 查看。`--dry-run` 只检查参数、验证记录和 GPU 初始化，不推进时间步、不创建算例结果目录。

每个算例包含 `run_request.json`、`config.json`、`progress.jsonl`、`run_summary.json`、`results.npz`、历史和中心线 CSV、`timing_and_population.json`。`latest_gpu_run.json` 指向最近一次结果。GPU 主程序不生成图像，以免影响计算流程；现有 CPU 输出格式和诊断算法复用冻结代码。

`run_summary.json` 区分 `converged`、`max_iter`、`diverged` 和 `cancelled`。`Ctrl+C` 会尝试导出当前结果并标为取消；强制结束进程不保证写出完整结果。NPZ 是宏观场输出，不是支持续算的完整分布函数检查点。

## 保持不变的科学设置

- 默认 D2Q9 / BGK＋节点式非平衡外推 NEE；可选择 MRT 和 halfway bounce-back，具体尺度与角点定义见模型说明。
- 壁面重构、对角内部点角点供体、上下壁角点优先级保持原规则；壁面宏观量从重构后的分布重新求得。
- 采用原始双精度平衡态和碰撞表达式。融合内核指定 `--fmad=false`，不启用 fast-math，也不改用 FP32。
- 默认顶盖速度 `0.04`、线性启动 `500` 步、最少 `2000` 步、内部流体节点的**相邻时间步**速度残差阈值 `1e-6`。每步检查，不以输出间隔代替残差间隔。
- 质量漂移只报告、不补偿；默认不是收敛门槛。NaN/Inf、非正密度、实际最大 Ma 超过 `0.1` 均停止。
- 优化版每步仍执行异常和收敛检查、回传诊断量。前一步已完成的状态检查可复用于下一步输入，没有降低检查频率。不要绕过 `load_state()` 直接修改设备数组。

当前支持无体力、无障碍物的 BGK/MRT 顶盖驱动方腔及矩形测试域，四边统一为 NEE 或 HBB。HBB 限定侧壁/底壁静止、顶盖切向运动。外力、障碍物、混合边界或其他案例类型会明确拒绝。

## 重现与更改代码

```powershell
& $python -X utf8 check_gpu.py
& $python -X utf8 -u validate_gpu.py --backend array --full
& $python -X utf8 -u validate_gpu.py --backend fused --full
& $python -X utf8 -u validate_models.py
& $python -X utf8 -u benchmark_gpu.py --grid 256 --warmup 100 --steps 500 --repeats 3
& $python -X utf8 -u verify_delivery.py
& $python -m pip check
```

运行入口核验完整正确性记录和 GPU 核心文件 SHA-256；改动核心 Python 或 CUDA 文件后，旧验证记录不能继续放行，必须重新执行对应的 `--full` 对照。仅运行不带 `--full` 的测试会替换验证记录，但不会获得生产运行许可。

更换 GPU、驱动或 Python/CuPy/NumPy 版本后也应重新运行完整对照；现有源码哈希门禁并不代替环境变化后的数值验证。

`verify_delivery.py` 是本次本机部署验收脚本，包含原任务路径和进程身份检查；原任务以后结束或更换机器时，该审计部分可能不再适用，不应据此判断 GPU 数值实现失效。通常的移植正确性复核使用 `validate_gpu.py`。

本次完成时，原 CPU 求解器和调度器保持仪表盘此前设置的暂停状态，未被停止、续跑、重启或替换；18 个原冻结文件哈希一致。GPU 结果不会写入原 CPU 搜索结果或排名。
