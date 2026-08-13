# LBM Research Workspace

这个文件夹已经整理成一个格子 Boltzmann 方程仿真工作台。当前版本提供：

- `src/lbm_lab`: 可复用 Python 包，包含 D2Q9/BGK 基础算子、示例仿真、数据库记录和命令行入口。
- `configs`: 每个实验一份 TOML 配置，便于复现实验和批量扫描参数。
- `database`: SQLite 运行数据库，记录配置、状态、耗时、输出文件和关键指标。
- `results`: 每次仿真的结果目录，默认保存 `fields_final.npz`、`metrics.csv`、`run_summary.json`。
- `data`: 原始数据和处理后数据。
- `logs`: 长任务日志。
- `notebooks`: 后处理和画图 notebook。
- `docs`: 研究记录、流程说明和模型说明。

## 快速开始

建议先创建虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e .[analysis,dev]
```

如果 PowerShell 提示找不到 `python`，可以安装 Python/Miniconda，或临时设置：

```powershell
$env:LBM_PYTHON="C:\Path\To\python.exe"
```

初始化数据库：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\init_database.ps1
```

运行一个小规模 lid-driven cavity 示例：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_example.ps1
```

查看最近运行记录：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\show_runs.ps1
```

已安装本项目后，也可以直接使用模块入口：

```powershell
python -m lbm_lab.runner --config configs\lid_driven_cavity.toml
```

## 新版可配置 LBM 程序

根据实际需求，项目根目录新增了完整第一版程序：

- `main.py`: 命令行运行入口。
- `lbm_solver.py`: D2Q9-BGK 核心求解器，数组形状为 `(NY, NX, 9)`。
- `boundary_conditions.py`: 边界条件注册系统，第一版实现 `periodic`、`no_slip_bounce_back`、`moving_wall_bounce_back`、`non_equilibrium_extrapolation`、`full_developed_outlet`。
- `postprocess.py`: 涡量、速度、流线、矢量图、残差图和结果保存。
- `desktop_app.py`: PySide6 桌面端 UI，不使用浏览器端口。
- `ui_app.py`: Streamlit 前端 UI，保留为可选旧入口。
- `config.py`: 参数、稳定性诊断和自动推荐器。

命令行快速验证：

```powershell
$env:PYTHONPATH="."
python main.py --case lid_driven_cavity --nx 64 --ny 64 --re 100 --u-ref 0.05 --max-iter 500 --min-iter 100
```

启动桌面端 UI：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_desktop.ps1
```

这个入口会直接打开桌面窗口，不使用 `localhost:8501` 或任何网页端口。

更方便的方式是直接双击根目录下的：

```text
start_lbm_desktop.bat
```

如果双击时提示缺少 `matplotlib`、`PySide6` 等依赖，先双击：

```text
install_lbm_dependencies.bat
```

该脚本会在项目内创建 `.venv` 并把依赖安装进去。之后 `start_lbm_desktop.bat` 会优先使用这个本地虚拟环境，避免系统 Python 版本或 Codex runtime 变化导致打不开。

也可以创建桌面快捷方式：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\create_desktop_shortcut.ps1
```

可选旧版网页 UI：

```powershell
$env:PYTHONPATH="."
streamlit run ui_app.py
```

或者使用项目脚本启动：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_ui.ps1
```

该命令会保持运行；打开 `http://localhost:8501` 使用界面，停止时在终端按 `Ctrl+C`。

如果系统没有 `python` 或 `streamlit`，先安装依赖：

```powershell
python -m pip install -r requirements.txt
```

## 推荐工作流

1. 在 `configs/` 复制一份实验配置，例如 `configs/re100_n128.toml`。
2. 修改网格、雷诺数、步数、输出间隔等参数。
3. 运行 `python -m lbm_lab.runner --config configs\你的配置.toml`。
4. 在 `database/lbm_runs.sqlite` 中查看运行记录。
5. 在 `results/<run_id>/` 中做后处理和出图。

更详细的目录和数据库说明见 `docs/workflow.md`。

## GitHub 发布策略

本仓库适合公开源码、文档、配置模板、测试和启动脚本；本地仿真输出、数据库、日志、缓存、原始数据和个人环境状态应保留在本地。详细规则见 `docs/publication_policy.md`。

## Project Docs

- `docs/about.md`: project scope, goals, and limitations.
- `docs/workflow.md`: simulation and publication workflow.
- `docs/package.md`: installation, packaging, and build notes.
- `docs/release.md`: release checklist and tag process.
- `docs/contribute.md`: contribution guide.
- `docs/publication_policy.md`: public vs local-only file policy.

CI is defined in `.github/workflows/ci.yml`.
