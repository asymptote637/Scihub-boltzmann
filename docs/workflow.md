# Workflow

This document describes the recommended research workflow for configuring, running, saving, and reviewing LBM simulations.

## 目录约定

| 路径 | 用途 |
| --- | --- |
| `src/lbm_lab/lbm` | LBM 低层算子，例如离散速度、权重、平衡态、碰撞、流动 |
| `src/lbm_lab/simulations` | 具体物理问题，例如 lid-driven cavity、Poiseuille、Rayleigh-Benard |
| `configs` | 实验参数，每次运行都应该能从配置复现 |
| `database` | SQLite 数据库，记录运行元数据 |
| `results` | 自动生成的结果文件 |
| `data/raw` | 外部或原始数据，不建议在代码里直接修改 |
| `data/processed` | 后处理后的可复用数据 |
| `logs` | 长时间任务日志 |
| `notebooks` | 探索性分析和画图 |

## Daily Simulation Workflow

1. Start from a known case preset: `lid_driven_cavity`, `poiseuille_channel`, `couette_flow`, `periodic_channel`, `cylinder_flow`, or `custom`.
2. Choose grid size, Reynolds number, reference velocity, and boundary conditions.
3. Check derived values before running:

   - `Ma`
   - `nu_lattice`
   - `tau`
   - `omega`
   - stability level

4. Run a short smoke simulation before a long one.
5. Review residual, convergence factor `q`, rolling `q_avg`, mass drift, and maximum velocity.
6. Save long-run configs and summaries, but keep generated result fields local unless they are intentionally curated for publication.

## Desktop UI Workflow

Start the desktop UI:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_desktop.ps1
```

Recommended UI workflow:

1. Select the case type.
2. Set grid and flow parameters.
3. Use `Recommend U_ref` when exploring new `Re` and grid combinations.
4. Confirm the diagnostics area has no blocking errors.
5. Configure boundaries and obstacles.
6. Start the simulation.
7. Watch residual, mass drift, and preview figures.
8. Inspect saved output under `results/desktop_runs`.

## Command-Line Workflow

Run a small case:

```powershell
$env:PYTHONPATH="."
python main.py --case lid_driven_cavity --nx 64 --ny 64 --re 100 --u-ref 0.05 --max-iter 500 --min-iter 100
```

Run the older package-style example:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_example.ps1
```

## 数据库表

- `simulations`: 每次仿真的主记录，包含配置、状态、开始/结束时间、输出目录。
- `metrics`: 关键时间序列指标，例如最大速度、残差、质量误差。
- `artifacts`: 输出文件索引，例如 `fields_final.npz`、`metrics.csv`。

## 后续扩展建议

- 新增模型时，把通用算子放到 `src/lbm_lab/lbm`，把问题设置放到 `src/lbm_lab/simulations`。
- 新增实验时，优先复制配置文件，而不是复制 Python 脚本。
- 批量参数扫描可以新增 `scripts/sweep.py`，循环生成配置并调用 `lbm_lab.runner`。
- 大型结果不要提交到 Git；只提交配置、代码、图表脚本和小型摘要。

## Publication Workflow

When preparing material for GitHub or a paper:

1. Commit code, configs, tests, and documentation.
2. Keep raw data, generated fields, local databases, and logs out of Git.
3. If figures are needed in the repository, curate a small set under `docs/figures/`.
4. Store large reproducibility datasets in an external archive and cite the DOI or link.
5. Document exact configs used for published runs.

## GitHub Actions Workflow

The repository includes `.github/workflows/ci.yml`.

The CI workflow runs on:

- pushes to `main`
- pull requests targeting `main`

It installs dependencies from `requirements.txt` and runs:

```powershell
python -m pytest -q
```

Keep the workflow lightweight. Long CFD runs should not be part of default CI; use small smoke tests only.
