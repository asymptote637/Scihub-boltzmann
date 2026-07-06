# LBM 仿真工作流

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

## 数据库表

- `simulations`: 每次仿真的主记录，包含配置、状态、开始/结束时间、输出目录。
- `metrics`: 关键时间序列指标，例如最大速度、残差、质量误差。
- `artifacts`: 输出文件索引，例如 `fields_final.npz`、`metrics.csv`。

## 后续扩展建议

- 新增模型时，把通用算子放到 `src/lbm_lab/lbm`，把问题设置放到 `src/lbm_lab/simulations`。
- 新增实验时，优先复制配置文件，而不是复制 Python 脚本。
- 批量参数扫描可以新增 `scripts/sweep.py`，循环生成配置并调用 `lbm_lab.runner`。
- 大型结果不要提交到 Git；只提交配置、代码、图表脚本和小型摘要。

