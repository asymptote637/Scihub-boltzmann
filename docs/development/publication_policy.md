# GitHub 公开与本地保留策略

本项目建议作为公开科研代码仓库发布，但只公开能复现实验环境和模型逻辑的内容，不公开本地运行产物、数据库、缓存和个人环境状态。

## 建议公开

- `README.md`
- `pyproject.toml`
- `requirements.txt`
- `.env.example`
- `.gitignore`
- `main.py`
- `lbm_solver.py`
- `boundary_conditions.py`
- `config.py`
- `postprocess.py`
- `desktop_app.py`
- `ui_app.py`
- `configs/`
- `scripts/`
- `src/`
- `tests/`
- `docs/`
- `data/raw/.gitkeep`
- `data/processed/.gitkeep`
- `database/.gitkeep`
- `logs/.gitkeep`
- `results/.gitkeep`
- `notebooks/.gitkeep`

这些文件包含模型实现、界面入口、参数模板、测试和复现实验所需说明，适合公开。

## 建议本地保留

- `results/**`: 仿真输出、图像、`npz` 场数据、残差历史。
- `database/**`: SQLite 运行数据库。
- `logs/**`: matplotlib/Streamlit/桌面端运行日志和缓存。
- `data/raw/**`: 原始数据，可能包含未整理或未授权材料。
- `data/processed/**`: 中间处理数据。
- `notebooks/.ipynb_checkpoints/**`
- `__pycache__/`、`.pytest_cache/`、`.ruff_cache/`
- `.agents/`、`.codex/`
- `.env`、密钥、token、证书文件。

这些内容要么体积会快速增长，要么带有机器路径、运行历史、个人环境或潜在敏感信息，不应进入公开仓库。

## 后续建议

- 需要展示结果时，优先提交小尺寸示例图片到 `docs/figures/`，不要直接提交完整 `results/`。
- 论文复现实验建议提交配置文件和脚本，而不是提交全部原始输出。
- 大型数据可使用 Zenodo、OSF、Figshare 或 Git LFS，并在 README 中给出 DOI 或下载链接。

