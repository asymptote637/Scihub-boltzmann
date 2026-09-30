# 独立解压验收记录

日期：2026-09-28T19:33:08.378372+08:00

在包含空格与中文的独立临时目录解压 ZIP 后，使用本机 Python 3.9.6 / macOS / NumPy 1.26.4 / Matplotlib 3.6.3 / PySide6 6.10.3 / psutil 6.0.0 验收。psutil 仅安装在临时验证目录，未改全局环境。

- `transfer.py check`：包清单及参数预检通过；search 自检通过。
- `transfer.py boundary-check`：NEE 矩、角点、壁长/启动、Couette、半格反弹逐位回归、质量判据和独立 pull 演化等检查通过。
- `transfer.py smoke`：N16 / Re100 / 300 步运行正常；停止原因为 max_iter，明确不是收敛结果。已独立打开 NPZ，确认 rho/ux/uy 全有限且 rho>0，顶壁速度为 0.04×300/500。详细指标见 validation/local_numerical_checks.json。
- `QT_QPA_PLATFORM=offscreen python -m unittest discover -s tests -v`：7 项隔离真实进程和 Qt 集成测试通过。临时进程专用；没有向原科研进程发送控制信号。
- 数值核心与入口保留 simulation/plan.json 中的原始 18 项冻结哈希，未修改物理参数、残差、边界和求解核心。

日志保留在 validation/。这些测试仅验证本机/POSIX 分支，不构成 Windows 原生验收。Windows msvcrt 锁、bat、psutil suspend/resume/terminate、PySide6 可见窗口需要目标 Codex 重测。Windows requirements 使用 Python 3.11 兼容版本，和本机验证的 matplotlib/Qt 版本不同，必须记录目标机实际版本。

没有在打包机运行新的 N256/Re5468 百万步计算，也没有证明该高 Re 工况收敛。历史 N64/Re100 证据与原机未完成高 Re 进度存放在 evidence，均不导入新搜索状态。
