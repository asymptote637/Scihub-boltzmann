# Windows 部署与验收记录

部署日期：2026-09-29（Asia/Shanghai）。正式队列已于 01:46:27 启动。

## 部署目录与来源

- 根目录：`D:\BaiduSyncdisk\我的文件\Leon\科研\BGK_NEE_Windows_20260928_191621`。
- 来源 ZIP：`D:\xwechat_files\wxid_lq8hb9i6lzfz21_507d\msg\file\2026-09\BGK_NEE_Windows_20260928_191621.zip`。
- ZIP SHA256：`9f3cc76289f4fbd66df097001e91f06c0751fda6a25d2962e01f333da1915d43`。
- 已阅读 `README_FIRST.md`、`CODEX_HANDOFF.md`、`VALIDATION.md`、`BUILD_PROVENANCE.json`。
- 使用包内独立 `.venv`；原科研工程、全局 Python 环境和系统字体未修改。
- 本次从静止初场重算，没有导入原电脑的 PID、STOP、搜索状态或未完成计算作为初场。

## 本机环境

| 项目 | 实测值 |
|---|---|
| 系统 | Windows 11 家庭版中文版，10.0.26200，64 位 |
| CPU | AMD Ryzen 9 7945HX，32 逻辑处理器 |
| 内存 | 总计 15.69 GiB，部署开始时空闲约 4.77 GiB |
| D 盘 | 部署开始时可用约 138.24 GiB |
| Python | 3.12.14，64 位 |
| NumPy / Matplotlib | 1.26.4 / 3.8.4 |
| psutil / PySide6 | 6.0.0 / 6.8.3 |

Python 3.11 安装命令被自动审批拒绝，工具仅返回 `blocked by policy`，没有进一步原因；安装命令未执行。依据交接文件允许记录并验证其他 Python 版本的约定，使用本机已有 Python 3.12.14 创建专用虚拟环境。依赖按原 requirements 固定版本安装，`pip check` 通过。解释器与版本记录见 `validation_windows/numerical_audit.json`。

专用解释器：`.venv\Scripts\python.exe`。其基础解释器位于 `C:\Users\hongt\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`。此虚拟环境依赖该基础解释器，不能直接当作跨电脑可搬移环境。

## 验收结果

| 清单项目 | 结果与证据 |
|---|---|
| 原包完整性 | 解压后 117/117 文件 SHA256 一致；`validation_windows/verify.log` |
| 冻结源码 | `simulation/plan.json` 所列 18 项全部一致；数值核心和参数未修改 |
| 参数与搜索自检 | 种子、扩展、下降搜索、二分、分类及正式参数 dry-run 通过 |
| NEE 边界验证 | 壁面和角点矩、顶盖启动、壁长、独立 pull 演化、质量策略、Couette 和半格反弹逐位回归通过 |
| 短跑 | N16/Re100，300 步，`max_iter`；仅为短跑验收，不是收敛结果 |
| 短跑输出审计 | NPZ 场有限、密度为正、摘要/进度/步数一致；顶壁速度 0.024，壁面与角点速度正确 |
| 低 Re 复核 | N64/Re100，第 16,148 步达到速度残差阈值；约 36.34 秒 |
| 历史结果对照 | 低 Re 的 rho、ux、uy 与包内已完成历史结果逐元素最大绝对差均为 0；诊断指标差值为 0 |
| 原包进程测试 | 6 项通过：暂停/恢复、运行中/暂停中终止、陈旧身份拒绝、STOP 保护、历史目录拒绝控制 |
| 原包 Qt 测试 | 1 项通过：按钮状态、暂停/恢复、取消终止、历史切换禁用控制 |
| Windows 补充测试 | 4 项通过：跨进程 msvcrt 锁、整个进程树退出、原生窗口重开识别暂停与确认终止、python/pythonw 身份一致 |
| 原生可视验收 | Qt 平台为 windows；中文及本地字体、真实曲线与流场、历史切换、1280×880 / 1040×760 截图已检查 |
| 本机批处理入口 | `check_deployment_windows.bat` 执行通过 |

低 Re 末态：残差 `9.998477937549365e-7`，质量相对漂移 `0.003104894017096238`，最大 Ma `0.06928203230275524`。质量仍为独立诊断，未加入停止判据，也未做质量修正。复现原结果不等于独立证明高 Re 收敛或网格独立性。

控制测试使用临时进程；未用正式计算执行暂停、终止或重启测试。Qt 验收截图中的历史曲线和流场来自实际短跑/低 Re 输出的隔离副本，未混入正式搜索结果。

## 两处本机修复

1. `dashboard/control.py`：原实现暂停 Windows venv 的转发进程，实际 Python 工作进程仍在计算。初始测试计数从 16 增到 22，复核从 17 增到 26，确认问题。修复后按解释器目录、命令、用户及父子关系识别实际工作进程，再使用 PID 和创建时间核验控制；兼容 python.exe 与 pythonw.exe。
2. `dashboard/dashboard.py`：小窗口挤压图表刻度。给图表增加最小高度和滚动区域，保留原来的数值及曲线来源。

原版保存在 `validation_windows/control_original.py`、`validation_windows/dashboard_original.py`。完整差异在 `validation_windows/WINDOWS_FIXES.patch`；原哈希、新哈希和修改原因在 `validation_windows/deployment_integrity.json`。

**原 `MANIFEST_SHA256.json` 未修改。** 目前 115 个原包文件保持不变，只有上述 2 项明确修复。原 `transfer.py verify/check` 或 `check_windows.bat` 会因这两项修复报告原包差异，这是预期行为。复核本机部署请运行新增的 `check_deployment_windows.bat`，它逐项校验原文件、修复后哈希、原版备份和全部 18 项冻结源码，再运行依赖与参数预检；任何未登记变化均失败。

## 正式计算

启动命令（包根目录）：

```powershell
.\.venv\Scripts\python.exe -X utf8 -u transfer.py search
```

已通过独立后台进程启动，启动命令所在的临时 PowerShell 退出后仍在运行。**当前不要重复启动。** 看板已经打开，关闭看板不会结束计算。

- 队列启动器 PID：46756。
- 实际搜索调度器 PID：48584。
- 首组实际计算 PID：28804；`search_state.json` 中 Popen 启动器 PID 为 44040，二者是 Windows 转发关系。
- 首组目录：`simulation/calculation/cavity_N256_Re5468_20260929_014631_137137`。
- 计算工作目录：`simulation/calculation`。
- 科学约定：D2Q9-BGK，256×256 节点，NEE，L=255，U=0.04，静止初场，顶盖 500 步线性启动，顶角随盖、下角静止，无体力、无质量修正。
- 当前 Re=5468，随后 Re=5312，再按原计划搜索；每组上限 100 万步，最多 16 组。
- 内部节点单步相对 L2 残差阈值 1e-6，最早第 2000 步判断，每 200 步落盘；Ma 保护 0.1。
- 最新核验快照：`validation_windows/production_status.json`；连续观察记录：`validation_windows/production_observations.jsonl`。

首个完整记录在第 200 步已确认；01:49:01 的独立核验显示第 3600 步、残差约 2.17314e-4、质量漂移约 4.28695e-5，最大 Ma 约 0.069282，仍在运行，调度器与真实工作进程均未暂停。该记录是启动验收快照，不是最终计算结果。

## 查看与操作

- 重开看板：双击 `open_dashboard_windows.bat`，或运行 `.\.venv\Scripts\python.exe -X utf8 transfer.py dashboard`。
- 暂停/继续：使用看板对应按钮；关闭再打开看板可识别原暂停状态。
- 终止：会同时停止当前计算和后续队列，并保留 STOP；未落盘的内存场无法恢复。不要自动清除 STOP。
- 当前进度：`simulation/search_state.json`、`simulation/搜索进度.md`。
- 当前计算日志：`simulation/logs/Re5468_00.log`。
- 每组结果：对应 `simulation/calculation/cavity_*` 目录；最终 NPZ 在该组结束时写入。
- 本机验收材料：`validation_windows/`，包括测试日志、哈希、修复补丁、数值审计和真实窗口截图。

该任务需要电脑持续运行。已核验当前接通交流电，当前电源方案的交流电自动睡眠设置为 0（不自动睡眠）；未更改系统电源设置。暂停不是磁盘检查点，关机或重启不能从暂停步数继续。

尚未验证：高 Re 队列最终收敛、全部搜索完成、网格独立性与物理临界 Re。当前完成的是部署、Windows 验收及正式计算启动。
