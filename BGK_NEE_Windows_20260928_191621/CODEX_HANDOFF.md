# 给目标 Windows Codex 的交接任务

请部署本目录中的 BGK–NEE 仿真与桌面看板。先读 README_FIRST.md、VALIDATION.md、BUILD_PROVENANCE.json，再做验收。用户已要求在另一台 Windows 电脑运行此仿真；本包必须从初始条件重算，不支持从原电脑的暂停步数续算。

## 建议顺序与完成标准

1. 确认 Windows 版本、CPU 架构、可用内存、磁盘与 `py -3.11`。不要使用原电脑的任何绝对路径/PID；不要复制原运行状态到 simulation。Python 未安装时先说明所需运行环境，使用官方安装渠道。
2. 在包内创建专用 `.venv`，安装 requirements-gui.txt。不要修改全局 Python、已有项目环境或系统字体。字体用应用本地加载；不部署 Web 端口。
3. 运行 `transfer.py verify` 和 `transfer.py check`。核对完整包清单及 simulation/plan.json 的源码哈希。解释任何差异，不自动更新哈希掩盖差异。
4. 运行 `transfer.py boundary-check` 和 `transfer.py smoke`，检查实际生成的 run_summary.json、results.npz、timing_and_population.json、progress.jsonl；核验有限场、正密度、步数、NEE/质量策略、壁面速度和参数。不将短跑称为收敛验证。
5. 单独执行 `tests/test_control.py` 中的临时进程测试。确认 Windows msvcrt 独占锁能防第二控制器；暂停后计数不再增长，恢复后同一 PID 接着增长；运行中与暂停中终止都不留子进程、不启动下组；历史记录与陈旧 PID 不得控制当前进程。
6. 启动实际 Qt 看板，验证中文/字体/图表/历史切换，检查暂停/继续按钮、取消终止、窗口缩放、关闭再打开后的暂停识别。用临时测试任务验收终止，不拿用户正式长算做终止测试。禁止用模拟曲线替代缺失数据。
7. 预检全部通过后，用 `transfer.py search` 开始正式队列，并另开 `transfer.py dashboard`。记录命令、解释器版本、PID、工作目录和首批进度。不要重复启动，也不要在验收失败时偷偷回退科学参数。
8. 给用户报告真实可运行/未验证部分、全部生成路径、启动方法。Windows GUI 或线程挂起若失败，保留日志并修正适配层；数值核心不应随意修改。

## 科学与来源边界

- 数值核心和 run_baseline.py 来自 2026-09-28 NEE 冻结版本，18 项冻结文件哈希应与 plan.json 一致。
- 工作边界是书附录 D 顶角随盖、下角静止，不是被保留但未打包的四角静止原型。
- 网格 256 指节点数，L=255；主残差排除节点壁面；质量仅诊断、不校正。
- 历史半格反弹通过点不能当作 NEE 已通过点；evidence 不能混进新搜索 results。
- 原机路径和 PID 仅作来源存档；没有 checkpoint，不能承诺无损续算。
- 适配层变化仅限操作系统锁、进程发现/控制、UTF-8 和路径/启动记录。任何求解器修改先说明原因，保留原版并重新做相称验证。

## 故障处理

`search.py` 遇到未完成旧结果会拒绝重跑，这是防重复机制。不要直接删除结果或 search_state.json。先解释中断原因；需要重算时建立新的独立运行目录。STOP 表示用户终止意图，不自动清除。

包需要联网下载 pip 依赖，不包含 Windows Python 安装器或离线 wheels。推荐 Python 3.11 x64；若使用不同版本，记录并验证。禁止宣称 Windows 原生验收已由打包机完成。
