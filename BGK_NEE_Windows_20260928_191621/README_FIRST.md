# BGK–NEE Windows 仿真迁移包

目标：在另一台 Windows 电脑上，由 Codex 部署并从初始条件重新运行当前 NEE 边界比较仿真。推荐 Windows 10/11 64 位、Python 3.11 64 位。**这不是检查点续算包**：旧电脑的 PID、暂停内存和进度记录不能在新电脑上恢复。

## 最短使用路径

1. 完整解压 ZIP，推荐 `C:\LBM\BGK_NEE_Windows`；不要在压缩包预览中运行。
2. 将 `CODEX_HANDOFF.md` 交给新电脑的 Codex，要求先验收再开始长算。
3. 确认安装了 Python 3.11 64 位，然后运行 `setup_windows.bat`。它只在包内创建 `.venv`、安装列明依赖并做预检，不自动开始百万步计算。需要访问 PyPI；本包不含离线安装轮子。
4. 运行 `smoke_windows.bat` 做 16×16 / Re100 / 300 步短跑；这不是收敛验证。Codex 应检查 NPZ、摘要、步数和有限值。
5. 正式运行 `run_search_windows.bat`。它按原计划从 NEE Re5468、5312 开始，每组上限 100 万步；每组重新初始化。
6. 另开 `open_dashboard_windows.bat` 查看真实记录，可暂停、继续或确认终止当前计算及队列。保持搜索终端开启；看板与计算是独立进程。

默认完整搜索可能需多小时甚至更久，不承诺耗时。不要重复双击启动搜索。包内没有原机 `.venv`、活动 PID、STOP 或 search_state 状态，首次启动是干净的新实验。

## 必须保持的科学约定

D2Q9-BGK；256×256 存储节点；节点壁面非平衡外推（NEE）；有效长度 L=255；U=0.04；静止初场；顶盖 500 步线性启动；顶角随盖运动、下角静止；无体力，无质量修正。黏度为 U×L/Re，tau=0.5+3nu。

主残差为内部节点相邻一步速度变化的相对 L2，以当前速度场归一化；每步检查，200 步落盘一次；阈值 1e-6，最早第 2000 步判断，一次达标。质量漂移只诊断，不参与速度停止判据；Ma 保护 0.1。

搜索种子 Re5468、5312；精度间隔 250；最小试探 Re100；安全上限 Re320000；最多 16 组。结果只对应这套离散参数与预算，不是物理临界 Re 或永久不收敛证明。

## 目录

- `simulation/`：新实验工作目录、计划、搜索控制器、运行入口、原样冻结的数值核心。
- `dashboard/`：真实数据看板、Windows/POSIX 进程控制及本地 Atlas 字体（含 OFL 授权）。
- `reference_halfway/`：仅供边界回归测试的旧半格反弹源码，非本次默认仿真。
- `evidence/`：原始说明、边界推导、书籍核对页、正式 NEE Re100/N64 已完成结果、旧半格反弹报告和原机未完成记录。
- `tests/`：隔离进程与 UI 检查；测试不应对正式任务发送控制信号。
- `MANIFEST_SHA256.json`：逐文件清单；`BUILD_PROVENANCE.json`：来源与改动；`VALIDATION.md`：本机验收范围。

`evidence/` 内原文绝对路径和 PID 只保留来源信息，不是部署路径。原机 Re5468 仅有进度和请求文件，未有末态 NPZ/分布函数检查点；不能把它标成已完成、已收敛或当作续算初场。旧“四角静止”原型和其他历史工程未打包。

## 命令（PowerShell，位于包根目录）

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-gui.txt
.\.venv\Scripts\python.exe -X utf8 transfer.py check
.\.venv\Scripts\python.exe -X utf8 transfer.py boundary-check
.\.venv\Scripts\python.exe -X utf8 transfer.py smoke
.\.venv\Scripts\python.exe -X utf8 transfer.py search
# 另开一个 PowerShell：
.\.venv\Scripts\python.exe -X utf8 transfer.py dashboard
```

只算当前 Re5468 单组可用 `transfer.py run-current`，不要与 search 同时运行。单组入口不创建搜索调度器，看板进程控制仅对 search 方式启用。

低 Re 正式复核（不要在运行高 Re 队列时并发执行）：

```powershell
.\.venv\Scripts\python.exe -X utf8 simulation/calculation/run_baseline.py --boundary nee --mass-policy diagnostic --grid 64 --re 100 --lid-speed 0.04 --tol 1e-6 --max-iter 50000 --min-iter 2000 --ramp-steps 500 --report-interval 200
```

原机此工况 16148 步速度达标，质量漂移约 0.003104894，Ghia 二次转录中线 RMSE 为 0.00447948 / 0.00427951。操作系统和数值库差异可能带来浮点差异，应记录偏差；不要把低 Re 检查等同于高 Re/网格独立性验证。

## 控制和限制

暂停/继续使用 psutil 的进程挂起/恢复接口（Windows 由系统线程挂起机制实现）；暂停仍占内存，重启电脑后不能恢复。终止按钮有默认取消的确认框，会同时结束调度器和当前计算，并创建 STOP 文件；Windows 终止不会执行 Python finally 或保存末态，已写文件保留。没有自动删除 STOP 或重启队列的功能。若任务已切换、PID/创建时间/父子关系不符则拒绝操作。

“冻结画面”仅停止看板刷新，不暂停计算。运行中只有真实标量报告；流场仍须等该组结束产生 NPZ。耗时沿用原始报告，可能包含暂停等待。

本次只在 macOS 上验证 Python 路径与跨平台适配的 POSIX 分支；Windows 原生锁、线程挂起/恢复、bat 和 Qt 显示必须由目标机 Codex 验收，不能将本机通过写成 Windows 已通过。详见 `CODEX_HANDOFF.md`。
