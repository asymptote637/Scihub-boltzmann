# LBM 科研仿真项目：新对话迁移 Prompt

> 2026-09-29 整理说明：本文件及研究文档已分类移动，操作性路径已更新；历史 Git 状态段保留原样。新旧位置见 [整理记录](../organization/2026-09-29.json)。

请把以下内容作为我从上一段对话迁移过来的项目上下文，继续担任我的长期学术与技术助手。它保留了旧对话的目标、互动节点、工程状态、验证结果和待办；不是要求你重新从头完成已做过的工作。

这份记录以原对话可见内容及 2026-09-08 的本地文件核查为依据，不代表已读取我账户下所有其他对话。若你能访问原工程，请以当前文件和实验记录为准；若不能，请明确说明需要哪些文件，不能声称已经读取、运行或验证。

## 一、我的工作方式与本项目目标

1. 我主要使用中文，重视结论准确、物理自洽和逻辑严谨。发现模型、答案或代码有问题时直接指出，不要为了迎合我默认现有结论正确。
2. 对物理问题按“物理建模→基本定律→数学推导→结果→物理解释”展开；检查量纲、符号、边界条件、极限和数量级，关键步骤不得跳过。
3. 沿用我提供的教材符号、模型和学习主线。新增方法、改变模型或边界时解释原因，区分教材内容、已有工程扩展、你的推导和文献结果。
4. 我需要实际可运行的程序、配置、数据、图像和验收结果。不能只给计划，也不能以“程序不报错”或“云图看起来合理”作为科研验证。
5. 信息充分时直接推进。任务是逐步指导我理解并实践，保留必要解释和小实验，但不要将所有代码执行工作推回给我。
6. 保护已有文件与未提交改动，不擅自重置、删除、重建工程或混用两套求解器。只有明确要求发布时才提交到远程、开 PR 或对外发送。
7. 我已明确表示背景窗口占用过多：后续按需读取文件和摘要，避免整文件反复输出、长日志、重复总结。事实记录留在文件，回答聚焦当次变化、验证、限制与下一步。
8. 做论文/基准检索时优先原论文和可靠一手来源；不能编造对照数据或把未经核实的二手表格当作基准。
9. 我的其他常用规则：竞赛题强调物理主线、逐层递进；批阅只按卷面明确评分点给分；名单以指定基准名单为唯一顺序；缺成绩填 0 须有明确授权；文档尊重原模板，中文优先宋体、英文及数学字符优先 Times New Roman。这些规则在对应任务出现时使用，不要机械套到所有编程讨论。

本项目总目标：依据《格子 Boltzmann 方法的理论及应用》的详细总结稿，在现有 Python LBM 工程上构建可复现、可定量验证、逐步深入的科研仿真训练体系，最终能够独立完成物理建模、离散模型选择、参数映射、误差分析与科研结论。

每关都应留下：模型卡、无量纲/格子参数表、可复现配置、解析或文献基准、守恒与稳定性记录、误差及独立性分析、一页研究记录。

## 二、已发生的关键互动节点

1. 2026-09-05，我提供总结稿，要求：“根据此md文件指引我构建科研仿真任务训练，并一步一步深入探索实践”。助手发现工作目录已是 LBM 工程，于是复用根目录求解器，建立十阶段训练路线。
2. 首关选择 D2Q9 矩审计和周期剪切波黏度标定，隔离体相碰撞/迁移误差。助手实际实现脚本、讲义和测试，完成定量实验；这部分已完成，不要只重新介绍 LBM。
3. 2026-09-08，我要求：“继续编写仿真程序”。助手接着完成第 2 关 Couette/Poiseuille 通道，发现并修复移动壁面符号及密度重构错误，新增半格反弹，修正 Guo 体力初始化，完成三网格与碰撞模型验证。
4. 我表示“背景信息窗口占用过多，压缩一些”，随后多次选中或发送 `/compact`。旧助手没有可直接执行的压缩工具；对话中没有确认成功的系统压缩事件。普通文字、代码格式或摘要均不等于已压缩。不要再反复让我尝试相同命令或声称已经释放上下文。
5. 我最终要求：“请根据我们所有历史对话，生成一份prompt，最大限度保留我们的互动节点、核心结论和待解决问题。我将使用它在新对话窗口进行数据迁移。”因此生成本文。当前重点是保留上下文并自然接续科研开发。

## 三、环境、原始材料与工程层次

平台：Windows，PowerShell，时区 Asia/Shanghai。
工程根目录 R = D:\BaiduSyncdisk\我的文件\Leon\科研
原始材料 = C:\Users\hongt\Downloads\格子Boltzmann方法_详细总结_仿真实践增强版.md
项目解释器 = D:\BaiduSyncdisk\我的文件\Leon\科研\.venv\Scripts\python.exe
最近实验记录中的环境：Python 3.12.14，NumPy 2.5.2。迁移后需从实际环境确认，不要假设永远相同。

本文后面的相对路径均相对于 R。原始总结稿在迁移整理时确认仍存在。它是学习型总结，不是原书逐字转录；其中页码映射记录为正文书页 1 对应 PDF 第 15 页，正文大致 PDF 页=书页+14。原始扫描 PDF 未作为本次迁移附件提供，不可声称重新核对了原扫描页。

源材料重点：3.5 D2Q9；4.2 热模型/Boussinesq；第 6 章边界；7.4 FD-LBM；8.1–8.6 算例；附录 B 格子张量、C 单位转换、D 顶盖驱动；文末统一仿真框架 Step 0–14。

根目录新版求解器是本项目训练基础：
- config.py：配置、Re/tau/物理单位/Ra 参数映射、预设和校验。
- boundary_conditions.py：D2Q9 速度/权重、平衡态、宏观量、边界注册与实现。
- lbm_solver.py：BGK/TRT/MRT、碰撞、迁移、Guo 体力、障碍物、残差及热耦合。
- thermal_lbm.py：D2Q5 温度双分布，适用低 Ma、Boussinesq 小温差模型。
- main.py：通用命令行及完整 JSON 配置入口。
- postprocess.py：场量、涡量、流线、残差和结果输出。
- desktop_app.py：PySide6 桌面端；ui_app.py：旧 Streamlit 入口。
- src/lbm_lab：较早的包、TOML、数据库和简单方腔框架。不能把它与根目录新版的参数定义、数据布局或边界实现混用。

现有预设还包括方腔、双盖方腔、通道、圆柱/方柱/后向台阶、Taylor–Green 涡、剪切波、自然对流、Rayleigh–Bénard 和加热通道等。存在预设或烟雾测试不等于已完成相应科研基准验证。

## 四、已建立的训练路线

0. D2Q9 矩与各向同性审计。
1. 周期剪切波衰减，标定黏度。
2. Couette/Poiseuille，验证壁面和体力。
3. 顶盖驱动方腔，中心线速度、涡心、网格独立性。
4. BGK/TRT/MRT 在边界误差、稳定性、耗时上的对照。
5. 被动温度扩散与受迫换热。
6. 方腔自然对流，Ra/Pr/Ma、平均 Nu、中心线极值。
7. Rayleigh–Bénard，临界 Ra、增长率与热流。
8. 圆柱绕流/曲线边界，CD/CL/St、阻塞比及几何误差。
9. 自选科研问题，参数扫描、不确定度和可复现报告。

阶段 0–2 已有本次实际开发与验证；阶段 3–9 尚未按此训练体系完成。阶段 4 的少量体相/通道测试不代表完整碰撞模型比较已经完成。此前给过约八周的建议节奏，但没有设立定时任务或硬性期限；晋级以验收证据为准。

## 五、必须保持一致的物理与数值约定

- 分布数组布局：(NY, NX, 9)。轴 0 是 y，轴 1 是 x。
- D2Q9 顺序：0=(0,0)，1=(1,0)，2=(0,1)，3=(-1,0)，4=(0,-1)，5=(1,1)，6=(-1,1)，7=(-1,-1)，8=(1,-1)。
- 权重：w0=4/9；w1–w4=1/9；w5–w8=1/36。
- 对向索引 OPP=[0,3,4,1,2,7,8,5,6]。
- 格子单位 dx=dt=1，cs²=1/3。
- feq_i = rho*w_i*[1+3(e_i·u)+4.5(e_i·u)²−1.5|u|²]。
- 无外力时 rho=sum(f_i)，rho*u=sum(f_i*e_i)，p=rho*cs²。
- nu=cs²*(tau−1/2)=(tau−1/2)/3；omega=1/tau；Re=UL/nu；Ma=U/cs。
- 基本 D2Q9 是弱可压等温模型，其低 Ma 宏观极限近似不可压；不等于完整可压缩热气体模型。
- 体相正确不等于边界正确；残差下降不等于解正确；稳定不等于达到精度阈值。

## 六、第 1 关：剪切波，已经完成

文件：
- training/shear_wave_viscosity.py
- tests/test_training_shear_wave.py
- docs/training/01_shear_wave_viscosity.md

模型：二维全周期，rho=1，ux(y,0)=U0*sin(ky)，uy=0，k=2*pi/Ny。该速度场对流项为零；宏观目标为 ∂t ux=nu*∂yy ux，解为 ux=U0*exp(−nu*k²*t)*sin(ky)。通过正弦基模投影取得 A(t)，拟合 ln A 的斜率，nu_fit=−slope/k²。

默认 Nx=Ny=64，U0=0.02，Ma≈0.034641，运行 400 步，每 5 步采样，从第 20 步起拟合，以减小初始动力学瞬态影响。

已验证记录：
tau | nu_theory | nu_fit | 有符号相对误差
0.6 | 0.0333333333 | 0.0333590451 | 7.713529e−4
0.8 | 0.1000000000 | 0.1000514443 | 5.144428e−4
1.0 | 0.1666666667 | 0.1666666380 | −1.721788e−7

D2Q9 0–4 阶各向同性最大残差与平衡态密度/动量/动量通量矩最大残差均约 1.665e−16。最大相对质量漂移约 3.575e−14。默认验收黏度误差 <0.2%，矩残差 <1e−14，质量漂移 <1e−12。

结果目录：results/training/01_shear_wave/
包含 fit_summary.json、amplitude_decay.csv（默认 243 条采样记录）、amplitude_decay.png；旧对话已结构及视觉核验。

此前提出 Ny=32/64/128、幅值 0.01/0.05/0.10、TRT/MRT 对照作为练习；不要把提出的练习当成全部已经执行，也没有证据证明我已经独立完成讲义问题。

物理边界：改变 Ny 同时改变格子波数 k，是长波极限扫描；若要称为同一个物理问题的网格收敛，需明确尺度映射。剪切波验证只覆盖体相剪切模和周期迁移，不验证固壁、高 Re、角点或热耦合。高幅值扫描的实际结果必须测量，不预设一定出现密度扰动或非线性偏差。

## 七、第 2 关：通道，已经完成

文件：training/channel_validation.py、tests/test_training_channels.py、docs/training/02_channel_validation.md。
主要函数：build_channel_config、analytical_profile、run_channel_case、observed_orders、source_hashes、save_experiment、plot_results、main。

物理模型：ux=u(y,t)，uy=0，x 周期；∂t u=nu*∂yy u+gx。
- Couette：下壁静止，上壁以 U 向右移动，gx=0。u=U*y/H，单位展向宽度体积流量 Q=UH/2。
- Poiseuille：两壁静止，恒定 gx。u=gx*y(H−y)/(2nu)，U=u_max=gx*H²/(8nu)，Q=2UH/3。
- 这里 Re=UH/nu 中 U 是顶盖速度或 Poiseuille 最大速度，不能偷偷换成平均速度。
- body_force_x 是单位体积力 Fx=rho0*gx，不是加速度；设置 Fx=rho0*8nu*U/H²。
- 训练脚本以 custom 配置构造两算例，从静止物理速度启动；解析解只用于后处理。通用 Couette 预设原本可以初始化为线性速度，不要混淆预设和训练脚本。

### 已修复的具体问题

1. 旧 moving_wall_bounce_back 速度修正符号错误，且从迁移后缺失入射分布的数组计算密度。旧测试 U=+0.02、Nx=8、Ny=16、tau=0.8、即时启动、2500 步，顶部流速约 −0.0166667。现在先用反射关系重构密度；当 i 指向流体内部，速度修正为正号。四面壁、正负切向速度的平衡态不变性已有测试。
2. 新增 halfway_bounce_back / moving_halfway_bounce_back，通过 apply_halfway_boundaries(streamed,f_post,rho,cfg,ramp) 处理。缺失入射分布取同一流体点碰撞后的反向分布，而不是迁移后的相反分布：f_i(x,t+1)=f_opp_i*(x,t)+2*w_i*rho*(e_i·u_wall)/cs²。
3. 半格几何：全部 Ny 行为流体点，y_j=j+1/2，真实壁面 y=0,H，H=Ny。最外层流体点不在壁面上，不能强制要求其速度等于壁速。纵向周期通道对应 H=Nx。
4. 新半格外边界目前仅允许两个相对平壁+另一个方向周期；移动壁速度必须切向；不兼容热场、障碍物和一般角点拓扑时配置校验会拒绝。不要直接套到四壁方腔。
5. couette_flow 和 periodic_channel 新预设改用半格反弹，characteristic_length 相应变为 Ny（有显式 l_ref 时仍以显式值为先）；显式旧 JSON 按自己的边界类型加载，不自动转换。
6. Guo 体力物理动量为 rho*u=sum(f_i*e_i)+F/2。初始化改为 f_i=feq_i−S_i/2，S_i 是未乘碰撞 prefactor 的源项，以保持给定初始物理速度。均匀全周期加速、非单位 rho0=1.7、二维正负体力和 BGK/TRT/MRT 已做回归。

### 默认扫描和验收标准

默认 BGK、tau=0.8、nu=0.1、Re=3.2、rho0=1、Nx=8、Ny=16/32/64。由于流向均匀，Nx=8 是隔离横向剖面误差的设置，不代表一般二维流动只需 8 列。
固定 Re 与 tau 的扩散尺度：U∝1/H，Fx∝1/H³，物理固定长度时 dx_phys∝1/H、dt_phys∝1/H²。
H=16/32/64 对应 U=0.02/0.01/0.005、Ma≈0.03464/0.01732/0.00866；Poiseuille Fx=6.25e−5/7.8125e−6/9.765625e−7。

以 t_nu=H²/nu 计时；每约 0.01*t_nu 采样一次，至少运行 t_nu。连续三次 RMS[u(t)−u(t−报告间隔)]/U <1e−9 才算 converged。默认最多 3*t_nu，未满足则 max_iter 并验收失败；不是固定步数结束就算成功。
逐算例阈值：Couette 剖面相对 L2、流量误差均 <1e−6；Poiseuille 两者均 <0.005；采样最大相对质量漂移 <1e−9；横向速度/U 与流向不均匀度/U 均 <1e−8；必须达稳态。
Poiseuille 误差可分辨时 p=ln(Ecoarse/Efine)/ln(Hfine/Hcoarse)，判 1.8≤p≤2.2。Couette 线性解剩余误差主要是停止容限及舍入，不强行计算空间阶数。单网格及低于误差底限时明确标记没有检验阶数。
流量由中点求积得到，因此包含场误差和求积误差。壁面滑移是近壁三点二次外推到真实壁面得到的诊断，不是最外层流体点速度。

### 默认 BGK 六个已验证实验

Ny | Couette 相对 L2 | Poiseuille 相对 L2 | Poiseuille 流量有符号相对误差
16 | 1.340004147e−8 | 2.781394393e−3 | −1.093760240e−3
32 | 1.310124805e−8 | 6.953598542e−4 | −2.734476232e−4
64 | 1.333900560e−8 | 1.738471073e−4 | −6.836965681e−5

Poiseuille 收敛阶为 1.9999766465 和 1.9999407156。采样最大相对质量漂移 4.803046849e−12。六个实验均 converged 且 passed=true。
Couette 16/32/64 分别在 4628/18564/74210 步停止；Poiseuille 分别 4758/19074/76260 步。参数变化后不能把这些步数当成通用停止条件。

最终基准目录：
results/training/02_channels/20260907T205808_886472Z/
较早同结果目录：results/training/02_channels/20260907T205127_545470Z/。
顶层包含 summary.json、profiles.png、convergence.png；各 couette_n16 等子目录包含 config.json、metrics.json、history.csv、profile.csv、fields.npz。
summary 记录命令参数、Python/NumPy/平台、四个核心源码 SHA256、运行期间源码是否变化、每项验收与阶数。旧对话已核验 CSV/NPZ 一致性、图片布局和指纹。UTC 目录日期为 09-07，对应北京时间 09-08，不能据此误判运行日期。

### MRT 未达标结果必须保留

同 tau/Re，rho0=1.7，MRT Lallemand–Luo 默认参数：
- Ny=16：L2=0.005123674854（约 0.5124%），横向速度最大约 1.13554e−9，归一化后也超阈值；虽然 converged，但 passed=false。
- Ny=32：L2=0.001280922120（约 0.1281%），横向速度最大约 3.88795e−11；所有判据通过。
- 两网格阶数约 1.999996163；扫描总体 passed=false，退出码 1，因为包含未达标粗网格。
目录：results/training/02_channels_mrt/20260907T205656_410749Z/。
这是被保留的精度限制，不是遗漏修复，也不应通过放宽阈值抹掉。测试中对 MRT H=16 预期 false、H=32 预期 true；整个软件测试集仍可通过。此证据也不等于完整解释了 MRT 误差机制或证明所有参数都可靠。

## 八、测试、版本状态与复现命令

截至上一次开发完成：完整 pytest 为 69 passed in 26.81s；所改文件 Ruff 检查通过；git diff --check 通过（存在 Windows LF/CRLF 提示）。本迁移整理轮只核对了文件/结果/源码指纹，没有重新跑 69 项测试，也没有重算全部仿真。

测试增长：最初 41 passed+2 errors，两个错误为 pytest 默认临时目录权限；第 1 关完成后 45 passed；第 2 关完成后 69 passed。不能把旧的环境错误当成数值错误。
当前 tests 覆盖原有碰撞守恒/等价性、烟雾测试、热边界、数据库等，加上矩审计、剪切黏度、移动壁正负方向、Guo 均匀加速、半格通道、Re 缩放、Poiseuille 二阶收敛、MRT 粗细网格、旋转通道、非法拓扑和未收敛状态。

2026-09-08 迁移整理时再次核对：config.py、boundary_conditions.py、lbm_solver.py、training/channel_validation.py 与最终基准 JSON 中 SHA256 全部匹配。此为该时刻事实，后续改动后要重验。

Git 未提交状态：README.md、boundary_conditions.py、config.py、lbm_solver.py 为 modified；docs/lbm_research_training.md、docs/training/、tests/test_training_channels.py、tests/test_training_shear_wave.py、training/ 为 untracked。本文也是新增文件。没有在旧对话中 stage、commit、push 或发布 PR。代码和 results 数据尚不能指望通过 git clone 自动获得；results、.tmp、.venv 等被忽略，迁移到另一台机器时需另行复制结果。

PowerShell 常用命令（先进入 R）：

```powershell
Set-Location -LiteralPath 'D:\BaiduSyncdisk\我的文件\Leon\科研'
$env:PYTHONPATH='.'
.\.venv\Scripts\python.exe -m training.shear_wave_viscosity
.\.venv\Scripts\python.exe -m training.channel_validation
.\.venv\Scripts\python.exe -m training.channel_validation --case couette --grids 16
.\.venv\Scripts\python.exe -m training.channel_validation --case poiseuille --grids 16 32 --collision-model TRT
.\.venv\Scripts\python.exe -m training.channel_validation --case poiseuille --grids 32 64 --collision-model MRT
```

最后两条仅为可用命令，不代表所有列出的组合都有独立完整实验记录。MRT 16/32、rho0=1.7 的实际记录见前节。通道脚本每次创建 UTC 时间戳子目录；剪切波脚本默认使用固定结果目录，复验需保留旧结果或指定新 --output-dir。

Git 若出现 dubious ownership，单命令使用：
`git -c safe.directory='D:/BaiduSyncdisk/我的文件/Leon/科研' status --short`
不要为此擅自修改全局 Git 信任配置。

pytest 默认 C:\Users\hongt\AppData\Local\Temp\pytest-of-hongt 曾拒绝访问，项目旧缓存也曾权限异常。使用新的、未存在的项目内临时目录，避免覆盖旧目录：

```powershell
$testTemp=Join-Path (Get-Location) ('.tmp\pytest-migration-' + [guid]::NewGuid().ToString('N'))
if (Test-Path -LiteralPath $testTemp) { throw 'Test directory already exists' }
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp $testTemp
.\.venv\Scripts\python.exe -m ruff check boundary_conditions.py config.py lbm_solver.py training/channel_validation.py tests/test_training_channels.py --no-cache
```

## 九、待解决问题和后续优先级

优先接续阶段 3：顶盖驱动方腔的科研级基准。现有方腔预设和烟雾测试已经存在，但本训练尚未完成中心线与文献数据对比、涡心定位、角点审查、网格与 Ma 独立性。

建议实施顺序（这是接续建议，不是声称已经完成）：
1. 先检查现有方腔、非平衡外推、宏观量时间层、壁面位置和角点处理，保留当前模型与符号。
2. 特别核查：总结稿附录 D 的速度边界密度取邻近流体 rho；当前 boundary_conditions.py 的 non_equilibrium_extrapolation 用 bc.rho 或 rho0 填充固定 rho_b。迁移核查已确认代码有这一差异，但尚未证明其影响或实现修正；不要把“同名边界”当成逐式一致。
3. 当前 apply_boundaries 按 left→right→bottom→top 顺序更新，因此角点覆盖需要明确物理约定和单独验证；不能默认顺序无关。
4. 从低 Re 例如 100 开始验证，必要时再到 400/1000。Re=100 是后续合理起点建议，并非我已确认的新固定参数。
5. 获取可靠文献基准，例如原始 Ghia 中心线速度数据；核对 Re、坐标、速度归一化、原表数值与出处，不要编造。
6. 输出中心线 ux(y)、uy(x)、主涡位置、流线、残差、质量漂移，以及同 Re 下至少三网格比较；另检查 Ma 对结果的影响。
7. 视边界审计结果决定必要修正并添加有物理意义的回归；保持第 1–2 关基准有效。避免一次改动碰撞、壁面、网格和参数导致无法定位误差。

其他未完成：
- 第一关讲义中部分波数/幅值/碰撞模型练习的完整记录及我的个人学习反馈。
- 系统比较 BGK/TRT/MRT 的稳定区间、耗时和壁面误差；MRT 粗网格诊断的更深入机制分析。
- 非定常/ramp 体力、非均匀外力和热浮力耦合的专门定量验证；已通过恒定均匀体力测试不代表这些都已覆盖。
- 热场扩散率、Nu、Rayleigh–Bénard 临界失稳、曲线边界、圆柱力系数及 St 尚未在此训练完成科研基准。
- 没有确定最终论文研究方向、真实物性、具体几何、算力预算或发表目标，先完成通用验证再据我反馈选择。
- 工程缺少正式版本提交/发布；未获相应要求前不要自动推送。

## 十、新对话接手时怎么做

先用很短的回复说明你已理解“阶段 0–2 已完成，下一阶段是方腔基准，69 项测试是上次结果”。随后若我要求继续开发，直接开展必要只读核查并推进，不要重复给十阶段大纲或再次询问已经明确的模型选择。

按需读取 docs/training/lbm_research_training.md、docs/training/02_channel_validation.md、当前 git status、最终 summary.json，并核查相关源码指纹。只打开本阶段相关代码片段；不要把整个根目录、原始长 Markdown、所有日志和历史记录反复注入上下文。

把“历史已验证”“本轮新验证”“待核查假设”“建议下一步”分清。不要把本记录当成不可质疑的定理，也不要因为换了窗口就把所有已完成工作从头重做。若我的新消息改变方向，以最新明确要求为准。

已有参考来源仅作定位线索：
- Guo / Zheng / Shi, Physical Review E 65, 046308 (2002)：https://doi.org/10.1103/PhysRevE.65.046308
- 移动反弹实现说明：https://docs.aerosim.io/nassu/theory/LBM/bc/moving_wall.html
- 理论主线仍以我提供的本地总结稿为准；引用高精度基准时须再次核对原始来源和方向约定。
