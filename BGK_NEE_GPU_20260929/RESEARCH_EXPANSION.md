# LBM Atlas：二维、热流与三维扩展

更新日期：2026-09-30。入口仍为 `open_dashboard_windows.bat`。修改后的程序在重新打开窗口后加载；已运行的旧窗口不会自动替换 Python 模块。

本页记录基础速度集、热模型和第一轮验证。第二轮已增加开放边界尾流、周期驱动、Rayleigh–Bénard/混合对流、阵列、探针与频谱、快照回放、双参数矩阵及配置 JSON，共 14 类场景、20 个预设。当前完整操作与适用限制见 [ADVANCED_EXPERIMENTS.md](ADVANCED_EXPERIMENTS.md)。

## 直接使用

1. 在左侧 **内置实验** 中选择算例，程序填入相容的速度模型、碰撞方式、边界和主要参数。
2. 选择 CPU、GPU 融合 CUDA 或 GPU CuPy 对照。输入 Nx、Ny，三维还需 Nz；Ny=0 表示跟随 Nx。
3. 稳态任务使用速度/温度/质量判据；剪切波、Taylor–Green 涡和温度波必须选择“瞬态 · 固定步数”。最大步数用尽不会显示为“已收敛”。
4. 需要物性、外力、热流、障碍物或保存选项时，展开 **物性、热流与障碍物参数**。
5. 点击“开始计算”。结果仍由串行队列管理，支持暂停、继续、取消、模板、参数复用和导出。

运行中或完成后，在右侧“流场”选择速度模、ux、uy、uz、密度、温度、截面法向涡量或流线。三维可切换 xy/xz/yz 截面和归一化位置；标题给出实际选中的最近格点位置。速度模使用全部三个分量，流线只使用当前截面的两个切向分量。

长方形截面按真实边长比例显示；归一化坐标中的流线方向分别使用 ux/Lx、uy/Ly，避免把不同长宽的域误画成等尺度流动。

“截面法向涡量”按画面的水平轴与竖直轴确定右手法向：xy 对应 +z，xz 对应 −y，yz 对应 +x，数值采用格子单位。

## 速度模型和碰撞方式

| 流体模型 | 维数 / 分布数 | BGK | TRT | MRT | 当前适用范围 |
|---|---:|---|---|---|---|
| D2Q9 | 2 / 9 | 是 | 是 | 是 | 方腔、通道、障碍通道、周期涡、温度输运、自然对流 |
| D2Q25 | 2 / 25 | 是 | 是 | 否 | 无固体的全周期剪切波、Taylor–Green 涡、温度输运 |
| D3Q19 | 3 / 19 | 是 | 是 | 否 | 三维方腔、通道、球/立方体障碍物、周期涡、热流 |
| D3Q27 | 3 / 27 | 是 | 是 | 否 | 与 D3Q19 同类场景，可比较离散速度集 |

温度采用独立分布函数：二维 D2Q5、三维 D3Q7。它们求解标量输运方程，不能替代流体速度模型。

全部计算使用 float64。`research_solver.py` 提供 NumPy/CuPy 方程实现，`research_gpu.py` 与 `research_kernels.cu` 执行实际 CUDA 碰撞和迁移。界面选择后端会调用对应实现。原 D2Q9 BGK/MRT 方腔继续使用原有独立内核；`reference_cpu/` 的 18 个冻结文件保持原样。

MRT 沿用已验证的 D2Q9 原始矩基，未向三维模型套用二维矩阵。TRT 的偶、奇模态满足

\[
\tau_+=\frac12+\frac{\nu}{c_s^2},\qquad
\tau_-=\frac12+\frac{\Lambda}{\tau_+-1/2},\qquad \Lambda>0.
\]

默认 \(\Lambda=3/16\)。BGK 为单松弛；MRT 的剪切松弛率由黏度决定，其他非守恒模态通过 s_e、s_ε、s_q 编辑。

D2Q25 是 \(\{0,\pm1,\pm3\}\) 一维速度集的二维张量积，\(c_s^2=1-\sqrt{2/5}\)。权重通过零、二、四、六阶高斯矩确定并数值核验。当前仍使用二阶低马赫数平衡分布；增加速度数本身不构成高阶可压缩热模型，也不保证对所有问题精度更高。其长链跨越多个格点，因此当前只允许无障碍物的全周期域，避免误用单位链长的半格点反弹。

## 第一轮基础场景（第二轮增加六类场景）

| 场景 | 驱动 / 初态 | 流体边界 |
|---|---|---|
| 顶盖驱动腔 | 顶壁沿 +x 匀速运动，支持启动渐增 | 二维 D2Q9 BGK/MRT 可选 NEE/HBB；其他组合 HBB；三维六面封闭 |
| 平行壁剪切流 | 上壁速度 U，下壁静止 | y 方向 HBB，x 与三维 z 方向周期 |
| 体力驱动通道 | 自动加速度 \(a_x=8\nu U/N_y^2\)，可叠加手动加速度 | y 方向静止 HBB，其他方向周期 |
| 周期障碍通道 | 同上；圆/球或正方形/立方体障碍 | 周期障碍阵列；无入口/出口 |
| 剪切波衰减 | \(u_x=U\sin(2\pi y/L_y)\) | 全周期 |
| Taylor–Green 涡 | 二维涡或三维正交涡初态，配套低马赫数压力初值 | 全周期；各周期边长相等 |
| 温度对流扩散 | 均匀 \(u_x=U\)，\(T=0.5+0.1\sin(2\pi x/L_x)\) | 流体与标量全周期 |
| 差温腔自然对流 / 导热 | 左热 T=1、右冷 T=0；初始线性温度 | 流体各面静止 HBB；其余温度壁绝热 |

障碍物通过格点固体掩膜和流固链接反弹实现。中心坐标按各方向边长归一化；半径/半宽是“障碍尺度 × 最短边长”。障碍物必须与外边界之间留出流体格点。圆/球表面是阶梯近似，尚无插值曲壁；这类周期阵列不能解释为具有独立入口、远场和出口的单圆柱尾流。

HBB 外壁位于最外流体节点外侧半格；坐标为 \((i+1/2)/N\)。顶盖角点使用水平壁优先。三维通道在 z 方向周期，代表沿 z 周期重复的平行板通道，不是四侧均为固壁的矩形管道。

## 物性、加速度与热模型

长度、速度、黏度、时间均为格子单位，\(\Delta x=\Delta t=1\)。

- ν=0：由 \(\nu=UL/Re\) 求黏度。方腔 L 取最短边长，NEE 减去一格；其他场景取 Ny。
- ν>0：直接使用指定运动黏度。保留用户输入的 Re 用于记录，但实际配置中的 `flow.Re` 和显示的有效 Re 使用 \(UL/\nu\)。扫描 Re 时要求 ν=0，避免提交物理参数相同的多组任务。
- `force_x/y/z` 为加速度，不是力密度。通道的 x 向输入叠加在自动驱动之上。Guo 源项包含 \(\rho\mathbf a\)，物理速度满足 \(\rho\mathbf u=\sum_i f_i\mathbf e_i+\rho\mathbf a/2\)。非单位密度及恒加速度解析测试覆盖了这一约定。
- 温度波使用用户指定 α；自然对流使用 \(\alpha=\nu/Pr\)，输入的独立 α 在该场景不生效并被禁用。
- 自然对流的浮力为 \(a_y=Ra\nu\alpha(T-0.5)/N_y^3\)。Ra 以高度 Ny 定义，Nu 以横向热传导长度 Nx 归一化。Ra=0 是无浮力导热。

温度是无量纲标量，流体使用 Boussinesq 浮力耦合；这里没有密度随温度的完整状态方程、压缩功或黏性生热。该模型适用于低马赫数流动和相应的小密度变化近似，不能作为可压缩热流、激波、多相或湍流闭合模型。

热壁采用标量反向反弹，其他温度壁采用零通量反弹。壁面平均 Nu 由墙面与首层格点的半格温差梯度估计；输出 `nusselt_hot` 与 `nusselt_cold` 便于检查热平衡。它是离散壁面估计，精度仍需网格收敛研究。

## 任务、保存与续算

- 参数扫描支持 Re、Nx/Ny/Nz、ν、Ra、Pr、α、ax、U、周期、加速度幅值、障碍尺寸，最多两个轴，笛卡尔积不超过 100 组，整批校验后原子提交。Ra/Pr 扫描限浮力热对流，α 扫描限温度波。
- 新扩展内核每步检查非有限分布、正密度、实际最大 Ma；继承当前低马赫数保护上限 0.1。
- 稳态要求单步速度残差、温度残差（若有）以及闭域质量漂移达标，并通过连续步判据。温度残差和速度残差都写入日志。
- 瞬态不按速度残差提前结束，状态为 `completed_steps`；稳态步数用尽为 `max_iter`。
- `snapshot_interval=0` 关闭场快照；正整数会将完整场保存为 `snapshots/fields_<step>.npz`，并非显示预览。
- `checkpoint_interval=0` 关闭周期检查点；扩展内核在正常结束/取消时仍保存最后检查点。周期检查点原子替换 `checkpoint.npz`，其中包含完整 f、g、宏观场、总步数、物理参数和内核源码哈希。
- “从检查点读取参数”会填入原物理模型。允许调整后端、总步数预算、报告/保存间隔和停止设置；物理参数或内核源码改变时拒绝续算。最大步数表示续算后的总步数，必须大于检查点步数。续算总是新建任务与目录，原始数据保留。
- 旧 D2Q9 BGK/MRT 方腔内核不提供这次新增的检查点/场快照功能。要在二维方腔使用这些功能，请选择 TRT＋HBB，界面会对不相容设置给出说明。

`results.npz` 保存原分辨率的二维/三维密度、三个速度分量、温度（若有）、固体掩膜和实际坐标。三维数组顺序为 `(z,y,x)`。实时预览单方向最多约 41 个三维采样点；最终结果不抽样。交互分析沿用解压后 2 GB 的读取上限。

诊断新增平均动能、温度极值/均值、温度残差和冷热壁 Nu。CSV 包含完整扩展参数、有效 Re、实际 ν 和标量模型；ZIP 包含结果、检查点、场快照和 SHA-256 清单。三维差值指标覆盖完整体积，图示为 z 中截面；三维中心线也明确使用最接近 z/Lz=0.5 的 xy 截面。

## 可复现验证

在本目录中运行：

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
& 'D:/LBM/envs/bgk-nee-gpu-20260929/Scripts/python.exe' -X utf8 validate_research.py
& 'D:/LBM/envs/bgk-nee-gpu-20260929/Scripts/python.exe' -X utf8 -m pytest tests -q --basetemp=validation/manual_test_tmp -p no:cacheprovider
& 'D:/LBM/envs/bgk-nee-gpu-20260929/Scripts/python.exe' -X utf8 verify_research_ui.py
```

数值验收包含权重/矩、非单位密度恒加速度、TRT 到 BGK 的退化、剪切波衰减、二维/三维通道解析剖面、温度对流扩散、Ra=0 导热、CPU/CuPy/CUDA 非平衡态轨迹、封闭质量守恒、检查点无缝续算和 Ra=1000 自然对流基准。结果写入 [correctness_research.json](validation/correctness_research.json)，绑定实际内核源码 SHA-256；内核变化后需重新通过检查。

第一轮界面验收从真实 Qt 控件提交 12 个实验；历史记录见 [research_ui.json](validation/research_ui.json)。第二轮对全部 20 个预设及新增操作重新验收，当前记录见 [advanced_ui.json](validation/advanced_ui.json)。这些短任务证明入口和数据链路可用，不证明每个预设的高 Re/高 Ra 工况已经达到稳态或网格无关。

已完成的 Ra=1000、Pr=0.71、24×24 数值基准在第 8711 步收敛，热壁 Nu=1.113941573，冷壁 Nu=1.113943617；与二维方腔基准 Nu=1.118 的差约 0.36%。该对照只用于此次二维小 Ra 验收，不能外推为三维自然对流或大 Ra 的定量验证。

本次没有为新增三维/热流内核声明加速比。原 README_GPU 中的历史速度数据只针对原 BGK＋NEE 方腔。

## 方法依据

- Guo, Zheng & Shi, *Discrete lattice effects on the forcing term in the lattice Boltzmann method*, 2002，[DOI](https://doi.org/10.1103/PhysRevE.65.046308)：外力和半步速度修正。
- Lallemand & Luo, *Theory of the lattice Boltzmann method: Dispersion, dissipation, isotropy, Galilean invariance, and stability*, 2000，[NASA 原文入口](https://ntrs.nasa.gov/citations/20000046606)：D2Q9 MRT 矩方法。
- Chikatamarla & Karlin, *Lattices for the lattice Boltzmann method*, 2009，[原论文](https://doi.org/10.1103/PhysRevE.79.046701)：速度集与矩构造。这里采用的二阶等温平衡分布不等同于论文中全部高阶/熵方法。
- Bauer & Rüde, *An improved lattice Boltzmann D3Q19 method based on an alternative equilibrium discretization*, [预印本](https://arxiv.org/abs/1803.04937)：D3Q19/D3Q27 对照背景；本实现使用标准二阶平衡分布，没有宣称采用其改进模型。
- Xu, Shi & Xi, *Lattice Boltzmann simulations of three-dimensional thermal convective flows at high Rayleigh number*, [原文](https://arxiv.org/abs/1903.08882)：流体与温度采用不同速度集的热流背景；不代表本代码复现其高 Ra 验证。
- de Vahl Davis, *Natural convection of air in a square cavity: A bench mark numerical solution*, 1983，[原论文](https://doi.org/10.1002/fld.1650030305)：二维差温腔基准。

修改前备份在 `validation/research_extension_backup_20260930_002656/`。本次扩展集中在当前 GPU 工作台，不改写根目录求解器、独立资料站或冻结 CPU 源码。
