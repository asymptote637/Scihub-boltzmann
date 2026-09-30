# 第 2 关：Couette / Poiseuille 通道的边界验证

第 1 关已提供剪切波的体相黏度测量。本关加入两面平直壁，复用根目录求解器，分别验证移动壁面的动量传递、无滑移反射和 Guo 体力项。训练入口为 `python -m training.channel_validation`。

## 物理模型与解析解

设平行板之间距离为 \(H\)，\(x\) 方向周期，\(u_x=u(y,t),u_y=0\)，等温、常物性、低马赫数。两种流动的对流项均为零，方程为

\[
\partial_t u=\nu\partial_y^2u+g_x.
\]

Couette 流：下壁静止，上壁以 \(U\) 向右移动，\(g_x=0\)。稳态剖面和单位展向宽度体积流量为

\[
u(y)=U\frac{y}{H},\qquad Q=\frac{UH}{2}.
\]

Poiseuille 流：上下壁静止，恒定加速度 \(g_x\) 驱动。稳态满足

\[
u(y)=\frac{g_x}{2\nu}y(H-y),\quad
U=u_{\max}=\frac{g_xH^2}{8\nu},\quad Q=\frac{2UH}{3}.
\]

代码中的 `body_force_x` 是单位体积力 \(F_x=\rho_0g_x\)，不是加速度。程序按 `F_x = rho0 * 8 * nu * U / H**2` 设置驱动力；非单位密度也用同一个物理定义。本关的 Re 使用顶盖速度或 Poiseuille 最大速度，均为 \(Re=UH/\nu\)，不是平均流速定义的 Re。

两算例均从静止物理速度开始演化。解析解只用于后处理，不用它初始化速度以“跳过”建立过程。

## 壁面坐标与时间层

新增边界名为 `halfway_bounce_back` 和 `moving_halfway_bounce_back`。对于横向周期通道，全部 \(N_y\) 行都是流体点，物理坐标为

\[
y_j=j+\frac12,\quad j=0,\ldots,N_y-1,\quad H=N_y.
\]

壁面位于 \(y=0,H\)。因此最上面一行流体点并不在壁面上；不能要求该行速度精确等于顶盖速度。无滑移通过链路反射表达。

令 \(i\) 指向流体内部，\(\bar i\) 是相反方向。采用

\[
f_i(\mathbf x,t+1)=f_{\bar i}^{*}(\mathbf x,t)
+\frac{2w_i\rho}{c_s^2}\,\mathbf e_i\cdot\mathbf u_w.
\]

右侧必须取**同一流体点碰撞后的**分布，不能取迁移后的相反分布。按本程序的“入射方向”编号，速度修正项为正号。静止壁设 \(\mathbf u_w=0\)。[移动反弹技术说明](https://docs.aerosim.io/nassu/theory/LBM/bc/moving_wall.html)

新增格式当前只允许两面相对平壁与另一个方向的周期边界，移动速度必须平行于壁面。热场边界和曲线/角点不在本关格式的适用范围；配置验证会拒绝这些不兼容组合。

Guo 源项与宏观速度必须一致：

\[
\rho\mathbf u=\sum_i f_i\mathbf e_i+\frac{\mathbf F}{2}.
\]

本次也修正了体力初始化：\(f_i(0)=f_i^{eq}-S_i/2\)，其中 \(S_i\) 是未乘碰撞 prefactor 的源项，从而在 \(t=0\) 正确恢复指定的物理初速度。完整周期均匀体力测试验证了 BGK/TRT/MRT 的 \(u(t)=u(0)+Ft/\rho\)。[Guo、Zheng、Shi 原论文](https://doi.org/10.1103/PhysRevE.65.046308)

## 运行与参数扫描

在项目根目录执行默认的六个实验：

```powershell
.\.venv\Scripts\python.exe -m training.channel_validation
```

默认设置为 BGK、\(\tau=0.8\)、\(Re=3.2\)、\(N_y=16,32,64\)。\(x\) 方向均匀且周期，使用 8 列足以隔离横向壁面误差；这是充分发展的一维剖面验证，不是一般二维分辨率试验。

网格加密采用扩散尺度，固定 \(Re,\nu_{lattice},\tau\)，因而

\[
U_{lattice}=\frac{Re\nu}{H}\propto H^{-1},\qquad F_{x,lattice}\propto H^{-3}.
\]

如果物理通道高度固定，则 \(\Delta x_{phys}\propto H^{-1}\)、\(\Delta t_{phys}\propto H^{-2}\)。该方案同时减小格子速度与 Ma，在同一个无量纲问题下进行加密。

| H | U | Ma | Poiseuille 的 \(F_x\)，\(\rho_0=1\) |
|---:|---:|---:|---:|
| 16 | 0.02 | 0.03464 | 0.0000625 |
| 32 | 0.01 | 0.01732 | 0.0000078125 |
| 64 | 0.005 | 0.00866 | 0.0000009765625 |

只跑一个算例，或做碰撞模型/密度对照：

```powershell
.\.venv\Scripts\python.exe -m training.channel_validation --case couette --grids 16
.\.venv\Scripts\python.exe -m training.channel_validation --case poiseuille --grids 16 32 --collision-model TRT
.\.venv\Scripts\python.exe -m training.channel_validation --case poiseuille --grids 16 --rho0 1.7
.\.venv\Scripts\python.exe -m training.channel_validation --case poiseuille --grids 32 64 --collision-model MRT
```

每次运行都在 `results/training/02_channels/` 中创建新的 UTC 时间戳子目录。保存 `summary.json`、两张汇总 PNG，以及每个网格的配置 JSON、场数据 NPZ、速度剖面 CSV、收敛记录 CSV、指标 JSON。汇总含 Python/NumPy 版本、参数和核心源码 SHA256，便于判断两次实验是否用了同一程序。

## 收敛与验收

程序以 \(t_\nu=H^2/\nu\) 作为物理建立时间。每约 \(0.01t_\nu\) 检查一次速度变化；运行至少 \(t_\nu\) 后，连续三次满足

\[
r=\frac{\sqrt{\langle|\mathbf u(t)-\mathbf u(t-\Delta t_{report})|^2\rangle}}{U}<10^{-9}
\]

才标为 `converged`。达到默认 \(3t_\nu\) 仍未满足则标 `max_iter`，判验收失败。该量是固定无量纲采样间隔下的速度变化，不是方程残差。

默认逐算例验收：

- Couette 相对剖面 \(L_2\) 与流量误差均小于 \(10^{-6}\)；
- Poiseuille 两种误差均小于 0.5%；
- 采样点中最大相对质量漂移小于 \(10^{-9}\)；
- 横向速度与流向不均匀度相对 U 均小于 \(10^{-8}\)；
- 必须实际达到稳态判据。

若 Poiseuille 误差高于数值精度底限，程序计算

\[
p=\frac{\ln(E_{coarse}/E_{fine})}{\ln(H_{fine}/H_{coarse})}
\]

并用 \(1.8\le p\le2.2\) 检查二阶收敛。单网格或误差过小时明确标记未检验阶数。Couette 的稳态解是线性函数，误差主要来自停止容限及舍入，不能用其接近零的误差强行推断空间阶数。TRT 的特定参数也可能使 Poiseuille 误差接近精度底限。

流量采用流体中点求积，因此流量误差同时含速度场误差和求积误差。另用邻近壁面的三点二次外推估计真实壁面处的滑移量，只作诊断，不把最外层流体点速度当成壁面速度。

MRT 的 Lallemand–Luo 默认参数在 H=16、rho0=1.7 的对照中测得剖面误差约 0.5124%，超过 0.5% 阈值；横向速度诊断也未达标。H=32 时误差降至约 0.1281%，全部诊断通过。程序保留同一套阈值，因此包含前一个粗网格的 MRT 扫描会保存结果并以退出码 1 报告验收失败。这是精度验收的有效输出，不能把所有稳定运行都标为通过。

## 本次实现修正与适用边界

旧 `moving_wall_bounce_back` 用迁移后尚不完整的分布求密度，且入射方向的修正符号相反。修正前 \(U=0.02\) 的 Couette 测试得到顶部流体速度约 -0.0166667；本次修复了符号和密度重构，并用四面壁、正负切向速度的平衡态不变性测试防止回归。

`couette_flow` 和 `periodic_channel` 的新预设采用半格反弹，特征高度因此由原节点式定义改为 \(N_y\)。已保存的旧 JSON 按显式边界类型加载，不会自动转换成半格格式。旧节点式反弹与非平衡外推仍有自己的几何/误差特性，不能直接拿这里的半格坐标比较。

这一关验证的是直通道、等温、低 Ma、充分发展流动。通过结果还不能验证入口/出口、方腔角点、热通量边界、曲线边界或高 Re 稳定性。

## 下一步实践

先观察 `profiles.png` 中两种剖面的差别，再阅读 `summary.json` 的 `relative_l2_error` 与 `orders`。解释为何 Poiseuille 的误差具有二阶下降趋势，而 Couette 不应强求相同趋势。随后改变 \(\tau\) 或碰撞模型，对比外推滑移和网格误差，作为进入顶盖驱动方腔前的边界验证记录。
