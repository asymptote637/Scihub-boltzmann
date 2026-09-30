# 第 1 关：周期剪切波黏性标定

## 1. 本关要回答的问题

材料给出 D2Q9-BGK 的黏性关系

\[
\nu=c_s^2\left(\tau-\frac12\right)\delta t,
\qquad c_s^2=\frac13.
\]

本关不把它当作待背公式，而是把现有求解器当成一个“数值实验装置”，直接测量它产生的运动黏度，并检验 D2Q9 的矩条件。

## 2. 物理模型

取二维周期区域，初始速度为

\[
u_x(y,0)=U_0\sin(ky),\qquad u_y=0,\qquad \rho=1,
\]

其中

\[
k=\frac{2\pi}{N_y}.
\]

低马赫、线性剪切模满足

\[
\frac{\partial u_x}{\partial t}=\nu\frac{\partial^2u_x}{\partial y^2},
\]

因而解析解为

\[
u_x(y,t)=U_0\exp(-\nu k^2t)\sin(ky).
\]

将数值速度投影到初始正弦模，得到幅值 \(A(t)\)。于是

\[
\ln A(t)=\ln A_0-\nu k^2t,
\qquad
\nu_{fit}=-\frac{\mathrm{slope}[\ln A(t)]}{k^2}.
\]

这里没有固壁，所以如果 \(\nu_{fit}\) 与理论值不符，优先检查平衡态、碰撞、周期迁移、宏观重构或拟合过程，而不是把问题归因于边界格式。

## 3. 参数预报

默认设置为 \(N_x=N_y=64\)、\(U_0=0.02\)，因此

\[
Ma=\frac{U_0}{c_s}\approx0.0346.
\]

运行前先自行填写下表的理论列：

| \(\tau\) | \(\omega=1/\tau\) | \(\nu_{theory}=(\tau-0.5)/3\) | 预期衰减快慢 |
|---:|---:|---:|---|
| 0.6 |  |  |  |
| 0.8 |  |  |  |
| 1.0 |  |  |  |

## 4. 运行

在仓库根目录执行：

```powershell
$env:PYTHONPATH="."
.\.venv\Scripts\python.exe -m training.shear_wave_viscosity
```

输出目录为 `results/training/01_shear_wave/`：

- `fit_summary.json`：矩残差、理论黏度、拟合黏度、质量漂移等；
- `amplitude_decay.csv`：每个采样时刻的实测幅值与解析幅值；
- `amplitude_decay.png`：半对数坐标下的数值/解析衰减曲线。

## 5. 验收标准

默认算例同时满足以下条件才算通过：

1. D2Q9 的 0–4 阶各向同性恒等式最大残差不超过 \(10^{-14}\)；
2. 平衡态的密度、动量和动量通量矩最大残差不超过 \(10^{-14}\)；
3. 三组 \(\tau\) 的 \(|\nu_{fit}/\nu_{theory}-1|<0.2\%\)；
4. 最大相对质量漂移小于 \(10^{-12}\)；
5. \(\ln A(t)\) 近似直线，且 \(\tau\) 越大衰减越快；
6. 密度扰动保持在数值舍入量级，没有把横向剪切模污染为明显压缩波。

## 6. 必做探索

### A. 网格/波数效应

分别运行：

```powershell
.\.venv\Scripts\python.exe -m training.shear_wave_viscosity --tau 0.8 --ny 32 --steps 250 --output-dir results/training/01_shear_wave_n32
.\.venv\Scripts\python.exe -m training.shear_wave_viscosity --tau 0.8 --ny 64 --steps 400 --output-dir results/training/01_shear_wave_n64
.\.venv\Scripts\python.exe -m training.shear_wave_viscosity --tau 0.8 --ny 128 --steps 1000 --output-dir results/training/01_shear_wave_n128
```

比较相对黏性误差。注意：这里一个周期始终铺满整个计算域，改变 \(N_y\) 同时改变格子波数 \(k\)。请解释为什么长波极限更接近 Chapman–Enskog 的连续宏观极限。

### B. 低马赫限制

固定 \(\tau=0.8,N_y=64\)，将 `--amplitude` 改成 `0.01`、`0.05`、`0.10`。记录 \(Ma\)、黏性拟合误差和最大密度扰动，判断何时开始出现可观的非线性/弱可压效应。

### C. 碰撞模型对照

在其他参数不变时追加：

```powershell
.\.venv\Scripts\python.exe -m training.shear_wave_viscosity --tau 0.8 --collision-model TRT --output-dir results/training/01_shear_wave_trt
.\.venv\Scripts\python.exe -m training.shear_wave_viscosity --tau 0.8 --collision-model MRT --output-dir results/training/01_shear_wave_mrt
```

先判断三者是否应给出同一个剪切黏度，再根据矩空间中剪切模的松弛率解释结果。此对照只验证体相剪切模，还不能据此宣称三种碰撞模型在有壁问题中完全等价。

## 7. 本关研究记录模板

完成后用不超过一页回答：

1. 本算例隔离了哪些数值模块，又刻意排除了哪些误差源？
2. \(\tau\) 与衰减率、理论黏度之间的实测关系是什么？
3. 网格/波数扫描是否支持连续长波极限？
4. 提高 \(Ma\) 后最先偏离的是黏性拟合、密度扰动还是模态纯度？
5. 当前证据能验证什么，不能验证什么？

把 `fit_summary.json`、误差比较表和你的解释发回后，再进入第 2 关 Couette/Poiseuille 边界验证。

