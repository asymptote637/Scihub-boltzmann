# BGK / MRT 与 NEE / 半格点反弹

入口保持为 `open_dashboard_windows.bat`。重新打开工作台后，在“新建仿真”选择“碰撞模型”和“壁面条件”，再点击“开始计算”。两项独立选择，三个后端均执行相应模型。

| 碰撞模型 | 非平衡外推 NEE | 半格点反弹 HBB |
|---|---|---|
| BGK 单松弛 | 原模型，保留历史回归对照 | 单松弛＋半格点反弹 |
| MRT 多松弛 | 多松弛＋非平衡外推 | 多松弛＋半格点反弹 |

CPU、GPU 融合 CUDA、GPU CuPy 对照版均使用双精度。默认仍是 BGK＋NEE；旧模板缺少新字段时按原模型读取。这里的“切换”是为新算例选择模型，每次由静止初态开始；没有实现运行中改变碰撞模型、改变壁面位置或跨模型续算。

## 碰撞模型

BGK 保留原式：

\[
f_i^*=f_i-\omega(f_i-f_i^{eq}),\qquad \omega=1/\tau.
\]

MRT 使用 D2Q9 正交原始矩基，矩顺序为
\((\rho,e,\epsilon,j_x,q_x,j_y,q_y,p_{xx},p_{xy})\)：

\[
m=Mf,\quad m^{eq}=Mf^{eq},\quad
f^*=f-M^{-1}S(m-m^{eq}),
\]

\[
S=\operatorname{diag}(0,s_e,s_\epsilon,0,s_q,0,s_q,s_\nu,s_\nu),
\qquad s_\nu=\frac{1}{\frac12+3\nu}.
\]

密度和动量是守恒矩；两个剪切矩使用由黏度决定的松弛率。三个非流体模态参数可在界面编辑，要求严格满足 \(0<s<2\)，默认值为 \(s_e=1.64,s_\epsilon=1.54,s_q=1.9\)。这是本实现的默认数值选择，不是针对任意 Re 的最优参数或稳定性保证。BGK 模式下这三个输入框停用。矩阵与实现位于 `cavity_models.py`；CUDA 在格点上完成矩变换后再迁移。

矩空间模型依据 [Lallemand 与 Luo（2000）原论文](https://ntrs.nasa.gov/citations/20000046606)。测试包括各模态衰减、质量/动量守恒，以及所有非守恒松弛率相同时退化为 BGK；通过周期剪切波衰减另行测量实际黏度。

## 壁面位置、尺度与角点

NEE 保留原节点式边界，壁面就在最外层格点上。HBB 将全部存储节点作为流体节点，壁面位于数组外半个格距。对于方腔：

| 项目 | NEE | HBB |
|---|---|---|
| 格子腔长 \(L\) | \(N-1\) | \(N\) |
| 归一化节点坐标 | \(j/(N-1)\) | \((j+1/2)/N\) |
| 黏度 | \(U(N-1)/Re\) | \(UN/Re\) |
| 残差求和节点 | 去掉边界节点的内部流体点 | 全部流体点 |

所以切换边界后，界面的 \(L,\nu,\tau_\nu\) 会随之更新。预览、最终 NPZ、中心线 CSV 都使用实际坐标。HBB 最外层节点的速度是流体速度，不等于壁面速度，不能把它当成壁面滑移直接判断。

对来源落在域外的未知入射方向 \(i\)，HBB 使用本流体节点的碰撞后反向分布：

\[
f_i(\boldsymbol x,t+1)=f_{\bar i}^{*}(\boldsymbol x,t)
 +6w_i\rho(\boldsymbol x,t)\,\boldsymbol e_i\cdot\boldsymbol u_w.
\]

侧壁与底壁静止，顶盖沿 x 方向移动，顶盖速度沿用原来的线性启动。直接来自域内的方向保持普通迁移。对恰好穿过顶角的对角格链，明确采用上下壁优先约定：与顶盖相交的入射对角方向采用顶盖修正。每条缺失格链只反射一次；此规则在 CPU、CuPy 与 CUDA 中一致，写入 `halfway_corner_policy`，并用独立逐格链测试和总质量测试检查。顶盖与静止侧壁交点本来具有不连续壁速，角点附近的精细结构仍须网格敏感性检验。

半格点壁面的经典讨论可见 [Zou 与 He 原论文](https://arxiv.org/abs/comp-gas/9611001)。本实现限定于无体力、无障碍物的顶盖方腔，不接受混合 NEE/HBB 壁面。

## 模板、档案与对比

模型、壁面和 MRT 参数随模板、队列、配置、运行摘要和导出指标 CSV 保存；复用参数时会恢复这些选项。默认算例名称及流场标题标明 BGK/MRT 和 NEE/HBB。

BGK 与 MRT 在相同壁面、相同物理参数及相同网格上可以直接作场差，界面会注明这是方案差异。NEE 与 HBB 的节点坐标不同，仍拒绝直接作节点差；可以使用归一化中心线图比较。未收敛结果继续标明状态，不将它们标成稳态比较。

## 命令行

```powershell
Set-Location -LiteralPath 'D:\BaiduSyncdisk\我的文件\Leon\科研\BGK_NEE_GPU_20260929'
$python = 'D:\LBM\envs\bgk-nee-gpu-20260929\Scripts\python.exe'
& $python run_gpu.py --backend fused --collision MRT --boundary halfway --grid 64 --re 100 --max-iter 50000
```

`--collision BGK|MRT` 与 `--boundary nee|halfway` 独立组合。MRT 可再指定 `--mrt-s-e`、`--mrt-s-eps`、`--mrt-s-q`。`--dry-run` 只检查，不推进时间步。

## 验证与复现

新功能的数值证据为 `validation/correctness_models.json`。它记录源码 SHA-256、独立物理/格链测试、三个后端完整收敛对照、四种组合的 Re100 方腔中心线比较。基准数据使用原冻结参考中的 Ghia 表格，对应 [Ghia、Ghia 与 Shin（1982）](https://doi.org/10.1016/0021-9991(82)90058-4)。HBB 比较时只使用处在实际节点跨度内的内部表格点，不把近壁流体值当成壁面值。

原 BGK＋NEE 两个 GPU 后端的完整回归仍由 `validation/correctness_array.json` 和 `validation/correctness_fused.json` 控制；新模型须同时通过新增验证。修改数值核心后旧凭据不会放行。界面验收记录为 `validation/models_ui.json`，使用独立工作空间提交真实计算。

```powershell
& $python -X utf8 -u validate_gpu.py --backend fused --full
& $python -X utf8 -u validate_gpu.py --backend array --full
& $python -X utf8 -u validate_models.py
& $python -m pytest tests -q --basetemp validation/pytest-models-recheck -p no:cacheprovider
& $python -X utf8 -u verify_models_ui.py
```

原 `reference_cpu/` 的 18 个冻结文件保持完整，可继续核验；新增 CPU 模型通过独立子类实现。修改前的入口、核心文件、界面和原验证凭据备份于 `validation/model_extension_backup_20260929_201337/`。旧报告及打包交付件仍代表其生成时的版本。

上述验证范围是模型实现、后端一致性和低 Re 基准；不据此宣称高 Re 已稳定、临界 Re 已确定或已经网格无关。
