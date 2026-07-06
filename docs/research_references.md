# 相似模型调研记录

本项目的第一版实现以 D2Q9-LBGK 不可压缩低马赫数流动为核心。网络调研后，最接近本需求的现有模型是“二维 D2Q9 通道绕圆柱/方腔类基准算例 + NumPy 实现 + 反弹障碍物边界”的组合。

## 参考基线

1. Benchpress 的 `Lattice Boltzmann D2Q9` 示例提供了 Jonas Latt 风格的 Python/NumPy D2Q9 通道绕圆柱模型，包含 D2Q9 权重、速度、opposite、BGK 松弛、圆柱障碍物 mask 和反弹处理。当前项目借鉴其模型层级，但改为 `(NY, NX, 9)` 数组布局，并避免对非周期边界滥用 `np.roll`。
2. D2Q9-BGK 的基本碰撞、平衡态和宏观量关系来自标准 LBM 公式。
3. 边界处理的第一版选取稳健核心：周期、静止反弹、运动壁反弹、非平衡态外推、充分发展出口。Zou-He、滑移、镜面、混合反射和质量修正出口保留为第二版。
4. Streamlit 用于前端参数设置、运行控制、实时指标和结果下载。

## 本项目的个性化改动

- 支持 UI 选择 `lid_driven_cavity`、`poiseuille_channel`、`couette_flow`、`periodic_channel`、`cylinder_flow`、`custom`。
- 增加稳定性自动诊断：`tau`、`omega`、`Ma`、网格/Re 匹配、周期边界成对检查。
- 增加参数推荐器：按低马赫数和目标 `tau` 自动推荐 `U_ref`，并提示所需特征网格长度。
- 增加残差、收敛因子 `q`、滚动 `q_avg`、质量漂移、最大速度监控。
- 结果统一保存为 `results.npz`、`config.json`、`residual_history.csv` 和五类图像。

