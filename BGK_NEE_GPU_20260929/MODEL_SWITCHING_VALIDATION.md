# 模型切换验收记录

生成时间：2026-09-29T21:40:58+08:00。

自动测试：22 项模型测试 + 60 项既有回归测试 = 82 项通过。原生 Qt 界面提交 12 组真实任务，覆盖 3 后端 × 2 碰撞模型 × 2 边界方案；这些 101 步短任务均正确标记为步数用尽，未冒充收敛。

## 三后端完整收敛对照

N=24，Re=100，U=0.04。各组 CPU、CuPy、CUDA 收敛步数一致。

| 模型 | 边界 | 收敛步数 | 三后端最大场量绝对差 |
|---|---|---:|---:|
| BGK | nee | 7059 | 0 |
| BGK | halfway | 7271 | 0 |
| MRT | nee | 7037 | 7.77156e-16 |
| MRT | halfway | 7222 | 7.77156e-16 |

## 低 Re 方腔基准

N=64，Re=100，U=0.04，CUDA 后端。Ghia 比较使用实际物理中心线及 15 个内部表格点；速度以顶盖速度归一化。

| 模型 | 边界 | 收敛步数 | 相对质量漂移 | u 最大绝对差 | v 最大绝对差 |
|---|---|---:|---:|---:|---:|
| BGK | nee | 16148 | 0.00310489 | 0.00830417 | 0.00637478 |
| BGK | halfway | 16184 | 1.55431e-12 | 0.00484788 | 0.00305705 |
| MRT | nee | 16142 | 0.00297668 | 0.00836704 | 0.00642707 |
| MRT | halfway | 16182 | 0 | 0.00444286 | 0.00323749 |

表中的质量漂移 0 表示保存的双精度数值为 0，不代表无限精度证明。NEE 的质量漂移继续如实报告，没有补偿。此次不验证高 Re 稳定区间、临界 Re 或网格无关性。

## 证据

- [模型数值及源码指纹](validation/correctness_models.json)
- [新增测试](validation/model_unit_tests.xml)
- [既有回归](validation/models_full_tests.xml)
- [界面验收与截图索引](validation/models_ui.json)
- [命令行 MRT＋HBB 配置检查](validation/models_cli_dry_run.json)
- [原 BGK/NEE CUDA 完整回归](validation/correctness_fused.json)
- [原 BGK/NEE CuPy 完整回归](validation/correctness_array.json)

三个后端的当前源码验证门禁均通过。原 18 个冻结参考文件校验通过；正式工作空间未提交测试任务。界面另检查 1440×940、960×680，以及 HBB 流场坐标、模板恢复、全零质量漂移显示。

方法、参数与文献说明见 [MODEL_SWITCHING.md](MODEL_SWITCHING.md)。
