# GraphCast：官方 JAX / H100 与海光

曙光项目分支 `zyx/sugon-graphcast`，对应 2026-10-12～16 周报中的官方 JAX GraphCast 推理测试。

| 路径 | 40 步耗时中位数 | 对官方 JAX 严格 FP32 的整体相对 MAE | 状态 |
| --- | ---: | ---: | --- |
| H100 JAX TF32 | 14.46 秒 | 1.5146%～1.5825% | 三轮通过 |
| 海光 BW1200 JAX FP32 | 51.97 秒 | 见逐轮明细 | 三轮通过 |
| 海光 BW1200 JAX XF32 | 32.12 秒 | 约 22.79%～22.81% | 三轮未通过 |
| 海光 BW1200 JAX RNE＋XF32 | 42.73 秒 | 1.5926%～1.8265% | 三轮通过 |

这些是整理前已完成的周报测量。整理后的代码验证单独记录，不混入上述统计。H100 相对当前海光 RNE＋XF32 的速度为 2.955×；两平台各三轮的九组直接比较，整体差异为 0.4839%～0.6893%。两平台的 JAX/编译器、CPU 主机及依赖不同，这是当前部署的实测比较。

**模型版本：**Google DeepMind GraphCast v0.2，提交 `08cf73625c9d12bd9aaa038868bcb2fe488f2a22`。仓库包含推理所需的 16 个官方 Python 模块，文件字节未修改，附原许可证及逐文件 SHA-256。模型通过官方 Haiku/JAX 实现执行。

**工作负载：**operational 0.25°，721×1440 网格，13 个气压层，起报 2021-01-01 06:00，每步 6 小时，共 40 步。参数、输入、输出均以 FP32 存储。

## 从哪里开始

- [复现步骤、数据和环境位置](docs/REPRODUCE.md)
- [性能、误差口径与原始结果](docs/RESULTS.md)
- [XF32 精度问题、舍入实现与限制](docs/XF32.md)
- [整理后的验证记录](docs/VALIDATION.md)

在仓库根目录执行；`DATA` 是按复现说明准备的数据目录，输出目录必须不存在：

```bash
# H100：每轮预热 5 步，再计时 40 步，默认三轮
python3 scripts/run.py benchmark --platform h100 --device 7 \
  --data "$DATA" --output "$OUT/h100-tf32"

# 海光：同一进程按轮换顺序测 FP32、XF32、RNE＋XF32，各三轮
python3 scripts/run.py benchmark --platform hygon --device 9 --mode all \
  --data "$DATA" --output "$OUT/hygon-three-modes"
```

## 目录

| 目录 | 内容 |
| --- | --- |
| `third_party/graphcast/` | 固定版本的官方模型、来源和许可证 |
| `src/graphcast_bench/` | 公共推理、计时、校验、误差计算及海光路由审计 |
| `scripts/` | 容器启动、数据准备、环境探针、RNE 构建与输出比较 |
| `configs/`、`environment/` | 案例清单、两平台默认配置、镜像及依赖版本 |
| `kernels/xf32_rne/` | 已验证的 HIP 舍入核、rocBLAS 路由和权重指纹 |
| `tests/` | CPU 正确性与输入完整性检查 |
| `examples/rounding/` | 中文注释的最近偶数舍入 C++ Demo |
| `evidence/20261016/` | 周报原始测量的小型摘要和逐层指标 |
| `evidence/consolidation-20261016/` | 整理后两平台回归、探针和审查修正验证 |

数据、权重、镜像、编译库、完整预测和跟踪文件保存在外部目录，不提交 Git。项目当前交付范围是推理与基础验证，完整训练流程仍待开展。
