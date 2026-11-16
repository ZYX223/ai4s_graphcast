# GraphCast 训练与推理

版本 `graphcast-20261114`，固定 **Operational 0.25°／13层／batch=1**，网格721×1440。
BW1100和H100的推理、训练各使用1张卡，统一入口为 `scripts/run.sh`。

| 平台 | 精度 | 已部署源码根目录 |
|---|---|---|
| BW1100 | FlagOS JAX fl_only，RNE＋XF32 | `/recovery/graphcast-delivery/releases/graphcast-20261114` |
| H100 | TF32，无显式RNE | `/data2/graphcast-delivery/releases/graphcast-20261114` |

1. [环境与材料](docs/ENVIRONMENT.md)：平台条件、镜像恢复与材料路径。
2. [运行说明](docs/RUN.md)：两平台通用命令、参数、checkpoint续训及输出。
3. [结果与验收](docs/RESULTS.md)：验证范围、参考、精度及计时定义。
4. [训练日程](docs/FULL38.md) · [更新与交付](docs/UPDATING.md)。

完整40步使用 `--workload infer-full40`：先做单步检查，再执行3轮独立40步，各轮预热5步。
自定义推理使用 `infer`；训练使用 `train` 或固定预设 `train-full38`。
两平台参数相同，切换 `--platform`、本机路径与物理卡号即可；单步检查的内部配方和结果子目录有所不同。

代码阅读从 [run.sh](scripts/run.sh) → [delivery.py](scripts/delivery.py) → [宿主机入口](src/graphcast_bench/delivery/cli.py) → Docker → [容器分发](src/graphcast_bench/delivery/runtime.py) 开始。

| 平台 | 阶段编排 | 推理入口 | 训练入口 |
|---|---|---|---|
| BW1100 | [operational_bw1100.py](src/graphcast_bench/delivery/operational_bw1100.py) | [inference/entry.py](src/graphcast_bench/inference/entry.py) | [training/entry.py](src/graphcast_bench/training/entry.py) |
| H100 | [operational_h100.py](src/graphcast_bench/delivery/operational_h100.py) | [inference/h100/entry.py](src/graphcast_bench/inference/h100/entry.py) | [training/entry.py](src/graphcast_bench/engines/operational_h100/training/entry.py) |

推理入口继续进入各自的runner和rollout；训练入口进入逐阶段、逐更新循环。训练日程与计时说明见[FULL38.md](docs/FULL38.md)。

交付包含运行、编译、校验所需源码和来源记录。开发测试、设计计划、详细代码导读与历史实验另行归档，不随运行包分发。
`release.json`固定当前源码身份，`configs/artifacts.lock.json`固定材料身份；GPU结果必须对应实际执行SHA，不能由CPU测试推定。
