# GraphCast · BW1100 训推代码

本包包含官方 JAX GraphCast 的 BW1100 推理优化、短程训练和训练 RNE 融合实现。
支持 operational 0.25°、13 层模型；FP32 存储，RNE＋XF32 计算。

- [运行说明](docs/RUN.md)：准备配套材料、进入可复用环境、构建、短检查、完整训推及结果验收。
- [版本说明](docs/CHANGELOG.md)：版本内容与验证状态。

## 使用入口

| 文件 | 用途 |
| --- | --- |
| `scripts/reproduce.sh` | 选择卡号后自动完成构建、训推和精度对照 |
| `scripts/inference.py` | 推理及 profile |
| `scripts/train.py` | 训练与 checkpoint 接续 |
| `scripts/build_kernels.py` | 构建推理和训练原生库 |
| `scripts/validate.py` | 文件完整性、运行完成性及 H100 精度对照 |
| `environment/paths.example.json` | 数据和原生库路径模板 |
| `environment/container.sh` | 自动创建或复用当前账户的 Docker 环境 |

本目录提供源码；本机完整交付目录 `/recovery/graphcast-testuser-20261113/delivery/` 已包含固定镜像、运行时快照、输入数据与 H100 参考结果。
先按运行说明启动 Docker，再在容器内检查包的完整性：

```bash
python scripts/validate.py release "$PWD"
python scripts/validate.py code
```

## 代码结构

```text
scripts/                    运行和验证入口
configs/                    案例、平台和优化配置
src/graphcast_bench/
  data/                     数据读取、校验和上传
  model/                    官方模型装配、参考图和重计算
  inference/                推理流程和计时
  training/                 训练日程、梯度累积和恢复
  optimizations/            推理优化、训练优化、RNE lowering
  platforms/                设备配置、精度路由和库构建
  validation/               执行与结果检查
kernels/                    原生路由和 RNE 核
third_party/graphcast/       官方源码、版本和许可证
environment/                依赖清单和路径模板
tests/                      核心回归测试
docs/                       运行与版本说明
release.json                文件清单和校验值
```

阅读实现可从 `inference/entry.py`、`training/entry.py` 开始，再进入对应的 `runner.py`。
训练 RNE 位于 `optimizations/rne/`，在自动微分后的 lowering 阶段对 GEMM 操作数舍入。
官方 GraphCast 源码版本和逐文件校验记录在 `third_party/graphcast/UPSTREAM.json`；
训练数据准备依赖的两个同版本官方模块另记录在 `third_party/graphcast/TRAINING_UPSTREAM.json`。
