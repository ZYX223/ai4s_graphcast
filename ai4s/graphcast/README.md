# GraphCast · NVIDIA H100 Native baseline

Baseline 复现分支：`zyx/nvidia-native-baseline`。本分支只新增固定配置、
启动入口和使用说明，不复制或改写训练/推理主循环。多平台开发仍在
[zyx/flaggems-profile-routing](https://github.com/ZYX223/ai4s_graphcast/tree/zyx/flaggems-profile-routing)。

这里的“N 卡”指 **NVIDIA**，本次固定为 **单张 H100 80GB HBM3**，
不是任意 N 张卡。两项均为 Native PyTorch、eager、FP32 storage + TF32 矩阵计算；
**不使用 FlagGems、BF16、torch.compile 或 CUDA Graph**。

## 先看哪些文件

| 内容 | 文件 |
| --- | --- |
| 一份固定训练/推理配方 | [h100_native_tf32.json](configs/baseline/h100_native_tf32.json) |
| 统一启动入口（不实现模型算法） | [run_nvidia.py](scripts/baseline/run_nvidia.py) |
| 环境、数据、诊断、E2E/MFU 和验证范围 | [使用说明](docs/nvidia-baseline-handoff.md) |
| 训练主循环 | [run_official_epoch_training.py](scripts/training/run_official_epoch_training.py) |
| 训练模型：G2M → 16 步 Mesh → M2G | [training_model.py](src/graphcast_compat/training_model.py) |
| 推理自回归循环 | [run_operational_40step_inference.py](scripts/inference/run_operational_40step_inference.py) |
| 推理模型 | [google_small.py](src/graphcast_compat/google_small.py) |
| 最新训练基线原始口径与参数 | [冻结记录](docs/training-native-baseline-20261012.md) |

`quickstart train`、`scripts/training/legacy/` 和 `scripts/training/lib/`
保留兼容用途，**不用于本分支 baseline**。其他平台与 FlagGems 模块仍保留，
但下列入口不会启用；不做大范围删代码，以维持已验证的依赖关系与源码哈希。

## 环境和资产

已测训练环境：`nvcr.io/nvidia/pytorch:25.11-py3`、
Torch `2.10.0a0+b558c986e8.nv25.11`、CUDA 13.0、
NumPy 1.26.4、SciPy 1.16.3。在已准备的环境内运行；
本分支不自动创建容器、下载数据、安装 FlagGems 或占用其他人的设备。
[requirements.txt](requirements.txt) 是通用依赖范围，不是精确性能环境锁文件。

需要另行提供：转换后的 PyTorch 权重、三份 normalization stats；
训练还需要 ERA5 corpus manifest 及其引用的数据文件；
推理需要 inputs、rollout forcings、target template。
这些资产和诊断参考均不在 Git 内。必须使用原 baseline 对应资产，不能用随机权重、
合成输入或 quickstart 小样本替代。格式和诊断用法见使用说明。

## 训练：固定 38 次更新

- 单卡 MP=1/DP=1；global batch=2，microbatch=1，梯度累积 2 次。
- 6 次 R1 warmup + 10 次 R1 + R2…R12 各 2 次，共 340 sample-leads。
- 保留当前 checkpoint/chunk、grouped gather=4、每 microbatch 输入驻留优化。
- 每轮从相同预训练权重开始，优化器重新初始化；不接续上一轮。

从仓库根目录先执行 `cd ai4s/graphcast`，再预览（不要求资产存在，不导入 Torch、不运行 GPU）：

```bash
python scripts/baseline/run_nvidia.py train --dry-run \
  --gpu 0 --weights /assets/weights.pt --stats /assets/stats \
  --corpus /assets/era5/corpus.json --output-dir /runs/train-r1
```

去掉 `--dry-run` 才会运行。对三轮分别使用新的目录：

```bash
for r in 1 2 3; do
  python scripts/baseline/run_nvidia.py train \
    --gpu 0 --weights /assets/weights.pt --stats /assets/stats \
    --corpus /assets/era5/corpus.json --output-dir "/runs/train-r$r" || break
done
```

物理卡号以**当前容器可见设备**为准，也可填完整 GPU UUID；启动前确认空闲。
原训练测量使用 GPU 本地 NUMA1。新机器先检查拓扑，再酌情加入
`--numa-node 1`，不要直接照抄 NUMA 编号；未绑定时应在结果中注明。

## 推理：40 步真实自回归

40 步是 benchmark 配方，不是模型写死的上限；支持 `--steps N`，
要求 forcing 恰好有 N 步。默认每步 6 小时、3 次不计时 warmup。
H100 使用 `baseline` 模型执行配置，不开启国产平台的投影分解/full-grid 路径。

```bash
python scripts/baseline/run_nvidia.py infer --dry-run \
  --gpu 0 --weights /assets/weights.pt --stats /assets/stats \
  --inputs /assets/inputs.nc --rollout-forcings /assets/rollout_forcings.nc \
  --target-template /assets/target_template.nc --steps 40 \
  --output-dir /runs/infer-r1
```

去掉 `--dry-run` 执行；正式统计同样用三个独立进程和全新的输出目录。
入口会显式清理历史 FlagGems/平台/TF32 覆盖变量、指定 Native 和单卡，
运行前验证 H100 型号、资产存在及冻结源码哈希；不自动下载缺失资产。

## 结果与口径

| 项目 | 三轮 E2E 中位数 | 有效吞吐 TFLOP/s/GPU | MFU（495 TFLOP/s） | 证据状态 |
| --- | ---: | ---: | ---: | --- |
| 训练，38 updates | 659.728441 s | 44.615685 | 9.013270% | 最新已测，核心源码与本分支一致 |
| 推理，40 leads | 11.808169 s | 97.752988 | 19.748078% | 2026-09-06 历史结果；当前分支需设备复测 |

历史训练约 10.01% 不是当前稳定复现值，应与最新结果分开记录。
本次代码整理没有新跑 H100 性能；CPU 测试不能替代实机复测。

只采用一套 MFU：`有效模型 FLOPs / E2E秒 / 单卡峰值 × 100%`。
训练不把 checkpoint 重算计入有效 FLOPs，不减去数据等待。
训练与推理各自的 E2E 边界、精确数值及计算代码见
[使用说明](docs/nvidia-baseline-handoff.md#e2e-与-mfu)。

输出包含 `baseline-launch.json`（展开命令/环境/源码哈希/硬件）和
`baseline-exit.json`。训练保留 `metrics/`、`checkpoints/`；
推理保留 `report.json`、`prediction.nc`。
**退出成功、finite 和 MFU 都不等于正确性通过**；还需执行使用说明里的诊断。

## 不用设备的代码检查

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python -m pytest tests -q
```

源模型来自本仓库已发布提交 `50fa1808928ff8e751fe30b13e7659e30d686ef3`。
137 文件冻结清单仍有效；本分支只增加可复现入口/配置/测试/文档。
代码许可证见 [LICENSE](../../LICENSE)；权重、ERA5 和上游资产的授权需分别确认，
本仓库许可证不替代这些资产的使用条件。
