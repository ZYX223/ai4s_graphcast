# NVIDIA H100 baseline 使用说明

本分支提供 baseline 复现入口，不是另一套模型或新性能优化。
完整计算代码沿用本仓库已公开的 `50fa180`；正式 Native eager TF32 配方见
[配置](../configs/baseline/h100_native_tf32.json) 和 [首页](../README.md)。
本文命令均从 `ai4s/graphcast/` 目录运行。
本次不启动 GPU 测试；不同机器和运行环境的性能需要分别验证。

## 结构与执行

```text
scripts/baseline/run_nvidia.py + configs/baseline/h100_native_tf32.json
  ├─ train → training/run_training.py（平台/配置适配）
  │          → torchrun，1 rank / 1 GPU
  │          → training/run_official_epoch_training.py（AdamW 主循环）
  │          → src/graphcast_compat/training_model.py
  └─ infer → inference/run_operational_40step_inference.py（自回归计时）
             → src/graphcast_compat/google_small.py
```

二者共享图构造、数据规范化、精度/后端等模块，训练另含 autograd、loss、
梯度累积与 checkpoint。共享同一模型参数结构，不表示训练和推理是同一个函数。

入口只暴露 GPU、资产位置、输出位置、可选 NUMA、推理步数和 dry-run。
训练配方固定，不接受任意透传参数；更改 batch、优化器、chunk、精度、compile
应在另一实验配置中验证，不要继续标成此 baseline。
原训练配置的 peak=0 保留，避免改变已测 engine contract；训练 MFU 在下文
按统一的 495 TFLOP/s 后处理。推理报告中 MFU 是比例，展示百分数需乘 100。

入口运行前清除 `GRAPHCAST_*`、`FLAGGEMS_*`、`FLAGOS_REPLACE`、
`PYTHONPATH`、TF32 强制覆盖、CUDA visibility/launch-blocking 和 allocator
旧设置，再设置已记录的配置。清理仅影响子进程，不修改当前 shell。
不接受在已有 torchrun rank 内再次调用。不会自动挑卡或终止其他任务。
原 GPU 接口/封装仍保留在共同源码中，但 baseline 不导入/启用 FlagGems。
依赖应安装在所选 Python 环境内，不能只依赖旧实验目录的 `PYTHONPATH` 注入；
预检记录 Torch/CUDA/Python 和主要 Python 包版本，不擅自升级依赖。

设备必须是已记录的 NVIDIA H100 80GB HBM3 全卡，单卡峰值固定 495。
PCIe、MIG、其他 NVIDIA 型号不能套用这个分母，入口会拒绝不匹配型号。
预检验证设备/环境，不检查 GPU 是否被其他任务占用，也不验证 TF32 指令占比；
运行者必须另查 GPU 使用情况和拓扑，并保存频率、功耗、温度及主机负载记录。

## 环境与输入

固定训练环境详情在[原始训练记录](training-native-baseline-20261012.md)。
训练和历史推理均使用 Torch `2.10.0a0+b558c986e8.nv25.11` / CUDA 13.0。
基准镜像为 `nvcr.io/nvidia/pytorch:25.11-py3`；训练记录的镜像 ID：
`sha256:06faed719d1bbd31d7053c96d0c94fce4f2b5f92fd95f07d60ec323c9744f118`。
这不是一个可直接拿来 pull 的 registry RepoDigest。另需核对实际安装包版本；
容器标签和宽范围 requirements 不能单独证明环境一致。

| 资产 | 要求 |
| --- | --- |
| 模型配置 | 仓库内 operational 0.25°、721×1440、mesh6、latent512、16 mesh steps、13 pressure levels |
| 初始化权重 | 对应 operational 25km 的转换后 PyTorch checkpoint；每轮一致；不随机初始化 |
| stats 目录 | `mean_by_level.nc`、`stddev_by_level.nc`、`diffs_stddev_by_level.nc` |
| 训练 corpus | 原 ERA5 2017-01-01…07 训练 manifest 及引用 shard；保留采样/时间窗口/变量和单位 |
| 推理 inputs | 同一固定起报案例的两帧历史输入，原记录起报时刻为 2021-01-01 06:00 |
| rollout forcings | N 帧、6h 间隔、对应同一起报时刻的未来 forcing；不是把一个输入重复 N 次 |
| target template | 对应模型输出变量、层次和空间坐标的单步模板 |
| 诊断参考 | 经来源/精度资格校验的双参考 NPZ + metadata；训练另需同配方 reference run |

数据、权重、诊断参考、完整环境镜像和训练 checkpoints 均需另行准备。
没有这些资产时，只能 dry-run 或做 CPU 回归，不能宣称复现性能/正确性。
统计数据的格式解析见 `official_training_corpus.py`、`official_training_data.py`；
推理 case/参考工具见 `scripts/inference/inference_reference.py` 和
[参考构建说明](../tools/inference_reference/README.md)。
不要使用 quickstart 自动下载的小样本代替正式训练 corpus。

## E2E 与 MFU

全程只报告同一公式：

`吞吐 = 有效模型 FLOPs / E2E秒 / GPU数`

`MFU(%) = 吞吐(TFLOP/s/GPU) / 495 × 100`

这里是沿用项目的 **Linear 主导 FLOPs 估算**，不是 profiler 枚举的全部硬件操作。
计 FMA=2 FLOPs，不将 checkpoint 重算重复放进有效工作量。
张量 FP32 存储并不意味着使用普通 FP32 峰值；本配方矩阵乘请求 TF32，故用 495。

### 训练

- 每样本每预报步前向 `F = 28.857094475776 TFLOP`。
- 前向加反向按此模型的 Linear 估算 `3F = 86.571283427328 TFLOP`。
- `sum(R) = 6×1 + 10×1 + 2×(2+3+…+12) = 170`。
- global batch=2，总有效工作 `3F × 2 × 170 = 29434.23636529152 TFLOP`。
- global batch 已包含两次 microbatch 累积，不再额外乘累积次数。
- E2E 为 **全部 38 次 update 的 `end_to_end_step_seconds` 之和**：
  暴露的 CPU 准备/等待、H2D、前反向、重算、optimizer 和同步均计入。
- 不计初始化、完整语料预加载、计时间隔外 checkpoint 写盘/管理开销；
  因此称“update E2E”，不是完整进程墙钟时间。学习率 warmup 的 6 次更新必须计入。

从完成的、无恢复重叠的本配方 run 目录复算：

```python
import json, math
from pathlib import Path
run = Path('/runs/train-r1')
rows = [json.loads(line) for file in sorted((run / 'metrics').glob('segment_*.jsonl'))
        for line in file.read_text().splitlines() if line.strip()]
assert len(rows) == 38
assert [r['optimizer_step'] for r in rows] == list(range(1, 39))
assert all(math.isfinite(r['loss']) and math.isfinite(r['grad_norm_before_clip']) for r in rows)
assert all(math.isfinite(r['end_to_end_step_seconds']) and r['end_to_end_step_seconds'] > 0
           for r in rows)
e2e = sum(r['end_to_end_step_seconds'] for r in rows)
work = sum(r['training_flops'] for r in rows) / 1e12
assert math.isclose(work, 29434.23636529152, rel_tol=1e-10)
print({'e2e_s': e2e, 'effective_tflops_per_gpu': work / e2e,
       'mfu_percent': 100 * work / e2e / 495})
```

三轮各自完整运行，取总 E2E 的中位数后计算 MFU；不可对每个 update 的 MFU
求算术平均，也不可减掉数据时间、只取 R12 或删掉慢更新。NUMA/主机争用会影响
CPU 预取和 E2E，必须一起记录。

### 推理

- 默认 `40F = 1154.28377903104 TFLOP`；其他步数为 `N×F`，无需改模型。
- 计时前完成加载权重、图初始化、CPU 预处理、初始 H2D 和 3 次 warmup。
- 计时内是依赖前一步预测的 N 步自回归、forcing 选择、FP32 state 更新、
  最终 finite 检查、最后一次 D2H 和设备同步。
- 输出物理量转换、NetCDF 写盘、离线诊断不计时；不是完整业务服务 E2E。
- 使用 `report.json: measurement.e2e_seconds`；不将逐步轨迹导出的诊断运行
  当作性能运行。默认只输出最后一帧，足够做固定 day10 诊断。

```python
import json
from pathlib import Path
report = json.loads(Path('/runs/infer-r1/report.json').read_text())
assert report['status'] == 'passed'
assert report['runtime']['flagos_mode'] == 'off'
assert report['precision']['float32_math_mode'] == 'tf32'
assert report['measurement']['performance_eligible'] is True
m = report['measurement']
print({'e2e_s': m['e2e_seconds'], 'effective_tflops_per_gpu': m['effective_tflops'],
       'mfu_percent': 100 * m['mfu']})
```

## 正确性诊断

成功退出/finite 只是第一层检查，不能替代对参考的诊断。

训练使用现有 `diagnose_training_loss.py`，对齐完整 1…38 更新、配方、数据、
初始化参数和 optimizer 状态，再计算
`100 × sum(abs(candidate_loss-reference_loss)) / sum(abs(reference_loss)) ≤ 5%`。
必须保留 metrics JSONL 对应的 `.start.json`，不得只复制 loss 列或用自己对自己比较。

```bash
python scripts/training/diagnose_training_loss.py /runs/train-r2 \
  --baseline /runs/train-r1 --start-step 1 --end-step 38 \
  --output-json /runs/train-r2-vs-r1.json
```

同平台重复对比验证一致性；国产平台对 H100 比较需另核验可比性状态。
这不是完整收敛或训练后 checkpoint 天气技能验证。

推理继续使用固定参考算法：六类气压层变量 × 13 层，各层 `MAE/mean(abs(ref))`
转百分数，再取 78 项等权平均；同时独立检查 Z500。
在本项目 day10 双参考协议下，两项均须 **严格小于 5%**，分别对合格的
Torch FP32/JAX FP32 参考执行。某单层超过 5% 不等同于这个整体门槛失败。
使用当前案例对应的 NPZ、metadata，校验权重、输入、stats、坐标、单位、时间和来源；
不能随意用候选结果重建参考再自行宣布通过。

```bash
python scripts/inference/diagnose_inference.py \
  --run-dir /runs/infer-r1 \
  --baseline /assets/reference/baseline.npz \
  --baseline-metadata /assets/reference/baseline.json \
  --output-dir /runs/infer-r1-diagnosis
```

此固定 day10 参考只用于 40×6h；改变步数时须另提供匹配时效的合格参考。
天气预报对真实目标的 RMSE/ACC 属于另一评估层，不以 5% 一致性诊断替代。

## 历史证据与新分支的边界

- 训练最新三轮 E2E：660.386422 / 656.841734 / 659.728441 秒，MFU 中位数
  9.013270%。同源码历史 10.012132% 为不同 GPU/主机窗口，不混作本轮结果。
- 推理历史三轮 E2E：11.8119776123 / 11.8054953380 / 11.8081687340 秒，
  中位数 MFU 19.748078%。原记录 `graphcast-h100-native-tf32-baseline-20260906`，
  源快照摘要 `97a2f36934f5855c663e5223b37143d03eb778ab2b16b74ac37364211724ad5e`；
  原 case manifest 摘要 `4c7ee6989fed959501a5f5368f813259589b8cc6da899d813708e6014d0bffc0`。
- 当前 operational 配置哈希与历史推理一致，但 `google_small.py`、`graph.py`、
  `run_operational_40step_inference.py`、`diagnose_inference.py` 与当时不同。
  本分支保持目前统一源码，**未退回旧快照，也没有声称历史 11.808 秒已在新源码复现**。
- 本次新增入口的验证是 CPU 测试、命令/参数/环境检查和冻结源码哈希，
  不是新 H100 性能/正确性测量。正式验证时，仍需在准备好的 H100 环境重跑
  三轮、正确性诊断，保留启动记录、退出码、metrics/report 和诊断产物。

## 公开分支范围（2026-10-12）

- 从个人仓库 `50fa1808928ff8e751fe30b13e7659e30d686ef3` 新建
  `zyx/nvidia-native-baseline`，保持已有 `ai4s/graphcast/` 目录结构。
- 只增加 baseline 配方、启动入口、说明及 CPU 测试，调整 README；
  不改模型、loss、计时逻辑或已有的 137 文件冻结清单。
- 数据、权重、诊断参考、容器镜像及私有仓库的提交历史均未上传。
- 测试结果见[公开分支验证记录](nvidia-baseline-validation.md)；
  不将 CPU 检查或历史结果表述为此次新跑的 H100 测量。
