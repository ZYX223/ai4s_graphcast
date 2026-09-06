# 重建推理双参考（不是修改历史基线）

使用同事诊断包的原公式：每个变量/层计算
`mean(abs(candidate-reference)) / mean(abs(reference)) * 100`，均匀空间权重，
六变量 × 十三层的 78 个百分数等权平均。整体与 Z500 都必须 **严格小于 5%**，
分别对 JAX FP32 和 H100 Torch FP32 检查。不改成 RMSE，不改变阈值。

这是一致性验收，不是对真实天气的预报技能验收。一个起报时刻/时效通过，不代表
所有天气案例通过。TF32/BF16/FlagGems 的同平台 Native 比较是补充诊断，不能替代双参考门槛。

## 流程与边界

1. 审计原始 JAX 与现有 Torch 权重：名字映射、线性权重转置、FP32 类型、全部参数逐值相等。
2. 冻结一个共享 case，复制权重、审计记录/代码、config、stats、两帧输入、完整 forcing 和单步模板。
3. 官方 JAX 严格 FP32 跑两次；H100 Native eager 严格 FP32 跑两次，每次启动新进程。
4. 检查来源、同引擎复跑稳定性、Torch 对 JAX 的 5% 一致性，导出选定时效的双参考 NPZ。
5. 用现有 `diagnose_inference.py` 检查各平台候选结果，实际算子命中数仍来自候选运行报告。

`provenance_status=verified` 只表示资料/运行来源绑定通过；必须再有
`qualification.status=qualified` 才可用于正式验收。复跑稳定性默认要求整体和 Z500
差异均小于 **0.1%**，在 case 冻结前确定；它是工程稳定性保护，不是官方天气指标，
也不替代 5%。可以预先收紧到低于 0.1%，不能调宽。不要求跨次逐位相同。
至少两次独立执行，复制同一 execution ID 不算复跑。

步骤数没有写死：`steps`、`step_hours`、`initialization_time`、`lead_hours` 由 case 指定。
40 × 6h、选择 +240h 只是当前案例。按实际时效选帧，不按数组下标 39 选帧。

## 1. 权重对应关系审计

在项目 `ai4s/graphcast` 目录中执行；该步骤只需要 CPU Torch，不运行模型：

```bash
python scripts/inference/audit_reference_conversion.py \
  --jax-weights /assets/GraphCast_operational_25km.npz \
  --torch-weights /assets/google_graphcast_operational_25km_compat_state.pt \
  --output /evidence/new-reference/conversion.json
```

只支持本项目现有的 bare state_dict 和 Haiku 命名映射。不靠相同 shape 猜参数对应关系。
这是重新验证现有权重的对应关系，不声称找回了历史转换脚本，也不重写权重。

## 2. 共享 case 资料

创建 JSON 规格（以下路径需指向实际已核验文件）：

```json
{
  "initialization_time": "2021-01-01T06:00:00",
  "steps": 40,
  "step_hours": 6,
  "lead_hours": 240,
  "repeatability_percent": 0.1,
  "assets": {
    "config": "/project/ai4s/graphcast/configs/inference/google_graphcast_operational_25km.yaml",
    "jax_weights": "/assets/GraphCast_operational_25km.npz",
    "weights": "/assets/google_graphcast_operational_25km_compat_state.pt",
    "converter": "/project/ai4s/graphcast/scripts/inference/audit_reference_conversion.py",
    "conversion": "/evidence/new-reference/conversion.json",
    "inputs": "/assets/reviewed-case/inputs.nc",
    "rollout_forcings": "/assets/reviewed-case/rollout_forcings.nc",
    "target_template": "/assets/reviewed-case/target_template.nc",
    "stats/mean_by_level.nc": "/assets/stats/mean_by_level.nc",
    "stats/stddev_by_level.nc": "/assets/stats/stddev_by_level.nc",
    "stats/diffs_stddev_by_level.nc": "/assets/stats/diffs_stddev_by_level.nc"
  }
}
```

`converter` 字段保存上述对应关系审计器的源文件，哈希与审计结果绑定。
所有 case 数据变量要求 FP32；原始统计量可为 FP64，两条执行路径按 FP32 使用。
模板数值不是 truth，可以为空值；模板变量名、单位、坐标仍必须完整。
输入 time 为 `[-step_hours, 0]`，forcing 为全部正时效，模板为首个正时效。
三者均需 `datetime(batch,time)` 且等于起报时间加 time；网格/层次一致。
六个诊断变量需有来源可核实的单位属性。预检不会凭变量名补单位或猜起报时间。

```bash
python scripts/inference/inference_reference.py freeze \
  --spec /evidence/new-reference/case-spec.json \
  --output-dir /evidence/new-reference/case
```

冻结后两端均读取 `case/assets/` 中的同一份文件，不各自产生 forcing。
已有资产缺少属性时，先从原始资料补足可审核的时间/单位来源；若需转换 NetCDF，另存新文件，
记录原文件和变更原因，核对数据值未改变，再冻结。不要修改历史资料或直接勾选 verified。

可用 `tools/inference_reference/complete_metadata.py` 生成这样的副本：

```bash
python tools/inference_reference/complete_metadata.py \
  --spec /evidence/new-reference/metadata-spec.json \
  --output-dir /evidence/new-reference/metadata-completed
```

规格的 `assets` 对象按输出文件名组织；每项提供 `path`、原文件 `sha256`、
`datetimes`（batch/time 二维 ISO 时间字符串）、`units`（变量名到单位的映射），
以及 `provenance.time_basis` / `provenance.unit_basis` 两个非空来源说明。
单位声明可只涉及缺失属性的变量；不能覆盖冲突单位。工具拒绝时间溢出、精度丢失、
哈希变化和已有输出目录，并在写出后核对原数组、坐标、类型及新增时间。
终态 `metadata_audit.json` 绑定原/新哈希；它证明元数据变更过程，不证明单位来源本身真实。
例如按 ERA5 通道约定补单位，必须明确说明这是约定声明，而非原 H5 自带属性。

先对照此前运行报告的输入哈希选文件，再补元数据。代码包的旧输入可能与实际测试使用的
`inputs_official_forcings.nc` 存在 forcing 数值差异，不能因为起报时间相同就当作同一 case。

## 3. 分别运行官方 JAX 与 H100 Torch

官方源固定到 [GraphCast v0.2 / 08cf7362](https://github.com/google-deepmind/graphcast/tree/08cf73625c9d12bd9aaa038868bcb2fe488f2a22)，
不能直接使用后续重构的 WeatherNext2。入口逐文件核对固定 commit 的源码字节，拒绝改动或额外 Python 文件。
在独立 JAX 环境安装与这份官方源码匹配的依赖；工具不会自动安装/替换任何设备环境。

```bash
/envs/jax/bin/python scripts/inference/run_inference_reference.py \
  --case /evidence/new-reference/case --engine jax \
  --jax-source-root /sources/official-graphcast-v02 --device gpu:0 \
  --output-dir /evidence/new-reference/jax-r1 --check-only
```

`--check-only` 只检查资料和源文件，不导入 Torch/JAX、不分配 GPU，也不生成运行结果。
在确认设备空闲后去掉 `--check-only` 实际运行，并以新目录 `jax-r2` 再运行一次。
JAX 使用官方 `GraphCast → InputsAndResiduals → autoregressive.Predictor` 和
[`chunked_prediction_generator`](https://github.com/google-deepmind/graphcast/blob/08cf73625c9d12bd9aaa038868bcb2fe488f2a22/graphcast/rollout.py)，
FP32 参数/输入，无 BF16 wrapper；显式设置 `jax_default_matmul_precision=highest`，关闭 TF32 override。
JAX 的 `jit` 是官方执行方式，不是本项目的 `torch.compile` 实验组。

```bash
/envs/torch/bin/python scripts/inference/run_inference_reference.py \
  --case /evidence/new-reference/case --engine torch --device cuda:0 \
  --output-dir /evidence/new-reference/torch-r1
```

再以 `torch-r2` 新目录运行一次。Torch 必须实际识别为 H100：入口强制 Native、eager、
严格 FP32、baseline profile，清除继承的 GRAPHCAST/FlagGems 优化环境变量，并核对运行报告的实际 dtype。
复用原 `run_operational_40step_inference.py`，不复制一份模型或 Torch 自回归循环。
入口按生产约束执行一次丢弃的 forward warmup；它不推进自回归状态，也不计入测量步数。

默认选最终时效时只保存最终原始 prediction；如果选择中间时效，Torch 额外输出完整 trajectory，
该文件作为 `prediction.nc` 供按时效选择。需预留磁盘容量。JAX 保留选定时效和最终时效。
这两种执行都是诊断，**不得把有逐步检查/输出的时间填入性能 E2E/MFU 表**。

## 4. 资格检查、导出与使用

```bash
python scripts/inference/inference_reference.py qualify \
  --case /evidence/new-reference/case \
  --jax-run /evidence/new-reference/jax-r1 --jax-run /evidence/new-reference/jax-r2 \
  --torch-run /evidence/new-reference/torch-r1 --torch-run /evidence/new-reference/torch-r2 \
  --output-dir /evidence/new-reference/baseline

python scripts/inference/diagnose_inference.py \
  --baseline /evidence/new-reference/baseline/baseline.npz \
  --baseline-metadata /evidence/new-reference/baseline/baseline.json \
  --run-dir /results/c550-flaggems-tf32 \
  --output-dir /results/c550-flaggems-tf32/new-baseline-diagnostics
```

导出目录包含 `baseline.npz`、`baseline.json`、`qualification.json` 及 `references/` 原始预测和报告副本。
资料或结果目录存在时拒绝覆盖。输入变化、单位/时间/精度/来源不一致时退出 2。
资格检查数值失败退出 1，仍保存 `unqualified` 证据；合格退出 0。
候选诊断遇到 `unqualified_reference` 退出 2，不能显示 passed，即使它与其中一个参考完全相同。

不修改 `configs/inference/diagnostic_baseline_day10.json` 和历史参考。
本次工具的 CPU 测试通过不等同于官方 JAX/H100 设备运行通过；真实双参考完成前不得替换正式验收基线。
