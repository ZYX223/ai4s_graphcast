# 运行说明

固定Operational 0.25°／13层／batch=1，无需指定case。以下命令在宿主机同一个Bash终端执行。
入口管理Docker与底层环境；`PLATFORM`、`WORK`、`DEVICE`仅用于简化命令，无需导出环境变量。

## 1. 选择本机平台并检查材料

BW1100：

```bash
cd /recovery/graphcast-delivery/releases/graphcast-20261114
PLATFORM=bw1100
WORK="/data_add/$(id -un)/graphcast-20261114-work"
```

H100：

```bash
cd /data2/graphcast-delivery/releases/graphcast-20261114
PLATFORM=h100
WORK="/data2/$(id -un)/graphcast-20261114-work"
```

两平台接着执行：

```bash
mkdir -p "$WORK"
python3 scripts/verify_delivery.py --platform "$PLATFORM" --plans
bash scripts/run.sh --platform "$PLATFORM" --work "$WORK" --check
read -r -p 'Assigned physical GPU index: ' DEVICE
```

`--check`验证环境与依赖，不计算模型。启动时还会检查输入、模型、参考与选定GPU的占用。
首次校验建立 `$WORK/.verification-cache/`，后续复用未变化材料；需完整重验时加 `--verify-full`。
work必须由当前账号拥有、可写且位于交付根目录之外；`--devices`接收已分配的物理卡号。
更换账号时使用该账号自己的work路径，不复用交付者的运行目录；父目录不存在或不可写时，改用已分配的个人存储路径。
同一平台环境和work上的任务串行执行。并行任务应使用分别分配的GPU和独立work；不同卡仍共享主机资源，性能对比宜避免并发。

```bash
bash scripts/run.sh --list
bash scripts/run.sh --platform "$PLATFORM" --workload infer-full40 --plan
```

`--plan`只解析计划，不启动容器或GPU；可在任一主机检查两平台参数。

## 2. 推理与训练

完整复现使用固定预设：

```bash
# Single-step validation
bash scripts/run.sh --platform "$PLATFORM" --workload infer-single \
  --devices "$DEVICE" --work "$WORK"
# Single-step preflight, then 3 independent 40-step forecasts
bash scripts/run.sh --platform "$PLATFORM" --workload infer-full40 \
  --devices "$DEVICE" --work "$WORK"
# 38 consecutive optimizer updates over stages 1-12
bash scripts/run.sh --platform "$PLATFORM" --workload train-full38 \
  --devices "$DEVICE" --work "$WORK"
```

固定预设不接受自定义步数；旧 `single`、`full40`别名继续可用。
H100单步检查为同进程基线与候选各1步，分别预热5步；BW1100为1轮单步检查。
两平台full40均先做上述单步检查，再进行3轮40步、每轮预热5步。

自定义执行：

```bash
# Predict 8 steps twice; warm up 3 steps before the first round only
bash scripts/run.sh --platform "$PLATFORM" --workload infer \
  --steps 8 --warmup-steps 3 --repeats 2 --warmup-scope once \
  --devices "$DEVICE" --work "$WORK"
# Stop at stage 2 or cumulative update 18, whichever comes first
bash scripts/run.sh --platform "$PLATFORM" --workload train \
  --end-stage 2 --max-updates 18 --warmup-iters 2 \
  --devices "$DEVICE" --work "$WORK"
```

| 工作负载 | 参数 | 默认值／含义 |
|---|---|---|
| infer | `--steps` | 40；每轮正式预测1～40步 |
| infer | `--warmup-steps` | 5；每次预热1～40步 |
| infer | `--repeats` | 3；独立预测轮数，至少1 |
| infer | `--warmup-scope` | per-repeat；每轮预热，once仅首轮预热 |
| train | `--end-stage` | 12；最后执行阶段1～12 |
| train | `--max-updates` | 38；累计上限1～38，包含续训前已完成的更新 |
| train | `--warmup-iters` | 1；每阶段预热次数，至少1 |
| train | `--save-every` | 默认阶段结束保存；正整数表示每N次累计更新另存一次 |
| train | `--resume` | 默认从初始权重开始；接受work内的断点文件或单次运行目录 |

推理每次预热及正式轮次都重置原始输入与随机数；预热不计入正式predictor时间或capture。
比较性能时保持轮数、步数、预热策略一致。1步预热可能遗漏自回归路径，最小值需按平台实测。
自定义infer默认也是3×40，但不包含固定预设的单步前置验收及完整参考验收。
训练的stage表示预测长度，update表示一次优化器更新；`--max-updates 20`是累计到第20次，不是再执行20次。
每阶段在独立worker中编译、预热，预热更新只作用于副本，不计入正式更新编号；学习率日程保持。
BW训练自动应用进程级THP设置，无需额外环境变量。完整日程、计时与策略记录见[FULL38.md](FULL38.md)。

## 3. checkpoint生成与续训

首次训练不传 `--resume`，自动使用交付权重。新checkpoint包含参数、优化器状态和累计更新编号。
`--save-every`按全局编号保存，达到阶段或累计上限时总会保存；保存与回读校验在正式计时之外。

```bash
bash scripts/run.sh --platform "$PLATFORM" --workload train \
  --devices "$DEVICE" --work "$WORK" \
  --max-updates 20 --warmup-iters 1 --save-every 4
```

若在更新10中断，更新8的完整断点仍可恢复。将该次输出的 `Results:`目录填入 `RESUME_RUN`：

```bash
read -r -p 'Results directory printed by the previous run: ' RESUME_RUN
bash scripts/run.sh --platform "$PLATFORM" --workload train \
  --devices "$DEVICE" --work "$WORK" \
  --resume "$RESUME_RUN" --max-updates 20 --warmup-iters 1 --save-every 4 --plan
bash scripts/run.sh --platform "$PLATFORM" --workload train \
  --devices "$DEVICE" --work "$WORK" \
  --resume "$RESUME_RUN" --max-updates 20 --warmup-iters 1 --save-every 4
```

目录恢复只扫描这次运行的r01～r12，按已提交且校验通过的更新编号选择，跳过损坏或未完成文件。
没有有效断点就报错；不扫描整个work，不按修改时间选择。再次中断时使用最新一次启动输出的运行目录。
`--max-updates`始终是累计上限；续训上限须大于已完成编号。恢复必须使用同平台、同case、同一work。

也可直接传宿主机checkpoint路径。周期断点 `rXX/checkpoints/checkpoint-NNNN.pkl` 须保留同名JSON；
旧式 `rXX/checkpoint.pkl` 须保留同目录report及退出码。不要传初始 `jax_weights.npz` 或容器内路径。

下面是从训练到续训的完整示例：独立work中先完成18次更新，再恢复执行19～20。

```bash
(
set -euo pipefail
: "${PLATFORM:?Initialize PLATFORM first}"
: "${WORK:?Initialize WORK first}"
: "${DEVICE:?Set the assigned physical GPU index first}"

TRAIN_WORK="$(mktemp -d "${WORK%/}/resume-example.XXXXXX")"
printf 'Training work directory: %s\n' "$TRAIN_WORK"
bash scripts/run.sh --platform "$PLATFORM" --workload train \
  --end-stage 2 --max-updates 18 --warmup-iters 2 \
  --devices "$DEVICE" --work "$TRAIN_WORK"

# This new directory contains only this run; require exactly one stage-2 checkpoint.
checkpoints=("$TRAIN_WORK"/runs/graphcast-20261114/*/r02/checkpoint.pkl)
if [[ ${#checkpoints[@]} -ne 1 || ! -f "${checkpoints[0]}" ]]; then
  printf 'Expected exactly one stage-2 checkpoint; inspect training logs: %s\n' "$TRAIN_WORK" >&2
  exit 1
fi
CHECKPOINT="${checkpoints[0]}"
printf 'CHECKPOINT=%s\n' "$CHECKPOINT"

# Expect completed_updates=18, next stage 3, and updates=[19,20].
bash scripts/run.sh --platform "$PLATFORM" --workload train \
  --resume "$CHECKPOINT" --end-stage 3 --max-updates 20 \
  --devices "$DEVICE" --work "$TRAIN_WORK" --plan

bash scripts/run.sh --platform "$PLATFORM" --workload train \
  --resume "$CHECKPOINT" --end-stage 3 --max-updates 20 \
  --devices "$DEVICE" --work "$TRAIN_WORK"
)
```

## 4. 输出与完成判据

启动时打印 `Results:`，输出位于 `$WORK/runs/graphcast-20261114/<case>-<workload>-<标识>/`。
每次运行使用新目录，保留历史结果。两平台训推均有 `host.log`、`console.log` 和 `run-summary.json`；
H100阶段另有 `engine-console.log`。`context.json`记录交付参数，阶段配置由入口生成，无需用户手写。

| 工作负载 | BW1100 | H100 |
|---|---|---|
| infer-single | `single/run/`、`single-comparison/summary.json` | `run/`、`single-step-validation.json` |
| infer-full40 | `full40/run/`、`comparison/summary.json` | `preflight-single/`、`run/`、`accuracy-h100/summary.json`、`accuracy-optimized-reference.json` |
| train-full38 | `r01/`～`r12/`、`training-loss-summary.json` | `r01/`～`r12/`、`training-validation.json` |
| infer | `inference/run/`、`execution-summary.json` | `run/`、`execution-summary.json` |
| train | 所选 `rXX/`、`execution-summary.json` | 所选 `rXX/`、`execution-summary.json` |

```bash
read -r -p 'Absolute results directory: ' RUN_DIR
tail -f "${RUN_DIR:?Results directory must not be empty}/host.log"
```

提示与注释为英文。日志区分Compile／Warmup／Inference／Training／Save；同步predictor耗时与rollout墙钟分开。
宿主机采集显存、利用率与进程归属；worker读取指定卡号或UUID的采样，过期或缺失记为unavailable。
监控在宿主机独立循环，每轮查询后等待1秒，实际间隔还包含查询耗时；不计入模型调用，但可能产生主机／驱动开销。
进程归属告警表示映射证据不足或发现外部进程，应查看监控记录；单条告警不等于模型执行失败。
失败时先看直接原因和退出码，完整异常在console.log。保存不计入正式推理／训练更新计时；
训练原始阶段trace可能包含周期checkpoint保存，GPU active复算只取正式更新窗口。

`execution=passed`表示请求完成；固定预设需全部参考检查通过才有 `accuracy=passed`。
自定义精度为not_evaluated。监控不完整或存在归属问题时performance为not_accepted，其余仍待trace分析，
不会因退出码0自动宣告性能达标。具体阈值、参考和已验证范围见[RESULTS.md](RESULTS.md)。
