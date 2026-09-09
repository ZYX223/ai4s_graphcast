# 训练脚本说明

正式入口 `run_training.py`、`train_official.sh`、`train_pipeline.sh` 均调用同一份
同事训练引擎 `run_official_epoch_training.py`（源提交 `b7667b4`）。
训练专用 `training_model.py` 使用 canonical graph；推理模型、图默认值和
inference FlagGems YAML 保持原样。来源和适配见
[provenance](../../docs/training-source-provenance.md)。

参数、激活和 loss 保持 FP32；公开入口默认请求 TF32/HF32，使用现有 backend
设置 CUDA、NPU 和 MUSA 的实际开关。`--tensor-core disabled` 请求 strict FP32。
默认全局 batch 16、每 rank micro batch 1，MP4 / DP1。MP2 可选，DP1/DP2 均可。
必须满足 global batch 整除 `DP × 每 rank micro batch`；没有不满的最后一个
micro batch。公开 `--micro-batch-size` 是每 rank，source runner 同名参数是
global micro batch，adapter 显式乘 DP，dry-run 同时显示两者。

从 `ai4s/graphcast` 执行：

```bash
# 不加载 Torch、FlagGems、数据或权重，也不创建输出目录。
python scripts/training/run_training.py --dry-run --device-type npu \
  --mp 4 --dp 1 --global-batch-size 2 --micro-batch-size 1

# 四种拓扑的 Native/FlagGems 命令；实际运行前替换数据和权重路径。
# 每个组合都在新进程和独立输出目录运行；DP2 自动启动两个 rank。
for backend in native flaggems; do
  for mp in 4 2; do
    for dp in 1 2; do
      python scripts/training/run_training.py \
        --device-type npu --mp "$mp" --dp "$dp" --backend "$backend" \
        --corpus /data/era5/corpus.json --stats /data/stats \
        --pretrained-checkpoint /data/operational_pytorch.pt \
        --global-batch-size 2 --micro-batch-size 1 \
        --warmup-updates 6 --one-step-updates 10 --curriculum-updates 22 \
        --max-updates 3 --checkpoint-every 1 \
        --output-dir "/runs/ascend-${backend}-mp${mp}-dp${dp}"
    done
  done
done

# 保持模型、数据、stats、batch、拓扑、精度、后端和 6/10/22 recipe 不变，
# 只改变绝对停止步数；从 prefix stop 的 numbered checkpoint 恢复。
python scripts/training/run_training.py \
  --device-type npu --mp 4 --dp 1 --backend native \
  --corpus /data/era5/corpus.json --stats /data/stats \
  --resume-checkpoint /runs/ascend-native-mp4-dp1/checkpoints/step_00000003.pt \
  --global-batch-size 2 --micro-batch-size 1 \
  --warmup-updates 6 --one-step-updates 10 --curriculum-updates 22 \
  --max-updates 6 --checkpoint-every 1 --output-dir /runs/ascend-native-mp4-dp1
```

这些是可复制的调用形式，不表示八种组合已通过实机验收。模型维持完整
721×1440、mesh6、latent512、16 mesh steps、13 层；不得用缩小模型的结果
代替正式设备验收。CLI、真实 tiny CPU 模型/数据、Gloo 和恢复已有回归覆盖。

2026-09-09 在冻结源码 `ffd34d532cf27d3319e18b1a93b0418167b15d25` 上完成
910C 四逻辑设备 MP4/DP1 短测：完整模型、FP32 storage/HF32 请求、global
batch2/micro1、seed27、6/10/22 schedule 停在第3次更新。Native/FlagGems
均正常完成并保存 checkpoint，梯度有限、无 OOM；单设备最大分配显存分别
47.3426/52.5519 GiB。完整区间 loss 比值差异0.0010006925%，配方/起始状态/
数据/运行库证据可比，低于5%门槛。第三步两组 loss 均上升至约12.217，
因此这只是配对流程正确性短测，不是收敛、12-lead、MP2/DP2、多平台或天气
技能验收；未计算正式 MFU，也未把注册数当作实际调用覆盖。
训练 YAML 的6个选中接口注册成功且无跳过；本次未另跑调用观察器。
独立证据目录标识为 `gct9-0lraxY`，含冻结清单、原始 metrics/start sidecar、
checkpoint checksum、进程退出记录和 `diagnosis/loss.json`；数据和权重不入库。

`--platform` 可选 `h100/metax/hygon/thead/mthreads/ascend910c`。
NPU/MUSA 可由 `--device-type` 推断；CUDA 兼容的海光、沐曦、平头哥须显式
指定 platform，避免被标成 H100。可设置 `GRAPHCAST_PLATFORM`。
`--devices` 为从 0 连续的逻辑卡号；`--physical-devices` 或厂商 visibility
变量选择物理卡。Hygon 用 `HIP_VISIBLE_DEVICES`，MUSA 用
`MUSA_VISIBLE_DEVICES`，Ascend 用 `ASCEND_RT_VISIBLE_DEVICES`；
所有 rank 必须看到完整卡组。运行环境须已安装匹配的厂商 Torch/工具链；
launcher 不修改环境安装、不同步源码、不选择部署目录。Shell 支持 `PYTHON_BIN`。

平台内存配置来自 pinned source launcher。MP4 保留 manual gradient sync，
MP2 默认 DDP；DDP accumulation 把 forward/backward 包在 `no_sync`，
最后一个 micro batch 才同步，没有第二次手动梯度平均。MUSA 多设备反向
保留源串行执行；Ascend 默认 `ASCEND_LAUNCH_BLOCKING=1`。
Ascend MP4 使用 mesh split16、stage checkpoint、reentrant outer checkpoint；
MP2 使用源 README 验证示例的 split8、step、non_reentrant，并保留显式 CLI
覆盖。此 MP2 设置复用不代表本次已做 MP2 设备验收。chunk、stage placement、
prefetch/cache 和 checkpoint 开关见公开 `--help`；完整 source-native 参数见：

```bash
python scripts/training/run_official_epoch_training.py --help
```

source-native 入口仍有 manifest schedule/HRES/validation/profiler 选项，调用时
自行设置 `GRAPHCAST_ACCELERATOR_TYPE`、`GRAPHCAST_BACKEND`、
`GRAPHCAST_PLATFORM_LABEL` 和设备可见性；DP1/DP2 均需 torchrun 提供进程组环境。
推荐使用公开 adapter 来应用平台策略。单独运行 source-native runner 默认 Native；
FlagGems 模式通过 `GRAPHCAST_FLAGOS_MODE=flaggems` 选择，其余仍使用
target `GRAPHCAST_FLAGGEMS_CONFIG/OPS/VARIANT` 变量和 training YAML。

默认 schedule `configs/training/official_2_to_12.json` 对应
1000 warmup、299000 half-cosine、11000 curriculum，LR 为 0→1e-3、
1e-3→0、3e-7。curriculum 在 2–12 lead 各占相等更新预算，覆盖预算必须为正，
curriculum 必须整除11。`--max-updates` 只停止前缀，不改 schedule digest。
每次 optimizer update 从当前 rollout 可用 trajectory 中按 seed/update 有放回
抽样；不沿用旧引擎每 epoch 无放回顺序。完整 BPTT、变量/纬度/气压权重、
AdamW 和 global norm clip32、CPU prefetch/cache 均保留源实现。

旧 raw tensor state_dict 与 source `model_state/metadata` envelope 均可通过
`--pretrained-checkpoint` 做模型初始化，优化器和 sampler 从新 run 开始。
exact resume 需要 source schema3 checkpoint 和 checksum sidecar，保留 source
recipe/topology/路径 contract 校验；旧引擎 optimizer checkpoint 不作 exact resume。
源 contract 记录路径，并不对同路径数据/stats 的全部内容重新建立哈希；
不要把它描述为可移动路径或任意数据改动检测。checkpoint 文件和状态有完整性校验。
prefix stop 写 `checkpoints/step_XXXXXXXX.pt`；`final_checkpoint.pt` 在完成 recipe
时写出。日志为 `metrics/segment_XXXXXXXX.jsonl`；恢复创建下一个无重叠 segment，
最终化会重写该 segment，保留 source 格式。

恢复的 replacement identity 只包含稳定的 backend/op/version 策略，不包含
rank 卡号或进程录制路径。完整运行时证据和 cache 实测状态保存在
`processes/from_XXXXXXXX/rank_R/runtime.json`；immutable run_config 仅记录
cache 预算，重开数据产生的新预加载/校验耗时不会阻断恢复。
source-native validation 请求分变量/逐 lead loss；普通训练仍关闭这些诊断。

FlagGems 唯一 op 策略仍是 target training YAML，公开 config/ops/variant 参数
转发给现有 backend；training bridge 将 `layer_norm` 映射为实际注册名
`native_layer_norm`，验证所有选中 op 的注册证据，缺失则失败。
Native 不导入 FlagGems。metadata 保留 runtime profile/versions、注册名和导出函数
来源；注册和导出来源不等于实际 forward/backward 调用覆盖。

每次训练在 update 计时之前保存 `metrics/segment_XXXXXXXX.start.json`，
与同名 JSONL 绑定。它记录实际每 rank 参数/优化器指纹、数据 shard 的有序
内容哈希、三份 stats 和模型 config 哈希、source resume contract、变量权重、
内存/缓存策略及实际后端和精度。哈希会增加启动 I/O；不要在配对运行期间
修改数据。元数据不参与 source checkpoint 的 immutable resume identity。
恢复使用新 sidecar，旧段不会被覆盖；source 收尾重写 JSONL 不会丢失元数据。

```bash
python scripts/training/diagnose_training_loss.py /runs/flaggems \
  --baseline /runs/native --start-step 1 --end-step 3 \
  --output-json /runs/paired-loss.json
```

输入可以是 run 目录、metrics 目录或单个 segment JSONL；单文件自动读取
同名 `.start.json`。旧 start/update JSONL 和显式 metadata CSV 仍可读。
独立 reader 仅需 Python 标准库。必须提供完整区间，不能只比较历史交集；
每 update 的有放回 global/local 样本顺序、权重和时间来源必须一致。
使用 `sum(abs(candidate-reference))/sum(abs(reference))*100`，默认包含 5%
边界；只在该边界容许最多 8 个 binary64 ULP（约 7.1e-15 个百分点）舍入差。
例如 reference `[2,4]`、candidate `[2.1,4.2]` 为 5% 通过，`[2.2,4.4]`
为 10% 失败。退出码 0=合格，1=合格配方但 loss 超阈值，2=不可比/证据缺失/
无效输入。Native reference 必须有禁用替换的实际 runtime 证据。
配对恢复段也必须有相同实际起始参数/优化器；分别演化后存在微小状态差异的
Native/FlagGems 恢复轨迹会保守拒绝，不推断共同 lineage 来放宽该条件。

调用观察是单独的实验，不能用于性能计时。先用公开入口 `--dry-run` 获取
完全相同的平台 environment 和 engine_argv，再在独立目录下替换启动目标：

```bash
# 导出 dry-run 中的 environment，包括训练 YAML/variant 与设备可见性。
# 以下值举例对应 Ascend；其他平台使用其实际 dry-run 值。
export GRAPHCAST_FLAGOS_MODE=flaggems GRAPHCAST_BACKEND=ascend
export GRAPHCAST_ACCELERATOR_TYPE=npu
python -m torch.distributed.run --standalone --nproc-per-node 1 \
  scripts/validation/trace_training_flaggems.py --trace-dir /runs/calls -- \
  <dry-run 的 engine_argv；使用新的 run-dir/save-checkpoint>
```

wrapper 通过 runpy 执行唯一 canonical runner，不建立新训练循环。
`FLAGOS_REPLACE=1` 单独设置不够；缺少 target FlagGems/backend 环境会失败。
`rank_R.jsonl` 按真实 AdamW update 保存 logger/function/device/source 路径和
logged function calls；启动调用标为 initialization 且 update=null，
forward、backward、checkpoint recompute、optimizer、unscoped 分开。
仅统计发出 GEMS 日志的函数，既不是 kernel 数，也不是全算子覆盖证明。
未完成的 optimizer update 不会产生已完成 update 证据。

loss 比较始终返回 `weather_skill=not_evaluated`。评估 trained checkpoint 的
天气能力，需要另行固定验证数据、lead、变量和统计口径，比较该 checkpoint
对应的输出；固定 pretrained 权重的推理参考不能直接当成已训练权重的数值
等价目标。历史 small-1deg JAX CSV 不作为默认 reference。本次不执行长天气评估。

性能计时默认不启用 profiler；`--profile-hotspots` 为独立诊断实验。
线性 FLOPs/recompute 估算沿用源定义，并不是完整算子 FLOPs。
默认 peak=0（unknown），MFU 为 null；显式正数必须提供 `--peak-tflops-source`，
来源字符串本身不是峰值已验证的证明。不默认使用 engineering peak，不声称天气技能。
`mfu_sys` 分母是明确给定的设备总峰值乘 compute wall time；`mfu_e2e` 使用
包含数据处理的端到端时间，`hfu_sys` 分子还计入估算的 recompute FLOPs。
profiler 的 kernel-busy 分母仅供另一次观测实验，不能与无 profiler 的计时混用。

以下保留既有短训练探针说明；它们不属于新的正式 epoch runner 验收路径。

## 脚本一览

| 脚本 | 用途 | 卡数 | 模型并行 | Grid 分区 | 适用场景 |
|------|------|:---:|:---:|:---:|------|
| `train_25km_single.sh` | 25km 单卡训练 | 1 | - | - | smoke test / 轻量训练 |
| `train_100km_single.sh` | 100km 单卡训练 | 1 | - | - | 快速验证 / JAX 基线对比 |
| `train_ddp.sh` | 多卡 DDP 训练 | N | - | - | 数据并行训练 |
| `train_mp.sh` | 模型并行训练 | 4 | ✅ | ✅ | **25km 全尺度训练** |

## 运行前必读

### 1. 检查 NPU/GPU 占用

在提交任务前，**必须**检查目标设备是否空闲。以 Ascend 为例：

```bash
# 查看 HBM 占用（不能只看进程列表！）
npu-smi info 2>&1 | grep -E "^\| [0-9]+\s+[0-9]+\s+"

# HBM > 10 GB 表示有任务占用，不可使用
# 选择 HBM < 5 GB 且进程列表显示 "No running processes" 的 NPU
```

### 2. 通用环境变量

所有脚本支持以下环境变量：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `GRAPHCAST_BACKEND` | `auto` | 后端选择：`ascend`（Ascend）、`ptg`（PPU）、`auto`（自动检测） |
| `DEVICE` | `auto` | 设备类型：`npu`、`cuda`、`cpu`、`auto` |
| `GRAPHCAST_FLAGOS_MODE` | `off` | FlagOS 模式：`off` 或 `flaggems` |
| `GRAPHCAST_FLAGGEMS_CONFIG` | `configs/flaggems/graphcast_training.yaml` | 按平台选择已验证的训练前向算子 |
| `GRAPHCAST_FLAGGEMS_OPS` | 空 | 可选覆盖项；非空时替代 YAML 算子列表 |
| `GRAPHCAST_FLAGGEMS_VARIANT` | `installed` | `installed`、`official` 或 `internal` |

训练 profile 只描述 forward 接管范围，不代表 backward 已验证。需要对比
同基线 official/internal 时，先运行 `bash tools/prepare_flaggems_runtime.sh`。

### 3. Docker 运行

所有脚本需在 Docker 容器内运行。以 Ascend 为例：

```bash
# 先检查 NPU 占用！
docker run --rm --runtime=ascend --privileged --security-opt label=disable \
  --device=/dev/davinci8 --device=/dev/davinci9 \
  --device=/dev/davinci10 --device=/dev/davinci11 \
  --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/dcmi:/usr/local/dcmi:ro \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v /path/to/repo:/ws -w /ws \
  -e ASCEND_RT_VISIBLE_DEVICES=8,9,10,11 \
  -e "PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256" \
  quay.io/ascend/vllm-ascend:v0.20.2rc1-a3 bash -c "
pip install xarray netCDF4 scipy trimesh rtree PyYAML h5py pandas
bash scripts/training/train_mp.sh
"
```

---

## train_25km_single.sh — 25km 单卡训练

**调用**：`lib/train_single.py`

**适用**：快速 smoke test（小步数、decoder-grid 模式）

```bash
STEPS=10 TRAIN_SCOPE=decoder-grid bash scripts/training/train_25km_single.sh
```

**常见错误**：

| 错误 | 原因 | 解决 |
|------|------|------|
| OOM（全尺度训练） | 25km 全尺度需要 >64 GiB | 改用 `train_mp.sh`（模型并行），或设置 `TRAIN_SCOPE=decoder-grid` |
| `missing required file` | cases / weights / stats 不存在 | 确认 `cases/`、`outputs/checkpoints/`、`data/raw/` 目录完整 |

---

## train_100km_single.sh — 100km 单卡训练

**调用**：`lib/train_single.py`

**适用**：快速验证训练流程，100km（1°网格）显存需求 ~20-30 GiB，单卡即可全尺度训练。

```bash
STEPS=10 bash scripts/training/train_100km_single.sh
```

**常见错误**：

| 错误 | 原因 | 解决 |
|------|------|------|
| `missing required file` | 100km 配置文件或权重缺失 | 确认 `configs/training/google_graphcast_100km.yaml` 和权重文件存在 |
| `PytorchStreamReader failed` | 权重文件损坏 | 重新从可靠源拷贝权重文件 |

---

## train_ddp.sh — 多卡 DDP 训练

**调用**：`lib/train_single.py`（通过 `torch.distributed.run`）

**适用**：数据并行多卡训练，每张卡处理完整模型，数据分片

```bash
NPROC_PER_NODE=4 DEVICE=npu DISTRIBUTED_BACKEND=hccl \
  STEPS=10 bash scripts/training/train_ddp.sh
```

**常见错误**：

| 错误 | 原因 | 解决 |
|------|------|------|
| `libcublas.so not found` | CUDA 驱动库未加载 | Ascend 使用 `hccl` 后端，PPU 使用 `nccl` |
| `torch.distributed not initialized` | 通信后端不可用 | 确认 `DISTRIBUTED_BACKEND` 与平台匹配 |
| 25km DDP OOM | 单卡无法放下全模型 | DDP 不做模型切分，25km 需用 `train_mp.sh` |

---

## train_mp.sh — 模型并行训练

**调用**：`lib/train_mp.py`

**适用**：25km 全尺度训练。将模型三个 GNN 分布到 4 张 NPU，并通过 grid 分区减少反向传播显存。

**核心参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MESH2GRID_GRID_PARTITIONS` | 8 | grid 分区数，增大可减少反向传播显存 |
| `ACTIVATION_CHECKPOINTING` | 1 | 激活检查点，以时间换空间 |
| `GRID2MESH_NODE_CHUNK_SIZE` | 8192 | grid2mesh 节点分块大小 |
| `MESH2GRID_EDGE_CHUNK_SIZE` | 2048 | mesh2grid 边分块大小 |
| `TRAIN_SCOPE` | all | `all`=全参数，`decoder-grid`=仅训练输出头 |
| `STEPS` | 3 | 训练步数 |

**运行示例**：

```bash
ASCEND_RT_VISIBLE_DEVICES=8,9,10,11 \
MESH2GRID_GRID_PARTITIONS=8 \
STEPS=3 \
bash scripts/training/train_mp.sh
```

**常见错误**：

| 错误 | 原因 | 解决 |
|------|------|------|
| OOM（grid2mesh forward） | 25km grid2mesh 边过多 | 减小 `GRID2MESH_NODE_CHUNK_SIZE`（如 4096） |
| OOM（backward 碎片化） | PyTorch 内存碎片 | 设置 `PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256` |
| `No running processes` 但 OOM | 其他容器挂载了设备，预占了 HBM | 换用真正空闲的 NPU |
| `Failed to start the device` | NPU 被其他进程占用 | 重新检查 `npu-smi info`，换空闲 NPU |
| `libcublas.so not found` | Docker 镜像不匹配 | Ascend 用 `vllm-ascend` 镜像，PPU 用 `inference-xpu-pytorch` |

### 模型并行原理

```
grid2mesh GNN  →  NPU 0（编码器+1步处理器+解码器）
mesh GNN       →  NPU 0→1→2→3（16步处理器均匀分布）
mesh2grid GNN  →  NPU 3（编码器+1步处理器，按 grid 分区逐段反向）
```

### Grid 分区原理

将 1,038,240 个 grid 节点切为 N 个连续分区。每个分区只处理相关的 mesh2grid 边，前向和反向独立完成，大幅减少单卡峰值显存。

---

## lib/ 目录

存放训练相关的 Python 模块，供 Shell 脚本调用：

| 文件 | 功能 |
|------|------|
| `train_single.py` | 单卡训练 loss 计算（Google 风格归一化加权 MSE） |
| `train_ddp.py` | DDP 辅助函数（数据加载、训练范围设置等） |
| `train_mp.py` | 模型并行训练（多平台，支持 Ascend/PPU/CUDA） |
| `summarize.py` | 训练指标汇总（mean loss、step time 等） |
