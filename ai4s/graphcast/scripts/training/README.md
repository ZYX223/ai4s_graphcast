# 训练脚本说明

## 正式 TF32 自回归入口

`run_training.py` / `train_official.sh` 连接真实 ERA5 corpus、可微多 lead
反馈、加权 loss、AdamW 和优化器边界恢复。默认参数、激活与 loss 使用 FP32，
通过现有 `configure_float32_math` 请求 TF32（Ascend 为 HF32）；
`--tensor-core disabled` 请求 IEEE FP32。不使用 autocast 或 BF16。
旧的短 `train_mp.sh` 等入口继续用于已有算子回归；下文旧脚本的设备历史记录
不构成此正式入口的显存或性能验收。

本次证据仅为 CPU 张量/真实小网格数据集成、两进程 CPU Gloo 和 CLI dry-run。
尚未验证完整模型的四卡显存适配、12 lead TF32 + AdamW、加速器通信、
训练天气技能或性能。前向/反向 FlagGems coverage 标为 `unverified`，
training MFU 为 `null`（没有已验证的训练 FLOPs 分子与匹配峰值）。

从 `ai4s/graphcast` 目录执行：

```bash
# 仅检查参数、阶段预算和逻辑拓扑，不加载数据、权重、torch 或 FlagGems。
bash scripts/training/train_official.sh --dry-run --dp 1 --devices 0,1,2,3
python scripts/training/run_training.py --dry-run --dp 2 \
  --devices 0,1,2,3,4,5,6,7 --global-batch-size 6 --micro-batch-size 2

# 后续完整模型设备验证的调用形式（本次未执行）。
python scripts/training/run_training.py \
  --corpus /data/era5/corpus.json --stats /data/stats \
  --pretrained-checkpoint /data/operational_pytorch.pt \
  --device-type cuda --backend native --dp 1 --global-batch-size 4 \
  --micro-batch-size 1 --checkpoint-segments 2 --loss-grid-chunk-size 16384 \
  --max-updates 2 --checkpoint-every 1 --output-dir /runs/native-tf32

# 一个 DP 进程拥有四个 MP 设备；两进程拥有八个可见逻辑设备。
torchrun --standalone --nproc-per-node=2 scripts/training/run_training.py \
  --dp 2 --devices 0,1,2,3,4,5,6,7 --global-batch-size 6 --micro-batch-size 2 \
  --corpus /data/era5/corpus.json --stats /data/stats \
  --max-updates 2 --output-dir /runs/native-dp2

# 恢复时保持 config、corpus 内容、stats、种子、拓扑、batch 和阶段预算一致。
python scripts/training/run_training.py \
  --corpus /data/era5/corpus.json --stats /data/stats \
  --resume-checkpoint /runs/native-tf32/update-000000002.pt \
  --device-type cuda --backend native --dp 1 --global-batch-size 4 \
  --micro-batch-size 1 --checkpoint-segments 2 --loss-grid-chunk-size 16384 \
  --max-updates 4 --checkpoint-every 1 --output-dir /runs/native-tf32
```

`--pretrained-checkpoint` 是已有转换工具生成的 PyTorch state dict，不能与
`--resume-checkpoint` 同时提供；两者都不提供时从固定种子初始化。
恢复文件需要同目录 `.sha256` sidecar。`--max-updates` 是绝对停止步数，
可从 2 改为 4 而不改变 schedule。只有成功完成 optimizer update 后才能保存；
每个保存包含模型、AdamW、schedule、下一批 cursor 和各 DP rank 的 RNG。
保存的是内容 SHA256 与训练语义，因此原样搬迁 corpus/stats 路径可恢复，
同路径改动数据、stats、配置或精度策略会被拒绝。首次正式运行/恢复会读取完整
shard 文件计算内容哈希，可能增加启动时间。

`--global-batch-size` 是一个 optimizer update 的全局样本槽位数；必须整除 DP。
`--micro-batch-size` 是每个 rank 一次前向/反向的最多样本数。例：DP=2、
全局 batch=6、micro=2 时，全局微批次分别为 4 和 2 个样本，累积两次。
epoch 尾部补齐样本权重为零；每次更新按全局有效样本数归一化，
loss 缩放为 `DP × 本 rank 当前微批次有效数 / 整次更新全局有效数`。
全补齐 rank 仍执行可微零 loss 和 collective，全局全补齐微批次一致跳过。
每次更新仅手动同步一次梯度、clip global norm 到 32、AdamW step 一次，
没有额外 DDP 包装。rollout lead 数与累积次数、optimizer update 数分开记录。

正式模型固定 MP=4：输入阶段在卡 0，16 个 mesh steps 的前/后八步在卡 1/2，
输出阶段在卡 3。DP rank r 使用逻辑卡 `4r..4r+3`。`--devices` 只接受从 0
起连续的完整逻辑序列；通过 CUDA/NPU/MUSA 的可见设备环境变量选择物理卡。
所有 rank 必须在同一节点看到完整设备列表，不能设置为每个进程仅可见四卡。
此入口会检查 operational 配置和真实 721×1440 四分之一度全球坐标与 13 层，
裁剪数据不会被当作完整模型验收。

默认 schedule 位于 `configs/training/operational_tf32_schedule.json`：
1000 次一步线性 warmup（0→1e-3）、299000 次一步半余弦（1e-3→0），
11000 次恒定 3e-7 的 curriculum（从 2 到 12 lead，每 1000 次加一）。
`--warmup-updates`、`--one-step-updates`、`--curriculum-updates` 可覆盖预算，
0 表示省略阶段；`--curriculum-every-updates` 可改变递增间隔。
epoch universe 始终取最大配置 rollout 可用的窗口，短阶段只缩短提取 horizon，
阶段切换不改变 epoch 顺序。每个 epoch 固定种子洗牌、有效窗口各用一次。
这没有采用同事特殊 recipe 的按 update 有放回采样；短运行不等于复现论文训练。

Native 与 FlagGems 必须在两个新进程分别启动，后者使用 `--backend flaggems`。
入口显式选择现有 training allowlist，严格要求注册成功；可用
`--flaggems-config`、`--flaggems-ops`、`--flaggems-variant` 选择现有运行时。
这些开关不证明反向覆盖；不自动降级到 Native，也不修改已安装运行时或 overlays。
日志记录请求/解析的 backend 与 precision、拓扑、有效样本、阶段/epoch/rollout、
loss/分变量 loss、有限性/裁剪前 norm、数据/计算时间、各卡峰值显存与保存状态。
下一步边界是代码审查后的完整模型 Native TF32 短更新，再取得 FlagGems
真实前向/反向 trace 与数值对照；七天 ERA5 handoff 不足以支持长期训练结论。

以下保留既有短训练脚本说明，其部署示例以 Docker 容器为主。

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
