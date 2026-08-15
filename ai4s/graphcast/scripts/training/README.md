# 训练脚本说明

本目录包含 GraphCast 在不同配置下的训练入口脚本。所有脚本应在 Docker 容器内运行。

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
