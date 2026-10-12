# 训练脚本说明

已测 H100 Native eager TF32 优化快照、完整复现参数及源码哈希见
[固定训练 baseline（2026-10-12）](../../docs/training-native-baseline-20261012.md)。
最新三轮 MFU 中位数9.0133%，历史同源码10.0121%；两者不能混写。
新增 grouped gather / microbatch 数据驻留均需显式启用，不修改下面的默认配方。
不包含后续“两样本一起上传 GPU”或 CPU buffer 复用实验。

正式入口 `scripts/train.sh`、`run_training.py`、`train_official.sh`、`train_pipeline.sh` 均调用同一份
同事训练引擎 `run_official_epoch_training.py`（源提交 `b7667b4`）。
训练专用 `training_model.py` 使用 canonical graph；推理模型、图默认值和
inference FlagGems YAML 保持原样。来源和适配见
[provenance](../../docs/training-source-provenance.md)。

参数、激活和 loss 保持 FP32；公开入口默认请求 TF32/HF32，使用现有 backend
设置 CUDA、NPU 和 MUSA 的实际开关。`--tensor-core disabled` 请求 strict FP32。
默认全局 batch 16、每 rank micro batch 1，MP4 / DP1。MP2/MP4 支持 DP1/DP2；
单卡使用 MP1/DP1，不启用 MP1/DP2。
必须满足 global batch 整除 `DP × 每 rank micro batch`；没有不满的最后一个
micro batch。推荐使用明确单位的参数：

| 含义 | 公开入口 | 底层 runner |
|---|---|---|
| 每次参数更新的全局样本数 | `--global-batch-size` | `--global-batch-size` |
| 每个 rank 一次前反向的样本数 | `--micro-batch-size-per-rank` | 全局 micro batch 除以 DP |
| 一次前反向所有 rank 的样本总数 | 每 rank micro batch 乘 DP | `--global-micro-batch-size` |

梯度累积次数 = global batch / (DP × 每 rank micro batch)，MP 不参与样本数计算。
默认 16/1/DP1 累积16次；16/1/DP2 每 rank 累积8次。
公开入口旧 `--micro-batch-size` 仍表示每 rank；底层旧 `--micro-batch-size`
仍表示全局，旧 `--batch-size` 仍表示一次更新的全局 batch。新旧别名不能同时传入，
不会采用后一个参数静默覆盖前一个。adapter 只向底层发送明确的全局参数。
dry-run 新增 `global_micro_batch_size/gradient_accumulation_steps`，保留
`canonical_micro_batch_size/accumulation_steps` 兼容字段；checkpoint schema 和字段不变。

## 顶层入口迁移

`scripts/train.sh` 现在进入正式训练，不再默认运行三步 SGD。
该顶层入口会拒绝旧 `STEPS/LR/OPTIMIZER/DEVICES/TRAIN_SCOPE`、旧 chunk/输出变量，
以及旧后端和 FlagGems 来源选择变量，防止旧环境被忽略后误跑新配方。
先清除这些旧变量，再使用明确的 CLI：

- `STEPS`：正式入口用 `--max-updates` 限制绝对 optimizer step；不等同于旧 SGD 配方。
- `DEVICES=npu:0,...`：改用 `--device-type npu --mp 4 --dp 1`；
  `--devices` 只接受从0连续的逻辑序号，物理卡选择用 `--physical-devices`。
- FlagGems：用 `--platform/--backend/--flaggems-config/--flaggems-ops/--flaggems-variant`，
  不再通过顶层旧入口的 `GRAPHCAST_BACKEND/GRAPHCAST_FLAGOS_MODE` 等变量选择。
- `LR/OPTIMIZER/TRAIN_SCOPE`：不自动迁移；正式配方为完整模型、调度学习率和 AdamW。
  需要原行为时显式运行 `scripts/training/legacy/train_mp.sh`。

`PYTHON_BIN`、`GRAPHCAST_PLATFORM` 和厂商设备可见性变量继续支持。
`FLAGGEMS_RUNTIME_DEST` 和 `GRAPHCAST_FLAGGEMS_SOURCE_ROOT` 也继续传递：
前者选择 official/internal 的已准备运行库目录；后者显式选择源码，沿用现有 backend
优先于 variant 的规则，并在实际运行时记录为 custom。两者不是旧训练配方选项。
直接调用公开 Python 入口也应使用 CLI 选择配方和后端；不要沿用旧 Shell 配置。

从 `ai4s/graphcast` 执行：

```bash
# 不加载 Torch、FlagGems、数据或权重，也不创建输出目录。
python scripts/training/run_training.py --dry-run --device-type npu \
  --mp 4 --dp 1 --global-batch-size 2 --micro-batch-size-per-rank 1

# 四种拓扑的 Native/FlagGems 命令；实际运行前替换数据和权重路径。
# 每个组合都在新进程和独立输出目录运行；DP2 自动启动两个 rank。
for backend in native flaggems; do
  for mp in 4 2; do
    for dp in 1 2; do
      python scripts/training/run_training.py \
        --device-type npu --mp "$mp" --dp "$dp" --backend "$backend" \
        --corpus /data/era5/corpus.json --stats /data/stats \
        --pretrained-checkpoint /data/operational_pytorch.pt \
        --global-batch-size 2 --micro-batch-size-per-rank 1 \
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
  --global-batch-size 2 --micro-batch-size-per-rank 1 \
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

平台内存配置以 pinned source launcher 为基础。MP4 保留 manual gradient sync，
MP2 默认 DDP；DDP accumulation 把 forward/backward 包在 `no_sync`，
最后一个 micro batch 才同步，没有第二次手动梯度平均。MUSA 多设备反向
保留源串行执行；Ascend 默认 `ASCEND_LAUNCH_BLOCKING=1`。
Ascend MP4/DP1 使用 mesh split16、逐消息步 checkpoint、reentrant outer checkpoint；
不再在反向重算时同时保留16步的 chunk 输入。MP4/DP2 保留原 stage 策略。
MP2 使用源 README 验证示例的 split8、step、non_reentrant，并保留显式 CLI
覆盖。此 MP2 设置复用不代表本次已做 MP2 设备验收。chunk、stage placement、
prefetch/cache 和 checkpoint 开关见公开 `--help`；完整 source-native 参数见：

```bash
python scripts/training/run_official_epoch_training.py --help
```

### 六平台统一单卡配置

`--mp 1 --dp 1` 在 H100、910C、BW3000、S5000、C550、810E 使用同一份
训练配置，不复制模型或训练主循环；已有 MP2/MP4 默认行为不变。

- 外层 rollout checkpoint：`reentrant`，单步预测也启用。
- Mesh：16 个消息步均在同一设备，逐消息步 checkpoint。
- 完整 Grid→Mesh / Mesh→Grid stage checkpoint 均启用；内部 chunk 的
  checkpoint 仍为 `non_reentrant`，不是把所有 checkpoint 都改成 reentrant。
- edge / mesh edge chunk 均为 8192，grid node chunk 为 4096，loss chunk 为 65536。
- `gradient_sync=manual`；DP1 无跨 rank 梯度平均，仍按 global/micro batch 累积。
- 保持 FP32 storage 和 TF32/HF32 请求，默认拼接投影；不启用 BF16 或推理投影分解。

先检查 H100 Native 单卡短测配置（不启动训练）：

```bash
python scripts/training/run_training.py --dry-run \
  --platform h100 --backend native --mp 1 --dp 1 \
  --global-batch-size 2 --micro-batch-size-per-rank 1 --seed 27 \
  --warmup-updates 6 --one-step-updates 10 --curriculum-updates 22 \
  --max-updates 3 --checkpoint-every 1
```

其它平台只需将 `--platform` 换为 `ascend910c/hygon/mthreads/metax/thead`。
实际运行时去掉 `--dry-run` 并提供 corpus、stats、checkpoint 和独立输出目录；
选择物理卡用 `--physical-devices`。`--backend flaggems` 仍使用既有训练 YAML，
不意味着训练算子已全部接管。显式 CLI 覆盖仍优先，例如
`--rollout-checkpoint-implementation non_reentrant`；当前项目拒绝
`--gradient-sync ddp` 与 `reentrant` 外层组合，不会静默更改 checkpoint 策略。

此配置来自冻结实验 `training-mp1-20260913-A9I4Ie`：910C 单逻辑 NPU、完整
25km 模型、global2/micro1、单步预测、3 次更新（其中2次非零学习率）。
Native/FlagGems 峰值已分配显存分别为 38.2293/42.5583 GiB，配对完整区间
loss 比值差异为 0.0049423%，通过现有 5% 短测门槛。冻结实验源码和证据保持原样；
本次只是将同一单卡策略接入六平台入口，其它五平台尚未做单卡设备验收。
不能据此宣称 global16、12-lead、收敛或训练后天气技能已经通过。

训练 edge chunk 的 checkpoint 包含原始边编码（首步）、节点 gather、跨设备
拷贝、cat 和 edge MLP；保存的是原始/上一消息步的边视图与共享节点引用，
不再保存大 cat 输入。该边界统一用于 chunked Grid→Mesh、Mesh 和 Mesh→Grid。
共享聚合 buffer 的 `index_add_` 留在 checkpoint 外，重算不会重复累加；多步
Mesh 的残差边状态仍保留。参数、FP32 storage、TF32/HF32 开关、FlagGems
调用路径和推理模型不变。`--mesh-checkpoint-granularity stage` 只覆盖 Mesh
的分组策略，不关闭上述 chunk 边界；此次未加入投影分解或其它模型优化。

2026-09-10 显存对照：910C、完整25km模型、FP32 storage/HF32请求、MP4/DP1、
global batch2/micro1、seed27，同一6/10/22 schedule停在第3次更新。
以下是每个逻辑设备的峰值已分配显存（GiB）；各设备峰值不一定同时发生，不能相加
推算单卡所需显存。分配历史无截断，重建峰值与四设备运行时计数逐一吻合。

| 方案 | 设备0 | 设备1 | 设备2 | 设备3 |
|---|---:|---:|---:|---:|
| Native 旧边界 | 3.5042 | 46.6715 | 21.6843 | 47.3426 |
| Native 仅逐消息步 checkpoint | 3.5042 | 15.1122 | 21.6843 | 47.3078 |
| Native 仅完整 edge chunk checkpoint | 3.5042 | 25.3305 | 21.6843 | 22.2530 |
| Native 两项组合 | 3.5042 | 13.6122 | 21.6843 | 22.2530 |
| FlagGems 两项组合 | 3.5042 | 13.7552 | 21.8067 | 25.7471 |

Native最高峰减少53.00%；FlagGems对比9月9日原配置52.5519 GiB减少51.01%。
同配置的完整chunk边界对照loss诊断差异0.00071094%，逐消息步配置下边界对照
差异0.00947797%；最终FlagGems/Native配对为0.00696260%，均通过现有5%门槛。
FlagGems源码树摘要与先前冻结版本一致，原6个接口注册、无跳过；未另采集调用覆盖，
不能把注册数等同于前/反向实际接管数。证据标识为
`training-memory-attribution-20260910-gMfZ3v`，含原始分配快照、阶段计数、冻结源码、
训练metrics和诊断JSON。此轮是显存/正确性短测，包含冷编译和观察器开销，不报正式
性能/MFU；尚未验收global batch16、12-lead、MP2、其它平台或训练后天气技能。

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

## Mesh 投影分解实验（默认关闭）

`--mesh-edge-input-projection factorized --mesh-checkpoint-granularity step`
只改变 Mesh 边 MLP 的第一层：原权重按 edge/sender/receiver 切片，节点先投影，
各 edge chunk gather 后与边投影相加，bias 只加一次，再执行原 SiLU/Linear/LayerNorm。
Grid→Mesh、Mesh→Grid、参数键、优化器和 FlagGems 库不变；不是低秩近似。
节点投影仅在当前消息步内复用，不跨步或 optimizer update 缓存。chunk/stage
checkpoint 保留；推荐逐消息步 checkpoint，否则整段重算可能同时保留多步投影。
分解会改变浮点累加顺序，需要重新比较梯度和训练 loss，不能保证逐位一致。
forward/recompute 线性 FLOPs 按分解后的执行计算，仍不是完整算子 FLOPs。
启用状态写入恢复契约，exact resume 禁止切换；旧 checkpoint 未记录该字段时
视为 concatenated。需要切换实验时用同一 pretrained 权重重新初始化配对训练。

四卡 Native/FlagGems 显存、同数据 loss 对照及无观察器计时证据保存在
`training-mesh-factorized-20260910-QGE0NH` 实验目录；不据单块 cat 的容量推算
整卡峰值下降，不据此宣称两卡或单卡训练已经可用。

910C MP4/DP1、global=2/micro=1、单 lead 三次更新的 allocator 对照：
Native Mesh 卡峰值从 13.6122 增至 13.9247 GiB（约 +320 MiB），四卡中最高单卡
均为 22.2530 GiB。分解新增的节点投影/反向中间量未带来峰值收益，因此不默认启用。
Native 分解/拼接受控 A/B 的 loss 比值为 0.004218%；项目严格配方比较器仍返回
incomparable_training，仅有 source_recipe 的投影开关不同，实验另行审计了其余
起始状态、数据、采样/更新契约相同，没有修改诊断门槛或原始元数据。
同为分解路径的 FlagGems/Native 正式 loss 配对通过，差异 0.005933%，仍不代表
长训练收敛或天气技巧验收。
另外各跑五次无观察器更新，排除第一次后，Native compute 均值为
16.6429 → 16.1176 s/update（耗时 -3.16%），含数据处理的 E2E 为
16.6766 → 16.1512 s/update；两版最高单卡仍为 22.2530 GiB。
这是各一次运行、四个稳态样本的初步结果，不是多轮正式性能验收。
五次更新的受控 loss A/B 差异为 0.012074%。

## 历史探针

旧单卡、DDP、MP 测试及其环境变量说明已移至 [legacy/README.md](legacy/README.md)。
它们不属于正式训练验收路径；旧 Shell 路径仅作带提示的兼容转发。
`quickstart train` 仍执行旧 SGD 探针，不会自动转换成正式 AdamW 配方。
