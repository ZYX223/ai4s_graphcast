# 2026-10-26～30：FlagOS JAX 推理与38次训练交付

本轮在旧提交 `47129e4b2654bc1c8e99c5b77bb7d54b4a191611` 上新增独立实验归档，不改旧 baseline、官方源码或冻结输出。归档源码字节及小型结果的 SHA-256 见 `evidence/20261030/SOURCE_MANIFEST.json`；运行 `python3 scripts/verify_release_20261030.py` 校验。

## 代码导航

| 目录 | 用途 |
| --- | --- |
| `experiments/flagos_gemm_ex/` | 推理 SGEMM/GEMM_EX RNE 路由、两个 Linear 探针、79处激活侧融合、40步及H100误差比较 |
| `experiments/flagos-inference-20261030/baseline/` | 本次 FlagOS 推理实际使用的配置及冻结依赖；与仓库顶层旧 baseline 分开 |
| `experiments/h100-gpu41-tf32-20261029/` | H100 TF32 的实际启动器、clone3 兼容包装及数据读取适配 |
| `experiments/mfu-40step-20261030/` | 三路40步卡上采集、事件并集、MFU、表格与复核脚本 |
| `experiments/flagos-inference-final-20261030/` | 推理逻辑算子汇总、Linear补测及kernel附件生成 |
| `experiments/flagos-training38-20261030/experiment/` | 两平台共用38次训练、AdamW、checkpoint完整性检查与算子重放 |
| 同目录 `h100-policy/`、`hygon-policy/` | H100阶段/消息/时间重计算；海光额外LayerNorm重计算 |
| 同目录 `router/` | 训练专用RNE路由；每次重新舍入更新后的权重及反向张量，不使用推理权重缓存 |
| 同目录 `analysis/`及根目录分析脚本 | 有效前后向FLOPs、完整XSpace/Nsight计时、loss对照和MFU |
| 同目录 `source-snapshots/` | 实测版本与最终恢复守卫版本分别保存 |
| 同目录 `layernorm-debug/` | 编译断言最小复现、规避验证与资源检查；5个native GPU前向耗时待补测 |
| `evidence/20261030/` | 小型结果摘要、38次loss、计时及环境版本；完整预测/checkpoint/trace在交付目录 |

历史脚本保留原设备号、绝对路径、启动命令。不要直接在原实验目录运行或覆盖结果。下面按容器挂载复用归档代码；新结果必须写入新的可写目录。此次整理验证源码、CPU检查、编译和交付文件，不声称迁移后GPU复跑。

## 结果与评价口径

| 路径 | 推理40步卡上(s) / MFU | 训练38次卡上(s) / MFU | 对H100训练loss最大/平均误差 |
| --- | --- | --- | --- |
| H100 TF32，无显式RNE | 12.658 / 18.387% | 536.957 / 10.7896% | 参考 |
| BW1100 native RNE＋XF32 | 29.568 / 22.854% | 1837.960 / 9.1516% | 0.2646% / 0.1578% |
| BW1100 FlagOS JAX fl_only RNE＋XF32 | 54.675 / 12.359% | 2975.674 / 5.6526% | 0.1880% / 0.0832% |

native训练使用10月23日历史结果复算，发行版/重计算/采集工具不同，不当作同环境严格A/B。推理各三轮中位数；训练每条路径一条完整轨迹，未评估整轮方差。训练单卡覆盖1～12步、38次优化器流程，2次学习率为零、36次实际参数变化；短程loss通过不代表收敛。

推理固定同一预训练权重/输入/统计，独立自回归40步，比较+240h末帧。参考为10月29日gpu41 H100 TF32；各字段相对MAE为 mean(abs(candidate-reference))/mean(abs(reference))，整体按6个高空变量×13层等权平均，Z500单列。FlagOS整体0.5155%～0.6036%、Z500 0.005827%～0.007779%；native整体0.5600%～0.6760%、Z500 0.006629%～0.008206%。不是天气真值精度或所有时效评价。

训练只比较对应更新的loss（两个微批次的更新前loss均值）：abs(candidate−H100)/abs(H100)。不再使用两份分别训练的checkpoint输出互比评价训练。内部两项<5%，阶段≤10%；MFU比值为候选峰值归一化，正式效率验收口径待确认。当前训练FlagOS/H100 MFU=52.3894%，不能宣称达到60%目标。

MFU=有效GEMM FLOPs/(卡上计算kernel区间并集×稠密峰值)。H100/BW1100采用494.7/170.3936 TFLOP/s；40步有效量1151.39795697664 TFLOPs，38次训练约28.6608 PFLOPs。训练有效前后向量剔除重计算重复计数，RNE、重计算、优化器时间保留在分母；CPU、拷贝及kernel间隙排除。同步更新墙钟和端到端时间分别记录。

## 环境与数据

本轮后端是 **FlagOS JAX（fl_only），JAX/XLA→FlagTree，GEMM仍由rocBLAS执行**，不称为FlagGems算子库接入。

- 官方GraphCast：`08cf73625c9d12bd9aaa038868bcb2fe488f2a22`，0.25°、721×1440、13层、batch1、FP32存储。无BF16。
- BW1100、DTK26.04；实测FlagOS镜像 `flagos-r6-optimization-test:20261026-c93db685-bw59-hal642`，ID `sha256:6d4813ee61bb39a439d96bb28b62371628d9eb0a2e62cc4a84a5218403bb0aad`。JAX/jaxlib 0.6.0；FlagTree `0.7.0rc2+hcu3.6.glibc228.compat1.git76201fd9`。完整版本见 `evidence/20261030/flagos-training38-20261030/environment-version.json`。
- H100镜像及Python路径沿用冻结baseline的 `configs/h100.json`，H100 TF32不加显式RNE、不加载海光路由。
- SCNET根目录：`/work2/ac9844np6r/zyx/delivery/graphcast-jax-weekly-20261030/`。`code/`为本次Git提交；`results/`为本周原始结果；`assets/inference/`、`assets/training/`关联已经存在的相同输入/权重；`runtime/`为实测训练依赖补充；`validation/`保存逐文件核验及权限检查。
- 甲方共用账号为`ac9844np6r`。交付不包含容器镜像；GPU复跑需匹配运行环境和空闲设备。SCNET海光不能重新执行H100 CUDA任务；可读取冻结H100结果并重算误差。

## 容器内复现：海光推理

在独立容器/作业中将本次推理baseline只读挂到 `/tmp/zyx-flagos-rne-inference-20261029/baseline`，将`experiments/flagos_gemm_ex`只读挂到 `/tmp/zyx-flagos-rne-gemm-ex-20261030`。后者是实测策略校验使用的固定位置。另将`assets/inference`挂为`/weekly`，新建输出目录挂为`/output`；保留厂商`/opt/hyhal`及DTK库映射。设备选择应先确认空闲，以下变量必须由操作者设定。

```bash
set -euo pipefail
: "${GC_CARD:?选择已确认空闲的物理卡}" "${GC_PCI:?对应PCI地址，小写}"
export ROCR_VISIBLE_DEVICES="$GC_CARD" HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0
export GRAPHCAST_PHYSICAL_DEVICE="$GC_CARD" GRAPHCAST_EXPECTED_PCI="$GC_PCI"
base=/tmp/zyx-flagos-rne-inference-20261029/baseline
candidate=/tmp/zyx-flagos-rne-gemm-ex-20261030
export PYTHONPATH="$candidate/model-experiment:$base/src:$base/third_party/graphcast"
export LD_LIBRARY_PATH="/opt/hyhal/lib:${LD_LIBRARY_PATH:-}"
unset LD_PRELOAD ROCBLAS_MATH_MODE NVIDIA_TF32_OVERRIDE JAX_DUMP_IR_TO
export XLA_MODE=fl_only JAX_PLATFORMS=rocm JAX_ENABLE_X64=false
export JAX_DEFAULT_MATMUL_PRECISION=tensorfloat32
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export XLA_FLAGS='--xla_gpu_enable_command_buffer= --xla_gpu_graph_level=0'
export XLA_FLAGTREE_COMPILE_WORKERS=12 XLA_FLAGTREE_PERSISTENT_WORKERS=12
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 PYTHONNOUSERSITE=1
export GRAPHCAST_EXPERIMENT="$candidate/model-experiment" NATIVE_OUTPUT=/output/native
python "$candidate/build.py"
export GRAPHCAST_ROUTER_LIBRARY=/output/native/libxf32_rne_router.so
export GRAPHCAST_STAGE_WEIGHTS="$base/kernels/xf32_rne/weights.tsv"
export GRAPHCAST_FUSED_LAYERS="$candidate/model-experiment/fused_layers.txt"
export GRAPHCAST_STAGE_LOG=/output/gemm.jsonl LD_PRELOAD="$GRAPHCAST_ROUTER_LIBRARY"
python "$candidate/model-experiment/entry.py" --platform hygon --mode rne_xf32 \
  --variant fused --data /weekly --output /output/benchmark-new --repeats 3
```

上述模型入口保留历史官方JAX FP32参考的内置诊断。**本周H100对照必须另运行下面的比较**：将交付的`results/h100-gpu41-tf32-20261029/benchmark-r1/run/`只读挂为`/reference`，三轮预测和各轮report需完整存在；使用同一冻结参考，不换成重新训练的checkpoint。

```bash
unset LD_PRELOAD
python "$candidate/compare_h100.py" --candidate /output/benchmark-new \
  --reference /reference --case "$base/configs/case.json" \
  --output /output/h100-comparison-new
```

比较器检查模型版本、输入标识和逐轮输出SHA，生成同轮三组及全部九组末帧配对、内部<5%和阶段≤10%结果。

独立路径探针、单步及40步卡上采集沿用归档`adapter_probe.py`、`native_probe.py`、`run-model.sh`和`experiments/mfu-40step-20261030/`；将原输出和设备号改到新的工作目录。K=4的两个GEMM保留rocBLAS FP32回退，不将math-mode标志当作真实XF32执行证据。H100复跑使用`experiments/h100-gpu41-tf32-20261029/launch.py`，其冻结目录结构见同目录`RUN_RECORD.json`；在H100上执行，不在海光上运行CUDA脚本。

### H100工作目录准备

在有H100和已登记镜像的主机上准备新目录。`GC_REPO`为当前Git仓库（包含旧提交），`GC_DATA`为交付`assets/inference`的可读实际目录，`GC_WORK`必须不存在。Git快照保留原H100配置；不会在源码归档下写入结果。

```bash
set -euo pipefail
: "${GC_REPO:?本机Git仓库}" "${GC_DATA:?推理数据}" "${GC_WORK:?新的H100工作目录}"
mkdir "$GC_WORK"
mkdir "$GC_WORK/code"
git -C "$GC_REPO" archive 47129e4b2654bc1c8e99c5b77bb7d54b4a191611 | tar -xf - -C "$GC_WORK/code"
cp "$GC_REPO/experiments/h100-gpu41-tf32-20261029/launch.py" "$GC_WORK/"
cp -a "$GC_REPO/experiments/h100-gpu41-tf32-20261029/runner" "$GC_WORK/runner"
ln -s "$(realpath "$GC_DATA")" "$GC_WORK/data"
# 不启用GPU的环境检查；随后选择已确认空闲的H100卡。
python3 "$GC_WORK/launch.py" check --label check-new
python3 "$GC_WORK/launch.py" benchmark --label benchmark-new --device "$GC_CARD"
```

SCNET `code/`为源码快照，另提供`repository.bundle`供离线`git clone`，可取得上述冻结旧提交。没有H100时直接用交付冻结输出进行数值比较，不声称已复跑H100。

## 容器内复现：38次训练

将`code/`挂到`/code`、本交付`runtime/`挂到`/runtime`、`assets/inference/`挂到`/weekly`、`assets/training/`挂到`/training-data`，新的可写目录挂到`/output`。使用上述匹配FlagOS镜像及厂商设备/库映射。训练路径**不加载推理producer-RNE或冻结权重缓存**。

```bash
set -euo pipefail
ulimit -c 0
: "${GC_CARD:?选择已确认空闲的物理卡}" "${GC_PCI:?对应PCI地址，小写}"
root=/code/experiments/flagos-training38-20261030
export PHYSICAL_DEVICE="$GC_CARD" EXPECTED_PCI="$GC_PCI"
export ROCR_VISIBLE_DEVICES="$GC_CARD" HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0
export TRAINING_PLATFORM=hygon TRAINING_DATA=/training-data TRAINING_WEEKLY=/weekly
export PYTHONPATH="/runtime/training-deps:/runtime/training-official:$root/hygon-policy:$root/experiment"
export LD_LIBRARY_PATH="/opt/hyhal/lib:${LD_LIBRARY_PATH:-}"
unset LD_PRELOAD ROCBLAS_MATH_MODE NVIDIA_TF32_OVERRIDE JAX_DUMP_IR_TO XLA_PYTHON_CLIENT_ALLOCATOR
export XLA_MODE=fl_only JAX_PLATFORMS=rocm JAX_ENABLE_X64=false
export JAX_DEFAULT_MATMUL_PRECISION=tensorfloat32 XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS='--xla_gpu_enable_command_buffer= --xla_gpu_graph_level=0'
export XLA_FLAGTREE_COMPILE_WORKERS=12 XLA_FLAGTREE_PERSISTENT_WORKERS=12
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 PYTHONNOUSERSITE=1
export TRAINING_JAX_TRACE=1 MEMORY_COMPILE_ONLY=0
export TRAINING_REMAT_SHA256=28d4a02d1401d154d77e510cedcbf91df126f3a9beaa3d2a0743684b4da49a86
export TRAINING_ROUTER=/output/training-router.so
/public/software/compiler/dtk-26.04/bin/hipcc --offload-arch=gfx938 -O2 -fPIC -shared \
  -I"$root/router" "$root/router/training_router.cc" "$root/router/rne_kernel.hip" \
  -o "$TRAINING_ROUTER" -ldl -pthread
export LD_PRELOAD="$TRAINING_ROUTER"
for horizon in $(seq 1 12); do
  label=$(printf 'flagos-r%02d' "$horizon")
  export TRAINING_OUTPUT="/output/$label"
  mkdir "$TRAINING_OUTPUT"
  export TRAINING_GEMM_LOG="$TRAINING_OUTPUT/gemm.jsonl"
  export TMPDIR="$TRAINING_OUTPUT/tmp"
  mkdir "$TMPDIR"
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.70
  if [ "$horizon" -ge 11 ]; then export XLA_PYTHON_CLIENT_MEM_FRACTION=0.73; fi
  args=(--mode rne_xf32 --rollout "$horizon")
  if [ "$horizon" -gt 1 ]; then
    previous=$(printf '/output/flagos-r%02d/checkpoint.pkl' "$((horizon-1))")
    args+=(--resume "$previous")
  fi
  bash "$root/run-child.sh" python "$root/entry.py" "${args[@]}"
done
```

`run-child.sh`保存真实Python退出码；后续阶段校验前一阶段报告完成状态、退出成功和checkpoint SHA，失败时停止。单步检查可先只运行horizon1并加`--limit 1`，但该日程首次学习率为零；有效更新验证应至少包含第二次更新，不把零学习率流程称为参数已更新。

H100采用同一`experiment/`、`h100-policy/`和数据，`TRAINING_PLATFORM=h100`、`JAX_PLATFORMS=cuda`、`--mode tf32`，取消全部海光LD_PRELOAD。按归档`run-h100.sh`与`launch_h100.py`挂载；1～10步额度94%，11～12步使用`run-h100-r11plus.sh`的96%。每个阶段使用独立进程并从自己的checkpoint恢复。H100卡上时间使用Nsight CUDA kernel事件；海光使用完整XSpace PB，不使用被100万事件上限截断的Chrome导出。

主机采样整卡显存、原始数据SHA、各阶段命令及退出记录见归档启动器；只运行上面的容器内命令时不会自动得到主机整卡显存采样，勿与JAX分配器峰值混用。

## 原始结果复算及待补

结果目录中保留实际版本的`calculate.py`、`summarize_training.py`、`compare_historical_native.py`与表格生成器。复算请复制到新目录后执行，避免覆盖冻结摘要；历史绝对路径按`SOURCE_MANIFEST.json`和交付索引映射。参考原始CSV、trace、FLOPs与源码校验值，不用本机新运行输出覆盖周报。

LayerNorm仅保留待补测。已证明禁止`horizontal-loop-fusion-for-copy`可规避一个native输出拷贝融合断言；尚未完成全部5形状GPU耗时对照，不将中断样本纳入统计。外部HIPPROF退出清理139是历史工具限制，本轮训练普通退出及JAX原生采集通过，不宣称修复了HIPPROF。
