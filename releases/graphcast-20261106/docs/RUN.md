# GraphCast 开发与复现说明

适用版本：GraphCast 20261106，BW1100、operational 0.25°、13 层，FP32 存储、RNE＋XF32 计算。
主流程使用固定 Docker 环境：准备材料 → 启动容器 → 校验与构建 → 短检查 → 完整推理与训练 → 结果验收。

完整材料位于 `/recovery/graphcast-testuser-20261113/delivery/`，包含镜像归档、运行时快照、输入和 H100 参考结果。
执行前，确认当前账户可读取交付目录、可使用 Docker，且拥有命令中指定的工作目录。

完整材料约 48 GB；单次完整运行的结果、checkpoint、缓存和 trace 约需 15 GiB，另存于工作目录。
首次加载镜像还需 Docker 自身的存储空间；重复运行会创建新的结果目录。

## 完整复现

```bash
cd /recovery/graphcast-testuser-20261113/delivery/graphcast-20261106
bash environment/container.sh .. /recovery/graphcast-testuser-20261113/work \
  -- bash scripts/reproduce.sh
```

启动后会列出设备使用率和显存占用，输入已分配给本任务的卡号即可；PCI 地址由脚本自动读取。
执行前确认所选卡可用于本任务。命令依次完成环境校验、构建、短检查、三轮 40 步推理、
12 阶段 38 次训练更新和 H100 精度对照；任何执行失败会停止。全部结果及 `reproduce.log` 写入工作目录，
每次运行自动创建新目录。下文提供同一流程的分步命令，整套运行时不需要再重复执行。

已经进入容器时，直接运行 `bash scripts/reproduce.sh`。
非交互运行可指定卡号，例如 `bash scripts/reproduce.sh 3`，同样无需填写 PCI 地址。

## 1. 环境与材料准备（宿主机）

使用已安装匹配海光驱动、可执行 Docker 的 BW1100 机器。先由管理员分配一张空闲卡，并确认卡号。
设备名称可能显示为 BW1200；代码按配套平台配置核对实际设备。

配套材料与源码包放在同一交付根目录，布局如下：

```text
delivery/
├── graphcast-20261106/          本源码包
├── runtime-image.tar           固定 Docker 镜像归档
├── runtime/                    与镜像配套的运行时目录快照
│   ├── work1/
│   ├── public/
│   └── opt/flagos/
├── data/
│   ├── prepared-inference/     manifest.json、reference.npz、assets/
│   ├── training/               source.json、hres-20220101-025-13-12.nc
│   ├── reference-graph/        推理 reference-graph.json 与 reference-graph.pkl.gz
│   ├── reference-graph-training/  训练的同名两文件
│   └── h100-reference/         冻结的 reference/、runs/、training/ 等完整相对目录
└── SHA256SUMS                  配套材料的文件校验清单
```

`prepared-inference/assets/` 包含 `jax_weights.npz`、`inputs.nc`、`rollout_forcings.nc`、
`target_template.nc` 及 `stats/` 下三份归一化统计。使用配套的预处理输入，无需重新下载原始气象数据。

H100 参考版本为 `h100-earth-tf32-20261103-v1`：
清单位于 `h100-reference/reference/h100-earth-tf32-20261103-v1/REFERENCE.json`；
训练报告位于 `training/r01/report.json` 至 `training/r12/report.json`。
推理预测、报告、计时文件按该清单的相对路径保留，不能只复制单个预测文件。
H100 参考结果用于离线比较，运行时无需额外的 H100 机器。

本版固定镜像 ID 为 `sha256:6d4813ee61bb39a439d96bb28b62371628d9eb0a2e62cc4a84a5218403bb0aad`。
**该 ID 是本地镜像身份，不是下载地址。** 它需要上述运行时快照共同工作；镜像归档、快照和校验清单应由交付方一起提供。
宿主机的 `/opt/hyhal` 使用该机器已安装且兼容的驱动库。

## 2. 进入运行环境（宿主机）

进入交付材料中的源码目录，运行以下命令打开容器终端：

```bash
cd /recovery/graphcast-testuser-20261113/delivery/graphcast-20261106
bash environment/container.sh .. /recovery/graphcast-testuser-20261113/work
```

交付目录用于只读访问，工作目录用于保存配置、构建产物和运行结果。
迁移到其他机器时，完整复制交付目录，将两条命令中的路径改为实际位置，并选择当前账户拥有的独立工作目录。
当前账户须能直接执行 Docker，宿主机须已安装匹配的海光驱动。

入口首次校验配套材料、按需加载固定镜像并创建容器，此后重复执行同一条命令即可复用。
容器名自动使用当前账户的 UID；同一个源码版本复用原生库，每次进入生成独立结果目录。
每次运行及各训练阶段仍需进行 JAX 编译和预热。
容器内路径、Python 环境和数据配置自动设置，不需要手工填写挂载参数或激活 Conda。

后续命令都在打开的容器终端执行。完成后输入 `exit` 回到宿主机；工作文件会归还当前账户，
容器保留待下次使用。退出前请结束本次运行，不要把任务放到后台后退出。
更新源码版本时，将源码包放在同一交付目录的另一个版本子目录中，从新目录执行相同入口即可。
若环境版本或数据约定变化，应使用该版本配套的运行说明。

## 3. 选择已分配的设备（容器内）

```bash
set -euo pipefail
selection=$(python -m graphcast_bench.platforms.device)
read -r DEVICE DEVICE_PCI <<< "$selection"
export DEVICE DEVICE_PCI
cat "$CONFIG_PATH"
```

入口已核对源码完整性、Python 3.10、JAX/jaxlib 0.6.0 和 dm-haiku 0.0.14。
`CONFIG_PATH`、`BUILD_ROOT`、`RUN_DIR` 分别指向本账户的配置、当前源码构建目录和本次结果目录。
数据统一从只读的 `/data` 读取；代码和运行时同样只读。
GPU 环境不得安装 CPU 测试依赖来替换已有 JAX。

## 4. 构建原生库（容器内）

首次运行会构建推理和训练原生库；再次运行时校验并复用已有构建。

```bash
for kind in inference training; do
  if [ ! -d "$BUILD_ROOT/$kind" ]; then
    python scripts/build_kernels.py --kind "$kind" --hipcc "$HIPCC" \
      --include "$CONDA_PREFIX/include" --library-dir "$CONDA_PREFIX/lib" \
      --output "$BUILD_ROOT/$kind"
  fi
done
python - <<'CHECK'
import os
from pathlib import Path
from graphcast_bench.platforms.native import verify_native
build = Path(os.environ["BUILD_ROOT"])
verify_native(build / "inference/libxf32_rne_router.so", "inference")
verify_native(build / "training/training-router-candidate.so", "training")
print("两套原生库与当前源码一致")
CHECK
```

先预览完整推理的执行计划：

```bash
python scripts/inference.py --paths "$CONFIG_PATH" --device "$DEVICE" --pci "$DEVICE_PCI" \
  --output "$RUN_DIR/inference" --profile full40 --dry-run
```

`--dry-run` 只生成计划、检查参数格式，**不检查全部文件是否存在，也不执行 GPU**。实际校验在正式启动时进行。

## 5. 基础运行检查（容器内）

依次执行一轮单步推理和训练前两次更新。它们使用独立输出目录，完整训练会从初始权重重新开始。

```bash
python scripts/inference.py --paths "$CONFIG_PATH" --device "$DEVICE" --pci "$DEVICE_PCI" \
  --output "$RUN_DIR/smoke-inference" --profile single
python scripts/validate.py completion "$RUN_DIR/smoke-inference" single

python scripts/train.py --paths "$CONFIG_PATH" --device "$DEVICE" --pci "$DEVICE_PCI" \
  --output "$RUN_DIR/smoke-training" --rollout 1 --limit 2
python scripts/validate.py completion "$RUN_DIR/smoke-training" training-smoke
```

两次检查均输出 `"status": "execution_complete"` 后继续。首次启动含编译和预热，短检查也会加载完整模型。

## 6. 完整推理（容器内）

```bash
python scripts/inference.py --paths "$CONFIG_PATH" --device "$DEVICE" --pci "$DEVICE_PCI" \
  --output "$RUN_DIR/inference" --profile full40
python scripts/validate.py completion "$RUN_DIR/inference" full40
python scripts/validate.py inference --candidate "$RUN_DIR/inference/run" \
  --reference /data/h100-reference --steps 40 --output "$RUN_DIR/inference-comparison"
```

每轮预热 5 步，随后独立执行 40 步，共 3 轮。比较 +240h 末帧，H100 作分母，6 个高空变量 × 13 层等权相对 MAE，Z500 单列。
`full40` 采集 trace；只需独立计时时可用 `--profile none` 和另一个全新输出目录。

## 7. 完整训练：12 个阶段、38 次更新（容器内）

以下循环串行执行，每个 horizon 启动新进程，检查成功后把 checkpoint 传给下一阶段。
horizon 1 执行更新 1～16；horizon 2～12 各两次，每次更新累积两个微批梯度。

```bash
(
  set -euo pipefail
  for horizon in {1..12}; do
    printf -v tag 'r%02d' "$horizon"
    output="$RUN_DIR/train-$tag"
    resume_args=()
    validation_args=()
    if [ "$horizon" -eq 1 ]; then
      updates="$(seq -s, 1 16)"
    else
      printf -v previous 'r%02d' "$((horizon - 1))"
      checkpoint="$RUN_DIR/train-$previous/checkpoint.pkl"
      resume_args=(--resume "$checkpoint")
      validation_args=(--predecessor "$checkpoint")
      first_update=$((2 * horizon + 13))
      updates="$first_update,$((first_update + 1))"
    fi
    python scripts/train.py --paths "$CONFIG_PATH" --device "$DEVICE" --pci "$DEVICE_PCI" \
      --output "$output" --rollout "$horizon" "${resume_args[@]}"
    python scripts/validate.py completion "$output" training-stage \
      --updates "$updates" --rollout "$horizon" "${validation_args[@]}"
  done
)
```

任何阶段失败，循环立即停止；完成第 12 阶段后，最终 checkpoint 位于 `$RUN_DIR/train-r12/checkpoint.pkl`。
训练完成检查包括更新连续性、前序 checkpoint 哈希和保存后重新加载。

再比较全部 12 个阶段的对应更新 loss；即使某一阶段未达误差线，也会继续保存其余阶段的比较结果：

```bash
(
  comparison_failed=0
  for horizon in {1..12}; do
    printf -v tag 'r%02d' "$horizon"
    python scripts/validate.py training-loss "$RUN_DIR/train-$tag" \
      --reference-root /data/h100-reference --output "$RUN_DIR/train-$tag-loss.json" \
      || comparison_failed=1
  done
  test "$comparison_failed" -eq 0
)
```

这是 38 次更新的短程一致性验证，不代表模型收敛。

## 8. 结果检查与输出文件

| 检查内容 | 成功判据与文件 |
| --- | --- |
| 执行完成 | 推理及所有训练阶段的 `python-exit-code.txt` 为 0；`completion-validation.json` 为 `execution_complete` |
| 推理精度 | `inference-comparison/summary.json` 中 `all_pairs_under_5_percent` 为 true |
| 训练精度 | 全部 `train-rXX-loss.json` 中 `internal_under_5_percent` 为 true，共覆盖更新 1～38 |
| 阶段误差线 | 推理 `all_pairs_at_most_10_percent`、训练 `stage_at_most_10_percent` 分别记录是否 ≤10% |
| 性能记录 | 推理 `inference/run/report.json`、各训练目录 `report.json` 及 trace；精度通过不等于性能达标 |

比较命令以 <5% 为退出成功条件；若退出非零，应查看已保存的数值和 ≤10% 判定，不能把进程执行完成当成精度通过。
性能达标需另按双方约定的计时、显存和 MFU 口径判断，当前验证命令不会自动给出性能达标结论。

保留入口显示的本次结果目录、构建目录、`paths.json` 及本源码包的 `release.json`。
它们保存在宿主机工作目录中；容器路径 `/workspace/` 对应本账户选择的工作目录。
反馈问题时，提供失败步骤、终端错误和对应运行目录。

## 9. 中断与常见问题

- **重复运行提示目录已存在**：输出禁止覆盖。退出并再次进入会自动创建新的结果目录。
- **构建中断**：将入口显示的构建目录中对应的失败子目录移走后，重新执行第 4 节；不要跳过原生库校验。
- **终端意外断开**：重新运行同一环境入口可恢复文件所有权。先检查旧进程是否仍在运行，再开始新任务。
- **找不到 Python、hipcc 或动态库**：检查 runtime 三处挂载及宿主 `/opt/hyhal`，不要用系统 Python 或临时安装其他 JAX 替代。
- **设备/PCI 不符或显存不足**：核对宿主机分配的卡号及占用，不能通过修改参考配置跳过设备检查。
- **输入、图或原生库哈希不符**：核对配套材料和版本，在当前源码包重新构建；不要修改校验值绕过检查。
- **训练中断**：从最后一个完成验证的 checkpoint 接续，在新目录重跑失败 horizon。只完成短检查时仍停留在 horizon 1，应先继续 horizon 1；不能直接跳到 horizon 2。
  修改第 7 节循环起点和前序 checkpoint 路径时，要同步修改完成检查的 `--predecessor` 与后续阶段引用。

CPU 自检是可选开发检查，需单独的 Python 3.10 虚拟环境；安装
`environment/requirements-cpu.txt` 和 `environment/requirements-test.txt` 后执行：

```bash
JAX_PLATFORMS=cpu LD_LIBRARY_PATH= \
XLA_FLAGS='--xla_cpu_multi_thread_eigen=false --xla_cpu_use_thunk_runtime=false' \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:third_party/graphcast \
python -m unittest discover -s tests -q
```

CPU 自检不替代上述 GPU 执行与验收。
