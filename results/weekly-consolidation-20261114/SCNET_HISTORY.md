# GraphCast 本周交付：SCNET 副本与复现

记录日期：2026-11-13。固定源码为 `graphcast-20261113-r3`，原 BW1100、H100 源码和结果不改动。

## 服务器与目录

在 SCNET 账号 `ac9844np6r` 下操作。已有登录配置的研发机可用 `ssh scnet-hygon`；计算任务提交到 Slurm 分区 `hx1hdnormal01`，不要在登录节点运行 GPU 任务。

| 材料 | SCNET 路径 |
|---|---|
| 本周交付根目录 | `/work2/ac9844np6r/zyx/delivery/graphcast-delivery/` |
| 固定源码 | 根目录下 `releases/graphcast-20261113-r3/` |
| 输入、权重与统计 | 根目录下 `data/cases/`、`data/models/` |
| 对应 H100 参考 | 根目录下 `references/h100/<case>/v1/`、`full38-v1/` |
| 已完成训练的历史结果索引 | 根目录下 `results/full38-20261113-r1/` |
| 新loss规则离线复算 | 根目录下 `results/training-loss-20261113-r3/` |
| SCNET 适配代码、校验与新作业 | `/work2/ac9844np6r/zyx/outputs/graphcast/scnet-delivery-20261113/` |

结果索引包含报告、loss 和来源路径，不等于已复制原服务器的全部大体积 trace/checkpoint。H100 CUDA 镜像在 H100 服务器；本目录的 H100 冻结数值用于对照，不能在 SCNET 的海光卡上运行 CUDA。

## 当前验证范围

- SCNET 实际卡为 BW / gfx936 / 65520 MiB；与 BW1100 的 gfx938、约144 GiB不同。性能单列，不套用BW1100峰值计算MFU。
- Slurm作业 `1105828`：新版JAX/ROCm wheel的36项兼容性检查通过，真实退出码0，包含设备算术、RNE位运算、求导和AdamW。此记录不替代整模型测试。
- 整模型候选固定为官方 Small 1°、13层、batch=1。`1105884` 在环境文件校验时失败，尚未执行GPU模型；复制时产生的目录别名问题已修复，原路径另行保留。
- `1105925`的模块导入失败已修复，原记录保留。r3单步`1107046`通过，真实子进程及Slurm退出码均为0。对同case H100的+6h整体误差0.04937%、Z500误差0.0001753%；GPU active 0.170803秒，synchronized apply 0.183456秒。
- 单步记录83次rocBLAS调用、81个XF32 GEMM kernel、43个匹配执行的FlagTree kernel；RNE输入检查通过。仅代表本case单步，不代表所有GEMM均使用同一种硬件内核。
- 三轮40步`1107142`已通过，真实退出码0。三轮候选与三轮H100参考共9对末帧比较：最大整体误差0.90723%、Z500 0.011051%，内部<5%及阶段≤10%均通过。每40步GPU active中位数6.892666秒、同步apply合计中位数7.177592秒、rollout wall中位数10.387625秒，三种计时不混用。
- h1两次更新`1107146`已通过，真实退出码0；整段loss差异0.0016653%，两次更新GPU active合计3.446144秒、同步update合计3.842096秒。首次非零学习率更新生效，checkpoint与optimizer恢复检查通过；每次804次rocBLAS调用、792个XF32 kernel，206个FlagTree kernel匹配执行。两次更新不代表连续38次训练。
- 训练整卡显存采样峰值26663 MiB（每2秒采样），JAX allocator峰值9.410 GiB、池峰值23.998 GiB；整卡采样和allocator统计分别记录。
- r3源码339个文件及新loss结果240个文件已逐文件核对SHA通过；全量环境245,217个文件、源码／模型输入／参考1,505个文件已核对通过。SCNET离线loss复算与发布结果完全一致。
- 原环境校验6260条软链接误报来自`Path.readlink()`对链接文本的规范化；现按`os.readlink()`核对原文本。未更改原环境manifest及文件内容。

## 环境适配

沿用SCNET已验证的Podman基础镜像缓存，挂载固定新版conda与依赖、节点 `/opt/hyhal` 驱动及DTK26.04。原始BW镜像另作归档，SCNET实际执行环境单独记录。

基础镜像ID：`3d1a9e3d13198df0fd4c28762d9acffe05f29f180e7e4ae1a8b6089c06e41797`。

固定新wheel、原始模型和RNE实现保持不变；原生库按实际 `gfx936` 重新编译，保留源码和构建哈希。精度为FP32存储、FlagOS JAX `fl_only`、RNE＋XF32，不启用BF16。独立适配文件为 `model_job.py`、`model_inside.py`、`analyze_scnet.py`，不写入冻结release。

## 提交单步验证

先检查材料校验报告，之后在SCNET登录节点提交：

```bash
GC_SCNET_RUN=/work2/ac9844np6r/zyx/outputs/graphcast/scnet-delivery-20261113
sbatch --partition=hx1hdnormal01 --nodes=1 --ntasks=1 \
  --cpus-per-task=16 --mem=56G --gres=dcu:1 --gres-flags=disable-binding \
  --time=01:00:00 --job-name=graphcast-single \
  --output="$GC_SCNET_RUN/model-%j.slurm.log" \
  --wrap="python3 $GC_SCNET_RUN/model_job.py single --probe $GC_SCNET_RUN/compat-1105828"
```

每次作业生成独立目录 `single-h1-<jobid>/`。启动器只使用Slurm分配的单卡，检查空闲显存、利用率与PCI；记录真实子进程退出码。账号若触发节点数限额，等待已有作业释放或协调分配，不直接占用未分配设备。

单步完成条件：退出码0、模型报告完成、RNE输入检查、实际FlagTree／rocBLAS／XF32路径通过，且与本case冻结H100的+6h输出对照通过。仅Slurm作业结束或日志生成不能认定成功。

每个作业会先核验约22万份环境依赖文件，启动检查可能耗时数分钟，日志会输出已检查数量。Slurm的Elapsed包含环境校验、编译与结果导出，不能作为模型性能；GPU active和同步调用时间从正式采样窗口读取。

## 单步通过后

从真实单步结果取得目录，将 `--single` 指向该目录，沿用上面的Slurm资源配置：

```bash
# 示例参数；将 <jobid> 替换为已通过单步作业号。
python3 "$GC_SCNET_RUN/model_job.py" full40 \
  --probe "$GC_SCNET_RUN/compat-1105828" \
  --single "$GC_SCNET_RUN/single-h1-<jobid>"

python3 "$GC_SCNET_RUN/model_job.py" train --horizon 1 \
  --probe "$GC_SCNET_RUN/compat-1105828" \
  --single "$GC_SCNET_RUN/single-h1-<jobid>"
```

这两条命令应作为 `sbatch --wrap` 的内容提交，不能直接在登录节点运行。40步入口包含三轮独立rollout；训练入口是对应horizon的两次更新预验证，不是连续38次。

连续38次定义及原服务器运行说明见固定源码 `docs/FULL38.md`。当前独立SCNET适配尚未验证连续38次，也未验证四个case全部覆盖。37层双卡在BW1100已测每卡工作集超过SCNET单卡容量，不能仅凭卡数相同承诺可运行；后续必须按实际显存和运行证据登记。

## 正确性与结果读取

- 推理：同一case、初始权重、输入、统计与静态图，各自独立自回归，比较第40步末帧。按变量／层计算相对MAE，13层为78项等权平均，Z500单列；分母为该case冻结H100输出。
- 训练：固定数据计划、初始参数、AdamW状态和学习率，对应update比较loss，检查参数实际变化、checkpoint恢复和正常退出。独立训练checkpoint之间不做推理输出互比。
- 推理阶段线≤10%、内部回归线<5%。训练按完整指定区间 `100×Σabs(candidate-reference)/Σabs(reference)≤5%`；逐次和最大误差仅诊断，不能平均阶段百分比。完整38次需覆盖1～38次，短训只判预先指定的两更新区间。短程loss一致性不代表收敛，输出一致性不代表天气真值精度。
- 原始结果保留在每个新作业目录；`scnet-analysis.json` 给出卡上时间、路径证据和同case误差。SCNET峰值未确认，MFU留空。

## 历史SCNET入口

已实跑的2026-10-20 native JAX复现说明：
`/work2/ac9844np6r/zyx/delivery/graphcast-jax-scnet-20261020/README.md`。

当时完成0.25°、13层、三种精度各三轮40步；不是本周FlagOS新wheel、新case或训练的通过证据。2026-10-30目录为材料归档，其原服务器结果也不改标为SCNET实测。

## 离线复算与新运行目录

新loss结果可以在有Python 3.10及代码依赖的环境中离线重算，不使用GPU：

```bash
GC_DELIVERY=/work2/ac9844np6r/zyx/delivery/graphcast-delivery
GC_RELEASE="$GC_DELIVERY/releases/graphcast-20261113-r3"
GC_EVIDENCE="$GC_DELIVERY/results/training-loss-20261113-r3"
PYTHONPATH="$GC_RELEASE/src" PYTHONDONTWRITEBYTECODE=1 \
  python3 "$GC_EVIDENCE/recompute.py" "/work2/ac9844np6r/zyx/outputs/graphcast/loss-recheck-r3"
```

输出目录须为未使用的新目录。SCNET模型作业通过Slurm启动Podman；交付包通用`run.sh`用于BW1100/H100的Docker主机，不能直接替代SCNET提交脚本。所有新日志和模型输出放`/work2/ac9844np6r/zyx/outputs/graphcast/`，不写入冻结源码、输入或参考。

## 中间进度日志

每个作业的`launcher.log`记录环境核验与启动，`console.log`记录模型编译及运行。训练每次optimizer update完成后打印`UPDATE <horizon> <update> loss <值> seconds <同步耗时> change <参数最大绝对变化>`；两个microbatch的loss另存`report.json`的`micro_losses`。推理目前每轮结束打印`INFERENCE <round> <steps> <rollout wall秒数>`，逐步同步耗时保存为`apply_seconds`，不逐预测步打印。编译日志与正式进度分开阅读，不把编译或环境核验用时计入模型性能。

```bash
# 在SCNET登录节点查看已有作业的日志，不占用GPU。
tail -f /work2/ac9844np6r/zyx/outputs/graphcast/scnet-delivery-20261113/train-h1-1107146/console.log
```
