# GraphCast C550 训练与验收复现

本目录封存智源 GraphCast 当前 C550 验收版本的四类入口。模型核心和训练集成
共 144 文件与验收记录逐 SHA 一致；数学逻辑沿用已有模型，不复制另一项目的实现。
本次新增的是路径配置、设备检查、独立输出管理、CPU 测试和说明，**没有重新跑 GPU**。
它不是任意平台/任意 Torch 版本都已验证的通用启动器。

| 模式 | 实际工作 | 用途 |
|---|---|---|
| `full38` | 原 seed27、B2/microbatch1、完整 38 次更新，保留 CP36/CP38 | 一次正式端到端 MFU |
| `short8` | 单进程 R1×3、R6×3、R12×2；末对不稳定才追加第9步 | 日常优化的条件性 MFU 预测 |
| `recovery` | 读取原 CP36，只执行更新37、38 | 对照同源连续训练的真实恢复 |
| `evaluation` | 原 CP38、111 个 R12 窗口、microbatch1、no_grad、0更新 | 与固定 H100 参考比较完整验证损失 |

## 1. 代码和环境

GraphCast 基线为本仓库 `f8b26dbd2c3a252dd4bb6f461a7b5cde5ab059ac`；
本次补入的 7 份集成文件使 `locks/model-files.json` 的 144 文件全部可校验。
模型入口是 `scripts/training/run_official_epoch_training.py`。

FlagGems 使用 [flagos-common 的固定快照](https://github.com/flagos-adaptation-lab/flagos-common/tree/f2a276f3741026f1234574f3beccde55c5bb5f8c/patches/flaggems/wip-metax-c550-training-c03e)，
提交 `f2a276f3741026f1234574f3beccde55c5bb5f8c`。先按该目录 README 将
**完整项目父版**与八份源码/配置组合，运行其校验工具；完整 2984 文件清单必须是
`c03eb85f35abaf90665d922013f3ffe67f5c97b9bd919df60dd4d23c11510227`。
父版和私有仓库访问权限须从项目维护者取得，不能用任意最新版 FlagGems 替代。
固定 LN 依赖仍是 PR4916@`7f97e605a179b3017326f981f0da9e59e49e4df4`。

原训练环境为 Linux、Python 3.12.11、Torch `2.10.0+metax3.8.1.0`、
MACA 3.8.1 系列容器/驱动和其已安装的 Triton/依赖。
评估迁移机器使用同镜像、驱动3.3.12；该评估结果不产生另一驱动的新 MFU。
完整镜像/依赖资产沿用项目归档，不在此自动安装、升级或替换共享环境。
`requirements.txt` 是模型依赖说明，不能代替这份已验收环境。
CPU 包装检查可用 Python 3.11；实际硬件复现应使用上述原解释器。

训练调用链：`reproduce.py` → 相应 `run_*.py` → `timing_worker.py` →
`ClipRouting` → `run_training_with_native_io.py` 的 LN 作用域 → 原训练入口。
短测只在原入口外围替换采样课程及保存策略；恢复增加状态/RNG和真实指标前缀检查。
评估复用原入口的模型构造 AST 和 `evaluate_validation`，不调用 `train()`。
`provenance.json` 分别记录原始与整理后的 SHA。固定计算模块保持原字节，
worker 改动仅用于显式设备映射和启动门禁；启动器另统一父/子进程的依赖路径
并限定超时清理范围，CPU检查点工具另导出原恢复路径契约。

## 2. 外部资产

代码仓库不包含权重、数据、CP、编译缓存、容器镜像和私人部署日志。
请从智源 GraphCast 项目归档取得下列**原文件**。只有代码没有这些资产不能独立复现。
复制已有文件后保留对应来源/校验记录；不要为了检查代码重复传输或重算未变大文件。
训练数据若已有包含真实SHA、bytes、mtime_ns的部署收据，在settings添加
`training_data_receipt`绝对路径即可复用。无收据时第一次`check`做一次流式完整SHA，
在job生成收据；随后`run`复用该收据和stat，文件变化直接拒绝。

| 资产 | 固定身份 |
|---|---|
| 模型初值 `weights.pt` | `2bd1786e197b432d0f471ce4118f34b231c73c322dbaba0684e5f71097e010b4` |
| 2017训练ERA5，约5.9GB | `f277accdd54a8d652b2357035fde107fc80f7cdcfceb71d0ba155ed92b3c02a3` |
| 原CP36及 `.pt.sha256` | `211b19a84964215a65694d4ad1a7d88299c19c60cffe9636d142f406ef13935f` |
| 原连续CP38及 `.pt.sha256` | `94dbe71828ecc7793114fad04ba0a38b25ee4db4515c63eee26c74bd0bea3a2d` |
| 2022年1月HRES，19,600,887,174字节 | `ff6287e487371e45035a52d50a9fd83f8b683dc824f9464f16a46281ed6552ee` |
| 三份统计量 | 模板 `input_file_sha256` / `input_files` 内的逐文件 SHA |
| 固定mesh6资源 | 已在本仓库 `src/graphcast_compat/assets/`；不要重新生成 |
| 原连续训练指标 | 原 `candidate/run/summary.json`、`metrics/segment_*.jsonl` 和对应 `.start.json` |
| 已核编译/调优缓存 | 只读种子，运行前复制至本任务目录；不得直接写共享缓存 |

配置 `example-settings.json` 中的路径，所有路径均是**执行容器内部**路径。
训练数据目录应包含：

```text
data-r4/
  corpora/era5_2017-01-01_07_train.json
  official_era5/era5_2017-01-01_07_graphcast_operational_pretrain.nc
  stats/{mean_by_level,diffs_stddev_by_level,stddev_by_level}.nc
```

训练清单的原字节副本在 `metadata/`，相对数据位置不可随意改写。
验证目录应包含：

```text
validation/
  data-verified.json
  official_hres/hres_2022-01_graphcast_operational_validation.nc
  corpora/hres_2022-01_validation.corpus.json
  validation-windows-r12.json
```

将本目录 `metadata/` 的两份冻结验证小文件放入相应位置；已有同 SHA 文件直接复用。
`data-verified.json` 来自完成部署时的真实完整 SHA 检查，含 `sha256`、`bytes`、
`mtime_ns`。启动时复用收据并核对 stat，**不要手填已通过或复制旧机器mtime**。
新复制导致mtime改变时，必须在目标重新完成部署校验才能生成新收据。

`cache_seed` 可以是显式空目录，但冷启动会影响首步与短测成本，必须如实记录，
不能据此声称与历史暖缓存严格同条件。依赖路径和 `python_site_paths` 只引用
已安装路径；bootstrap 使用 `-S`，标准库优先，不执行额外 `.pth` 文件。

**恢复的额外限制：**原CP36包含严格的路径、环境和训练契约，指标前缀也保留这些字段。
复现原CP36恢复时，容器内 model/source/data/stats 的逻辑路径必须保持检查点记录值，
可通过受控挂载提供；更换宿主不等于允许改写这些字段。不要修改检查点或放宽比较来迁移。
`cp36_probe.py`会导出`required_resume_paths`；`check`在GPU前拒绝不匹配的路径。
其他三种模式支持配置路径；路径可配置不表示旧检查点契约允许任意重定位。
重新训练生成的CP字节可能不同，不会被本工具默认为历史原CP36/38；需要另立来源锁和审核。

## 3. 准备、检查、运行

以下命令在仓库根目录执行。准备与检查不会使用 GPU；`run` 才会开始相应工作。
运行前使用独立 tmux 会话，确保退出码与本地/远端实际状态可查；断连后先查同一
job 的 `launcher.started`、`launcher-exit.json`、`candidate/exit.json` 和日志，禁止盲目重启。

```bash
export GC_REPRO="$PWD/ai4s/graphcast/tools/c550_reproduction"
cp "$GC_REPRO/example-settings.json" /mnt/graphcast-settings.json
# 编辑该副本中的实际路径、Python、依赖、单卡映射。
python3 "$GC_REPRO/reproduce.py" prepare short8 \
  --settings /mnt/graphcast-settings.json --job /mnt/jobs/c550-short8-001
python3 "$GC_REPRO/reproduce.py" check --job /mnt/jobs/c550-short8-001
```

依次把 `short8` 换为 `full38`、`recovery`、`evaluation` 并使用不同的新目录。
**不需要每次都运行 full38**。`check` 核对完整源、输入、CP和数据身份；任何缺失或
变化都拒绝。模型文件和 c03e 源清单是锁定的；优化新候选须新建并评审来源锁，
不能把旧来源/覆盖证据直接贴到新版本。

启动前由操作者/资源管理器在宿主核对完整设备进程表、显存、利用率、健康、
PCI/UUID，并核实容器可见映射及环境。0%利用率但有PID必须拒绝。
保留实际输出为 `resource-evidence.txt`，填写 `gate.json`，其中：

```json
{
  "allowed": true, "platform": "c550", "owner_user_id": "实际负责人标识",
  "healthy": true, "process_count": 0, "utilization_percent": 0,
  "memory_used_mib": 0, "epoch": 0,
  "physical_gpu": 7, "pci_bus": "0000:da:00.0", "device_uuid": "实际设备UUID",
  "container_mapping_verified": true, "environment_verified": true,
  "job_digest": "prepare输出的package-manifest SHA256",
  "raw_evidence_sha256": "实际resource-evidence.txt的SHA256"
}
```

示例值不能直接作为门禁。`epoch` 必须是本次实际检查时刻（Unix秒）；有效期120秒。
最多1024MiB仅容纳无进程的已核驱动保留显存，不允许忽略未知驻留进程。
记录是人工/调度系统的资源声明，代码本身**不从文本自动判断宿主空闲**；它检查声明、
原始证据SHA和新鲜度，并在工作进程创建模型前调用真实MACA接口再次核对单卡PCI。
配置中的物理卡号与容器mask不能混用。过期时重新检查资源，不是改时间戳。

```bash
python3 "$GC_REPRO/reproduce.py" run --job /mnt/jobs/c550-short8-001 \
  --gate /mnt/gate.json --resource-evidence /mnt/resource-evidence.txt
python3 "$GC_REPRO/summarize.py" --job /mnt/jobs/c550-short8-001
```

运行有锁和存在排重，失败目录永久保留，重试另建目录。不使用 `python -O`。
外层上限分别7500/1800/2100/43350秒；内部原任务上限7200/1500/1800秒，
评估由外层管理。启动器按父子关系记录本任务进程及PID出生标识；即使worker
另建session，超时也只清理本任务的已核后代，不按进程名全局终止，不停止共享容器。
所有输出都在job里；数据、检查点和种子缓存不在输出目录内。

## 4. 恢复与验证的具体核对

恢复前，在原环境执行只读CPU检查，保留完整响应，再提取 `audit` 到 settings 的
`resume_audit` 路径。该检查本身不证明真实恢复：

```bash
python3 "$GC_REPRO/harness/recovery/cp36_probe.py" \
  /path/to/step_00000036.pt /path/to/weights.pt /path/to/model-root > /mnt/cp36-probe.json
python3 -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1]))["audit"],indent=2))' \
  /mnt/cp36-probe.json > /mnt/checkpoint-input-audit.json
```

随后prepare/check/run恢复job。原36步指标只作为可核验历史前缀，实际执行必须恰好37、38；
恢复参数、优化器和RNG起点严格一致。结束后对原连续CP38和恢复CP38做全状态CPU比较：

```bash
python3 "$GC_REPRO/harness/recovery/checkpoint-comparison-probe.py" \
  /path/to/continuous/final_checkpoint.pt \
  /mnt/jobs/c550-recovery-001/candidate/run/final_checkpoint.pt > /mnt/cp-comparison.json
python3 "$GC_REPRO/summarize.py" --job /mnt/jobs/c550-recovery-001 \
  --continuous-run /path/to/continuous/run --checkpoint-comparison /mnt/cp-comparison.json
```

模型/优化器相对L2≤1e−4，每步loss/梯度范数相对差≤1e−3，
游标/调度/RNG/来源契约精确一致；容差在运行前冻结。恢复不产生新正式MFU。

验证的结果必须是全部111个有序唯一有效窗口的样本加权mean；要求模型全量hash
前后不变、零梯度/更新、逐变量/lead及总loss有限、实际导入与TF32退出回读通过。
比较 `abs(C550_mean − 31.3537019609331) / 31.3537019609331 × 100 ≤ 5%`，
不是逐窗口都必须≤5%、末窗口loss或训练loss。
`validation_journal.py`逐窗原子写入并fsync，有身份/hash链；中断结果可审计，
**没有自动跳过已保存窗口的续评估功能**。不把残缺窗口拼成完整验收。

## 5. MFU口径、八步方法和覆盖率

正式MFU为 `29434.23636529152 TFLOP / sum(38个update E2E秒) / 1 / 132.49 ×100%`。
包含6次LR warmup和完整sumR170，不丢前两步、不平均逐步MFU。
初始化及update区域之外的CP I/O不计入该E2E口径，另外记录作业总耗时。

八步只加载一次初值，不写新CP：R1采第2、3步；R6采第5、6步；R12先看第7、8步，
相对极差>5%才执行第9步并用第8、9步。所有实际更新和采集成本都计入采样成本。
令s1、s6、s12为各采样对均值，线性模型 `t(R)=a+bR` 用R1/R12拟合，
R6残差及各对极差均须≤5%。`T38=25*s1+12*s12+t_first`；再用上述FLOP/峰值估计MFU。
任一门槛失败则 `conditional_mfu_pct=null`，不能挑一个快样本报收益。
短测初值/中间状态不同于完整课程；条件性预测不等于正式MFU或统计置信区间。

训练保留 `coverage_extension.py` 的作用域ClipRouting，调用已有FlagGems公共函数，
保留fallback及退出还原。注册列表不是实际覆盖率：原验收为Native观察到41个算子类型，
其中33个有实际FlagGems执行证据，33/41=80.4878%。证据由新版R1/R12与未变R2–R11
的历史实际派发/源码关联组成，**不是新完整38步profile**。
本目录的计时入口不采全量算子trace，`summarize.py`不会自动认证覆盖率超过80%；
新版本必须重新建立缺失/变更路径的实际证据。仅保留ClipRouting不能宣称新版本已覆盖。

## 6. 历史结果与本次检查

历史当前版本：完整38步2618.7143秒、MFU8.48362683%、H100的84.34206430%；
实际类型覆盖80.4878%；CP36两步恢复通过；111窗C550mean31.348498834146035，
相对H100差0.016594936%，完成验收。历史八步预测8.33944865%，相对此次完整课程误差
−1.69949%；含采集成本为完整课程的37.4268%，尚未达30%成本目标。
这是一组前瞻对照，不是所有模型/平台的误差保证。

`fixtures/`只含可重算的精简数值，不包括原始运行日志、CP和私人记录。
完整原始证据仍在项目归档。新包装的CPU测试：

```bash
python3 -m unittest discover -s "$GC_REPRO/tests" -v
```

覆盖144源文件、四模式参数、重复/漂移拒绝、资源声明、8/9步停止规则、
实际历史MFU回归、无效预测拒绝、恢复前缀、111窗冻结函数输出、损坏/中断及
真实SIGKILL子进程后的结果保存。CPU替身测试没有运行真实模型前向。
源码封存和这些测试不冒充本次新GPU验收；换设备/软件栈需独立核验。
