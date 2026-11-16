# 结果与验收

结果必须绑定实际 `release.json` SHA、平台、设备、环境和输入；同名目录更新不继承旧GPU结论。
## 2026-11-15 完整推理复验

两平台均实际完成单步检查＋3轮独立40步；每轮5步预热后重置，正式窗口的JIT cache均为2→2，退出码均为0。
case为Operational 0.25°／13层／batch 1，精度参考为冻结的 `h100-earth-tf32-20261103-v1`。
GPU执行的 `release.json` SHA256：`5e3b49492c893daad6a61d7c8fae1f4cfa78fc6eb03570a52244f80e15e17700`。
实际执行清单保存在 [executed-release.json](../results/full40-20261115/executed-release.json)。
随后训练入口、日志、监控及trace校验有更新；旧推理结果与执行清单保持原字节。
以下推理结论仅属于上述执行SHA；训练验证单独记录，不由推理或CPU测试推定。

| 平台／设备 | 3×40步GPU active总计 | 汇总MFU | 最大整体误差 | 最大Z500误差 |
|---|---:|---:|---:|---:|
| BW1100入口／GPU 15 | 143.144112 s | 14.16185% | 0.655275% | 0.008302% |
| H100／GPU 2 | 34.456517 s | 20.26438% | 0.631235% | 0.007625% |

误差为各自3轮对冻结参考3轮、共9组的最大值，比较+240h末帧；全部通过内部<5%及阶段≤10%数值门限。
本次H100与原H100参考另有9组对比：最大整体0.598384%、Z500 0.007582%，均通过。
本次BW与历史BW完整输出的9组距离：最大整体0.617927%、Z500 0.007182%；历史轮间整体差异最高0.611392%，本次0.580190%。
因此旧结果的精度结论得到复现，输出并非逐位一致；轮次差异原因仍未归因。

MFU采用有效GEMM FLOPs近似：每步84个逻辑GEMM、`28,784,948,924,416 FLOPs`，每轮40步。
按 `ΣF / (ΣGPU active seconds × dense TF32 peak) × 100%` 汇总，不平均／取中位数各轮MFU。
BW使用项目NMZ峰值170.3936 TFLOP/s，H100使用494.7 TFLOP/s；本次相对MFU为 **69.88541%**。
BW运行时设备名称为 `BW1200`，交付入口沿用 `bw1100`；峰值与具体设备／频率的映射尚待确认，所以BW MFU及相对比值采用项目峰值估算。

计时与限制：

- BW逐轮active为47.687／47.748／47.709 s；同步调用合计54.948／47.811／47.797 s；rollout墙钟84.800／54.898／195.802 s。
- H100逐轮active为11.486／11.486／11.485 s；同步调用合计11.827／11.854／11.804 s；rollout墙钟16.342／16.349／16.150 s。
- BW Chrome导出达到约100万事件上限，复算采用完整XSpace protobuf：每轮25,821个计算kernel、3,360次GEMM及40步审计，无protobuf告警。H100每轮16,341个kernel，Nsight launch关联与40步签名检查通过。
- BW单步预检查138次采样中，4次报告外部PID映射、1次归属不明；full40的279次采样没有同类问题。原运行汇总的性能状态保留 `not_accepted`。H100监控253次无异常，原状态为 `not_evaluated`，离线MFU另记于此。
- **69.89%是研发峰值归一化对照，不代表正式“≥60%”性能验收。** profiler、共享主机活动和未锁频会影响计时；墙钟与active之差不能当作CPU耗时。5步预热已验证，最小预热数仍待验证。
- BW第三轮rollout比predictor累计多约148秒，等待原因尚未定位；训练的THP诊断不能直接作为推理结论，当前推理没有应用THP策略。

### 包内复算

结果只有7个文件、约2 MB：[summary.json](../results/full40-20261115/summary.json)给出配置、公式、逐轮结果和门限；
[evidence.json](../results/full40-20261115/evidence.json)保留执行命令、环境／材料／预测／trace身份、原始状态和旧结果对比。
压缩CSV保存全部6轮计算kernel区间和18组×78项误差分子／分母，`gemm-flops.csv`列出84个逻辑GEMM。

从本release目录运行（Python 3.6+，标准库，不使用GPU）：

```bash
python3 results/full40-20261115/recalculate.py
```

脚本验证文件SHA、步数／kernel计数，使用两种区间并集算法复算MFU，并从逐层MAE及参考均值复算精度。
这条命令复算已保存的统计；重新读取NetCDF比较原始数组需使用完整结果与冻结环境，不能由此替代GPU复跑。

大型原始预测、完整trace和日志保留在原工作目录，包内记录SHA256，不重复打包：

- BW：`/data_add/zhangyuxin/graphcast-gpu-validation-20261115-ecso_gv4/runs/graphcast-20261114/operational-025deg-13l-b1-infer-full40-9ow82ttr`
- H100（h100-earth）：`/data2/earth_1/graphcast-20261114-work/runs/graphcast-20261114/operational-025deg-13l-b1-infer-full40-nxt03wcb`

规则来源：[统一MFU／精度规范](https://app.notion.com/p/41773db917b681c28454f3269be6fc15)、[GraphCast测量配置](https://app.notion.com/p/40873db917b681ebb2f0d8f250a60bdf)。

## 2026-11-15～16 训练验证

两平台均完成12阶段、38次更新，参数／optimizer的checkpoint连续性与回读已核对。
BW按原始、trace修订、THP修订三个执行版本续跑；最终续跑退出码0。H100一次完成，退出码0。

| 平台 | 正式更新GPU active | MFU | 完整loss误差／优化H100参考 |
|---|---:|---:|---:|
| BW1100 | 1908.937905 s | 8.81136% | 0.153991%（通过） |
| H100 | 379.982578 s | 15.24694% | 0.134656%（通过） |

H100相对原H100参考的完整loss误差为0.012718%。所有loss门限均为≤5%。
分子采用有效GEMM FLOPs近似；按38次更新的总FLOPs除以GPU active总时间与项目峰值，不平均各阶段百分比。BW／H100相对值为57.79104%。
BW有宿主机监控中断、归属告警及短时CPU诊断采样，物理设备峰值映射仍待确认；上述MFU为研发估算，不能视为正式性能验收。
BW前16次曾因PostGSU辅助归约的GEMM计数校验失败；修订后使用原始trace复验通过，辅助kernel时间仍计入GPU active。

包内[summary.json](../results/full38-20261115/summary.json)记录公式及结果，
[evidence.json](../results/full38-20261115/evidence.json)记录逐阶段源码SHA、checkpoint链、输入、计时与原始状态。
原始执行清单分别为[原版](../results/full38-20261115/executed-release-original.json)、
[trace修订](../results/full38-20261115/executed-release-tracefix.json)、[THP修订](../results/full38-20261115/executed-release-thp.json)。
共9个结果文件；大型trace、checkpoint及日志仍在记录的原工作目录。复算使用Python 3.6+标准库，无需GPU：

```bash
python3 results/full38-20261115/recalculate.py
```

整段耗时同时保留输入准备、编译和阶段main统计；这些耗时不能用GPU active替代，也不将跨次中断等待计为训练计算。

BW进程级THP策略已用相同GPU、checkpoint及第29～30次更新做顺序对照：

| 两次更新的统计 | 修改前 | 进程THP关闭后 |
|---|---:|---:|
| 输入准备合计 | 2200.98 s | 58.53 s |
| 阶段main耗时 | 51.78 min | 9.88 min |
| GPU active | 181.354 s | 181.790 s |

对照执行SHA为 `6c3c7457c3476bef7a5284721a70216803d05aa8466c57c9718b4803168e2eef`。
GPU active变化约+0.24%；参数与optimizer相对L2分别为2.91e-10、5.43e-7，通过1e-5检查但非逐位一致。
这是顺序诊断对照，不是独占性能验收；重复的29～30次不计入完整38次结果，正式续跑从原checkpoint30开始。
策略仅用于BW训练进程，不修改全局THP、H100或推理；说明见[ENVIRONMENT.md](ENVIRONMENT.md)。

## 冻结材料

以下路径相对交付根目录，manifest身份由 `configs/artifacts.lock.json`固定：

| 内容 | 路径 |
|---|---|
| 优化H100推理及38次训练参考 | `references/h100/operational-025deg-13l-b1/v1/` |
| 原H100推理／训练参考 | `references/h100/operational-025deg-13l-b1/original-v1/` |
| 输入与训练计划 | `data/cases/operational-025deg-13l-b1/v1/` |
| 权重、统计与固定图 | `data/models/operational-025deg-13l/v1/` |

参考编号 `h100-earth-tf32-20261103-v1`。H100原版→优化版与优化H100→BW1100的比较分别记录。
历史baseline保持原版本标签，其他wheel或case的结果不能作为当前包实测。

## 精度与完整性

固定推理使用相同权重、输入和统计，各自独立40步，比较+240h末帧。
每变量层相对MAE为 `mean(abs(candidate-reference))/mean(abs(reference))*100%`；
整体为6个高空变量×13层等权平均，Z500单列。阶段门限两项均≤10%，内部回归线两项均<5%。
H100单步基线／候选筛选两项均<0.1%，仅为进入完整预测前的检查。

训练先核对初始权重、数据计划、更新编号和checkpoint接续，再比较完整1～38次更新：
`100*Σabs(candidate_loss-reference_loss)/Σabs(reference_loss) ≤ 5%`。
缺步、重复、乱序、非有限loss或零总分母不能通过；不平均阶段百分比，不附加逐次≤5%的要求。
BW汇总为 `training-loss-summary.json`；H100为 `training-validation.json` 中的loss_validation。
此短程检查不代表收敛，不比较分别训练后的checkpoint推理输出。

## 计时与证据

逐步predictor计时包含模型调用和同步；rollout墙钟还包含输入更新、数据处理、审计与循环开销。
推理正式capture排除编译、预热、末帧传回、精度计算和保存。
训练原始阶段trace可能包含周期checkpoint保存，但正式更新窗口排除保存；GPU active仅对这些更新窗口内的计算kernel求并集。
主性能口径为GPU active MFU：
`F_useful/(T_gpu_active*P_total)`；有效模型FLOPs不重复计算重计算、RNE、optimizer，但其kernel时间计入分母。
同步wall、GPU active、端到端耗时分别报告。执行、精度、性能状态及日志路径见[RUN.md](RUN.md)。

原始trace、日志、checkpoint及结果保留在运行者work目录；开发测试和旧文档保留在内部归档。
CPU测试、文件校验与计划检查均不等于GPU复跑。SCNET的本case适配及GPU验证仍待完成。
