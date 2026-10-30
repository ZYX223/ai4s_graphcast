# GraphCast｜完整40步卡上MFU补测｜2026-10-30

本次只扩展profile采集长度：每平台3轮，每轮预热5步后重置输入、独立自回归40步。原baseline、融合策略和冻结输出保留，既定Notion工作步骤未增改。

| 路径 | 40步GPU活跃时间中位数(s) | 稠密峰值(TFLOP/s) | 40步MFU估算 | 相对H100的MFU比值 |
|---|---:|---:|---:|---:|
| H100 JAX TF32（无显式RNE） | 12.658042 | 494.7 | 18.3873% | 100.0000% |
| BW1100 原生JAX融合 RNE＋XF32（恢复环境） | 29.567501 | 170.3936 | 22.8538% | 124.2912% |
| BW1100 FlagOS fl_only RNE＋XF32 | 54.675367 | 170.3936 | 12.3589% | 67.2146% |

相对H100的MFU比值＝本路径三轮中位MFU／H100三轮中位MFU×100%，使用未四舍五入值计算；H100自身为100%。该值是峰值算力归一化比较，不是实际推理速度比，也不直接作为正式阶段验收判定。

分子：40×28,784,948,924,416 = 1,151,397,956,976,640 FLOPs（1151.39795697664 TFLOPs）。GPU时间取每轮全部计算内核区间并集，包含RNE、Gather、Scatter与自回归更新的设备计算，重叠不重复计入；拷贝/CPU/空隙另列。

原始卡上吞吐比：23.1513%；按峰值归一化的MFU比：67.2146%。后者是候选效率口径，不能据此直接宣布阶段60%验收通过。

H100峰值采用[NVIDIA H100架构白皮书v1.04表1](https://resources.nvidia.com/en-us-hopper-architecture/nvidia-h100-tensor-c)的SXM5稠密TF32 494.7 TFLOP/s；同表989.4为稀疏值，不使用。海光继续采用用户指定170.3936。原件已保存为nvidia-h100-architecture-v1.04.pdf。

## 原始测量

| 平台 | 轮次 | GPU活跃(s) | kernel次数 | MFU(%) | profiler期间rollout墙钟(s) |
|---|---:|---:|---:|---:|---:|
| h100 | 1 | 12.658269 | 16501 | 18.3869 | 14.641555 |
| h100 | 2 | 12.658042 | 16501 | 18.3873 | 14.638966 |
| h100 | 3 | 12.657395 | 16501 | 18.3882 | 14.645756 |
| native | 1 | 29.567487 | 19901 | 22.8538 | 33.240742 |
| native | 2 | 29.665307 | 19901 | 22.7784 | 32.962120 |
| native | 3 | 29.567501 | 19901 | 22.8538 | 32.865147 |
| hygon | 1 | 54.675367 | 14261 | 12.3589 | 58.272765 |
| hygon | 2 | 54.702007 | 14261 | 12.3529 | 58.093253 |
| hygon | 3 | 54.667091 | 14261 | 12.3608 | 58.059670 |

## 算子主表

新版Excel的主表依次列JAX（同一FlagOS容器native）、FlagOS fl_only、加速比JAX÷FlagOS、FlagTree覆盖（含标记用例/总例）及执行说明。大于1为fl_only更快，小于1为更慢。Linear 15/15仅代表用例包含FlagTree的bias等部分；GEMM仍为rocBLAS，独立RNE为HIP。旧10月22日原生JAX基线数字保留在历史版本，不混作当前同容器native对照。

独立算子计时为同步调用墙钟，模型MFU使用设备trace，两者不混用。16类的原始采样/HLO与校验见上一版目录flagos-inference-final-20261030，新的列结构见operators.csv。

## 复算与原始材料

- `python3 calculate.py`：核对采集边界、有效GEMM量、内核数并复算MFU。
- `python3 build_tables.py`：生成新版Excel；原版Excel不覆盖。
- `h100-profile/model-trace.1/2/3.sqlite`及`.nsys-rep`：三轮Nsight原始事件。
- `hygon-profile/trace.json`及`trace.db`：三轮HIPPROF原始事件；`run/capture-ranges.json`用于逐轮范围校验。
- `kernel-events.csv`、`rounds.csv`、`summary.json`、`validation.json`：事件、逐轮值与复算证据。
- `profile40.py`与各平台启动脚本、instrumentation.diff：仅采集范围与输出留存的改动；官方模型源文件未改。

## 适用边界

- GPU-active union excludes CPU, transfers and gaps; includes all traced device kernels across full rollout including state-update auxiliaries. Not end-to-end MFU.
- Each platform captured 3 complete 40-step rounds after 5 warmup steps and resetting input; no warmup, compile, metrics or file writes included.
- Useful model FLOPs approximate 84 logical GEMMs per step; elementwise/reduction FLOPs omitted. Same numerator across platforms; auxiliary overhead stays in measured time.
- H100 494.7 TFLOP/s from NVIDIA v1.04 dense TF32 SXM5 specification; BW1100 170.3936 from user-designated peak. No use of sparse/BF16 peak or frequency scaling.
- H100 GPU2 and BW1100 GPU9 are individually available but other cards on hosts had workloads. No frequency lock or power configuration changes.
- BW1100 is the user-confirmed project device label; the raw Hygon runtime device name reports BW1200 and is preserved verbatim in summary.device, not silently corrected.
- Profiler can perturb timings. Profiled rollout wall time is recorded separately and does not replace the earlier unprofiled 3-round performance results.
- Peak-normalized MFU ratio is a diagnostic candidate for relative efficiency; equivalent-compute definition and contractual acceptance timing need agreement. Not a formal 60% pass declaration.
- This run extends timing scope, not model/precision. Previous 40-step accuracy results retain their original labels; fresh final outputs are retained without relabeling old accuracy as newly computed.
- H100 raw traces and metadata collected locally; new H100 final predictions remain at /data/zyx/graphcast-mfu40-20261030/h100-profile/run/. Baseline predictions and all previous results preserved.
- No GitHub push or testuser/SCNET delivery sync; Notion work-step list not changed.
- Native vendor JAX restored from all 40 frozen Python distributions in a different base image with the same DTK-26.04-DCC2602-0317 release. Frozen Oct22 model/experiment code and RNE library used unchanged. This is a recovered-runtime remeasurement, not proof that the missing original image was restored bit-for-bit.

## 原生JAX补测与环境恢复

原生JAX三轮40步端到端中位数为32.962120秒（带profiler）；历史未带profiler值为33.311110秒。卡上时间中位数29.567501秒；FlagOS相较原生的卡上加速比为0.540783倍，即耗时约1.849倍。

以10月29日冻结H100 TF32为精度参考，同轮整体误差0.559964%～0.676028%，Z500为0.006629%～0.008206%；全部9组配对通过内部5%线与阶段10%线。详情见native-accuracy/results/。

原镜像缺失；使用本机已有DTK镜像的基础运行时、离线冻结的40项Python包和原版JAX四个wheel恢复独立venv。DTK版本与原记录均为DTK-26.04-DCC2602-0317。原实验源码逐文件SHA及原RNE动态库SHA通过校验。基础镜像ID不同，不能声称逐字节恢复原镜像；native-profile/launch.json及environment-probe.json保存差异。

采集器启动失败留存在native-profile-startup-failed/：最小用例确认其对继承的Python/临时目录环境敏感。隔离采集器启动环境、在子进程恢复相同模型环境后采集完成；模型代码与精度策略没有修改。

复现入口：launch-native.py、run-native.sh、native-profile-entry.sh；离线环境恢复命令见native-recovery/setup-command.json。原生完整trace位于native-profile/。独立复算核对运行verify_results.py。
