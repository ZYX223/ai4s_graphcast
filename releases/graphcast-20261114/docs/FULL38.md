# 连续38次训练

在 BW1100 或 H100 各用1张卡；命令见 [RUN.md](RUN.md) 的 `train-full38`。
固定预设从固定预训练权重和相同数据计划开始；batch=1、梯度累计2。38指优化器更新次数，不是预测步数。
两个 microbatch 的梯度取平均，再做一次 AdamW 更新。

| 阶段 | 预测长度 | optimizer update |
|---|---|---|
| h1 | 1×6小时 | 1～16，共16次 |
| h2～h12 | h×6小时 | 每阶段2次，合计第17～38次 |

例如h10执行第33～34次更新，h11执行35～36，h12执行37～38。每个阶段启动独立worker，重新编译并预热；默认预热1次，不改变正式参数与更新计数。
预测长度改变目标数据形状；同阶段的更新复用编译结果，不同阶段分别编译。独立worker用于释放前阶段可执行程序与内存。
当前持久编译缓存位于每次输出的 `rXX/cache`，新运行／续跑不共享旧缓存，因此编译开销在这种每阶段仅2次更新的短测中占比较高。

同一平台按阶段恢复自己的参数、optimizer 状态和更新计数，不能用另一平台的 checkpoint 替代。
保留既有学习率、重计算及数据计划，h1 日程中学习率为0的更新也按原编号记录。
这是短程流程与 loss 一致性验证，不代表重新训练到收敛。

实现入口：[BW阶段编排](../src/graphcast_bench/delivery/operational_bw1100.py)、[H100阶段编排](../src/graphcast_bench/delivery/operational_h100.py)。
BW：[training/entry.py](../src/graphcast_bench/training/entry.py) → [runner.main](../src/graphcast_bench/training/runner.py) → run_update → record_update_diagnostics。
H100：[training/entry.main](../src/graphcast_bench/engines/operational_h100/training/entry.py) → [streamed_train.main](../src/graphcast_bench/engines/operational_h100/training/streamed_train.py) → run_update → record_update_diagnostics。
H100入口显式接收context，按原顺序安装remat与数值hook；固定和自定义训练共用阶段执行器。
入口和主循环是维护代码，原数值hook身份由ORIGINS保留；没有新增训练参数。
两平台主循环依次准备数据与状态、编译、在副本上预热、正式更新、诊断和保存。
BW 训练 RNE 在自动求导之后插入，保持前反向 GEMM 的实际路径检查。
BW入口在导入JAX前应用[进程级THP策略](../src/graphcast_bench/platforms/host_memory.py)，每阶段在 `host-memory-policy.json` 中记录是否生效；H100及推理不应用此策略。

需检查38次更新完整、loss有限、首次非零学习率发生参数更新、阶段checkpoint／optimizer接续、
真实退出码及 RNE／FlagOS 路径。整段loss验收见 [RESULTS.md](RESULTS.md)。
通过 `train --resume` 续跑时，单次执行成功只覆盖所请求的更新；完整验收还需合并前后记录，核对1～38无缺失／重复及checkpoint连续性。

| 计时字段 | 范围 |
|---|---|
| `input_preparation_and_h2d_seconds` | 两份输入准备、摘要及首份传输，可能包含主机分配等待；第二份传输单列 |
| `compute_phase_sync_seconds` | 两次前反向及梯度累计／优化器调用的同步墙钟和，不等于纯GPU计算 |
| `synchronized_update_seconds` | 整次更新调用及同步，另含第二份输入传输等开销 |
| GPU active | 从原始trace计算正式更新内GPU计算kernel的区间并集，用于MFU |
| `total_main_seconds` | 单阶段main计时，包含准备、编译、预热、正式更新及checkpoint保存等；不含进程启动／导入和最终report写出 |

编译、预热和保存不计入正式更新GPU active；它们仍影响任务总耗时。跨阶段worker耗时之和不包含宿主机启动、阶段间隔或中断等待。

训练日志按Compile／Warmup／Training输出当前阶段、更新编号、loss、梯度范数和同步耗时；
根目录 `run-summary.json` 分别记录执行、完整loss验收和监控状态。
宿主机监控按物理卡号／UUID取样，`observed_device_peak_mib`为采样的整卡用量峰值；
`observed_allocator_peak_mib`为JAX allocator用量峰值，两者缺失均记null。
BW的 `training-trace-summary.json` 从原始XSpace检查更新标签、物理GPU计算流及GEMM调用数，
不依赖可能被截断的Chrome导出；H100沿用Nsight完整性和更新窗口检查。
异常会关闭已启动的profile，并保留原始错误；保存窗口不计入正式更新区间。
