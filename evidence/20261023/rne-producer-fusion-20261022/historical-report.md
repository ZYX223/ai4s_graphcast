# BW1100：JAX Concat／SiLU＋RNE 融合可行性实验

2026-10-22。结论：本次12组独立算子组合中，XLA确实将JAX表达的RNE融合进Concat或SiLU；rocBLAS GEMM及权重RNE保留。全部融合producer及组合完整输出与baseline逐位一致。数据来自同一张GPU12（PCI 0000:b0:00.0），项目型号BW1100，驱动/JAX设备名BW1200，gfx938、DTK26.04、JAX 0.6.0+das.opt1.dtk2604。

这是可行性实验，代码和结果均位于独立输出目录，没有接入正式GraphCast推理入口。原仓库及当前baseline的115个文件SHA-256核对不变，rocBLAS库、官方模型代码均未修改。

## 性能结果

以下是“Concat／SiLU＋Linear”的独立组合耗时，含权重RNE、rocBLAS GEMM和bias，以及JAX调度与同步。每条路径测3轮，每轮5次预热、20次测量，轮次顺序为B/C、C/B、B/C；先取轮内中位数，再取三轮中位数。编译、随机输入准备和正确性检查不计时。性能数据仅采用未开启profiler的`bench-r2`。

| 组合（M×K，输出宽度N） | 图中频次 | baseline ms | 融合后 ms | 加速倍数 | 耗时下降 |
|---|---:|---:|---:|---:|---:|
| Concat → Linear：327660×1536 → 512 | 16 | 13.995 | 9.183 | 1.524× | 34.38% |
| SiLU → Linear：327660×512 → 512 | 17 | 5.804 | 4.112 | 1.411× | 29.15% |
| Concat → Linear：3114720×1536 → 512 | 1 | 137.195 | 85.163 | 1.611× | 37.93% |
| Concat → Linear：1038240×1024 → 512 | 1 | 30.497 | 20.141 | 1.514× | 33.96% |
| Concat → Linear：40962×1024 → 512 | 17 | 1.410 | 0.964 | 1.463× | 31.64% |
| Concat → Linear：1038240×184 → 512 | 1 | 9.247 | 7.460 | 1.240× | 19.33% |
| Concat → Linear：1618745×1536 → 512 | 1 | 68.778 | 44.287 | 1.553× | 35.61% |
| SiLU → Linear：3114720×512 → 512 | 2 | 54.418 | 40.621 | 1.340× | 25.35% |
| SiLU → Linear：40962×512 → 512 | 17 | 0.976 | 0.702 | 1.391× | 28.10% |
| SiLU → Linear：1618745×512 → 512 | 2 | 28.448 | 21.354 | 1.332× | 24.94% |
| SiLU → Linear：1038240×512 → 83 | 1 | 13.394 | 8.910 | 1.503× | 33.48% |
| SiLU → Linear：1038240×512 → 512 | 3 | 18.010 | 13.407 | 1.343× | 25.55% |

12组代表形状对应原图79次直接连接，其中Concat 37次、SiLU 42次。每组使用原图形状和一份真实权重、bias，激活为固定种子的uniform[-4,4]生成值，不是模型捕获的中间激活。79次是这些形状在图中的频次，未单独测量79套权重。

按图中频次对独立组合耗时加权：842.018 ms → 575.293 ms，下降31.68%。**该加权结果不是整模型单步延迟，更不是40步推理加速比例**；组合间的融合、显存复用、上游运算和运行时调度均可能改变最终收益。

## 融合证据

12组优化HLO中，producer＋RNE均只有一个入口fusion，其中同时包含Concat或SiLU计算以及bitcast、整数移位/加法/掩码、NaN/Inf选择。后面继续有GEMM custom call。HLO沿用`__cublas$gemm`名称，但本环境的运行时调用由原生rocBLAS执行；已用拦截日志和Tensile设备kernel交叉验证。

设备trace为`profile-r3`，每种形状、每种路径各捕获3次，合计72次GEMM和324次GPU kernel：

| 每条组合的实际内核 | baseline | 候选 |
|---|---:|---:|
| producer（候选含RNE） | 1 | 1 |
| 权重RNE | 1 | 1 |
| 激活RNE | 1 | 0 |
| rocBLAS GEMM | 1 | 1 |
| bias | 1 | 1 |
| 合计 | 5 | 4 |

各组两路径的GEMM kernel名字相同。下面为trace中各项GPU内核耗时的3次中位数（ms），用于解释融合效果；不能与上面的同步JAX计时相加，也不作为性能主结果。

| 组合 | 原 producer | 激活 RNE | 融合 producer＋RNE | 原 GEMM | 候选 GEMM |
|---|---:|---:|---:|---:|---:|
| Concat → Linear：327660×1536 → 512 | 2.575 | 4.607 | 2.574 | 4.775 | 4.787 |
| SiLU → Linear：327660×512 → 512 | 1.050 | 1.624 | 1.189 | 1.893 | 1.891 |
| Concat → Linear：3114720×1536 → 512 | 24.096 | 61.966 | 23.668 | 50.207 | 53.258 |
| Concat → Linear：1038240×1024 → 512 | 5.324 | 10.811 | 5.299 | 12.051 | 12.353 |
| Concat → Linear：40962×1024 → 512 | 0.210 | 0.495 | 0.211 | 0.531 | 0.530 |
| Concat → Linear：1038240×184 → 512 | 1.129 | 1.795 | 1.188 | 3.329 | 3.339 |
| Concat → Linear：1618745×1536 → 512 | 12.349 | 26.139 | 12.303 | 26.268 | 28.084 |
| SiLU → Linear：3114720×512 → 512 | 10.864 | 16.090 | 12.512 | 19.849 | 20.559 |
| SiLU → Linear：40962×512 → 512 | 0.128 | 0.275 | 0.144 | 0.321 | 0.319 |
| SiLU → Linear：1618745×512 → 512 | 5.761 | 8.415 | 6.711 | 10.126 | 10.301 |
| SiLU → Linear：1038240×512 → 83 | 3.132 | 4.861 | 3.567 | 3.009 | 3.119 |
| SiLU → Linear：1038240×512 → 512 | 3.379 | 5.315 | 3.563 | 7.087 | 6.689 |

## 精度与记录核对

- 100,017个FP32位模式：JAX RNE、原HIP RNE、独立商余数RNE参考逐位一致。覆盖最近偶数舍入、正负数、±0、次正规数、上溢、Inf和NaN payload。
- 12组producer完整张量：融合结果与“先物化producer、再单独JAX RNE”逐位一致。
- 12组完整Linear输出：候选与原HIP双输入RNE＋rocBLAS路径逐位一致，最大绝对差为0；另存32行输出样本供检查。
- 每次调用验证GEMM、舍入路由、跳过激活舍入的计数；计时区间JIT缓存不变。
- 最终profile的456条GEMM日志（含预热与校验）逐例核对实际权重SHA、XF32 mode=1、status=0，全部通过。保持所有代表权重驻留，避免用例间地址复用。
- 4项RNE测试、原仓库32项CPU测试通过；源文件、输入清单和结果身份均保存校验值。

尚未进行这条候选路径的整模型单步/40步测试，本次逐位相同不能替代真实自回归验收。

## 实现方式

`experiment-final-v2/rne_ops.py`以JAX整数运算表达原有舍入规则，张量仍存FP32：

```python
bits = jax.lax.bitcast_convert_type(x, jnp.uint32)
rounded = (bits + jnp.uint32(0xfff)
           + ((bits >> 13) & jnp.uint32(1))) & jnp.uint32(0xffffe000)
bits = jnp.where((bits & jnp.uint32(0x7f800000)) == jnp.uint32(0x7f800000),
                 bits, rounded)
y = jax.lax.bitcast_convert_type(bits, jnp.float32)
```

候选将`rne(producer(inputs))`和Linear放在同一个`jax.jit`中编译。实验版拦截器在已验证输入经过JAX RNE的调用上，仅跳过激活的HIP RNE；权重继续调用原HIP RNE，矩阵乘继续调用原生rocBLAS。没有自写GEMM，也没有修改rocBLAS。

实验拦截器的跳过开关仅适合本测试中每次执行一条已核验链。**不能把这个全局开关直接用于整模型**：正式接入需要按已覆盖的层/输入路由，保护未经过JAX RNE的其余GEMM，并重新检查完整图融合。

## 文件与复现

实验根目录：`/home/zhangyuxin/zyx/graphcast/outputs/rne-producer-fusion-20261022`。

| 路径（相对于实验根目录） | 内容 |
|---|---|
| `baseline-source/`、`baseline-manifest.json` | 保留的baseline及原文件校验值 |
| `experiment-final-v2/` | 计时/正确性实验源码、JAX RNE、实验拦截器补丁生成器、启动脚本 |
| `experiment-profile-v2/` | 相同计算代码，附采集器启动环境绕过脚本 |
| `bench-r2/report.json` | 正式3×20计时、全部逐位校验、源文件校验值 |
| `bench-r2/*.hlo.txt` | 每组producer、融合producer、baseline和候选的编译结果 |
| `profile-r3/trace.json`、`trace.hipkernel.csv`、`gemm.jsonl` | 实际设备内核与GEMM日志 |
| `summary.json`、`timings.csv`、`kernel-breakdown.csv` | 汇总核验、全部计时和内核拆分 |
| `check-final/console.log` | 最终4＋32项测试 |
| `analyze_results.py` | 重算计时中位数、验证HLO/trace/权重身份与baseline保留情况 |

输入来自冻结目录`/home/zhangyuxin/zyx/graphcast/outputs/logical-operators-20261022/capture-r3/run`，以及`/home/zhangyuxin/zyx/graphcast/outputs/logical-operators-20261022/bench-full-r2/logical`，启动器只读挂载。镜像固定为`sha256:919cf075fadd5fab21b2ea5978374be00caf1db7d6f02fc90846633c4c4bd546`。运行前确认卡空闲；启动器会再次检查。输出目录必须不存在：

```bash
python3 /home/zhangyuxin/zyx/graphcast/outputs/rne-producer-fusion-20261022/experiment-final-v2/launch.py bench --device 12 --output /home/zhangyuxin/zyx/graphcast/outputs/rne-producer-fusion-20261022/reproduce-bench
python3 /home/zhangyuxin/zyx/graphcast/outputs/rne-producer-fusion-20261022/experiment-profile-v2/launch.py profile --device 12 --output /home/zhangyuxin/zyx/graphcast/outputs/rne-producer-fusion-20261022/reproduce-profile
```

当前会话需要用`sg docker -c '上面的命令'`获取已有Docker组权限；新登录通常不需要。正式结果以`status=completed`、`complete_catalog=true`、两项`all_*_bits_equal=true`为准；profile还要求每行`all_runtime_weight_hashes_match=true`，并验证真实设备trace。

旧版`bench-r1/profile-r2`因跨用例权重指针复用导致日志身份标签不可靠，已由本次完整重跑取代，保留作诊断。`profile-r1`是采集器启动失败记录；空程序可复现该启动环境问题，最终通过父进程简化环境、子进程恢复相同JAX/XLA配置完成采集，未修改采集器。旧记录不参与本页性能结论。

## 后续接入

先以独立候选配置接入整模型，在目标Linear输入处加入JAX RNE并逐层取消重复的激活RNE；保留现有baseline入口。然后验证单步完整图的融合、内核数、耗时和输出，最后按既有40步口径复测性能和第40步精度。实际融合范围和整模型收益须以这些测试为准。
