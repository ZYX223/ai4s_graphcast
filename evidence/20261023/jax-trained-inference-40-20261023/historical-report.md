# 训练后 checkpoint 的 40 步推理验收

两组均在 BW1100 单卡完成。RNE＋XF32 训练后 checkpoint 相对于 FP32 训练后 checkpoint：**整体平均误差 32.264554%，未达到 5%；Z500 相对 MAE 0.926150%，达到 5%。综合验收未通过。**

| 指标 | 实测 | 门限 | 结果 |
|---|---:|---:|---|
| 六变量、十三层平均相对 MAE | 32.264554% | 严格小于 5% | 未通过 |
| Z500 相对 MAE | 0.926150% | 严格小于 5% | 通过 |
| Z500 MAE | 50.470632 m | 无独立米制门限 | 记录 |
| Z500 RMSE | 56.683963 m | 无独立米制门限 | 记录 |

各变量对 13 层取平均：U 风 34.022977%，V 风 58.188086%，位势 2.388747%，温度 0.852470%，比湿 26.793766%，垂直速度 71.341280%。78 项中 53 项达到或超过 5%；原规则针对整体平均和 Z500，不要求 78 项逐项全部小于 5%。

## 配置与口径

- 对照：`/home/zhangyuxin/zyx/graphcast/outputs/jax-training-38-20261023/fp32-r12/checkpoint.pkl`，FP32 推理。
- 候选：`/home/zhangyuxin/zyx/graphcast/outputs/jax-training-38-20261023/rne_xf32-r12/checkpoint.pkl`，RNE 舍入两侧 GEMM 输入后使用 rocBLAS XF32。
- 均为第 38 次更新后的最后 checkpoint，不是最佳 checkpoint。
- 官方源码 `08cf73625c9d12bd9aaa038868bcb2fe488f2a22`；0.25°、721×1440、13 层、batch=1、FP32 存储、无 BF16。JAX/JAXlib `0.6.0+das.opt1.dtk2604`。
- 使用历史初始场 2021-01-01 06:00。预热 5 步后重置，各自独立自回归 40×6 小时，比较第 40 步（+240h）。每步回填各自预测。
- 每个变量、每层：`mean(abs(candidate-reference)) / mean(abs(reference)) * 100`。全网格普通均值，不加纬度面积权重。整体为 78 项百分比等权平均。Z500 是 500hPa 位势的同一相对 MAE，换算米使用 9.80665。
- 本次参考为训练后的 FP32 权重，不使用原始预训练权重或 H100 历史输出。
- 本结果包含独立训练的参数差异与推理精度差异，是两条实现的输出一致性比较，不是相对于真实天气的预报误差。

## 结果与代码

- [汇总 JSON](/home/zhangyuxin/zyx/graphcast/outputs/jax-trained-inference-40-20261023/comparison/summary.json)
- [78 项误差明细](/home/zhangyuxin/zyx/graphcast/outputs/jax-trained-inference-40-20261023/comparison/by-variable-level.csv)
- [分变量平均](/home/zhangyuxin/zyx/graphcast/outputs/jax-trained-inference-40-20261023/comparison/by-variable.csv)
- [FP32 最终输出](/home/zhangyuxin/zyx/graphcast/outputs/jax-trained-inference-40-20261023/fp32/prediction.nc)
- [RNE＋XF32 最终输出](/home/zhangyuxin/zyx/graphcast/outputs/jax-trained-inference-40-20261023/rne_xf32/prediction.nc)
- [实验代码](/home/zhangyuxin/zyx/graphcast/outputs/jax-trained-inference-40-20261023/experiment/full_model.py)

各运行目录保存日志、配置、代码快照、源码/数据/权重/输出 SHA256、优化 HLO 和逐步 GEMM 记录。两组推理及比较进程退出码均为 0；正式 40 步每步 84 次 GEMM，共 3360 次/组；最终 11 个输出变量全部有限。误差公式已独立复算。

日志保留原有 HIP 编译警告和 XLA 常量折叠耗时提示，均未中断执行。同步耗时仅用于执行审计，不能用作纯卡上性能结果。

失败原因交叉对照单独保存在 `/home/zhangyuxin/zyx/graphcast/outputs/jax-trained-inference-diagnosis-20261023`。

## 本机复现

需要上述冻结资产、镜像 `sha256:919cf075fadd5fab21b2ea5978374be00caf1db7d6f02fc90846633c4c4bd546` 和空闲 GPU 12。启动器检查卡占用，拒绝覆盖已有结果。使用新的同级目录复现：

```bash
cd /home/zhangyuxin/zyx/graphcast/outputs
mkdir jax-trained-inference-40-recheck
cp -a jax-trained-inference-40-20261023/experiment jax-trained-inference-40-recheck/experiment
sg docker -c 'python3 /home/zhangyuxin/zyx/graphcast/outputs/jax-trained-inference-40-recheck/experiment/launch.py fp32 --output /home/zhangyuxin/zyx/graphcast/outputs/jax-trained-inference-40-recheck/fp32'
sg docker -c 'python3 /home/zhangyuxin/zyx/graphcast/outputs/jax-trained-inference-40-recheck/experiment/launch.py rne_xf32 --output /home/zhangyuxin/zyx/graphcast/outputs/jax-trained-inference-40-recheck/rne_xf32'
sg docker -c 'python3 /home/zhangyuxin/zyx/graphcast/outputs/jax-trained-inference-40-recheck/experiment/launch.py compare --output /home/zhangyuxin/zyx/graphcast/outputs/jax-trained-inference-40-recheck/comparison'
```
