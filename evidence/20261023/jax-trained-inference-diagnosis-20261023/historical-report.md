# 训练后 40 步整体误差未通过的定位

已完成两个额外的 40 步交叉对照。**本次 32.2646% 的整体误差主要来自训练后 checkpoint 已经不同。固定同一个 checkpoint 时，只切换 FP32 / RNE＋XF32 推理，整体误差约 0.04%，均通过 5%。**

## 交叉对照

四组均使用同一套官方 JAX 模型、同一历史输入/forcing/统计量、同一 BW1100 物理卡；每组预热 5 步后重置，独立推进 40 步，比较最终 +240h。没有重新训练或改变已有 checkpoint。

| 编号 | 训练 checkpoint | 推理模式 | 结果目录 |
|---|---|---|---|
| A | FP32，第 38 次更新 | FP32 | `/home/zhangyuxin/zyx/graphcast/outputs/jax-trained-inference-40-20261023/fp32` |
| B | RNE＋XF32，第 38 次更新 | RNE＋XF32 | `/home/zhangyuxin/zyx/graphcast/outputs/jax-trained-inference-40-20261023/rne_xf32` |
| C | 与 A 完全相同的权重 | RNE＋XF32 | `/home/zhangyuxin/zyx/graphcast/outputs/jax-trained-inference-diagnosis-20261023/train-fp32-infer-rne_xf32` |
| D | 与 B 完全相同的权重 | FP32 | `/home/zhangyuxin/zyx/graphcast/outputs/jax-trained-inference-diagnosis-20261023/train-rne_xf32-infer-fp32` |

| 比较（前者为候选，后者为参考） | 只改变的因素 | 整体平均误差 | Z500 相对 MAE | 5% 验收 |
|---|---|---:|---:|---|
| B 对 A | 原始比较：训练与推理均不同 | 32.264554% | 0.926150% | 整体未通过 |
| D 对 A | 两份权重都用 FP32 推理 | 32.265616% | 0.925877% | 整体未通过 |
| C 对 A | 固定 FP32 checkpoint，切换推理精度 | 0.038230% | 0.000739% | 通过 |
| B 对 D | 固定 RNE checkpoint，切换推理精度 | 0.042166% | 0.000684% | 通过 |
| B 对 C | 两份权重都用 RNE＋XF32 推理 | 32.259785% | 0.926693% | 整体未通过 |

统一推理精度后，约 32.26% 的差异仍然存在；固定权重后仅切换精度的差异很小。因此主因已定位到独立训练后形成的参数差异。上述是控制变量对照，不应把百分比直接相减或解释为精确的因果贡献率。

## 参数差异主要在前 16 次单步训练中形成

直接读取第 16 次与第 38 次更新的四份 checkpoint，逐参数用 FP64 复算：

- 第 16 次更新后，两条轨迹参数相对 L2 差异：**1.082993%**。
- 第 38 次更新后：**1.082449%**。
- 两次之间，参数差异向量的变化范数只占最终差异范数的 **0.500798%**，两个差异向量余弦为 **0.999987592**。

这比单看两个范数相近更直接：后续 2～12 步训练没有显著改变已经形成的参数差异。应优先检查前 16 次单步更新，而不是把问题归于最后的 12 步反向。

参数范数差异本身不等于输出误差。40 步推理的交叉对照才是其输出影响的证据；分模块参数差异也不能直接用来判断某层的因果贡献。

## 为什么 loss 接近，预测却相差较大

这次 loss 是官方模型对真实训练目标计算的归一化残差 MSE，还带纬度、气压层和变量权重，并对训练预测时效求平均。5% 验收则对两个模型在另一初始场上的 +240h 物理量输出计算相对 MAE，78 项等权平均。参考对象、时效、归一化、权重和范数均不同。

两条训练轨迹可以对真实目标产生大小相似但空间分布不同的误差，从而 loss 接近、彼此预测差异较大。最后 loss 仅相差约 0.3676%，不能替代输出一致性验证。

原始比较的变量平均相对 MAE：

| 变量 | 对 13 层平均 |
|---|---:|
| U 风 | 34.022977% |
| V 风 | 58.188086% |
| 位势 | 2.388747% |
| 温度 | 0.852470% |
| 比湿 | 26.793766% |
| 垂直速度 | 71.341280% |

差异主要体现在风场、比湿、垂直速度。较小的参考场平均绝对值会提高相对误差，但不是单纯的数值计算假象：例如 V500 的参考平均绝对值为 3.562678 m/s，MAE 为 4.935578 m/s，相对误差为 138.535620%。

Z500 通过不代表其他变量通过，也不等于真实天气预报技能达标。本次两训练后模型之间的 Z500 RMSE 为 56.683963 m；验收的百分比使用参考位势的平均绝对值作为分母。

## 训练阶段的首要疑点与尚未确认的部分

前 16 次使用压缩学习率日程：6 次从 0 增到 1e-3，再 10 次降到 0；之后 22 次使用 3e-7。起点是预训练权重，训练数据只有 14 个连续时刻，很多窗口重复。

第 2 次更新使用首次非零学习率 2e-4。FP32 记录的 loss 在第 2 次为 1.305405、第 3 次为 28.436411，裁剪前梯度范数从 20.5095 增至 1667.0909；RNE 路径出现相同方向的变化。第 16 次单步 loss 仍约 6.83 / 6.85。记录提示这一压缩训练方案存在稳定性问题，学习率和短样本设置需要优先核实。

**这些记录还不能单独证明学习率是唯一原因：第 2、3 次的采样窗口不同，尚未做固定同一 batch 的更新前后验证。** 不同预测长度的 loss 也不能直接拼接成收敛曲线。

在最初参数相同的两次梯度计算中，FP32 与 RNE 路径梯度相对 L2 差异约 2.290% / 3.538%；随后独立更新，差异累积。后期最大 76.11% 的梯度差异已经混合参数轨迹偏离，不能当作单次低精度反向算子的误差。

目前确认到“训练后的权重差异”这一层；尚未拆分前向精度、反向精度、优化器对扰动的响应，以及相同 FP32 训练重跑的非确定性底噪。

## 下一步建议

1. 固定同一初始权重、同一 batch、同一优化器状态，重放首次非零更新，测更新前后 loss 与参数变化，排除窗口变化影响。
2. 在相同参数点比较梯度和实际 AdamW 更新，分别控制前向/反向计算精度，定位训练阶段最敏感的环节。
3. 验证适合继续训练的学习率日程，再做 38 次流程验证与 40 步输出验收。现有 checkpoint、RNE 推理 baseline 和 5% 门限保留。

本轮仅做诊断，没有改变学习率、重新训练或修改原有模型实现。

## 证据与复现

- [交叉比较汇总](/home/zhangyuxin/zyx/graphcast/outputs/jax-trained-inference-diagnosis-20261023/comparison/summary.json)
- [交叉比较 CSV](/home/zhangyuxin/zyx/graphcast/outputs/jax-trained-inference-diagnosis-20261023/comparison/comparisons.csv)
- [参数差异定位](/home/zhangyuxin/zyx/graphcast/outputs/jax-trained-inference-diagnosis-20261023/parameters/summary.json)
- [分模块参数差异](/home/zhangyuxin/zyx/graphcast/outputs/jax-trained-inference-diagnosis-20261023/parameters/by-module.csv)
- [38 次训练记录](/home/zhangyuxin/zyx/graphcast/outputs/jax-training-38-20261023/by-update.csv)
- [交叉推理代码](/home/zhangyuxin/zyx/graphcast/outputs/jax-trained-inference-diagnosis-20261023/experiment/full_model.py)

各运行目录保存代码快照、配置、checkpoint/输出校验值、HLO、逐步日志、实际 GEMM 模式及退出码。四种配置各运行一次；未用多次运行估计非确定性底噪。

复现时复制本目录的 `experiment` 到新的同级目录，使用 `launch.py`：分别执行 `fp32 --checkpoint-mode rne_xf32` 和 `rne_xf32 --checkpoint-mode fp32`，输出子目录分别命名为 `train-rne_xf32-infer-fp32`、`train-fp32-infer-rne_xf32`；再执行 `compare` 和 `parameters`。比较脚本读取原始两组结果作为 A/B，所有输入目录只读挂载。具体完整参数及镜像挂载见每组 `launch.json`。默认使用空闲 GPU 12，既有输出目录会被拒绝覆盖。
