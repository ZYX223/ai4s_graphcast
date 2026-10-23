# BW1100 推理优化与端到端短程训练（2026-10-22～23）

本轮代码继续使用官方 JAX GraphCast，保留原 RNE＋XF32 推理 baseline。实验源码按实际测量版本归档，入口独立，不自动替换 baseline；没有引入 Torch、FlagGems 或自定义 GEMM。

## 当前结论与验收口径

| 工作 | 已核验结果 | 范围 |
|---|---|---|
| Concat／SiLU 的 JAX RNE 融合 | 37处 Concat＋42处 SiLU，共79处激活舍入移入 XLA producer；rocBLAS GEMM保留84次 | 独立算子及整模型均已验证；融合版本仅用于固定预训练权重推理 |
| 40步推理 | baseline 中位数43.799秒，融合版33.311秒，1.315×；单步GPU内核累计980.072→724.395毫秒 | 同卡，各3轮；端到端时间与设备内核时间分别记录 |
| 推理一致性 | 对既有官方JAX FP32参考，融合版整体误差1.670%～1.802%、Z500误差0.0209%～0.0222% | 独立自回归40步后比较第40步，整体及Z500均<5% |
| MFU估算 | GPU内核活跃时间口径17.24%→23.32% | GEMM FLOPs近似；项目指定峰值170.3936 TFLOP/s；不是端到端计时 |
| 单卡训练 | FP32、RNE＋XF32各38次优化器更新流程，覆盖1～12步；其中2次学习率为0、36次参数实际改变 | 官方预训练权重继续训练，完整模型反向传播，FP32存储，不用BF16 |
| 训练loss一致性 | 38/38组差异<5%，最大1.3203%，最后0.3676% | 按本次确认的loss口径通过；尚未验证收敛 |
| 训练性能和显存 | GPU内核累计2278.858→1837.963秒，1.240×；JAX峰值90.866GiB，RNE整卡采样峰值128.224GiB | microbatch=1、累计2，阶段/消息步骤重计算；含RNE和优化器开销 |
| 固定权重推理诊断 | 同一FP32训练权重：整体0.03823%、Z500 0.000739%；同一RNE训练权重：整体0.04217%、Z500 0.000684% | 两组仅切换推理精度，均通过5% |
| 不同训练权重比较 | 整体32.2646%、Z500 0.9262% | 包含独立训练轨迹差异，保留诊断结果；不作为loss口径的训练失败判定 |
| 首次非零更新重放 | 固定数据/参数/Adam状态，LR=2e-4时FP32 loss 1.3054→28.4836；LR=2e-6时→1.2906 | 支持原学习率对该样本过大；不证明后续训练已收敛 |

训练loss相对差异为 `abs(loss_rne-loss_fp32)/abs(loss_fp32)`。推理整体误差为六变量×十三层各自相对MAE的等权平均；Z500单独计算，两者均严格小于5%。FP32参考的一致性与对真实天气的预测效果是不同指标。

**运行限制：**原38次训练的24个采集进程均在保存完整结果后出现signal11退出警告；报告、checkpoint和设备时间线已独立核对，数据可用，但退出清理问题尚未解决。训练后推理和最终固定状态重放均有正常退出记录。历史报告保留当时结论；本页按用户最新确认更新训练验收解释。

## 查阅代码

| 目录/入口 | 内容 |
|---|---|
| `scripts/run.py`、`src/graphcast_bench/` | 原推理baseline，以及模型profile、逻辑算子目录和独立计时入口 |
| `kernels/xf32_rne/` | 原HIP最近偶数舍入及rocBLAS拦截；原baseline计算实现保留 |
| `experiments/rne-producer-fusion-20261022/experiment-final-v2/` | 独立Concat／SiLU＋Linear组合，JAX RNE与融合正确性测试 |
| `experiments/rne-model-fusion-20261022/experiment-r1/` | 整模型按层接入、79层名单、候选拦截器生成器、baseline/fused双入口 |
| `experiments/rne-model-fusion-20261022/analysis/`、`mfu/` | profile/HLO核验、40步对比和GPU活跃时间MFU复算 |
| `experiments/jax-training-38-20261023/experiment/` | 训练、AdamW配方、梯度累计、重计算、双精度GEMM路由、时间线分析 |
| `experiments/jax-trained-inference-40-20261023/experiment/` | 两份训练后checkpoint各跑40步并比较 |
| `experiments/jax-trained-inference-diagnosis-20261023/experiment/` | 权重×推理精度的交叉诊断及参数变化统计 |
| `experiments/jax-first-update-replay-20261023/experiment/` | 固定初态，重放首次非零更新；学习率扫描、梯度/参数变化 |
| `experiments/shared/` | 本轮实际使用的公共baseline源码与训练所需官方源码快照 |
| `evidence/20261022/`、`evidence/20261023/` | 小型结果表、kernel明细、数值核验和原始报告 |

`experiments/PROVENANCE.json`逐文件保存原路径和SHA256。实验内的重复小文件是实际运行快照，按原字节保留；格式检查排除该目录，用哈希校验防止无意修改。公共baseline只有一份归档，准备运行目录时复制给各入口。

训练使用通用双操作数RNE路由，支持更新后的权重和反向GEMM；**不能使用固定预训练权重指纹的推理融合路由替代训练路由**。

## 环境与外部材料

- 机器：Hygon_ZYX，项目型号BW1100、约144GiB；驱动字符串仍为BW1200/gfx938。
- 镜像：`sha256:919cf075fadd5fab21b2ea5978374be00caf1db7d6f02fc90846633c4c4bd546`，镜像ID不是公开下载地址。新机器需导入同一已验证镜像。
- JAX/JAXlib `0.6.0+das.opt1.dtk2604`，Python3.10，DTK26.04；训练另需Optax0.2.4、etils1.12.2，来源及wheel校验值在`environment/training/`。
- 官方源码提交 `08cf73625c9d12bd9aaa038868bcb2fe488f2a22`。原推理16模块与训练额外数据处理模块分别保留原许可证、来源和哈希。
- 本机材料根目录：`/home/zhangyuxin/zyx/graphcast/outputs`。下面路径均相对该根目录。

| 外部路径 | 用途 |
|---|---|
| `jax-reorganization-20261016/prepared/data/` | 权重、固定40步推理案例、归一化统计、FP32参考及manifest |
| `logical-operators-20261022/capture-r3/run/`、`bench-full-r2/logical/` | 独立producer融合所需的真实形状、图常量和代表权重 |
| `jax-single-card-training-20261022/dependencies/` | 固定版本训练依赖的已安装目录和wheel |
| `jax-single-card-training-20261022/data/` | 小型训练预检数据 |
| `jax-training-38-20261023/data/` | 真实训练标签与来源校验，HRES14时刻数据约4.89GB |
| `jax-training-38-20261023/{fp32,rne_xf32}-r12/` | 两份最终训练checkpoint及report |
| `jax-training-38-20261023/fp32-r01/` | 首步重放需要的前两次FP32梯度及report |
| 各同名实验目录 | 原始日志、预测、HLO、trace、显存采样和优化器状态 |

仓库不分发上述大文件、镜像或截图。仅克隆仓库不足以运行完整实验，必须准备表中对应材料；宿主启动器需要Python3.6+、Docker权限和`/opt/hyhal`。容器对输入材料只读挂载。旧登录会话可以用`sg docker -c '命令'`使用已有Docker组权限。

## 准备独立工作目录

在仓库根目录执行。`WORK`必须不存在，且位于代码仓库之外。准备过程不占GPU，也不启动训练。

```bash
export ASSETS=/home/zhangyuxin/zyx/graphcast/outputs
export WORK=/home/zhangyuxin/zyx/graphcast/outputs/graphcast-rerun-new
python3 scripts/prepare_experiments.py --verify-only
python3 scripts/prepare_experiments.py --assets-root "$ASSETS" --output "$WORK"
```

只准备一类任务可加`--suite inference`或`--suite training`。如直接使用历史checkpoint做推理/重放，加`--reuse-training-results`；该选项只取诊断需要的第1/12阶段材料，**不要在此目录执行完整38次训练**。重新训练使用不带此选项的独立目录。

准备器复制实测源码，链接外部输入；训练大文件在同文件系统上以硬链接复用，小型报告复制。复用checkpoint要求同文件系统；原始checkpoint必须保持只读。`workspace.json`记录实际来源和源码哈希。唯一启动适配是让复制后的课程脚本读取`GRAPHCAST_DEVICE`，默认仍为12，计算代码未改变。

## 推理复现

先检查所选卡空闲，以下示例使用GPU12；启动器还会检查占用并核对PCI。保留输出命名可直接使用后续分析脚本。

```bash
FUSION="$WORK/rne-model-fusion-20261022"
python3 "$FUSION/experiment-r1/launch.py" check --variant fused --output "$FUSION/cpu-check"
for variant in baseline fused; do
  python3 "$FUSION/experiment-r1/launch.py" profile --variant "$variant" --device 12 \
    --output "$FUSION/profile-$variant-r1"
  python3 "$FUSION/experiment-r1/launch.py" bench --variant "$variant" --device 12 --repeats 3 \
    --output "$FUSION/bench-$variant-r1"
done
python3 "$FUSION/analysis/run.py" /results/analysis/validate_suite.py
```

该核验检查真实kernel、84次GEMM、168→89次独立RNE、79层HLO融合及最终预测；不以采集器退出码代替模型完成证据。单轮结果写在各运行目录的`run/report.json`，逐轮最后输出为`run/roundN-rne_xf32/prediction.nc`。

MFU在上述profile完成后复算，峰值沿用用户指定的170.3936TFLOP/s：

```bash
python3 "$FUSION/mfu/launch.py"  # 实测参照GEMM；此历史入口固定GPU12，需空闲
python3 "$FUSION/mfu/calculate.py" --theoretical-peak-tflops 170.3936 \
  --peak-source '项目指定：NMZ 1300MHz TF32 Tensor峰值'
```

该MFU入口固定本次GPU12/PCI，未推广到其他卡；分子为84次GEMM的`sum(2MNK)`，分母用全部GPU内核活跃时间并集，包含RNE等内核，不包含CPU、拷贝与空隙。单步设备时间不能替代40步端到端时间。

独立producer组合可运行`rne-producer-fusion-20261022/experiment-final-v2/launch.py bench`，采集使用`experiment-profile-v2/launch.py profile`，均要求`--device`和新的`--output`。对应小型汇总在`evidence/20261023/rne-producer-fusion-20261022/`。其`analyze_results.py`以及整模型`analysis/finalize.py`保留历史机器路径和历史baseline完整性断言，是原始审计脚本；新工作目录采用上面的`validate_suite.py`，不直接执行历史报告生成器。

## 训练与诊断复现

重新训练：选不带`--reuse-training-results`准备的目录。

```bash
TRAIN="$WORK/jax-training-38-20261023"
python3 "$TRAIN/experiment/launch.py" cpu --output "$TRAIN/cpu-check"
python3 "$TRAIN/experiment/launch.py" probe --device 12 --output "$TRAIN/device-probe"
GRAPHCAST_DEVICE=12 python3 "$TRAIN/run_curriculum.py"
python3 "$TRAIN/summarize.py"
python3 "$TRAIN/verify_evidence.py"
```

原配方为16次单步更新、2～12步各2次，microbatch=1/累计2，AdamW、裁剪32，最大LR=1e-3。此处保留实测配方用于复现，不将其推荐为已收敛的继续训练方案。训练采集退出警告必须结合`report.json`、checkpoint哈希和trace确认，仍待单独排查。

已有完整训练结果或用`--reuse-training-results`准备后，运行四种权重/推理精度组合：

```bash
PAIR="$WORK/jax-trained-inference-40-20261023"
CROSS="$WORK/jax-trained-inference-diagnosis-20261023"
for mode in fp32 rne_xf32; do
  python3 "$PAIR/experiment/launch.py" "$mode" --device 12 --output "$PAIR/$mode"
done
python3 "$PAIR/experiment/launch.py" compare --output "$PAIR/comparison"
python3 "$CROSS/experiment/launch.py" rne_xf32 --checkpoint-mode fp32 --device 12 \
  --output "$CROSS/train-fp32-infer-rne_xf32"
python3 "$CROSS/experiment/launch.py" fp32 --checkpoint-mode rne_xf32 --device 12 \
  --output "$CROSS/train-rne_xf32-infer-fp32"
python3 "$CROSS/experiment/launch.py" compare --output "$CROSS/comparison"
python3 "$CROSS/experiment/launch.py" parameters --output "$CROSS/parameters"
```

固定状态重放：读取当前工作目录中FP32第1阶段的前两次梯度，重建同一Adam状态，固定窗口[2,3]。要复现历史初态请使用历史训练材料；若使用重训梯度，这是新的初态实验，应保留新的校验值。

```bash
REPLAY="$WORK/jax-first-update-replay-20261023"
python3 "$REPLAY/experiment/launch.py" replay --device 12 --output "$REPLAY/run-preallocated"
python3 "$REPLAY/summarize.py" "$REPLAY/run-preallocated"
```

重放默认预分配70%显存池，保留完成实测时的配置。五档学习率各自从同一初态出发；更新前后统一使用FP32前向评价loss，分别记录梯度、Adam更新方向和实际参数变化。
