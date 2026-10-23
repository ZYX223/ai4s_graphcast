# GraphCast 五国产平台推理性能表

主表数据采集：2026-10-20；910C FlagGems 行更新为 2026-10-21 官方新版 Mul 的实测结果，其他行不变。910C 官方 Cat 隔离试验单列，不混入主表。配套见[算子接入与覆盖率表](graphcast-inference-operator-coverage-20261021.md)。

另附[算子耗时与覆盖率 Excel 合并版（2026-10-23）](graphcast-operators-and-coverage-20261023.xlsx)，按推理、覆盖率、训练和证据分 sheet；其中算子累计耗时不同于本页模型 E2E。本次表格整理未改动本页性能数据。

## 性能结果

单设备、单样本、40 步连续自回归，三次独立运行取 E2E 中位数。执行模式为 eager，请求 TF32/HF32（fast-FP32），参数、输入输出和自回归状态为 FP32；没有使用 BF16 autocast、torch.compile 或图捕获。

| 平台 | 实现 | 40 步 E2E / s | 平均每步 / s | 等效吞吐 / TFLOP/s | 分母 / TFLOP/s | MFU | 末帧诊断 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| H100（历史对照） | Native | 11.8082 | 0.2952 | 97.7530 | 495 | 19.7481% | 历史通过 |
| C550 | Native | 31.2398 | 0.7810 | 36.9491 | 132.49 | 27.8882% | 3/3 通过 |
| C550 | FlagGems + 内部优化 | 41.7631 | 1.0441 | 27.6389 | 132.49 | 20.8611% | 3/3 通过 |
| S5000 | Native（排查记录） | 21.0232 | 0.5256 | 54.9054 | 224.07 | 24.5037%* | 0/3 通过 |
| S5000 | FlagGems + 内部优化 | 28.5038 | 0.7126 | 40.4959 | 224.07 | 18.0729% | 3/3 通过 |
| 910C | Native | 34.8585 | 0.8715 | 33.1134 | 199 | 16.6399% | 3/3 通过 |
| 910C | FlagGems + 内部优化 + 官方新版 Mul | 34.1451 | 0.8536 | 33.8053 | 199 | 16.9876% | 3/3 通过 |
| 海光 BW1101 | Native（排查记录） | 53.7974 | 1.3449 | 21.4561 | 236.94（暂定） | | 0/3 通过 |
| 海光 BW1101 | FlagGems + 内部优化 | 24.1941 | 0.6049 | 47.7094 | 236.94（暂定） | | 3/3 通过 |
| 810E | Native | 25.2885 | 0.6322 | 45.6447 | 61.5 | 74.2190% | 3/3 通过 |
| 810E | FlagGems + 内部优化 | 57.1711 | 1.4293 | 20.1900 | 61.5 | 32.8293% | 3/3 通过 |

- 海光为新 BW1101、约 144 GiB，不能当作旧 BW3000 同硬件复测。峰值暂记 236.94 TFLOP/s，尚未核实；按当前约定，Native 和 FlagGems 的 MFU 均留空，不计算，也不发布此前两种分母下的条件结果。
- `*`：S5000 Native 的 MFU 仅为耗时的算术换算，诊断未通过，不是合格性能结果。S5000、海光的 Native 均只保留作排查记录，不用于正式加速比。
- H100 沿用历史 Native TF32 eager 对照，本轮没有重测。其 MFU 的 80% 为 15.7985%；C550、S5000、910C、810E 的 FlagGems 均达到该值，海光暂不判断。910C 更新后的 MFU 约为 H100 的 86.02%。达到这一跨平台门槛不等于快于同平台 Native。

## 计时与 MFU 口径

每次先做 3 次 forward 预热，再计时连续 40 个预报步；每步 6 小时，末帧为 +240h。算子 trace 单独采集，不混入性能计时。新版 910C Mul 测试还预先指定了一轮完整运行用于预热缓存，保留记录但不计入三轮中位数。

- 计入：模型计算、残差还原、状态更新、forcing 选择、末帧有限值检查、末帧 D2H 和最终同步。
- 不计入：进程及模型初始化、数据读取和初始上传、预热编译、CPU 物理量转换、NetCDF 写盘及离线诊断。计时内如发生新编译，不事后扣除。
- 中间自回归状态常驻设备，不逐步回传 CPU。

沿用历史固定计算量，不改为投影分解后的实际 FLOPs：

$$F_{单步}=28.857094475776\;\mathrm{TFLOP},\qquad F_{40步}=1154.28377903104\;\mathrm{TFLOP}.$$

$$\text{等效吞吐}=F_{40步}/T_{E2E},\qquad \mathrm{MFU}=\frac{F_{40步}}{T_{E2E}\times P_{单设备}}\times100\%.$$

这里是单设备，不再乘卡数。投影分解减少了实际计算，因此该 MFU 是固定原工作量的等效效率，不是优化后实际执行 FLOPs 的硬件利用率。海光暂停此公式的 MFU 换算，其 E2E 和等效吞吐仍保留。其他平台分母为此前约定值，本轮未重新核定。

## 诊断与优化配置

诊断对象是 40 步自由自回归的末帧，分别对固定官方 JAX 严格 FP32 参考和 H100 Torch Native eager 严格 FP32 参考比较。6 个变量、13 个气压层的 78 项相对 MAE 等权平均，以及 Z500 相对误差，针对两个参考都必须严格小于 5%。输入、权重、参考身份和有限值同时检查。

表中的“通过”不是逐步误差或所有单层误差都小于 5%，也不等同于多案例天气技能验证。固定参考 SHA256：`10227f90bbd6dc94c032cf890ff6c6c9871e21d0a1678123c6ca183ae996be10`。

五平台均保留 `domestic-v1` 的边投影线性分解、full-grid mesh→grid 快速路径、`advanced_index` gather 和设备侧自回归。S5000、910C、海光的 FlagGems 组另有已测融合路径。精度请求相同不保证底层指令相同；S5000、海光部分内部 AddMM 路径含 TF32 舍入、FP16 打包及 FP32 累加，不能表述为全数值范围硬件 TF32 等价。

## 910C 官方新版 Mul

主表采用完整官方 Ascend `mul.py`，固定到 [FlagGems `367d193`](https://github.com/flagos-ai/FlagGems/blob/367d19367880eaf8843449abe11db4cfb7d47e03/src/flag_gems/runtime/backend/_ascend/ops/mul.py)，不再使用本地分块移植实现。只导出非原地 `mul`，并复制同一提交的 `mul_flat`、`mul_broadcast_tiled` 调优配置；`mul_` 及其他算子保持原实现，没有整体升级 FlagGems。

模型侧保留乘法前后的二维无拷贝视图：`[1,N,C] → [N,C] → [1,N,C]`，使实际调用命中官方分块广播内核。输入、状态和乘法仍为 FP32，矩阵运算请求 HF32；投影分解、full-grid 路径和原有融合配置均保留。Cat 仍用原包装路径，`copy_` 仍用 Native。

| 第 1 轮 / s | 第 2 轮 / s | 第 3 轮 / s | 中位数 / s | 双参考末帧诊断 |
| ---: | ---: | ---: | ---: | --- |
| 34.2245 | 33.9634 | 34.1451 | 34.1451 | 3/3 通过 |

另有一轮完整缓存预热，耗时 37.5220 秒，按测试前约定不计分。三轮均无 profiler；设备 trace 另采集。沿用原分子和 199 TFLOP/s 分母：

$$\mathrm{MFU}=\frac{1154.28377903104}{34.1451016003266\times199}\times100\%=16.9876\%.$$

三轮双参考诊断中，整体误差最大为 **0.1007705%**，Z500 最大为 **0.00110695%**，均小于 5%。这是 40 步末帧的固定案例诊断，不代表所有时效或多案例天气技能验证。算子专项测试 78 项、旧版与新版官方非原地 Mul 测试分别 176、224 项通过；两套官方测试各按平台规则跳过 48 项。

完整 40 步设备 trace 确认 **400 次 `mul_broadcast_tiled_kernel`**，来源为上述官方文件，没有发现 Native `aclnnMul`。此次只专项复核 Mul，没有重新发布全模型覆盖率；沿用既有 10/12＝83.33% 的统计，不将日志中的 `11/11` 当作覆盖率。

旧主表的 47.9672 秒、12.0925% 保留为历史记录，由上述实测结果替代。新方案较旧记录耗时下降约 28.82%；这一变化包含二维视图与分块 Mul 两部分，不能全部归因于“换成官方整文件”。与已验证分块移植版的 34.0446 秒相比，官方整文件版耗时高约 0.30%，三轮区间重叠，未观察到明显性能差异。

复查记录：

- 本地目录：`/mdata/data/zhangyuxin/wxt/inference-910c-mul-official-20261021-3X1LSo`；`result-summary.json` 为从原始报告、诊断和设备 trace 复算的汇总，`evidence/model-results/official-r1` 至 `official-r3` 为三轮原始记录。
- 容器内算子目录：`/workspace/FlagOS/campaigns/inference-910c-mul-official-20261021-3X1LSo/gems-official`；模型使用已验证二维视图副本 `inference-910c-mul-view-20261021-BHQVmn/candidate-project`，不是仅凭原提交号复现。
- 官方 Mul 文件 SHA256：`3d4269688e13ba1f3dda7160a95223010c39cbbcd9fd47ce3dfe0bcb4721519b`。
- 原始证据包 `completed-evidence.tar.gz`，SHA256：`87775588964fee179ffd64b9edfcdfe243bf6d1e15ecb41bde8d766f5e4c98ce`。原始证据未上传 GitHub；大体积预测文件保留在远端，哈希在证据包中。

本次更新的是文档中的已测配置和结果，没有修改远端运行环境或重新启动测试。

## 910C 官方 Cat 隔离试验

2026-10-21 补充：新版官方 Ascend Cat 数值测试通过，真实模型中也确认调用了 Triton；但在当前环境下，大尺寸末维拼接明显变慢，暂不替换正式配置。此 Cat 候选不纳入上表 E2E、MFU 和配套覆盖率。

测试源码来自已合并的 [PR #6548](https://github.com/flagos-ai/FlagGems/pull/6548)，固定为 [FlagGems `72b2a1a` 的 Ascend Cat](https://github.com/flagos-ai/FlagGems/blob/72b2a1a38b476df9bec0b20e8a498c1e4756d199/src/flag_gems/runtime/backend/_ascend/ops/cat.py)。官方文件保持字节不变；只在隔离源码中替换 Cat，没有升级整个算子库或运行环境，也不是此前的 common Cat 兼容改写。已验证的官方 Mul、二维无拷贝视图、HF32 AddMM 和模型侧优化保持不变。

环境：910C 单逻辑 NPU；Torch 2.8.0+cpu、torch_npu 2.8.0.post2、Triton 3.2.0、CANN 8.5.0。该结果不代表其他运行环境中的表现。

### 正确性与接管验证

- FP32、FP16、BF16 各 9 个定向用例，共 27 个，与 Native 完全一致。
- 固定环境和当前官方的 `tests/test_cat.py` 各 137 项通过；官方 core benchmark 对应尺寸、精度的数值补查 9 项通过。
- 官方 `benchmark/test_cat.py` 的 2 项测试通过，保留 Cat/Cat_out × 3 精度 × 3 尺寸的 18 条性能记录。这是 Cat 专项测试，不是全库 CI。
- 真实模型做了 1 次预热和 1 个预报步，输出 finite。测量窗口内 21 次 Cat 对应 42 个官方 `_cat_copy_strided_kernel` 设备任务，累计 **6.280632 秒**。

### 模型尺寸重放

五类实际输入，按每个预报步的调用次数 `1、1、1、17、1` 加权，共 21 次 Cat。每项先校验数值，再预热 3 次，取 5 次同步计时的中位数；输入已在设备上，计入输出分配，不计随机输入构造。

| 路径 | Cat 累计耗时 / ms·预报步⁻¹ |
| --- | ---: |
| Torch Native | 15.376 |
| 原 Ascend 包装路径（不计 Triton 接管） | 30.787 |
| 新版官方 Ascend Triton | 6279.729 |

Native 和新版官方在同一进程内对照；原包装路径另起进程重放。以上是算子耗时合计，不是模型 E2E，不能据此发布候选版本的 MFU。

设备 trace 的 6.281 秒与重放的 6.280 秒一致，主要耗时发生在设备内核，而非首次编译或 Python 调度。最重的 Grid 节点拼接为两个 `[1,1038240,512]` 沿末维拼接，单次约 3112.379 ms。此次官方 core benchmark 只测 `dim=0` 的 flat-copy 分支；模型调用的是 `dim=-1` 的 strided-copy 分支，不能用前者替代后者的性能验证。更深层的编译器、指令原因尚未确定。

由于算子层已出现明显退化，没有继续进行三轮 40 步 E2E 和完整双参考 5% 诊断。单步输出有限、Cat 数值测试通过，不等于完整自回归诊断通过。覆盖率证据与日志检查限制见[覆盖率表的 Cat 试验说明](graphcast-inference-operator-coverage-20261021.md#910c-官方-cat-隔离验证)。

### 试验记录

- 本地目录：`/mdata/data/zhangyuxin/wxt/inference-910c-cat-official-20261021-JnT9TN`；说明为 `README.md`，汇总为 `result-summary.json`，原始证据为 `evidence-final/`。
- 910C 容器 `zyx-graphcast-910c` 内目录：`/workspace/FlagOS/campaigns/inference-910c-cat-official-20261021-JnT9TN`。
- 官方 Cat 文件 SHA256：`dc9fc6b8ff5cc287557c29ec92049bbc911a914ea821508a584b5f6471c5e050`。
- 日志、trace、脚本与源码清单归档 `final-evidence.tar.gz`，SHA256：`3268e42ae56aabde302be995f00b92fa72ff9062220622610d206dba5974c0fe`。大体积 NetCDF 和首帧资产保留在远端，不在此归档中。

本次只补充文档，原始证据未上传 GitHub；上述本地与容器路径需要对应机器访问权限。

## 测试版本与数据来源

本轮基于 [flagos-graphcast-adaptation 的 c97b150](https://github.com/flagos-adaptation-lab/flagos-graphcast-adaptation/tree/c97b15036effc33721848561a6323d22513cd418)，使用以下本地隔离提交，包含采集修正、融合调用恢复及新海光配置：

- C550、810E：`ef885d531400f189faeba522197f260c35461207`
- S5000：`6f18d99e198b3176a14c6647b1a3c54c0537aac9`
- 910C：基础提交 `38814839d1a3f93cdf18d84f9e41e7c069408c37`；FlagGems 新结果另含已验证二维视图改动及官方 Mul，具体副本和文件哈希见[新版 Mul 说明](#910c-官方新版-mul)。Native 行仍为原结果。
- 海光 BW1101：`eb626ffc40599417d4381b6e8a1b19071a341605`

本次仅向本仓库提交结果文档，不包含上述隔离源码或运行环境；不能将本仓库当前代码直接视为这些结果的完整复现包。算子基线及内部来源见[覆盖率表的源码说明](graphcast-inference-operator-coverage-20261021.md#算子源码来源)。

原始五平台归档标识：`inference-coverage-perf-20261020-b0NcP0`；数据整理标识：`graphcast-inference-five-platform-report-20261021`。原始 `performance.csv` SHA256 为 `59dbb35d29acbea197390be7aa25268456e3d8302b4a39e906d0ecc7d803b489`，原文件未改。本文在该快照上将海光分母记为暂定并清空其 MFU，再以 `inference-910c-mul-official-20261021-3X1LSo` 的新实测结果更新 910C FlagGems 行；其他平台耗时、诊断不变。原始日志和 trace 未随本次两张表上传。
