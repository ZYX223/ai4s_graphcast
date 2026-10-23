# BW1100 JAX RNE＋XF32 推理逻辑算子表

按用户提供的 `GraphCast算子列表_v540训练_最新耗时_20261018.xlsx` 的逻辑/API粒度整理。主表采用独立算子耗时乘模型调用次数的口径；原有 GPU kernel 明细另作附件。

## 测量范围和输入

- 官方 GraphCast 提交 `08cf73625c9d12bd9aaa038868bcb2fe488f2a22`，16 个官方模型文件字节未改；operational 0.25°，721×1440，13 层，batch=1，单步＋6h。
- Hygon_ZYX 项目型号 BW1100；原始驱动/JAX 名 BW1200。GPU12，PCI `0000:b0:00.0`，gfx938，DTK26.04，JAX `0.6.0+das.opt1.dtk2604`。原 kernel 附件在 GPU11 采集，分别保留设备记录。
- FP32 存储；84 次运行时 Linear 的矩阵乘均先 RNE 再 XF32，偏置及非矩阵算子保持 FP32。
- 从冻结的单步未融合 JAXPR 提取形状、原语参数及调用路径；保留权重、偏置、归一化参数和图索引，激活用固定种子生成。它们是独立算子测试输入，不是捕获的真实中间激活。
- 相同形状、原语参数、标量/整数常量签名共享一个代表用例；不同浮点权重值可以共享同形状用例。

## 逻辑计数和耗时边界

共16类、108个用例、549次前向图浮点逻辑调用。原始展开图2280条方程，反向活跃性分析剔除260条，保留2020条。单步 scan 的输入/输出维度变换以 reshape 视图连接依赖；包含在布局类中。

Linear、LayerNorm、SiLU 各自拥有其内部原语。Linear 包含 RNE、GEMM 和偏置加法；LayerNorm 包含均值、方差、rsqrt、scale、offset；SiLU 包含 sigmoid 与乘法。外层 Add/Mul 等只计复合算子之外的调用，避免重复计时。

另外列出248项：246项设备放置或整数索引/控制辅助操作，2项在编译阶段预计算的 Linear 矩阵乘。后者同时以优化 HLO 中缺少对应 dot 和84个运行时权重指纹验证。43次SiLU、42次LayerNorm按活跃逻辑图计数；后续仍可融合、公共子表达式消除或预计算，不能与kernel次数一一对应。

每用例做3轮，每轮先预热5次，随后测20次同步JAX调用。取每轮中位数，再取三轮中位数。主表每类耗时为：

```text
Σ（该类用例的三轮中位数 × 该用例在单步逻辑图中的调用次数）
```

计时排除编译、输入准备/传输和正确性检查，包含JAX调用与同步。原图标量字面量保持编译期常量。检查计时区间JIT缓存不变；每个Linear计时与预热共75次，必须对应75次原生GEMM和75次RNE路由调用，且没有未知权重。

**这些独立加权耗时不能相加作为实际模型延迟，也不是GPU耗时占比。** 独立张量拼接/广播可能被物化，而整模型会融合；Reshape/Squeeze可能只有视图或调度成本。RNE子项已经包含在Linear中，不能再次相加。

正确性按每用例最多258个输出元素与独立NumPy参考比较，保存实际值、参考值和下标。Linear先对输入做独立RNE，再以FP64点积加偏置；LayerNorm和Scatter以FP64计算归一化/聚合；Gather/布局按索引关系核对。大张量按各维坐标取样，避免关闭JAX x64时超过2³¹元素的扁平下标溢出。这不是全输出数学参考校验，也不是新一轮40步精度验收。

## 正式补测结果

最终运行 `bench-full-r2`：108/108用例通过，源码、数据、目录和HLO结束时校验一致。以下单位为ms，采用独立同步调用三轮中位数乘逻辑频次；不要将本列之和当成整模型延迟。

| 逻辑算子 | 逻辑调用次数 | 用例数 | 独立加权耗时 |
|---|---:|---:|---:|
| Linear（含RNE＋XF32和bias） | 84 | 15 | 733.455 |
| ScatterAdd / SegmentSum | 18 | 3 | 106.075 |
| LayerNorm | 42 | 5 | 104.085 |
| Concat | 42 | 11 | 94.672 |
| SiLU | 43 | 5 | 67.877 |
| Gather | 36 | 6 | 43.277 |
| Add | 45 | 6 | 32.504 |
| Broadcast | 46 | 8 | 15.086 |
| Transpose | 50 | 10 | 12.176 |
| Reshape | 30 | 7 | 10.068 |
| Div | 22 | 4 | 7.311 |
| Sub | 22 | 4 | 5.124 |
| Slice | 22 | 13 | 3.897 |
| Squeeze | 21 | 2 | 3.569 |
| Mul | 16 | 7 | 3.123 |
| DynamicSlice | 10 | 2 | 1.656 |

完整Excel：`/home/zhangyuxin/zyx/graphcast/outputs/logical-operators-20261022/workbook/GraphCast_BW1100_JAX_RNE-XF32_推理逻辑算子耗时_20261022.xlsx`。

工作簿已按要求精简为4页：逻辑算子汇总、分形状耗时、附件_Kernel明细、说明与来源。16类、108用例、549次逻辑调用及全部保留测量数值均核对不变；保留56种kernel汇总。原16页版本归档在工作簿同目录的archive/，包含550次逐次执行、GEMM逐层及其他详细证据。重新计算三轮中位数和加权项、复核抽样输出容差均通过；未做桌面Excel渲染检查。

耗时列已区分JAX与FlagGems：JAX保留本次实测数据，FlagGems对应列留空待补测；Kernel附件的耗时明确标记为JAX。

仓库小型结果：`evidence/20261022/logical/`；完整原始样本和HLO留在外部输出目录。最终32项CPU测试通过；只读代码复核无遗留阻塞项。GPU12测试进程已退出，未调整共享设备的全局设置。

## 原始材料与复现

本机输出根目录：

```text
/home/zhangyuxin/zyx/graphcast/outputs/logical-operators-20261022/
```

| 相对路径 | 内容 |
|---|---|
| `capture-r3/run/logical/` | 最终 JAXPR、逻辑清单、冻结常量 |
| `capture-r3/run/optimized-hlo.txt` | 同一模型编译结果 |
| `bench-full-r2/logical/` | 正式补测，逐用例3×20原始计时、抽样输出、独立HLO |
| `source-final-r1/` | 采集和正式补测使用的冻结源码 |
| `check-review-green/` | 32项容器CPU检查及回归测试 |
| `workbook/` | 逻辑主表、kernel附件、工作簿来源与核验记录 |

代码目录：`/home/zhangyuxin/zyx/graphcast-sugon`。入口为 `scripts/run.py`、`scripts/logical_bench.py`；实现为 `src/graphcast_bench/logical_inventory.py`、`logical_bench.py`。沿用固定镜像，所有数值工作在容器内执行。

在上述仓库目录运行（输出目录必须不存在；设备需先确认空闲，启动器再次检查）：

```bash
DATA=/home/zhangyuxin/zyx/graphcast/outputs/jax-reorganization-20261016/prepared/data
OUT=/home/zhangyuxin/zyx/graphcast/outputs/logical-operators-reproduction

# 采集一份与当前代码/数据一致的单步逻辑图。
python3 scripts/run.py logical-catalog --platform hygon --device 12 \
  --mode rne_xf32 --data "$DATA" --output "$OUT/capture"

# 原图真实尺寸的全部逻辑用例；默认3轮，每轮5次预热和20次计时。
python3 scripts/run.py logical-bench --platform hygon --device 12 \
  --mode rne_xf32 --data "$DATA" --inventory "$OUT/capture/run/logical" \
  --output "$OUT/bench"
```

可增加 `--smoke` 只测每类一个用例；其报告明确标记不完整，不能替代正式表。当前会话用 `sg docker` 获取用户已经具备的Docker组权限，新登录会话通常无需该包装。

用于形成正式表的报告必须同时满足 `status=passed`、`complete_inventory=true`、`code_and_data_unchanged=true`。结束时复核代码、数据、逻辑目录和优化HLO的SHA-256。早期 `bench-full-r1` 因抽样下标问题失败，已排除；`capture-r1/r2` 和 `bench-smoke-r1` 仅保留诊断过程，不作为最终数据来源。

原kernel工作簿和550次逐次记录保留在完整备份；精简版保留56种kernel汇总附件，来源为 `outputs/inference-operators-20261022/`。本次不改动历史40步推理结果，不产生H100、训练、反向或新RNE算法的测量结论。
