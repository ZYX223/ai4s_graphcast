# GraphCast 五国产平台推理算子接入与覆盖率表

全模型覆盖率数据采集：2026-10-20；2026-10-21 补充 910C 官方新版 Mul 的专项设备验证，[推理性能表](graphcast-inference-performance-20261021.md)已采用该版实测结果。类型计数沿用原全模型统计，不声称已重新审计全部算子。910C 官方 Cat 隔离试验单列，未把它或关闭 mul 的排查配置计入正式结果。

Excel 合并版：[算子耗时与覆盖率工作簿（2026-10-23）](graphcast-operators-and-coverage-20261023.xlsx)。包含“推理总览”“推理覆盖率”“训练总览”“证据与历史”四个 sheet，独立与融合执行分别列出、不重复计时。本次仅整理已有数据：推理为 40 步设备累计耗时，训练沿用 v5.4.0 附件记录，不代表新增测试，也不能作为模型 E2E。

## 覆盖率汇总

以下为 eager 模式的设备 trace 结果，单列模式差异，不作为 Graph 模式验收结果。

| 平台 | Native 算子类型数 | 完整接管数 | 完整覆盖率 | 核心命中 | 部分接管（不计完整） | 未计入接管项 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| C550 | 14 | 12 | 85.71% | 7/7 | `copy_` | `fill_` |
| S5000 | 14 | 13 | 92.86% | 7/7 | 无 | `_copy_from` |
| 910C | 12 | 10 | 83.33% | 6/7 | 无 | `cat`、`copy_` |
| 海光 BW1101 | 14 | 12 | 85.71% | 7/7 | 无 | `copy_`、`fill_` |
| 810E | 14 | 13 | 92.86% | 7/7 | `copy_` | 无 |

五平台按此口径均超过 80%。覆盖率通过不等于性能目标通过；海光 MFU 暂不计算。910C 使用官方新版 Mul 后，MFU 为 16.9876%，已超过历史 H100 MFU 的 80% 门槛（15.7985%）。

## 逐算子接入情况

“完整”指本轮捕获范围内已确认完整接管；“部分”不计入完整接管数；“未计”表示走 Native 或接管证据不足，不代表官方没有该接口。“—”表示不在该平台本轮 Native 分母中，不表示不支持。

| 算子 | C550 | S5000 | 910C | 海光 BW1101 | 810E |
| --- | --- | --- | --- | --- | --- |
| `addmm` | 完整 | 完整 | 完整 | 完整 | 完整 |
| `cat` | 完整 | 完整 | 未计 | 完整 | 完整 |
| `native_layer_norm` | 完整 | 完整 | 完整 | 完整 | 完整 |
| `index_add_` | 完整 | 完整 | 完整 | 完整 | 完整 |
| `silu` | 完整 | 完整 | 完整 | 完整 | 完整 |
| `index` | 完整 | 完整 | 完整 | 完整 | 完整 |
| `add` | 完整 | 完整 | 完整 | 完整 | 完整 |
| `copy_` | 部分 | — | 未计 | 未计 | 部分 |
| `_copy_from` | — | 未计 | — | — | — |
| `fill_` | 未计 | — | — | 未计 | 完整 |
| `zero_` | — | 完整 | 完整 | — | — |
| `mul` | 完整 | 完整 | 完整 | 完整 | 完整 |
| `abs` | 完整 | 完整 | — | 完整 | 完整 |
| `eq` | 完整 | 完整 | — | 完整 | 完整 |
| `ne` | 完整 | 完整 | — | 完整 | 完整 |
| `all` | 完整 | 完整 | 完整 | 完整 | 完整 |
| `isfinite` | — | — | 完整 | — | — |

核心 7 类是前 7 行。融合路径按已核验的逻辑映射归属，不能把一个融合内核的耗时重复计给多个算子。

## 统计口径与未接管项

$$\text{完整类型覆盖率}=\frac{\text{完整接管的 Native 算子类型数}}{\text{Native 设备 trace 中的计算算子类型数}}\times100\%.$$

- 分母来自同平台、同模型配置的 Native 设备 trace，按规范名归并；排除视图、纯元数据、运行时事件和纯内存传输。`copy_` 若产生设备计算内核，仍进入分母。
- 按类型计数，不按调用次数或耗时加权。接管配置或日志出现算子名称，不足以证明接管；必须有设备事件及来源关联。结论仅覆盖本轮实际调用范围，不代表任意 shape、dtype 都已验证。
- 部分接管不计入分子。FlagGems 侧新增的边界检查、独立融合内核等不增补到 Native 分母。
- C550、810E 的 `copy_` 仍混有未归属或 Native 计算。S5000 Native 呈现为 `_copy_from`，候选 `copy_` 不能直接替换该分母类型。海光 `copy_`、`fill_` 未取得足够接管证据。
- 910C 主表配置的 `cat` 前端内部执行切片赋值和 Native 拷贝，未计为 FlagGems Triton 接管；`copy_` 有已知递归问题，保留 Native。新版官方 Cat 已在隔离试验中确认 Triton 接管，但因性能退化未采用，详见下文；本次没有改变 `copy_` 的结论。
- 910C 的有限值检查在 Native trace 中表现为 `isfinite`，其他平台可能展开为 `abs/eq/ne`，因此分母分别为 12 或 14，不强行统一。
- 910C `mul` 已采用官方新版分块实现并保留模型侧二维无拷贝视图，实际接管已专项验证；不是关闭 Mul 或回退 Native。`mul` 原本就在完整接管数中，因此换版不增加分子，仍保留 10/12。

## 910C 官方新版 Mul 接管

- 源码固定为 [FlagGems `367d193` 的 Ascend Mul](https://github.com/flagos-ai/FlagGems/blob/367d19367880eaf8843449abe11db4cfb7d47e03/src/flag_gems/runtime/backend/_ascend/ops/mul.py)，完整官方文件未修改；仅接入非原地 `mul`，不改变 `mul_`。
- 单独采集的完整 40 步 trace 确认 400 次 `mul_broadcast_tiled_kernel`，与 400 次反馈乘法一致；JIT 来源文件 SHA256 为 `3d4269688e13ba1f3dda7160a95223010c39cbbcd9fd47ce3dfe0bcb4721519b`。
- 未发现 Native `aclnnMul`、旧 `mul_broadcast_2d_kernel` 或通用高维 Mul 内核。这里仍依赖模型的二维视图，不能声称未改布局的三维输入也有同样性能。
- 日志中的 `11/11` 是配置接口的命中数，不是 Native 分母下的完整覆盖率。新版仅专项复核 Mul，其余类型统计沿用旧全模型 trace；Cat、`copy_` 继续不计入接管。

三轮性能、双参考诊断和原始证据位置见[性能表的新版 Mul 说明](graphcast-inference-performance-20261021.md#910c-官方新版-mul)。

## 910C 官方 Cat 隔离验证

2026-10-21 测试 [PR #6548](https://github.com/flagos-ai/FlagGems/pull/6548) 合入后的官方 Ascend Cat，固定提交 `72b2a1a38b476df9bec0b20e8a498c1e4756d199`，没有修改官方算子文件。

| 检查项 | 结果 |
| --- | --- |
| 实际模型输入 | 五类连续 FP32 输入，末维拼接 |
| 单个预报步的 Cat 调用 | 21 次，不含预热 |
| 官方 Triton 设备任务 | 42 个 `_cat_copy_strided_kernel`，有源码哈希关联 |
| Cat 范围内的 Native 拷贝 | 未观察到 `contiguous`、`clone` 或 `copy_`；与连续输入源码分支一致 |
| 是否采用 | 否；Cat 设备任务累计约 6.281 秒，明显慢于原路径 |
| 是否更新正式覆盖率 | 否；仍为 10/12＝83.33%，核心 6/7 |

这里的“不采用”是性能决定，不是未调用 FlagGems。单步 trace 证明了候选 Cat 的实际接管，但没有完成该候选版本的 40 步双参考诊断，也不能与旧配置混合，直接将正式结果改成 11/12。

新版官方 Cat 没有原 debug 日志，现有日志式接入检查因此报告 `cat` 未观察到，并将单步报告标为 `incomplete_diagnostic_only`。这与设备 trace 中已确认的 42 个 Triton 任务不是同一层面的结果：不能仅凭缺少日志判定未接管，也不能将该诊断报告当作正式性能结果。本次没有修改官方源码来补日志。

结论只限实际捕获的连续输入。官方实现对非连续输入会先调用 `.contiguous()`；单输入或非连续输出的 `cat_out` 还可能使用 Native `copy_`，不能据此宣称所有 Cat 调用均为纯 Triton。详细测试、计时和证据位置见[性能表的 Cat 试验记录](graphcast-inference-performance-20261021.md#910c-官方-cat-隔离试验)。

## 算子源码来源

这些结果来自官方基线与内部优化的组合，不是未经修改的官方 FlagGems。源码位置以固定提交和实际加载文件为准，不以包版本字符串或分支名作为一致性证明。

- 官方源码基线：[FlagGems v5.3.5 / a7620cc](https://github.com/flagos-ai/FlagGems/tree/a7620cc191a0b42e040194622c5758b22a7a25dc)。历史口头简称“3.5”，这里记录实际源码版本。
- 910C 新版 Mul：在上述派生环境中单独采用官方 `367d19367880eaf8843449abe11db4cfb7d47e03` 的 Ascend `mul.py`、后端 `mul` 导出和 `mul_flat`、`mul_broadcast_tiled` 调优配置，不代表整库升级；来源与接管证据见[新版 Mul 说明](#910c-官方新版-mul-接管)。
- 内部纯 Triton AddMM 归档：[flagos-common / 486c3d6 / wip-addmm-triton-only](https://github.com/flagos-adaptation-lab/flagos-common/tree/486c3d6c09dfc9645b9f2093645db6dfd585b573/patches/flaggems/wip-addmm-triton-only)。不是旧 Native redispatch 方案。
- 内部 Ascend Add 归档：[flagos-common / 486c3d6 / wip-add-ascend-triton-only](https://github.com/flagos-adaptation-lab/flagos-common/tree/486c3d6c09dfc9645b9f2093645db6dfd585b573/patches/flaggems/wip-add-ascend-triton-only)。
- 同事多平台候选：[YUANXIGUREN/FlagGems / 0518e51](https://github.com/YUANXIGUREN/FlagGems/tree/0518e51f320e1fde35e88246eff91a06e9d6698a)。S5000、910C、海光实际 AddMM 文件与该提交相应文件哈希一致；不代表所有平台整棵源码树相同。
- C550、810E 的 AddMM 和部分平台 index_add 使用 `graphcast-tf32-rne-20260906/candidate-v2` 冻结文件，不将同名远端文件自动认定为相同版本。
- C550、S5000、海光、810E 的实际 `cat` 匹配官方公共 `src/flag_gems/ops/cat.py`；910C 未接管的 vendor cat 单列。普通 LayerNorm 匹配官方公共 `src/flag_gems/ops/layernorm.py`，额外残差融合属于内部优化。

私有仓库链接需要相应访问权限。本次只发布两张结果表，不上传内部算子源码、不修改模型，也不声称上述远端分支最新 HEAD 已通过本轮验证。模型隔离提交见[性能表的版本说明](graphcast-inference-performance-20261021.md#测试版本与数据来源)。

原始全模型归档标识：`inference-coverage-perf-20261020-b0NcP0`；数据整理标识：`graphcast-inference-five-platform-report-20261021`。逐类型原表 `coverage.csv` 共 68 行，SHA256 为 `17d425fc7d406b58cc448e7034ffd03fdcbaca8bfb091fe624c67d2495cccbee`；原源码表 `operator_sources.csv` SHA256 为 `fad4b1c178eb89aa2a6abee15fc8ac6a728f0bb86e8e3c0c9aa562e81f699322`。两份原文件保留不改，新版 Mul 的源码和证据另见 `inference-910c-mul-official-20261021-3X1LSo` 归档。原始日志、trace 和源码快照未随本次表格上传。
