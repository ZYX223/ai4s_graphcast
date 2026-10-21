# GraphCast 五国产平台推理算子接入与覆盖率表

数据采集：2026-10-20；整理：2026-10-21。对应[推理性能表](graphcast-inference-performance-20261021.md)的已测配置；未把后续 910C cat 实验或关闭 mul 的排查配置计入正式结果。

## 覆盖率汇总

以下为 eager 模式的设备 trace 结果，单列模式差异，不作为 Graph 模式验收结果。

| 平台 | Native 算子类型数 | 完整接管数 | 完整覆盖率 | 核心命中 | 部分接管（不计完整） | 未计入接管项 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| C550 | 14 | 12 | 85.71% | 7/7 | `copy_` | `fill_` |
| S5000 | 14 | 13 | 92.86% | 7/7 | 无 | `_copy_from` |
| 910C | 12 | 10 | 83.33% | 6/7 | 无 | `cat`、`copy_` |
| 海光 BW1101 | 14 | 12 | 85.71% | 7/7 | 无 | `copy_`、`fill_` |
| 810E | 14 | 13 | 92.86% | 7/7 | `copy_` | 无 |

五平台按此口径均超过 80%。覆盖率通过不等于性能目标通过；海光 MFU 暂不计算，910C 的 MFU 尚未达到历史 H100 MFU 的 80%。

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
- 910C 当前 `cat` 前端内部执行切片赋值和 Native 拷贝，未计为 FlagGems Triton 接管；`copy_` 有已知递归问题，保留 Native。后续真正 Triton cat 仍是隔离实验，未纳入此表。
- 910C 的有限值检查在 Native trace 中表现为 `isfinite`，其他平台可能展开为 `abs/eq/ne`，因此分母分别为 12 或 14，不强行统一。
- 910C 的 `mul` 虽然造成性能退化，仍属于已接管算子。不能为提高性能或覆盖率而删去其真实调用。

## 算子源码来源

这些结果来自官方基线与内部优化的组合，不是未经修改的官方 FlagGems。源码位置以固定提交和实际加载文件为准，不以包版本字符串或分支名作为一致性证明。

- 官方源码基线：[FlagGems v5.3.5 / a7620cc](https://github.com/flagos-ai/FlagGems/tree/a7620cc191a0b42e040194622c5758b22a7a25dc)。历史口头简称“3.5”，这里记录实际源码版本。
- 内部纯 Triton AddMM 归档：[flagos-common / 486c3d6 / wip-addmm-triton-only](https://github.com/flagos-adaptation-lab/flagos-common/tree/486c3d6c09dfc9645b9f2093645db6dfd585b573/patches/flaggems/wip-addmm-triton-only)。不是旧 Native redispatch 方案。
- 内部 Ascend Add 归档：[flagos-common / 486c3d6 / wip-add-ascend-triton-only](https://github.com/flagos-adaptation-lab/flagos-common/tree/486c3d6c09dfc9645b9f2093645db6dfd585b573/patches/flaggems/wip-add-ascend-triton-only)。
- 同事多平台候选：[YUANXIGUREN/FlagGems / 0518e51](https://github.com/YUANXIGUREN/FlagGems/tree/0518e51f320e1fde35e88246eff91a06e9d6698a)。S5000、910C、海光实际 AddMM 文件与该提交相应文件哈希一致；不代表所有平台整棵源码树相同。
- C550、810E 的 AddMM 和部分平台 index_add 使用 `graphcast-tf32-rne-20260906/candidate-v2` 冻结文件，不将同名远端文件自动认定为相同版本。
- C550、S5000、海光、810E 的实际 `cat` 匹配官方公共 `src/flag_gems/ops/cat.py`；910C 未接管的 vendor cat 单列。普通 LayerNorm 匹配官方公共 `src/flag_gems/ops/layernorm.py`，额外残差融合属于内部优化。

私有仓库链接需要相应访问权限。本次只发布两张结果表，不上传内部算子源码、不修改模型，也不声称上述远端分支最新 HEAD 已通过本轮验证。模型隔离提交见[性能表的版本说明](graphcast-inference-performance-20261021.md#测试版本与数据来源)。

归档标识：`inference-coverage-perf-20261020-b0NcP0`；数据整理标识：`graphcast-inference-five-platform-report-20261021`。逐类型原表 `coverage.csv` 共 68 行，SHA256 为 `17d425fc7d406b58cc448e7034ffd03fdcbaca8bfb091fe624c67d2495cccbee`；源码表 `operator_sources.csv` SHA256 为 `fad4b1c178eb89aa2a6abee15fc8ac6a728f0bb86e8e3c0c9aa562e81f699322`。原始日志、trace 和源码快照未随本次表格上传。
