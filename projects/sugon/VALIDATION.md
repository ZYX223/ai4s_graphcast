# 2026-10-15 集成验证

工作目录 `/home/zhangyuxin/zyx/graphcast-sugon`，基础提交 `50fa1808928ff8e751fe30b13e7659e30d686ef3`。
完整日志和输出在 `/home/zhangyuxin/zyx/graphcast/outputs/sugon-integration-20261015/`。

## 回归检查

- 基础分支缺少待接入融合功能：新引入的 11 项融合测试首先失败，见 `fusion-red.log`。
- 合入后修正了旧快照测试中仍按海光小分块/inline 校验的过时预期；增加梯度开启时保持 `PrecisionLinear` 与梯度一致性的回归测试。
- 首轮基础分支：759 passed、2 failed、102 subtests passed。
- 同环境集成分支：853 passed、2 failed、102 subtests passed，失败项与基础分支完全一致。
- 其中推理 shell 测试失败来自只读挂载下创建输出目录；换成容器临时可写副本后该项通过。
- 另一项 `test_public_microbatch_aliases_cannot_silently_override_each_other` 在原分支和本分支都失败：Python 3.10 的 argparse 未拒绝默认值形式的两个互斥别名。通过将 argparse 的整数默认值改为经 type=int 转换的字符串默认值修复了该问题，训练算法及默认数值不变。相关训练入口 45 项测试通过，见 `microbatch-alias-green.log`。

最终可写副本全量检查：**855 passed、102 subtests passed**，无失败，见 `cpu-final.log`。

## 已知限制

可选的 post-LayerNorm 候选分析器只采集未融合路径。不要同时开启融合和 `--post-layernorm-residual-profile`；该组合目前会产生空观测，参数互斥提示留待后续改进。正式性能入口未开启该分析器。

## 关键算子

海光真实设备独立检查 **45 passed**（`operators-final-console.log` / `operators-final/operator-tests.xml`）。
其中项目新增 4 项 LayerNorm+residual 检查，先在梯度开启时证明 eligibility 不满足（4 failed、其余 41 passed），再在 no_grad 下禁止 fallback 并验证与 PyTorch 结果一致，4 项全部通过。原始 FlagGems 快照文件未改动。

## 40 步整模型

最终运行使用独立缓存目录，测量执行 3 次预热，报告不包含首次编译或启动耗时。单张 BW1200，40 × 6 小时，3 次预热，最终 rollout **28.590016 秒**，基础算子 **7/7**，输出有限，完整启动流程退出码 0。此为单次复测结果，未据此宣称统计意义上的加速。

| 固定参考 | 平均相对 MAE | Z500 相对 MAE | 既有门限 |
| --- | --- | --- | --- |
| Official JAX strict FP32 | 1.30586088% | 0.01535495% | 通过 |
| H100 Torch Native eager strict FP32 | 1.30619416% | 0.01535759% | 通过 |

逐变量/气压层指标见 [明细](evidence/20261015/acceptance-details.csv)。本次 W500 仍略超 5%，因此不能宣称所有层都低于 5%。

平均指标是 78 个变量/气压层相对 MAE 的等权平均；单项相对 MAE = mean(abs(candidate-reference))/mean(abs(reference)) × 100。门限要求两个参考的平均与 Z500 均严格小于 5%。

[运行报告](evidence/20261015/report.json)、[运行环境及 202 个源码/配置摘要](evidence/20261015/runtime.json)、[双参考检查](evidence/20261015/acceptance-summary.json)、[测试摘要](evidence/20261015/tests.json)、[审查记录](evidence/20261015/review.json)。
完整本机结果：`/home/zhangyuxin/zyx/graphcast/outputs/sugon-integration-20261015/run-final/`。

运行时 HEAD 仍是基础提交，工作区包含待提交改动；上述源码摘要已逐一核对与最终待发布文件相同。结果文件、说明文档不参与模型执行；发布提交记录同一份已核对代码。
