# NVIDIA baseline 公开分支验证记录

日期：2026-10-12。
基线提交：`50fa1808928ff8e751fe30b13e7659e30d686ef3`。
目标分支：`ZYX223/ai4s_graphcast:zyx/nvidia-native-baseline`。

## 改动范围

只增加固定训练/推理配方、统一启动入口、使用说明和专项测试，调整 README。
保持原有 `ai4s/graphcast/` 目录结构；没有改训练/推理引擎、模型、loss、
计时边界、算子实现或原有 137 文件的 SHA256 清单。

从上述公开提交直接创建分支，不导入其他仓库的提交历史。
新增内容不包含数据、权重、诊断参考、机器环境转储或凭据；没有创建 Release。

## 本次检查

| 检查 | 结果 |
| --- | --- |
| 冻结源码逐文件 SHA256 | 137/137 一致 |
| baseline 入口专项 CPU 测试 | 23 passed |
| 本仓库 GraphCast 全量 CPU 回归 | 784 passed，102 subtests passed |
| 训练、推理 README dry-run | 均通过；使用 `python -S`，无需资产或 Torch 导入 |
| 公开源提交、目录定位与固定配方检查 | 通过 |
| 新增文档相对链接与变更范围检查 | 通过 |
| `git diff --check` | 通过 |

从仓库根目录执行专项检查：

```bash
python -m pytest ai4s/graphcast/tests/test_nvidia_baseline_handoff.py -q
```

全量检查：

```bash
cd ai4s/graphcast
PYTHONPATH=src CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python -m pytest tests -q
```

此公开仓库的测试集合与其他仓库不同，以上计数只对应此分支实际运行结果。
CPU 回归验证代码和调用协议，不是 H100 性能、硬件精度或天气技能测试。
本次没有运行 GPU 任务；训练和推理表格仍是使用说明中标注的已有测量，
不能把推理历史成绩视为当前入口重新测出的成绩。

## 仍需外部准备

实际运行需匹配的 H100 环境、模型权重、stats、训练 corpus 或推理案例，
正确性检查另需匹配的诊断参考。见[使用说明](nvidia-baseline-handoff.md)。
本次只发布代码和说明，没有恢复先前暂停的数据打包/上传工作。
