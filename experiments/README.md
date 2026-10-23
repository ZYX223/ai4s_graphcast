# 已验证实验源码

完整入口、外部资产、结果解释与运行方法见 [实验复现说明](../docs/EXPERIMENTS_20261023.md)。

这里保存实际测量时的源码字节。`shared/baseline`供融合和诊断共用，`shared/training-official`包含固定版本官方训练数据处理模块及许可证。`PROVENANCE.json`提供原文件位置和SHA256；通过`python3 scripts/prepare_experiments.py --verify-only`核验。

使用仓库中的准备器建立独立工作目录后再运行。不要直接在归档目录生成结果，也不要将固定权重的融合推理路由用于训练。历史审计脚本中的绝对路径属于原始记录，新运行的推荐入口以复现说明为准。
