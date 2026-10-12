# GraphCast · NVIDIA H100 Native baseline

本分支：`zyx/nvidia-native-baseline`，基于本仓库公开提交 `50fa180`。
提供单卡 H100 的 Native PyTorch eager / TF32 训练和推理固定配置。
不启用 FlagGems、BF16、torch.compile 或 CUDA Graph。

- [运行入口与结果口径](ai4s/graphcast/README.md)
- [环境、数据和诊断说明](ai4s/graphcast/docs/nvidia-baseline-handoff.md)
- [固定配置](ai4s/graphcast/configs/baseline/h100_native_tf32.json)
- [本次代码验证](ai4s/graphcast/docs/nvidia-baseline-validation.md)

先进入 `ai4s/graphcast/`，再执行上述说明中的命令。
数据、权重和诊断参考需另行准备；本分支没有上传这些资产。
本次仅整理入口和配置，没有新跑 GPU 性能测试；历史结果与当前验证范围在说明中分开列出。

多平台开发保留在
[zyx/flaggems-profile-routing](https://github.com/ZYX223/ai4s_graphcast/tree/zyx/flaggems-profile-routing)，
该分支未修改。代码许可证见 [LICENSE](LICENSE)。
