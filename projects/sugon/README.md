# 曙光 GraphCast

本目录是曙光项目入口。基础代码为 `zyx/flaggems-profile-routing@50fa1808928ff8e751fe30b13e7659e30d686ef3`。
通用代码中仍有其他平台支持，但这些平台不属于本次曙光验证范围。

## 本次保留的实现

运行模型是 PyTorch `graphcast_compat`，JAX 用于已有参考输出。本项目通过 PyTorch 算子分发与显式融合 API 接入 FlagGems。

| 内容 | 本次配置 |
| --- | --- |
| FlagGems 基础算子 | `addmm`、`cat`、`native_layer_norm`、`index_add_`、`silu`、`index`、`add` |
| 推理融合 | `addmm_silu`、`add_add_silu`、大行数时的 `gather_gather_add_silu`、`post_layer_norm_residual` |
| 图计算 | factorized 边投影；两侧边分块 524288；mesh2grid full-grid；预校验静态 index_add |
| 推理规格 | 单卡，40 × 6 小时，3 次预热，FP32 存储，关闭 compile/profiler/graph capture |
| 数学精度 | CLI 名称 `tf32`；海光 K512 addmm 实际为 FP16 乘法、FP32 累加，不能标作严格 FP32 |
| 实验性 packed MLP | 保留源码和单元测试，正式入口关闭，未计入本次性能结果 |

`bw3000` 是继承代码中的海光配置名称；本机实际设备名为 **BW1200**，以 `runtime.json` 为准。
7/7 是基础算子分发覆盖；融合 API 返回次数包含可能的内部 fallback，不能等同于融合 kernel 命中次数。

训练继续使用基础分支的训练模型、`PrecisionLinear`、canonical graph 和官方训练入口。
新合入 MLP 融合只在关闭梯度时触发，保留训练工厂接口。**本次验证的是海光推理，不能据此宣称海光训练或训练反向 FlagGems 已完成。**

## 固定运行环境与源码

`runtime-lock.json` 固定已导入的海光镜像 ID、镜像归档摘要及原始来源说明。
完整 FlagGems 是镜像里的已验证快照，原始快照没有 `.git` 元数据，不能把来源说明中的短提交号当成精确版本。

- `flaggems-snapshot.sha256`：完整 FlagGems 快照逐文件校验；运行前校验镜像中的源码。
- `flaggems-overlay/`：从该快照原样保留的海光关键算子、公共导出、融合源码及相关测试，附 Apache-2.0 许可。
- `flaggems-overlay.sha256`：同时校验本目录文件和镜像对应文件，确保纳管源码与实际依赖一致。
- `assets.sha256`：数据、权重、归一化统计量和参考输出校验。

overlay **不是可独立安装的完整 FlagGems 包，也不是适用于任意上游版本的补丁**。其余公共依赖与其他后端仍在固定镜像中；本分支不重新复制整个多平台依赖项目。
后续改动海光算子时须一起更新运行镜像/快照校验并重新验证，不能只改 overlay 后仍声称镜像执行了新代码。

## 在本机复现

主机需要 Docker、`/opt/hyhal` 驱动和已导入的锁定镜像；无需 SCNET。
从仓库根目录执行：

```bash
bash projects/sugon/run_hygon.sh /absolute/path/to/new-output-directory 9
```

`9` 为海光工具/ROCR 的物理卡序号。脚本检查卡空闲、查询 PCI 地址并在容器内核对，已有输出目录会拒绝覆盖。
容器挂载当前仓库只读，将当前分支的 GraphCast 放入 Python 搜索路径，继续使用固定镜像的 FlagGems、输入和参考输出。
运行前验证依赖与资产摘要；输出记录实际模块路径和当前代码文件摘要。请避免运行中修改源码。

关键算子独立检查与 CPU 回归入口：

```bash
bash projects/sugon/run_hygon.sh /absolute/path/to/new-operator-output 9 operators
bash projects/sugon/run_cpu_checks.sh
```

CPU 检查使用容器内临时可写副本，并临时隔离会在导入时初始化 DCU 的厂商 Triton；不改变原镜像或真实推理环境。分布式 CPU 测试使用主机网络完成本机 rendezvous。

如果只剩本机归档而镜像尚未导入，可在核对 `runtime-lock.json` 中摘要后执行：

```bash
zstd -dc /home/zhangyuxin/zyx/graphcast/graphcast-hygon-selfcontained-20261013.tar.zst | docker load
```

| 材料 | 位置 |
| --- | --- |
| 本机原始部署和镜像归档 | `/home/zhangyuxin/zyx/graphcast/` |
| 本次开发目录 | `/home/zhangyuxin/zyx/graphcast-sugon/` |
| 容器输入、PyTorch 权重、JAX 权重、统计量 | `/opt/graphcast/case/assets/` |
| 容器 JAX / H100 Torch 严格 FP32 参考输出 | `/opt/graphcast/baseline/` |
| 10 月 14 日原部署运行结果 | `/home/zhangyuxin/zyx/graphcast/outputs/hygon-zyx/run-20261014T081200Z/` |
| 本次集成验证材料 | `/home/zhangyuxin/zyx/graphcast/outputs/sugon-integration-20261015/` |

新运行输出包含 `prediction.nc`、`report.json`、`runtime.json`、`flaggems_dispatch.log`、`acceptance/summary.json`、`acceptance/details.csv`，以及代码版本、工作区状态、设备和镜像记录。
`report.json` 的耗时是完成预热后的 40 步 rollout 测量，不包含容器启动、源文件校验、加载和 JIT 首次编译。

## 精度口径与后续工作

现有 `diagnose_inference.py` 比较最后一步输出与两个固定参考，记录相对 MAE；门限是平均相对 MAE 与 Z500 相对 MAE均低于 5%，并非要求每个变量/气压层都低于 5%。
原部署 W400/W500/W600/W700 曾略高于 5%，应保留逐层明细，不能写成全变量精度达标。
这是跨实现一致性检查，未代替对 ERA5 真值的逐时效 RMSE 或业务预报质量评估。

后续优先补齐：训练数据清单与小步训练验证、NV/DCU 固定配置对照、最佳/最后 LOSS 与 checkpoint 记录、按变量及气压层计算 RMSE。
本机验证结果与限制见 [验证记录](VALIDATION.md)。
