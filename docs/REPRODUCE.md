# 复现说明

2026-10-22～23新增的RNE融合、MFU、38次训练和精度诊断入口见[本轮完整复现说明](EXPERIMENTS_20261023.md)。本页继续保留原推理baseline及算子profile入口。

## 固定环境

H100 使用 `configs/h100.json` 的不可变镜像 ID：JAX/jaxlib/CUDA12 plugin/PJRT 0.4.34、Python 3.12。海光使用 `configs/hygon.json`：厂商 JAX 0.6.0+das.opt1.dtk2604、Python 3.10、DTK 26.04、gfx938。详细依赖在 `environment/`。主机启动器只需要 Python 3.6+、Docker 和设备管理工具。

本机和 H100 已存在对应镜像；镜像 ID 不是可下载地址。在新机器上，应从已验证机器导出/导入相同镜像并核对 ID。仓库没有假设这些镜像位于公开 registry，也不把已存在的基础镜像当作从零安装说明。海光厂商安装包来源与校验值保留在 `environment/hygon.json`。

容器固定 CPU 配额 16、内存 64 GiB、共享内存 8 GiB，数值计算使用单卡。选择设备前检查空闲状态，并在模型进程内核对 PCI 地址。海光启动器按周报配置关闭当前 Python 进程的 THP，不修改宿主全局设置。

## 准备独立数据目录

原始材料位置：

- 本机：`/home/zhangyuxin/zyx/graphcast/outputs/jax-hygon-20261015/case-40step/` 和同级 `baseline-existing/baseline.npz`。
- H100：`/workspace/graphcast-sugon-jax-tf32-20261016/work/case-40step/` 和同级 `baseline-existing/baseline.npz`。

这些原始材料只在准备时使用。准备后执行推理不需要任何历史实验目录、旧项目代码或敏感层实验报告。

```bash
python3 scripts/run.py prepare --platform hygon \
  --case /home/zhangyuxin/zyx/graphcast/outputs/jax-hygon-20261015/case-40step \
  --baseline /home/zhangyuxin/zyx/graphcast/outputs/jax-hygon-20261015/baseline-existing/baseline.npz \
  --output /absolute/path/new-prepared
```

H100 上换成 `--platform h100` 和对应原始材料路径。准备输出的 `data/` 即后续 `--data` 参数，包含：

- 原始官方 JAX 权重、输入、40 步驱动场、目标模板、三份归一化统计量；逐项核对 `configs/case.json` 的哈希。
- `reference.npz`：从已验证的冻结包提取官方 JAX 严格 FP32 六变量参考，数值保持不变。
- `manifest.json`：案例配置、参考来源及新参考文件校验值。

固定输入与参考只提供一个起报案例。数据目录可以复制到另一台机器；推理开始和结束都会重新核对校验值。原始数据/权重的许可由其上游决定，仓库不再分发这些大文件。

## 检查环境与执行

```bash
# CPU 正确性测试，不占用 GPU
python3 scripts/run.py check --platform hygon --output /absolute/path/check-new

# 海光：重新编译舍入库，验证位级舍入、原输入不变、数学模式切换与恢复
python3 scripts/run.py probe --platform hygon --device 9 --output /absolute/path/hygon-probe-new

# H100：确认设备可见、求梯度通过、TF32 数值路径实际生效
python3 scripts/run.py probe --platform h100 --device 7 --output /absolute/path/h100-probe-new

# 正式三轮；每轮都从相同原始输入重新开始
python3 scripts/run.py benchmark --platform h100 --device 7 \
  --data /absolute/path/new-prepared/data --output /absolute/path/h100-new
python3 scripts/run.py benchmark --platform hygon --device 9 --mode all \
  --data /absolute/path/new-prepared/data --output /absolute/path/hygon-new
```

海光也可指定 `--mode fp32`、`xf32` 或 `rne_xf32` 单独测量，默认 `rne_xf32`。`--repeats 1` 用于整理后的回归检查；三轮统计使用默认值 3。原始 XF32 预期可能不通过精度门限：程序保留实测结果，不把“推理完成”写成“精度通过”。

当前 H100 主机禁止非特权容器通过设备环境变量自动暴露 GPU，因此默认使用其已配置的设备卷挂载方式。其他正常支持 Docker GPU 设备请求的 N 卡主机可加 `--gpu-access standard`。两种方式都只申请所选设备，不需要特权容器。

每个结果目录保存 `launch.json`、`console.log`、退出码；海光另存 `native/build.json` 和重新构建的库。模型结果在 `run/report.json`、`run/step-audit.json` 及 `run/roundN-MODE/`。每轮包含最终 `prediction.nc` 和 `report.json`。

## 误差复算与平台直接比较

```bash
python3 scripts/run.py compare --platform hygon \
  --candidate /absolute/path/candidate.nc \
  --reference /absolute/path/new-prepared/data/reference.npz \
  --data /absolute/path/new-prepared/data --output /absolute/path/accuracy-new

# 平台直接比较：分母采用 --reference 指向的 H100 预测
python3 scripts/run.py compare --platform hygon \
  --candidate /absolute/path/hygon-prediction.nc \
  --reference /absolute/path/h100-prediction.nc --output /absolute/path/platform-comparison-new
```

比较在 CPU 上完成，输出 `comparison.json`，包含 78 项逐层指标。路径可以按实际位置传入，未要求历史日期目录。

## 独立 H100 模型跟踪

```bash
python3 scripts/run.py profile --platform h100 --device 7 \
  --data /absolute/path/new-prepared/data --output /absolute/path/profile-new
```

预热后只跟踪一个模型步骤，输出 `model-trace.nsys-rep`；它不计入正式性能统计。解析 trace 应使用运行镜像内相同版本的 Nsight Systems；旧版宿主工具可能无法读取。

## BW1100 的 RNE＋XF32 模型跟踪与独立算子

本轮海光 profile 和独立算子入口固定 `rne_xf32`；完整推理的 `benchmark` 仍保留原有各模式。先选择空闲卡，以下使用本次验证的 GPU11，启动器会重新检查占用并核对 PCI 和配置设备名。原始驱动名 `BW1200` 与项目型号 BW1100 分开记录。

```bash
# 整模型：预热 5 步后重置初始输入，只捕获一个 +6h 推理步
python3 scripts/run.py profile --platform hygon --device 11 --mode rne_xf32 \
  --data /home/zhangyuxin/zyx/graphcast/outputs/jax-reorganization-20261016/prepared/data \
  --output /absolute/path/model-profile-new

# 独立算子：15 种 GEMM 尺寸 + 14 项图操作，每项三轮
python3 scripts/run.py operators --platform hygon --device 11 --mode rne_xf32 \
  --data /home/zhangyuxin/zyx/graphcast/outputs/jax-reorganization-20261016/prepared/data \
  --output /absolute/path/operators-new
```

若旧登录会话尚未继承已恢复的 docker 组权限，可把整条启动命令放在 `sg docker -c '…'` 中；新登录会话通常无需此包装。不要修改 Docker 套接字或共享机器全局设置。

运行期间保持源码、配置、数据不变；建议使用冻结的独立源码副本。执行结束会再次校验源码和数据，新增分析脚本也会触发代码快照变化检查。所有输出目录必须是新目录。

模型 profile 输出：`model-trace.json` / `.db`、`model-gemm.jsonl`、`run/optimized-hlo.txt`、`run/report.json`、`run/step-audit.json` 和 `summary/{summary.json,kernels.csv,layers.csv}`。脚本验证模型成功完成、无区间内编译、84 次 GEMM 与 168 次 RNE，以及逐层权重指纹和设备事件关联，不能只看 hipprof 的退出码。已经保存的 profile 也可在宿主离线汇总：

```bash
python3 scripts/summarize_profile.py \
  --run /absolute/path/model-profile-new \
  --output /absolute/path/separate-summary-new
```

独立算子输出在 `operators/`：每种矩阵尺寸的 JSON、构建日志/二进制校验值、`gemm-timings.csv`、总 `report.json`，以及 `graph/` 中的真实图索引、索引校验值、输出抽样及参考、优化 HLO 和逐次时间。原生 GEMM 使用 HIP event，图操作使用同步 JAX 调用墙钟，两类时间口径不同。生成输入和正确性检查不计时；图操作独立执行，不重建模型中的融合 kernel。

不依赖 H100 连接即可复现本节。结果与历史周报的 40 步性能、误差分别查看，详见 [本轮测量结果与限制](INFERENCE_PROFILE_20261022.md)。

## 当前整理结果的位置

- 本地代码：`/home/zhangyuxin/zyx/graphcast-sugon/`。
- 本机新数据与回归结果：`/home/zhangyuxin/zyx/graphcast/outputs/jax-reorganization-20261016/`。
- H100 新代码与回归结果：`/workspace/graphcast-sugon-jax-20261016/`。
- 原始周报海光/H100 完整材料仍保留在 `outputs/jax-hygon-20261015/` 与 `outputs/jax-h100-tf32-20261016/`。

代码树中的历史实现已移除；需要查阅旧实验入口时，可使用 Git 提交 `39b6ee5`。本次没有改写 Git 历史或改动原始实验输出。
