# 环境与材料

| 平台 | 固定环境目录 | 计算精度 |
|---|---|---|
| BW1100 | `environments/bw1100/operational-v1/` | RNE＋XF32，FlagOS JAX fl_only |
| H100 | `environments/h100/tf32-v1/` | TF32，无显式 RNE |

本次不升级 wheel。镜像 ID、Python 路径、依赖、挂载和文件 SHA 以环境的 `manifest.json` 为准。
BW 路径为 JAX/XLA→FlagTree，GEMM 由 rocBLAS 执行；安装了 FlagGems 不表示模型调用了该算子库。
环境采用固定镜像及运行时快照恢复，目前未提供从基础镜像重建环境的 Dockerfile。

运行条件：Linux x86_64、宿主机 Python≥3.6、可访问 Docker、匹配的宿主机驱动；
H100 需 NVIDIA 容器 GPU 支持，BW1100 需 `/dev/kfd`、`/dev/dri` 及 hy-smi 可用。
镜像并不替代内核驱动。主机容器配额为8 CPU、128 GiB内存、4 GiB共享内存。
正式运行前检查工作盘和 Docker 镜像盘空间，工作目录建议预留100 GiB。

```bash
python3 --version
docker version
docker info --format '{{.DockerRootDir}}'
free -h
df -h
```

完整材料位于 [RUN.md](RUN.md) 的平台根目录。启动入口校验材料，若镜像尚未导入，
从环境中的 `runtime-image.tar` 执行 `docker load` 并核对身份；随后创建或复用当前账号的容器。
容器内源码、环境和输入只读；编译缓存与输出写入个人 work 目录。先执行 `--check` 再计算。
同一工作目录会自动复用材料校验记录，避免重复读取整个运行时及镜像归档；文件身份、大小或时间戳变化会触发重验。
首次部署或 `--verify-full` 仍做完整 SHA256 检查，通常比后续启动耗时更长。
缓存命中仍需遍历材料的文件元数据，不代表完全跳过检查；无需每次运行都加 `--verify-full`。

BW训练入口会在导入JAX前关闭当前worker进程的透明大页（THP），缓解固定主机内存页时的规整等待。
不修改宿主机全局设置，也不作用于推理或H100。每阶段的 `host-memory-policy.json` 记录实际读回状态；
若系统不支持或拒绝设置，会警告并继续，需结合该文件判断策略是否生效。

H100 CUDA 镜像归档保存在 H100 主机；海光副本不含该大型归档。
迁往新 H100 时需从 H100 原目录取得完整环境，不能只复制海光侧的 H100 元数据。
数据与模型目录的相对链接须按原样保留，链接目标不得越出交付根目录。

SCNET 是独立环境：Slurm／Podman、gfx936约64 GiB。本次单 case 尚未在该环境完成验证，
不能套用 BW1100 的显存配置或以已有其他 case 的结果认定通过。

已部署根目录：BW1100为`/recovery/graphcast-delivery`；H100账号earth_1为`/data2/graphcast-delivery`。
SCNET账号ac9844np6r的材料副本为`/work2/ac9844np6r/zyx/delivery/graphcast-delivery`；通用Docker命令不在SCNET登录节点执行。

宿主机协作权限：BW1100通过testuser的ACL、H100通过data_rw共享组开放当前源码及根目录说明读写，运行材料可读。
SCNET通过zhaozn1的ACL开放本次整个交付目录读写，目录及必要程序可执行，并设置默认ACL继承；有效权限与普通新建文件继承已核验，尚未切换该账号实写。
默认ACL不能保证同步工具或程序显式设置的限制模式仍然可写，复制后须检查有效权限。复跑结果写入各账号自己拥有的work目录。
材料的冻结版本与SHA约束保持；上述宿主机写权限不改变容器内源码、环境与输入的只读挂载。
文件权限不代替Docker使用权或Slurm资源分配。SCNET当前只提供材料检查入口：

```bash
cd /work2/ac9844np6r/zyx/delivery/graphcast-delivery/releases/graphcast-20261114
python3 scripts/verify_delivery.py --platform bw1100 --plans
python3 results/full38-20261115/recalculate.py
```

上述命令不申请GPU；Operational的Slurm／Podman运行入口、显存可行性和本地GPU复验仍待完成。
