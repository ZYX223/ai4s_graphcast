# GraphCast 交付材料

## 每周代码更新

固定仓库：[ZYX223/ai4s_graphcast](https://github.com/ZYX223/ai4s_graphcast)，固定分支：[`zyx/graphcast-delivery`](https://github.com/ZYX223/ai4s_graphcast/tree/zyx/graphcast-delivery)。

- 每周在 `releases/graphcast-YYYYMMDD/` 新增一个正式版本；同周修订更新原目录，以 Git commit 和 `release.json` SHA256 区分。
- BW1100 的整理入口为 `/recovery/graphcast-delivery/releases/`。验证后将相应目录提交到本分支，再向 H100 和 SCNET 同步已验证内容并复查协作权限。
- 本分支保存源码、配置、文档及已归档的精度／MFU复算材料。模型、输入数据、环境镜像和完整参考不上传 GitHub，仍从下列服务器材料根目录取得；依赖身份由各版本的材料清单固定。
- 新克隆只能进行源码检查及已归档结果的离线复算；运行训推前须补齐配套材料，参见[环境说明](releases/graphcast-20261114/docs/ENVIRONMENT.md)。
- `graphcast-20261106` 为历史快照，当前运行入口是 `graphcast-20261114`。仓库根目录原有 `src/`、`scripts/`、`experiments/` 等保留自旧开发分支，仅作历史来源；本分支每周更新以 `releases/` 为准。

```bash
# In a fresh source checkout; no GPU or external materials required.
python3 releases/graphcast-20261114/scripts/validate.py release releases/graphcast-20261114
```

当前版本：[graphcast-20261114](releases/graphcast-20261114/README.md)，固定 **Operational 0.25°／13层／batch=1**。
推理与训练各用1张卡；支持单步、三轮40步推理及连续38次训练。

- [运行命令](releases/graphcast-20261114/docs/RUN.md) · [环境与权限](releases/graphcast-20261114/docs/ENVIRONMENT.md)
- [结果与验证范围](releases/graphcast-20261114/docs/RESULTS.md) · [交付范围与更新](releases/graphcast-20261114/docs/UPDATING.md)

| 平台／协作账号 | 交付根目录 | 宿主机权限 |
|---|---|---|
| BW1100／testuser | `/recovery/graphcast-delivery/` | 当前源码与根目录说明可读写，运行材料可读 |
| H100／earth_1发布，data_rw组协作 | `/data2/graphcast-delivery/` | 当前源码与根目录说明可读写，运行材料可读 |
| SCNET／ac9844np6r发布，zhaozn1协作 | `/work2/ac9844np6r/zyx/delivery/graphcast-delivery/` | 本次交付目录可读写，目录及必要程序可执行，设置默认ACL继承 |

| 目录 | 用途 |
|---|---|
| `releases/graphcast-20261114/` | 当前源码、运行说明及最新结果；版本身份见 [release.json](releases/graphcast-20261114/release.json) |
| `data/`、`environments/`、`references/` | 模型、输入、环境与冻结参考；本版所需目录见 [material-paths.txt](releases/graphcast-20261114/material-paths.txt) |
| 外层 `results/`、其他 `releases/` | 历史结果、来源与每周交付归档；当前运行包按交付清单选取文件 |

最新复算材料：[40步推理](releases/graphcast-20261114/results/full40-20261115/summary.json) · [38次训练](releases/graphcast-20261114/results/full38-20261115/summary.json)。
BW1100与H100已完成上述完整验证；实际GPU执行SHA、精度、MFU口径与监控限制见结果说明。
交付说明修订不代表重新执行GPU验证。原始case与数值实现保持，H100完整镜像从H100主机取得。

运行输出写入各账号独立且可写的work目录；宿主机协作写权限不改变容器内源码、环境及输入的只读挂载。
SCNET当前提供材料检查与离线复算；Operational的Slurm／Podman适配、显存可行性及GPU复验仍待完成，见 [SCNET_REPRODUCE.md](SCNET_REPRODUCE.md)。
