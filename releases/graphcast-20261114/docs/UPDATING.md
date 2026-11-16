# 更新与材料交付

源码：`releases/graphcast-20261114/`。每周只保留一个正式交付版本，同周更新在该版本中整合。
版本目录采用`graphcast-YYYYMMDD`，日期为该报告周首次交付日期；同周修订保持目录名，以`release.json` SHA256区分实际内容。
本版固定一个case；配置、环境与冻结参考不变，首次材料齐全后无需每次重复传输。

`material-paths.txt` 列出本版依赖的六个材料目录：输入、模型、优化H100参考、
原H100参考，以及两平台环境。`configs/artifacts.lock.json`固定各manifest的SHA；
manifest再固定原始文件和链接。转交材料时按此清单复制，保留相对目录和相对链接。
只需在一种平台运行时，可仅携带该平台环境；其余四个数据／模型／参考目录均须保留。

对外运行包只包含以下项目，路径均相对交付根目录：

- `README.md`、`SCNET_REPRODUCE.md`：外层入口说明。
- `releases/graphcast-20261114/`：本版完整源码、六份公开文档与full40／full38复算材料。
- `material-paths.txt`列出的材料目录：保留四个数据／模型／参考目录及所用平台环境。
- `results/weekly-consolidation-20261114/SCNET_HISTORY.md`：SCNET入口引用的历史记录，仅作来源说明。

其余外层历史结果、旧release及其他case保留在服务器归档中，不随当前运行包重复分发。
`results/release-rename-20261114/before/`以及`results/delivery-startup-fix-20261114/`中的
`before-maintenance/`、`before-release/`属于修改前备份，保留原处，不加入对外运行包。
H100完整镜像须从H100主机取得；BW1100的镜像与运行时须同时保留。
将代码和材料迁入新根目录后，入口仍须位于 `releases/graphcast-20261114/scripts/run.sh`；
输出work目录独立于材料根目录。
同步源码后同时更新材料根目录的`README.md`和`SCNET_REPRODUCE.md`，避免入口继续指向已合并的说明。
保留接收账号／共享组权限；复制文件的模式可能收窄ACL的mask，应在同步后检查有效读写权限及上级目录的通行权限。
BW1100／H100协作账号可读写当前源码及根目录说明、读取运行材料；SCNET的zhaozn1可读写本次整个交付目录，并继承目录默认ACL。
运行账号须能读取完整依赖、在自己的work目录创建文件。冻结材料指内容及SHA绑定固定，并不要求宿主机文件只读；修改内容时须另建材料版本。
宿主机协作权限与容器内只读挂载分别检查。新机器上的账号及共享组可能不同，不能仅靠复制文件模式或照搬数字UID认定可读写。

同周更新先在临时目录核验；继承必要的脚本、说明和来源记录后更新正式目录，
清理同周已被替代的源码副本。发布时不得有进程正在使用被替换或清理的源码。
跨周保留每周一个版本。内部多case、原始实验和结果继续保留。
每次更新重新计算`release.json`，运行记录同时绑定版本名与manifest SHA256；
旧SHA和旧验证结果保留原标签，不将新包自动标为GPU已复跑。
改变wheel、输入、权重或参考时新增对应材料版本并重新核验。
`release.json`记录本次payload、scope及验证状态；GPU执行结果另记实际服务器、时间、源码版本和范围。

轻量交付检查（Python 3.6+，不使用GPU，不依赖旧release）：

```bash
python3 scripts/verify_delivery.py --platform bw1100 --plans
# Use h100 on H100; use bw1100 for the SCNET material-copy check.
```

检查源码SHA、材料manifest SHA、输入／权重／参考可读性和运行计划。
正式入口首次运行或传入 `--verify-full` 时检查环境与数据的完整payload SHA；后续复用有效缓存，文件变化触发重验。

兼容入口`python3 scripts/validate.py release .`支持v1／v2源码manifest，
只检查源码文件清单与SHA；材料manifest、可读性及运行计划使用上述
`verify_delivery.py`检查。两者均不代表GPU复跑。
SCNET范围见`docs/ENVIRONMENT.md`；历史Small结果不代表本版Operational已复现。
内部开发测试、设计计划、代码导读和历史实验不放入运行包；原文件、manifest与验证记录另行归档。
所有运行、编译、校验脚本，以及冻结源码、许可证、ORIGINS和材料身份清单必须保留。
本周历史整合证据保存在交付根目录`results/weekly-consolidation-20261114/`，当前入口不依赖旧release。
