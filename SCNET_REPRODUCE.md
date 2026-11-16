# GraphCast SCNET材料与复现

当前交付为 `releases/graphcast-20261114/`，固定Operational 0.25°／13层／batch=1。
根目录：`/work2/ac9844np6r/zyx/delivery/graphcast-delivery/`。

协作账号 `zhaozn1` 对本次整个交付目录可读写，目录及必要程序可执行；目录已设置默认ACL，普通新建文件继承协作权限。
有效权限及默认继承检查已通过；尚未以该账号登录实写。后续同步须保留ACL并复查有效权限，`755`本身不提供协作者写权限。
本次权限修复仅覆盖上述根目录；新运行的缓存、checkpoint及结果写入各账号自己的work目录。

- [当前SCNET环境、权限与材料检查命令](releases/graphcast-20261114/docs/ENVIRONMENT.md)
- [源码与材料检查](releases/graphcast-20261114/docs/UPDATING.md)
- [历史Small复现记录](results/weekly-consolidation-20261114/SCNET_HISTORY.md)

当前原始case的SCNET Slurm／Podman适配、显存可行性及GPU复验仍待完成。
历史Small结果不代表本case通过，旧源码目录不再作为运行入口。
