# GraphCast FlagOS推理验证汇总｜2026-10-30

推理任务的算子表、kernel附件、40步精度性能和MFU依据已汇总。两种后端各108用例通过：本次在GPU9补测全部15个Linear形状，其余93个非Linear用例复用10月29日GPU8记录。原baseline与修复前失败记录保留。

## 逻辑算子耗时

| 逻辑算子 | 历史JAX ms | FlagOS native ms | FlagOS fl_only ms | 前向调用 | 形状数 |
|---|---:|---:|---:|---:|---:|
| Linear | 733.455 | 802.309 | 763.221 | 84 | 15 |
| Concat | 94.672 | 96.723 | 184.849 | 42 | 11 |
| LayerNorm | 104.085 | 116.113 | 89.114 | 42 | 5 |
| ScatterAdd / SegmentSum | 106.075 | 108.856 | 369.827 | 18 | 3 |
| SiLU | 67.877 | 77.968 | 74.176 | 43 | 5 |
| Gather | 43.277 | 77.936 | 155.872 | 36 | 6 |
| Add | 32.504 | 33.892 | 42.560 | 45 | 6 |
| Mul | 3.123 | 2.962 | 2.668 | 16 | 7 |
| Sub | 5.124 | 4.978 | 4.710 | 22 | 4 |
| Div | 7.311 | 4.906 | 5.060 | 22 | 4 |
| Slice | 3.897 | 4.473 | 3.930 | 22 | 13 |
| DynamicSlice | 1.656 | 1.869 | 1.834 | 10 | 2 |
| Broadcast | 15.086 | 12.369 | 9.446 | 46 | 8 |
| Reshape | 10.068 | 8.092 | 7.237 | 30 | 7 |
| Transpose | 12.176 | 12.598 | 10.232 | 50 | 10 |
| Squeeze | 3.569 | 3.901 | 3.405 | 21 | 2 |

独立算子采用同步JAX墙钟，类别耗时按原图调用次数加权；不能相加当整模型延迟。Linear包含独立两输入RNE与bias，不含79处输入生成融合。原生历史列不是当前容器的native列。

## 整模型与MFU

- FlagOS三轮40步墙钟：58.766360、58.523623、58.616376秒，中位数58.616376秒；H100参考14.472323秒。
- 主要同轮配对整体误差0.515465%–0.603645%，Z500为0.005827%–0.007779%；全部9组通过内部<5%和阶段≤10%。
- 单步卡上内核活跃时间 1349.381105 ms，MFU 12.519228%；分子28.784948924416 TFLOPs，分母使用170.3936 TFLOP/s。
- Gather相关融合30.86%、Scatter相关融合26.92%，合计57.78%；GEMM23.39%；独立RNE1.12%，不包括融合中的RNE成本。
- 后端覆盖：165次FlagTree、84次rocBLAS、89次HIP RNE。81次GEMM为XF32 MX，3次K=4为原生FP32回退；显式RNE均保留。
- 同等算力60%效率的正式结论尚待峰值、设备等效与计时口径确认；当前单步MFU不代表完整40步卡上MFU。

## 文件入口

- [Excel总表](GraphCast_BW1100_FlagOS_推理算子与精度性能_20261030.xlsx)：逻辑算子、216行用例、模型性能、40步末帧精度、GPU分类、方法与来源，共6个sheet。
- `operators.csv`、`case-details.csv`：可计算表格；`summary.json`：机器可读汇总。
- `attachments/kernel-events.csv`：338个原始kernel区间；`kernel-details.csv`：按kernel汇总；`kernel-backend-evidence.csv`：后端与HLO位置；精度逐变量逐层明细同目录。
- `gemm-flops.csv`：84次GEMM有效FLOPs；`validation.json`与`source-manifest.json`：校验与来源SHA。
- 原始模型输出/环境/复现脚本：`/home/zhangyuxin/zyx/graphcast/outputs/flagos-rne-gemm-ex-20261030`；H100参考：`/home/zhangyuxin/zyx/graphcast/outputs/h100-gpu41-tf32-20261029`。
- 新补测命令在`run-linear.sh`与`replay-linear.py`；在已准备的容器中依次执行native、fl_only，固定GPU9（PCI 0000:88:00.0），需先确认空闲。脚本拒绝覆盖已有输出；再次复测须复制脚本到新的实验目录，并将run-linear.sh的replay_root改为该新目录。冻结的inventory和候选库只读复用。
- 离线重新汇总：`python3 /home/zhangyuxin/zyx/graphcast/outputs/flagos-inference-final-20261030/consolidate.py`。

## 测量边界与后续

- 算子表为历史JAX列＋FlagOS容器native/fl_only对照；不是FlagGems算子库耗时。native与fl_only均使用同一FlagOS镜像，不能将native称为已恢复旧海光baseline。
- 每个后端108用例：93项非Linear复用10月29日GPU8，15项Linear为10月30日GPU9补测；不是本次重跑全部108项。硬件同型号，跨日跨卡且机器有其他任务，非严格交错A/B。
- 独立算子计时为3轮，每轮5次预热＋20次同步JAX墙钟；取各轮中位数的中位数，再乘前向调用次数。包含调用、同步与审计开销，不是纯GPU时间，不能相加推导整模型耗时。
- Linear独测含两输入RNE＋rocBLAS＋bias，不使用整模型79处激活生成融合；其余算子为FP32。K=4可回退原生FP32，RNE仍执行。
- 算子正确性使用每例最多258个输出元素的独立参考＋全输出有限性；Linear参考先RNE输入再FP64点积，不是与未经舍入FP32输出的误差；全模型另以H100的40步末帧评价。
- 整模型单步338次kernel：165次FlagTree、84次rocBLAS、89次独立HIP RNE。81次GEMM为XF32 MX，3次K=4原生FP32回退。79处激活RNE已融合，不重复加到独立RNE耗时。
- 模型40步墙钟含预加载后传输、自回归更新和调用；不含加载、编译、预热、校验和写盘。三轮每轮预热5步后重置输入；profile单独采集，不混入三轮墙钟结果。
- MFU仅单步GPU内核区间并集、GEMM FLOPs近似；不含内核间空隙、CPU及拷贝。没有完整40步卡上实测MFU，不能将单步乘40冒充实测。
- H100参考为gpu41于10月29日官方JAX TF32（无显式RNE）；同一预训练权重/输入/统计，独立40步，+240h末帧6变量×13层等权平均；Z500单列，分母H100。不代表所有时效/变量或天气真值精度。
- 170.3936 TFLOP/s为用户指定截图峰值；设备映射与频率依据、H100同精度稠密峰值、等效算力公式及验收计时口径待确认。阶段60%不等于自身MFU60%；当前不作正式效率达标判定。
- 保留baseline与修复前失败记录。新结果未推送GitHub、未同步testuser/SCNET；按用户最后统一同步。训练反向与loss不在本次推理结论范围。

后续性能工作优先检查FlagTree的Gather/Scatter生成内核。完整40步卡上采集及正式效率判定，在对应测量约定明确后安排；训练最小验证由既有任务单独推进。本次未新增任务或统一同步。
