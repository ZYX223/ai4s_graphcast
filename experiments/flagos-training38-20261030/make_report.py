"""Render only complete paired evidence; retain failed-case limitations."""
import csv,json
from pathlib import Path
p=Path(__file__).resolve().parent
s=json.loads((p/'summary.json').read_text());assert s['status']=='completed'
o=json.loads((p/'operators-summary.json').read_text());h=s['backends']['h100'];d=s['backends']['flagos']
historical=json.loads((p/'historical-native.json').read_text())
stages=list(csv.DictReader((p/'horizons.csv').open(encoding='utf-8-sig')))
lines=['# GraphCast 38次短程训练：H100与BW1100 FlagOS','',
 '实验日期：2026-10-30。两端均在单卡完成38次优化器更新和1～12步预测长度；其中2次学习率为0、36次实际改变参数。每次更新包含2个微批次。各阶段checkpoint恢复及正常退出均通过。每个平台为一条完整38次轨迹，未做整轮重复方差评估。',
 '', '## 训练结果','',
 '| 项目 | H100 JAX TF32（参考，无显式RNE） | BW1100 FlagOS JAX fl_only RNE＋XF32 |',
 '| --- | ---: | ---: |',
 '| 卡上计算累计时间 | %.3f s | %.3f s |'%(h['gpu_kernel_seconds'],d['gpu_kernel_seconds']),
 '| 同步更新墙钟累计 | %.3f s | %.3f s |'%(h['synchronized_update_seconds'],d['synchronized_update_seconds']),
 '| 训练MFU（有效GEMM近似） | %.4f%% | %.4f%% |'%(h['mfu_percent'],d['mfu_percent']),
 '| 总峰值口径 | 494.7 TFLOP/s | 170.3936 TFLOP/s |',
 '| 整卡采样峰值 | %.3f GiB | %.3f GiB |'%(h['whole_card_peak_gib'],d['whole_card_peak_gib']),
 '| JAX分配器峰值 | %.3f GiB | %.3f GiB |'%(h['jax_peak_gib'],d['jax_peak_gib']),
 '| 本轮最低loss | %.10f | %.10f |'%(h['best_loss'],d['best_loss']),
 '| 第38次loss | %.10f | %.10f |'%(h['last_loss'],d['last_loss']),
 '', '对应38次更新的loss相对差异最大 **%.6f%%**、算术平均 **%.6f%%**。分母为对应H100 loss绝对值；本轮全部满足内部小于5%%及当前阶段≤10%%的loss要求。loss是本次参数更新前两个微批次前向loss的均值，不是最终checkpoint另行评价。短程一致性不代表收敛，且本轮最低loss跨预测长度，仅作记录。'%(s['max_loss_relative_percent'],s['mean_loss_relative_percent']),
 '', 'BW1100/H100的MFU相对比值为 **%.4f%%**。这是按所列峰值归一化的候选效率比值，尚非项目正式认可的60%%验收公式。卡上吞吐比为 **%.4f%%**；两者口径不同。'%(s['flagos_over_h100_mfu_ratio_percent'],100*h['gpu_kernel_seconds']/d['gpu_kernel_seconds']),
 '', '历史性能参照：10月23日海光native RNE＋XF32的38次GPU内核并集为%.3f秒，本轮增加%.2f%%。模型配置、数据及日程已核对；JAX发行版和LayerNorm重计算配置不同，不能把全部差值归因于单个因素，也不作为同版本A/B。历史原件保持不变。'%(historical['gpu_kernel_union_seconds'],historical['increase_percent']),
 '', '## MFU计算与采集','',
 '`MFU = Σ有效前后向GEMM FLOPs / (Σ实际GPU计算内核区间并集秒数 × 单卡峰值FLOP/s)`。先对完整loss求参数梯度，再在禁用模型、时间重计算的JAXPR中统计存活dot_general，每次按2MNK计数，乘2个微批次。使用各预测长度对应图，不能直接用推理84个GEMM的数值乘3。',
 '', '本轮有效工作量：H100 %.9f PFLOPs，BW1100 %.9f PFLOPs。1步训练每微批次253个有效GEMM；12步2960个。两平台内部grid2mesh边数差1条，因此分别计算有效FLOPs。'%(h['useful_gemm_flops']/1e15,d['useful_gemm_flops']/1e15),
 '', '分子不重复计入重计算，也不计RNE、优化器或通信；这些计算内核的实际时间计入分母。数据复制、CPU、内核间空隙、编译、预热及保存不计入卡上计算时间。同步更新墙钟另列，含两个微批次前反向、梯度平均和AdamW更新；输入准备与H2D在其外。',
 '', 'H100使用Nsight CUDA kernel记录；BW1100使用JAX原生profiler的完整XSpace。首阶段Chrome JSON导出受100万事件上限截断，但原始PB含完整16次更新；所有38次均使用原始记录计时，并逐次核对RNE内核数等于两倍已路由GEMM调用数。两端采集方式不同，数值为带采集运行的实测，未额外宣称无采集开销。',
 '', '## 各预测长度','',
 '| 预测步数 | 更新数 | H100卡上累计(s) | FlagOS卡上累计(s) | H100 MFU | FlagOS MFU |',
 '| ---: | ---: | ---: | ---: | ---: | ---: |']
for r in range(1,13):
 a=next(x for x in stages if x['backend']=='h100' and int(x['horizon'])==r);b=next(x for x in stages if x['backend']=='flagos' and int(x['horizon'])==r)
 lines.append('| %d | %s | %.3f | %.3f | %.3f%% | %.3f%% |'%(r,a['updates'],float(a['gpu_kernel_seconds']),float(b['gpu_kernel_seconds']),float(a['mfu_percent']),float(b['mfu_percent'])))
lines+=['', '## 逻辑算子与覆盖','',
 '共有221个形状/参数/前反向用例。FlagOS测得%d个，含FlagTree HLO调用%d个；%d个完成native GPU抽样对照，另5个LayerNorm前向用例完成CPU JAX独立抽样对照。'%(o['flagos_measured'],o['flagtree_cases'],o['flagos_pass']),
 '', '表格见[OPERATORS.md](OPERATORS.md)，按JAX/FlagOS fl_only/加速比/FlagTree覆盖列示；[operator-phases.csv](operator-phases.csv)按scope标记前向、反向或未分类辅助操作（不是严格完整的前反向时间拆分），[operator-cases.csv](operator-cases.csv)含每个形状参数用例状态及误差。[kernel-details.csv](kernel-details.csv)保留整模型训练内核明细，包含重计算与优化器。',
 '', '算子表为同一FlagOS容器内native与fl_only模式、同输入的独立同步墙钟测量；每轮5次预热、20次计时，共3轮，按调用次数加权。范围为预测1步的有效前后向及loss，不含重计算、优化器；不能把这些独立耗时相加当作整模型卡上时间。GEMM仍为rocBLAS，RNE仍为独立HIP内核；不称为FlagGems库接入。',
 '', '## 配置与已知限制','',
 '- 官方GraphCast源码08cf73625c9d12bd9aaa038868bcb2fe488f2a22，0.25°、13层、batch=1，固定原始权重、14帧真实样例和seed27。16次预测1步更新，随后预测2～12步各更新2次；沿用旧38次学习率日程，未使用最小验证的固定3e-7替代整个日程。',
 '- H100使用原有阶段/消息/时间重计算；BW1100增加已验证LayerNorm重计算。两平台均保留FP32参数与状态，未使用BF16。两端从相同初始状态独立更新，逐次比较loss；不进行两份训练后checkpoint的推理输出互比。',
 '- H100预测11步首次在94%显存额度下OOM，保留失败记录；从第34次checkpoint恢复后，11～12步采用96%，同时清除跨更新保留的梯度引用。此清理不改数学过程。BW1100的1～10步采用70%且关闭预分配；基于9步实测98.63GiB峰值及容量增长，11～12步调整为73%，保持模型与重计算方案；原baseline未修改。',
 '- native GPU的5个LayerNorm前向独立用例触发XLA布局编译断言，完整FlagOS训练未触发该错误。CPU JAX全形状、相同输入的抽样比较最大绝对差9.53674e-6；对应native GPU耗时及加速比留空。采样对照不是全部元素证明。一个大尺寸Linear反向用例在两后端连续测试均OOM；保持相同80%额度，各自独立进程重测后正常完成计时和精度对照，原失败与复测分别保留。',
 '- 外部HIPPROF退出139的历史故障未被宣称修复；本轮使用已验证正常退出的JAX原生采集。失败历史不纳入主训练轨迹与MFU。',
 '', '## 材料与复算','',
 '本机目录：`'+str(p)+'`；H100原始运行目录：`/data/zyx/graphcast-training38-20261030/`。`updates.csv`逐次记录loss/学习率/时间/MFU；`horizons.csv`记录显存与编译；`loss-comparison.csv`保存逐次精度对照；`flops/`与`flops-h100/`保存各平台有效计算量。',
 '', '`python3 summarize_training.py`重新核对报告、checkpoint链和原始时间后复算；`python3 summarize_operators.py`生成算子表。BW1100各阶段`jax-trace/*.xplane.pb`为原始时间证据，`xplane_timing.py`需在带TensorFlow protobuf依赖的容器中执行。H100使用各阶段`train-trace.sqlite`。',
 '', '原baseline10份源码哈希核对未变；源码实际版本及仅清理梯度引用的差异见`source-snapshots/manifest.json`。最终复现入口增加checkpoint成功退出/文件哈希校验；未将此额外守卫部署到正在运行的实验，现有轨迹另行完整核对。代码与结果按用户要求等待最后统一同步。']
(p/'REPORT.md').write_text('\n'.join(lines)+'\n')
print(p/'REPORT.md')
