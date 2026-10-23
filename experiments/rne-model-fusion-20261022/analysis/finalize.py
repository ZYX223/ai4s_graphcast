"""Verify immutable evidence and produce the concise experiment readout."""
from pathlib import Path
import collections,datetime,hashlib,json,statistics,csv
root=Path(__file__).resolve().parents[1]
def read(path):return json.loads((root/path).read_text())
def write(path,obj):(root/path).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
manifest=read('baseline-manifest.json');repo=Path('/home/zhangyuxin/zyx/graphcast-sugon')
for name,expected in manifest.items():
 assert digest(root/'baseline-source'/name)==expected,name
 assert digest(repo/name)==expected,name
frozen={p.name:digest(p) for p in (root/'experiment-r1').iterdir() if p.is_file()}
for phase in ['profile','bench']:
 for variant in ['baseline','fused']:
  folder=root/(phase+'-'+variant+'-r1');r=json.loads((folder/'run/report.json').read_text())
  assert (folder/'exit-code.txt').read_text().strip()=='0'
  assert r['status']=='completed' and r['variant']==variant
  assert r['experiment_sha256']==frozen
  assert r['pci'].lower()=='0000:b0:00.0'
  audit=json.loads((folder/'run/step-audit.json').read_text())
  assert len(audit)==(6 if phase=='profile' else 135)
  assert all(x['native_calls']==84 and x['rounded_calls']==84 for x in audit)
  if variant=='fused':assert all(x['fused_activation_calls']==79 for x in audit)
  hook=json.loads((folder/'run/hook-audit.json').read_text())
  assert len(hook['seen'])==(79 if variant=='fused' else 0)
  assert all(x['jit_cache_before']==x['jit_cache_after'] and x['jit_cache_before']>0 for x in r['rows'])
  assert digest(folder/'native/libxf32_rne_router.so')==r['router_build']['library_sha256']
  for x in r['rows']:
   assert digest(folder/'run'/x['run']/'prediction.nc')==x['output_sha256']
assert (root/'check-r1/exit-code.txt').read_text().strip()=='0'
check=(root/'check-r1/console.log').read_text()
assert all(('Ran %d tests'%n) in check for n in [3,4,32])
profiles={v:read('profile-'+v+'-r1/profile-summary.json') for v in ['baseline','fused']}
gemm=[{r['name']:r['calls'] for r in profiles[v]['kernels'] if r['category']=='gemm'} for v in ['baseline','fused']]
assert gemm[0]==gemm[1]
hlo=read('profile-fused-r1/hlo-fusion-audit.json')
assert hlo['same_fusion_count']==hlo['selected_runtime_calls']==79
assert all(g['observed']==g['expected'] for g in hlo['groups'])
comparison=read('40-step-comparison.json');single=read('single-step-comparison.json');repeat=read('repeatability.json')
metrics={}
for variant in ['baseline','fused']:
 rows=[read('bench-'+variant+'-r1/run/round%d-rne_xf32/report.json'%i) for i in [1,2,3]]
 assert all(r['accuracy']['metrics_within_limits'] for r in rows)
 metrics[variant]=[dict(round=r['round'],**{k:r['accuracy'][k] for k in ['overall_mean_percent','z500_percent','z500_rmse_m']}) for r in rows]
assert all(p['weekly_metrics']['metrics_within_limits'] for p in comparison['pairs'])
summary=dict(status='completed',date='2026-10-22',product_model='BW1100',driver_model='BW1200',pci='0000:b0:00.0',scope='Official JAX GraphCast inference; same frozen checkpoint/input; 79 selected Linear input producers; native rocBLAS unchanged',benchmark_order='baseline x3 followed by fused x3; sequential blocks, not randomized/counterbalanced',warmup_steps_each_round=5,measured_steps_each_round=40,performance=comparison['performance'],accuracy_vs_official_jax_fp32=metrics,accuracy_fused_vs_same_index_baseline=[dict(round=p['round'],**{k:p['weekly_metrics'][k] for k in ['overall_mean_percent','z500_percent','z500_rmse_m']},all_11_variables_bitwise_equal=p['all_11_variables_bitwise_equal']) for p in comparison['pairs']],within_variant_repeatability={v:[dict(round=r['round'],overall_mean_percent=r['accuracy']['overall_mean_percent'],z500_percent=r['accuracy']['z500_percent']) for r in rows] for v,rows in repeat.items()},profile={v:{k:p[k] for k in ['gpu_kernel_count','gpu_kernel_sum_ms','categories']} for v,p in profiles.items()},hlo_fusion=dict(selected=79,concat=37,silu=42,canonical_runtime_kernel_groups=len(hlo['groups']),same_gemm_kernel_names_and_multiplicities=True),baseline_files_unchanged=len(manifest),tests_passed=39)
write('summary.json',summary)
write('final-validation.json',dict(status='passed',verified_at=datetime.datetime.utcnow().isoformat()+'Z',baseline_files_unchanged=len(manifest),experiment_files_verified=len(frozen),profile_and_benchmark_runs_verified=4,model_steps_audited=282,tests_passed=39,all_6_fp32_comparisons_within_limits=True,all_3_fusion_comparisons_within_limits=True,same_native_gemm_kernels=True,limitations=['No bitwise equivalence: both variants have run-to-run output differences','Sequential blocks of 3 benchmark repeats, no counterbalancing','Only the fixed inference case; no training or new H100 run'] ))
with (root/'40-step-results.csv').open('w') as f:
 writer=csv.writer(f);writer.writerow(['variant','round','rollout_seconds','apply_sum_seconds','overall_error_percent_vs_fp32','z500_error_percent_vs_fp32','z500_rmse_m_vs_fp32'])
 for v in ['baseline','fused']:
  for i,m in enumerate(metrics[v]):writer.writerow([v,i+1,comparison['performance'][v]['rollout_seconds'][i],comparison['performance'][v]['apply_sum_seconds'][i],m['overall_mean_percent'],m['z500_percent'],m['z500_rmse_m']])
p=comparison['performance'];b=profiles['baseline'];c=profiles['fused']
text='''# BW1100 GraphCast：整模型 RNE 输入融合结果

已完成单步 profile 和两条路径各 3 轮的 40 步推理。原 baseline 和工作仓库的 115 个文件均未修改；候选实现位于独立冻结目录 `experiment-r1/`。

本次保留全部 84 次原生 rocBLAS GEMM、原有 XF32 模式和权重 RNE，仅把 37 处 Concat、42 处 SiLU 后的激活 RNE 放到 JAX 中，再按层跳过对应的独立 HIP 激活舍入。

## 性能

| 指标 | 原 RNE＋XF32 baseline | 输入融合版 | 变化 |
|---|---:|---:|---:|
| 40 步端到端，中位数 | %.3f s | %.3f s | 耗时减少 %.2f%%，%.3f× |
| 40 步同步模型调用总耗时，中位数 | %.3f s | %.3f s | 耗时减少 %.2f%% |
| 单步 profile：GPU kernel 累计 | %.3f ms | %.3f ms | 减少 %.2f%% |
| 单步 kernel 数量 | %d | %d | 净减少 %d |
| 独立 HIP RNE 次数 | 168 | 89 | 减少 79 |
| 单步原生 GEMM 次数 | 84 | 84 | 内核名称和调用分布相同 |

40 步逐轮耗时：baseline **%.3f / %.3f / %.3f s**；融合版 **%.3f / %.3f / %.3f s**。两条路径在同一张卡上按顺序分别运行 3 轮，每轮预热 5 步，排除编译和预热。这里没有随机交错或多机统计。profile 自带采集开销，其 GPU kernel 累计不能当作端到端耗时。

独立 RNE 内核累计从 **%.3f ms** 降为 **%.3f ms**。这不代表所有 RNE 算术只剩后一个数：移入 producer 的舍入成本已经计入融合内核。其他内核累计增加约 **%.3f ms**，因此净 GPU 累计收益约 **%.3f ms**。XLA 也重新组织了其他融合，最终净减少 71 个 kernel，而不是 79 个。

## 精度

第 40 步（+240h）对既有官方 JAX FP32 参考：

| 路径 | 整体相对误差，3 轮范围 | Z500 相对误差，3 轮范围 | Z500 RMSE，3 轮范围 |
|---|---:|---:|---:|
'''%(p['baseline']['rollout_median_seconds'],p['fused']['rollout_median_seconds'],p['rollout_reduction_percent'],p['rollout_speedup'],p['baseline']['apply_sum_median_seconds'],p['fused']['apply_sum_median_seconds'],(1-p['fused']['apply_sum_median_seconds']/p['baseline']['apply_sum_median_seconds'])*100,b['gpu_kernel_sum_ms'],c['gpu_kernel_sum_ms'],(1-c['gpu_kernel_sum_ms']/b['gpu_kernel_sum_ms'])*100,b['gpu_kernel_count'],c['gpu_kernel_count'],b['gpu_kernel_count']-c['gpu_kernel_count'],*(p['baseline']['rollout_seconds']+p['fused']['rollout_seconds']),b['categories']['rne']['total_ms'],c['categories']['rne']['total_ms'],(c['gpu_kernel_sum_ms']-c['categories']['rne']['total_ms'])-(b['gpu_kernel_sum_ms']-b['categories']['rne']['total_ms']),b['gpu_kernel_sum_ms']-c['gpu_kernel_sum_ms'])
for v,label in [('baseline','baseline'),('fused','融合版')]:
 vals=[[m[k] for m in metrics[v]] for k in ['overall_mean_percent','z500_percent','z500_rmse_m']]
 text+='| %s | %.3f%%–%.3f%% | %.4f%%–%.4f%% | %.3f–%.3f m |\n'%(label,min(vals[0]),max(vals[0]),min(vals[1]),max(vals[1]),min(vals[2]),max(vals[2]))
text+='''
六次结果的整体误差与 Z500 误差均小于现有 **5%** 门限。整体误差为六变量×十三层的 78 项相对 MAE 等权平均，比较各自完整自回归后的最后一步；不是 40 步平均，也不是对观测真值的预报误差。

融合前后不是逐位相同。单步直接整体差异 **0.03838%**、Z500 **0.00010165%**；第 40 步按轮号对应比较的整体差异 **0.520%–0.581%**、Z500 **0.00610%–0.00656%**。全部 11 个输出变量均完成有限值、坐标与逐位差异检查。

重复运行也有差异：以本路径第 1 轮为参考，baseline 第 2/3 轮整体差异 **0.4816%–0.5025%**，融合版 **0.5647%–0.6095%**。跨路径差异与这些重复差异量级接近。本实验没有定位首个偏差层，不能把全部差异归因于 RNE 融合，也不能据此声称数值完全等价。

## XLA 与原生库核验

优化后 HLO 中，79 个目标 GEMM 的输入 producer 均同时包含原 Concat/SiLU 和 RNE 位运算。运行时采用 13 组复用的 kernel 代码，各组实际 launch 数与目标逻辑位置数逐组相符，合计 79 次。每步 native 日志匹配全部 84 个原始权重 SHA256，XF32 模式读取正确，恰好跳过名单内 79 次激活 RNE。

权重 RNE 保留 84 次，未覆盖的激活 RNE 保留 5 次，合计 89 次。真实 rocBLAS GEMM 内核名称及各自调用次数与 baseline 完全相同。没有实现自定义 GEMM，也没有接入 FlagGems。

## 代码、复现和明细

- [复现说明](REPRODUCE.md)：环境、路径、单步 profile、40 步命令及必需验收步骤。
- [本次实际候选代码](experiment-r1/)；[按层 JAX 接入](experiment-r1/model_hook.py)；[候选原生路由生成器](experiment-r1/build_candidate.py)。
- [机器可读汇总](summary.json)、[逐轮性能精度 CSV](40-step-results.csv)、[最终核验](final-validation.json)。
- [40 步融合前后完整误差](40-step-comparison.json)、[单步误差](single-step-comparison.json)、[重复运行差异](repeatability.json)。
- [baseline kernel 明细](profile-baseline-r1/kernel-details.csv)、[融合版 kernel 明细](profile-fused-r1/kernel-details.csv)、[逐层 HLO 融合证据](profile-fused-r1/hlo-fusion-audit.json)。
- `profile-{baseline,fused}-r1/` 保存完整 trace、优化后 HLO、native 调用日志及单步输出。
- `bench-{baseline,fused}-r1/run/roundN-rne_xf32/` 保存三轮第 40 步输出、对 FP32 的逐变量逐层误差；每一步计数保存在各 `run/step-audit.json`。

39 项测试通过。4 次模型作业均完成，计入预热共核验 282 个预测步。当前结果支持将融合版保留为独立候选；现有 baseline 保持原样。结论限定于本次 BW1100、固定模型/权重/输入的推理实验，未测试训练或重新运行 H100。
'''
(root/'README.md').write_text(text)
print(json.dumps(dict(validation='passed',baseline_files_unchanged=len(manifest),rollout_reduction_percent=p['rollout_reduction_percent'],rollout_speedup=p['rollout_speedup']),indent=2))
