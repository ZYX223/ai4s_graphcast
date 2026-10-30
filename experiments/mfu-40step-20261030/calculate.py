"""Measured 40-step GPU-active MFU from HIPPROF and Nsight, excluding warmups."""
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import statistics

ROOT=Path(__file__).resolve().parent
F_STEP=28784948924416
F_40=F_STEP*40
sources={}
def read(p):
    sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    return json.loads(p.read_text())
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csvout(p,rows):
    with p.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def union_ns(intervals):
    total=0;end=0
    for a,b in sorted(intervals):
        assert b>=a
        total+=max(0,b-max(end,a));end=max(end,b)
    return total
peaks=read(ROOT/'peak-reference.json')
platforms=['h100','hygon']
if (ROOT/'native-profile/run/report.json').exists():
    native_report=read(ROOT/'native-profile/run/report.json')
    if native_report.get('status')=='completed':
        platforms=['h100','native','hygon']
        peaks['native']=peaks['hygon']
reports={p:read(ROOT/(p+'-profile')/'run/report.json') for p in platforms}
assert reports['h100']['original_case_sha256']==reports['hygon']['original_case_sha256']
assert reports['h100']['data']==reports['hygon']['data']
assert reports['h100']['source']==reports['hygon']['source']
assert reports['h100']['router_build'] is None
for platform,r in reports.items():
    assert r['data']==reports['h100']['data'] and r['source']==reports['h100']['source']
    assert r['status']=='completed' and r['measured_steps']==40 and r['warmup_steps']==5 and len(r['rows'])==len(r['warmups'])==3
    assert r['storage']=='float32' and r['jax_matmul_precision']=='tensorfloat32'
    assert r['mode']==('tf32' if platform=='h100' else 'rne_xf32')
    for row in r['rows']:
        assert len(row['apply_seconds'])==40 and row['jit_cache_before']==row['jit_cache_after']
    audit=read(ROOT/(platform+'-profile')/'run/step-audit.json')
    assert len(audit)==135
    for row in r['rows']:
        aa=[x for x in audit if x['label']==row['run']]
        assert [a['step'] for a in aa]==list(range(1,41))
        if platform!='h100':assert all(a['native_calls']==a['rounded_calls']==84 and a['fused_activation_calls']==79 for a in aa)

gemm_path=ROOT/'hygon-profile/gemm.jsonl';sources[str(gemm_path)]=hashlib.sha256(gemm_path.read_bytes()).hexdigest()
gemms=[json.loads(line) for line in gemm_path.read_text().splitlines()]
assert len(gemms)==84*135
for row in reports['hygon']['rows']:
    gg=[g for g in gemms if g['label']==row['run']]
    assert len(gg)==3360 and all(g['rne_inputs'] and g['mode']==1 and g['status']==0 for g in gg)
    assert sum(2*g['m']*g['n']*g['k'] for g in gg)==F_40
    assert collections.Counter(g['api'] for g in gg)==dict(sgemm=82*40,gemm_ex=2*40)
    assert all(n==40 for n in collections.Counter(g['layer'] for g in gg).values())

hip=read(ROOT/'hygon-profile/trace.json')['traceEvents']
hipkernels=[e for e in hip if e.get('cat')=='HIPOPS' and e.get('ph')=='X']
hipcopies=[e for e in hip if e.get('cat')=='HIPCOPY' and e.get('ph')=='X']
ranges=read(ROOT/'hygon-profile/run/capture-ranges.json')
result=dict(date='2026-10-30',status='completed',measurement='Three measured full 40-step independent autoregressions per platform; GPU-active interval union; not 40 times a single-step time.',
    useful_gemm_flops_per_step=F_STEP,useful_gemm_flops_per_40_steps=F_40,
    formula='MFU (%) = F_40 / (union_GPU_kernel_seconds * dense_peak_FLOPs_per_second) * 100',
    baseline_modified=False,steps_in_notion_modified=False,github_pushed=False,delivery_synced=False)
all_details=[];capture_counts={}
for platform in platforms:
    if platform!='h100':
        hip=read(ROOT/(platform+'-profile')/'trace.json')['traceEvents']
        hipkernels=[e for e in hip if e.get('cat')=='HIPOPS' and e.get('ph')=='X']
        hipcopies=[e for e in hip if e.get('cat')=='HIPCOPY' and e.get('ph')=='X']
        ranges=read(ROOT/(platform+'-profile')/'run/capture-ranges.json')
        matched_hip=0
        if platform=='native':
            native_gemm=ROOT/'native-profile/gemm.jsonl'
            sources[str(native_gemm)]=hashlib.sha256(native_gemm.read_bytes()).hexdigest()
            gg_all=[json.loads(line) for line in native_gemm.read_text().splitlines()]
            assert len(gg_all)==84*135
            for row in reports[platform]['rows']:
                gg=[g for g in gg_all if g['label']==row['run']]
                assert len(gg)==3360 and all(g['rne_inputs'] and g['mode']==1 and g['status']==0 for g in gg)
                assert sum(2*g['m']*g['n']*g['k'] for g in gg)==F_40
    per=[]
    for index,row in enumerate(reports[platform]['rows'],1):
        if platform=='h100':
            db=ROOT/'h100-profile'/('model-trace.{}.sqlite'.format(index))
            sources[str(db)]=hashlib.sha256(db.read_bytes()).hexdigest()
            con=sqlite3.connect(str(db))
            assert con.execute('pragma integrity_check').fetchone()[0]=='ok'
            assert con.execute('select count(distinct deviceId) from CUPTI_ACTIVITY_KIND_KERNEL').fetchone()[0]==1
            kk=con.execute('select k.start,k.end,s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.demangledName=s.id order by k.start').fetchall()
            copies=con.execute('select start,end from CUPTI_ACTIVITY_KIND_MEMCPY').fetchall()
            con.close()
            assert sum('tf32' in n.lower() for a,b,n in kk)==89*40
            assert not any('round_rne' in n for a,b,n in kk)
        else:
            cap=ranges[index-1];assert cap['label']==row['run']
            selected=[e for e in hipkernels if cap['wall_start_ns']<=int(e['args']['BeginNs']) and int(e['args']['EndNs'])<=cap['wall_end_ns']]
            matched_hip+=len(selected)
            kk=[(int(e['args']['BeginNs']),int(e['args']['EndNs']),e['name']) for e in selected]
            copies=[(int(e['args']['BeginNs']),int(e['args']['EndNs'])) for e in hipcopies if cap['wall_start_ns']<=int(e['args']['BeginNs']) and int(e['args']['EndNs'])<=cap['wall_end_ns']]
            assert sum(n.startswith('Cijk_') for a,b,n in kk)==84*40
            assert sum('round_rne' in n for a,b,n in kk)==89*40
            assert sum(n.startswith('Cijk_') and '_MX_' in n for a,b,n in kk)==81*40
        assert kk
        t=union_ns([(a,b) for a,b,n in kk])/1e9
        peak=peaks[platform]['peak_tflops']
        one=dict(round=index,label=row['run'],measured_steps=40,kernel_count=len(kk),
            kernel_active_seconds=t,kernel_sum_seconds=sum(b-a for a,b,n in kk)/1e9,
            kernel_span_seconds=(max(b for a,b,n in kk)-min(a for a,b,n in kk))/1e9,
            memcpy_sum_seconds=sum(b-a for a,b in copies)/1e9,
            kernel_and_copy_union_seconds=union_ns([(a,b) for a,b,n in kk]+copies)/1e9,
            rollout_wall_seconds=row['rollout_wall_seconds'],synchronized_apply_sum_seconds=row['apply_sum_seconds'],
            effective_tflops=F_40/1e12/t,mfu_percent=F_40/1e12/(t*peak)*100)
        per.append(one)
        for i,(a,b,n) in enumerate(kk,1): all_details.append(dict(platform=platform,round=index,index=i,name=n,start_ns=a,end_ns=b,duration_ns=b-a))
    if platform!='h100':
        assert matched_hip==len(hipkernels)
        capture_counts[platform]=matched_hip
    result[platform]=dict(label={'h100':'H100 JAX TF32（无显式RNE）','native':'BW1100 原生JAX融合 RNE＋XF32（恢复环境）','hygon':'BW1100 FlagOS fl_only RNE＋XF32'}[platform],
        device=reports[platform]['device'],pci=reports[platform]['pci'],physical_gpu=2 if platform=='h100' else 9,
        peak_tflops=peaks[platform]['peak_tflops'],rounds=per,
        median_gpu_active_seconds=statistics.median(x['kernel_active_seconds'] for x in per),
        median_profiled_wall_seconds=statistics.median(x['rollout_wall_seconds'] for x in per),
        median_mfu_percent=statistics.median(x['mfu_percent'] for x in per),
        aggregate_mfu_percent=3*F_40/1e12/(sum(x['kernel_active_seconds'] for x in per)*peaks[platform]['peak_tflops'])*100)
assert capture_counts['hygon']==42783
result['platform_order']=platforms
result['capture_kernel_counts']=capture_counts
nv,hg=result['h100'],result['hygon']
result['comparison']=dict(raw_gpu_active_throughput_ratio_percent=100*nv['median_gpu_active_seconds']/hg['median_gpu_active_seconds'],
    mfu_ratio_dcu_over_nv_percent=100*hg['median_mfu_percent']/nv['median_mfu_percent'],
    normalized_formula_status='Candidate only; no formal contractual 60% pass/fail verdict',formal_stage_efficiency_pass=None,
    phase_scope='Inference only; not training')
if 'native' in result:
    native=result['native']
    result['comparison'].update(native_mfu_ratio_over_nv_percent=100*native['median_mfu_percent']/nv['median_mfu_percent'],
        flagos_speedup_over_native_gpu=native['median_gpu_active_seconds']/hg['median_gpu_active_seconds'])
result['limits']=[
    'GPU-active union excludes CPU, transfers and gaps; includes all traced device kernels across full rollout including state-update auxiliaries. Not end-to-end MFU.',
    'Each platform captured 3 complete 40-step rounds after 5 warmup steps and resetting input; no warmup, compile, metrics or file writes included.',
    'Useful model FLOPs approximate 84 logical GEMMs per step; elementwise/reduction FLOPs omitted. Same numerator across platforms; auxiliary overhead stays in measured time.',
    'H100 494.7 TFLOP/s from NVIDIA v1.04 dense TF32 SXM5 specification; BW1100 170.3936 from user-designated peak. No use of sparse/BF16 peak or frequency scaling.',
    'H100 GPU2 and BW1100 GPU9 are individually available but other cards on hosts had workloads. No frequency lock or power configuration changes.',
    'BW1100 is the user-confirmed project device label; the raw Hygon runtime device name reports BW1200 and is preserved verbatim in summary.device, not silently corrected.',
    'Profiler can perturb timings. Profiled rollout wall time is recorded separately and does not replace the earlier unprofiled 3-round performance results.',
    'Peak-normalized MFU ratio is a diagnostic candidate for relative efficiency; equivalent-compute definition and contractual acceptance timing need agreement. Not a formal 60% pass declaration.',
    'This run extends timing scope, not model/precision. Previous 40-step accuracy results retain their original labels; fresh final outputs are retained without relabeling old accuracy as newly computed.',
    'H100 raw traces and metadata collected locally; new H100 final predictions remain at /data/zyx/graphcast-mfu40-20261030/h100-profile/run/. Baseline predictions and all previous results preserved.',
    'No GitHub push or testuser/SCNET delivery sync; Notion work-step list not changed.']
if 'native' in result:
    result['limits'].append('Native vendor JAX restored from all 40 frozen Python distributions in a different base image with the same DTK-26.04-DCC2602-0317 release. Frozen Oct22 model/experiment code and RNE library used unchanged. This is a recovered-runtime remeasurement, not proof that the missing original image was restored bit-for-bit.')
    result['native_recovery']=dict(environment=read(ROOT/'native-profile/environment-probe.json'),launch=read(ROOT/'native-profile/launch.json'),packages=read(ROOT/'native-recovery/versions.json'))
    accuracy=read(ROOT/'native-accuracy/results/summary.json')
    assert accuracy['all_nine_internal_pass'] and accuracy['all_nine_stage_pass']
    primary=[r for r in accuracy['rows'] if r['primary_pair']]
    result['native_accuracy']=dict(reference=accuracy['reference'],primary_overall_percent=[r['overall_mean_percent'] for r in primary],primary_z500_percent=[r['z500_percent'] for r in primary],all_nine_internal_pass=True,all_nine_stage_pass=True)
write(ROOT/'summary.json',result)
csvout(ROOT/'kernel-events.csv',all_details)
csvout(ROOT/'rounds.csv',[dict(platform=p,**r) for p in platforms for r in result[p]['rounds']])
write(ROOT/'source-manifest.json',sources)
write(ROOT/'validation.json',dict(status='passed',checks=[
    'Same source closure, prepared dataset metadata, original input case hash and FP32 storage on both platforms',
    'Three rounds per platform, 40 steps each, five excluded warmup steps, unchanged JIT cache',
    'Hygon 135-step routing audit: 84 GEMMs and 79 fused activations per step',
    'Hygon each captured round: 3360 GEMMs, 3240 XF32 MX kernels, 3560 standalone RNE kernels',
    'All 42783 HIP kernels assigned to exactly one measured capture; zero warmup kernels included',
    'Three SQLite files passed integrity check; each H100 capture has 3560 TF32 kernels and no independent RNE',
    'Per-round useful FLOPs recomputed from all 3360 real GEMM shape entries, total {}'.format(F_40),
    'MFU from actual 40-step interval unions, not extrapolated single-step time']))
lines=['# GraphCast｜完整40步卡上MFU补测｜2026-10-30','',
    '本次只扩展profile采集长度：每平台3轮，每轮预热5步后重置输入、独立自回归40步。原baseline、融合策略和冻结输出保留，既定Notion工作步骤未增改。','',
    '| 路径 | 40步GPU活跃时间中位数(s) | 稠密峰值(TFLOP/s) | 40步MFU估算 | 相对H100的MFU比值 |',
    '|---|---:|---:|---:|---:|']
for p in platforms:
    x=result[p];lines.append('| {} | {:.6f} | {} | {:.4f}% | {:.4f}% |'.format(x['label'],x['median_gpu_active_seconds'],x['peak_tflops'],x['median_mfu_percent'],100*x['median_mfu_percent']/result['h100']['median_mfu_percent']))
lines += ['', '相对H100的MFU比值＝本路径三轮中位MFU／H100三轮中位MFU×100%，使用未四舍五入值计算；H100自身为100%。该值是峰值算力归一化比较，不是实际推理速度比，也不直接作为正式阶段验收判定。']
lines += ['', '分子：40×28,784,948,924,416 = 1,151,397,956,976,640 FLOPs（1151.39795697664 TFLOPs）。GPU时间取每轮全部计算内核区间并集，包含RNE、Gather、Scatter与自回归更新的设备计算，重叠不重复计入；拷贝/CPU/空隙另列。',
    '', '原始卡上吞吐比：{:.4f}%；按峰值归一化的MFU比：{:.4f}%。后者是候选效率口径，不能据此直接宣布阶段60%验收通过。'.format(result['comparison']['raw_gpu_active_throughput_ratio_percent'],result['comparison']['mfu_ratio_dcu_over_nv_percent']),
    '', 'H100峰值采用[NVIDIA H100架构白皮书v1.04表1](https://resources.nvidia.com/en-us-hopper-architecture/nvidia-h100-tensor-c)的SXM5稠密TF32 494.7 TFLOP/s；同表989.4为稀疏值，不使用。海光继续采用用户指定170.3936。原件已保存为nvidia-h100-architecture-v1.04.pdf。',
    '', '## 原始测量','', '| 平台 | 轮次 | GPU活跃(s) | kernel次数 | MFU(%) | profiler期间rollout墙钟(s) |', '|---|---:|---:|---:|---:|---:|']
for p in platforms:
    for r in result[p]['rounds']:lines.append('| {} | {} | {:.6f} | {} | {:.4f} | {:.6f} |'.format(p,r['round'],r['kernel_active_seconds'],r['kernel_count'],r['mfu_percent'],r['rollout_wall_seconds']))
lines += ['', '## 算子主表','', '新版Excel的主表依次列JAX（同一FlagOS容器native）、FlagOS fl_only、加速比JAX÷FlagOS、FlagTree覆盖（含标记用例/总例）及执行说明。大于1为fl_only更快，小于1为更慢。Linear 15/15仅代表用例包含FlagTree的bias等部分；GEMM仍为rocBLAS，独立RNE为HIP。旧10月22日原生JAX基线数字保留在历史版本，不混作当前同容器native对照。',
    '', '独立算子计时为同步调用墙钟，模型MFU使用设备trace，两者不混用。16类的原始采样/HLO与校验见上一版目录flagos-inference-final-20261030，新的列结构见operators.csv。',
    '', '## 复算与原始材料','',
    '- `python3 calculate.py`：核对采集边界、有效GEMM量、内核数并复算MFU。',
    '- `python3 build_tables.py`：生成新版Excel；原版Excel不覆盖。',
    '- `h100-profile/model-trace.1/2/3.sqlite`及`.nsys-rep`：三轮Nsight原始事件。',
    '- `hygon-profile/trace.json`及`trace.db`：三轮HIPPROF原始事件；`run/capture-ranges.json`用于逐轮范围校验。',
    '- `kernel-events.csv`、`rounds.csv`、`summary.json`、`validation.json`：事件、逐轮值与复算证据。',
    '- `profile40.py`与各平台启动脚本、instrumentation.diff：仅采集范围与输出留存的改动；官方模型源文件未改。',
    '', '## 适用边界','']+['- '+s for s in result['limits']]
(ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
if 'native_accuracy' in result:
    acc=result['native_accuracy']
    extra=['','## 原生JAX补测与环境恢复',
        '', '原生JAX三轮40步端到端中位数为{:.6f}秒（带profiler）；历史未带profiler值为33.311110秒。卡上时间中位数{:.6f}秒；FlagOS相较原生的卡上加速比为{:.6f}倍，即耗时约{:.3f}倍。'.format(result['native']['median_profiled_wall_seconds'],result['native']['median_gpu_active_seconds'],result['comparison']['flagos_speedup_over_native_gpu'],1/result['comparison']['flagos_speedup_over_native_gpu']),
        '', '以10月29日冻结H100 TF32为精度参考，同轮整体误差{:.6f}%～{:.6f}%，Z500为{:.6f}%～{:.6f}%；全部9组配对通过内部5%线与阶段10%线。详情见native-accuracy/results/。'.format(min(acc['primary_overall_percent']),max(acc['primary_overall_percent']),min(acc['primary_z500_percent']),max(acc['primary_z500_percent'])),
        '', '原镜像缺失；使用本机已有DTK镜像的基础运行时、离线冻结的40项Python包和原版JAX四个wheel恢复独立venv。DTK版本与原记录均为DTK-26.04-DCC2602-0317。原实验源码逐文件SHA及原RNE动态库SHA通过校验。基础镜像ID不同，不能声称逐字节恢复原镜像；native-profile/launch.json及environment-probe.json保存差异。',
        '', '采集器启动失败留存在native-profile-startup-failed/：最小用例确认其对继承的Python/临时目录环境敏感。隔离采集器启动环境、在子进程恢复相同模型环境后采集完成；模型代码与精度策略没有修改。',
        '', '复现入口：launch-native.py、run-native.sh、native-profile-entry.sh；离线环境恢复命令见native-recovery/setup-command.json。原生完整trace位于native-profile/。独立复算核对运行verify_results.py。']
    with (ROOT/'REPORT.md').open('a') as f:f.write('\n'.join(extra)+'\n')
print(json.dumps({p:{k:v for k,v in result[p].items() if k!='rounds'} for p in platforms},ensure_ascii=False,indent=2))
print(json.dumps(result['comparison'],ensure_ascii=False,indent=2))
