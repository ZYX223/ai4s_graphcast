"""Build traceable inference tables. No GPU runs, model edits or remote writes."""
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import statistics
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OLD = ROOT / 'flagos-logical-operators-20261029'
MODEL = ROOT / 'flagos-rne-gemm-ex-20261030'
PROFILE = MODEL / 'model-single-r1'
H100 = ROOT / 'h100-gpu41-tf32-20261029'
HIST = ROOT / 'rne-model-fusion-20261022'
ATT = HERE / 'attachments'
ATT.mkdir(exist_ok=True)
sources = {}
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
    sources[str(p)] = sha(p)
    return json.loads(p.read_text())
def dump(p, obj): p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
def csvout(name, rows):
    with (HERE/name).open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def near(a,b): assert math.isclose(a,b, rel_tol=1e-9, abs_tol=1e-5), (a,b)

mapping = [('linear','Linear','矩阵乘与偏置'),('concatenate','Concat','张量拼接'),('layer_norm','LayerNorm','归一化'),('scatter-add','ScatterAdd / SegmentSum','图聚合'),('silu','SiLU','激活'),('gather','Gather','图索引'),('add','Add','逐元素'),('mul','Mul','逐元素'),('sub','Sub','逐元素'),('div','Div','逐元素'),('slice','Slice','切片'),('dynamic_slice','DynamicSlice','切片'),('broadcast_in_dim','Broadcast','布局 / 广播'),('reshape','Reshape','布局 / 视图'),('transpose','Transpose','布局 / 视图'),('squeeze','Squeeze','布局 / 视图')]
labels = {k:l for k,l,c in mapping}
original = read(OLD/'original-table.json')
old_ids = {r['id'] for r in original['rows']}
combined = {}; details = []
import numpy as np
for mode in ['native','fl_only']:
    old = read(OLD/('bench-'+mode)/'report.json')
    new = read(HERE/('linear-'+mode)/'report.json')
    manifest = read(HERE/('linear-'+mode)/'source-manifest.json')
    assert new['status'] == manifest['status'] == 'passed'
    assert len(new['rows']) == 15 and all(r['operator']=='linear' for r in new['rows'])
    assert old['inventory_sha256'] == new['inventory_sha256']
    rows = [r for r in old['rows'] if r['operator']!='linear'] + new['rows']
    assert len(rows)==108 and {r['id'] for r in rows}==old_ids and sum(r['calls'] for r in rows)==549
    combined[mode] = rows
    for r in rows:
        assert r['status']=='passed' and r['finite_full_output']
        samples = r['round_samples_ms']
        assert len(samples)==3 and all(len(a)==20 for a in samples)
        medians = [statistics.median(a) for a in samples]
        near(statistics.median(medians), r['median_ms'])
        linear = r['operator']=='linear'
        if linear: assert r['initial_router_delta']==[1,1,0]
        folder = HERE/('linear-'+mode) if linear else OLD/('bench-'+mode)
        sf = folder/(r['id']+'-samples.npz')
        sources[str(sf)] = sha(sf)
        arr = np.load(str(sf))
        np.testing.assert_allclose(arr['actual'],arr['reference'],atol=r['atol'],rtol=r['rtol'])
        near(float(np.max(np.abs(arr['actual']-arr['reference']))),r['max_abs_error'])
        hf = folder/(r['id']+'-hlo.txt'); hlo = hf.read_text(); sources[str(hf)] = sha(hf)
        assert r['flagtree_hlo'] == ('xla.flagtree' in hlo)
        details.append(dict(backend=mode,case_id=r['id'],operator=labels[r['operator']],
            calls=r['calls'],input_shapes=json.dumps(r['input_shapes']),output_shape=json.dumps(r['output_shape']),
            round1_ms=medians[0],round2_ms=medians[1],round3_ms=medians[2],median_ms=r['median_ms'],
            weighted_ms=r['median_ms']*r['calls'],oracle_samples=r['oracle_samples'],max_abs_error=r['max_abs_error'],
            atol=r['atol'],rtol=r['rtol'],status='passed',flagtree_hlo=r['flagtree_hlo'],
            date='2026-10-30' if linear else '2026-10-29',physical_gpu=9 if linear else 8,
            source_report=str(folder/'report.json'),sample_file=str(sf),hlo_file=str(hf)))

operators = []
for key,label,category in mapping:
    groups = {m:[r for r in combined[m] if r['operator']==key] for m in combined}
    orig = original['summary'][key]
    assert all(len(g)==orig['cases'] and sum(r['calls'] for r in g)==orig['calls'] for g in groups.values())
    native,flag = [sum(r['calls']*r['median_ms'] for r in groups[m]) for m in ['native','fl_only']]
    operators.append(dict(operator=label,category=category,historical_jax_ms=orig['weighted_ms'],
        flagos_native_ms=native,flagos_fl_only_ms=flag,native_over_fl_only=native/flag,
        forward_calls=orig['calls'],shape_cases=orig['cases'],passed_native=len(groups['native']),passed_fl_only=len(groups['fl_only']),
        flagtree_hlo_cases=sum(r['flagtree_hlo'] for r in groups['fl_only']),
        measurement='2026-10-30 GPU9补测' if key=='linear' else '2026-10-29 GPU8复用'))
csvout('operators.csv',operators); csvout('case-details.csv',details)

results=read(MODEL/'RESULTS.json'); prof=read(PROFILE/'profile-summary.json')
run=read(MODEL/'model-40-r1/run/report.json'); nv=read(H100/'summary.json')
hist_result=read(HIST/'summary.json')
acc=read(MODEL/'h100-comparison-r2/summary.json')
trace=read(PROFILE/'trace.json')
kernels=[e for e in trace['traceEvents'] if e.get('cat')=='HIPOPS' and e.get('ph')=='X']
assert len(kernels)==338
intervals=sorted((int(e['args']['BeginNs']),int(e['args']['EndNs'])) for e in kernels)
union=0;lo,hi=intervals[0]
for a,b in intervals[1:]:
    if a>hi: union+=hi-lo;lo,hi=a,b
    else: hi=max(hi,b)
union+=hi-lo
near(union/1e6,prof['gpu_kernel_busy_union_ms'])
near(sum(b-a for a,b in intervals)/1e6,prof['gpu_kernel_sum_ms'])
event_rows=[dict(index=i+1,name=e['name'],begin_ns=e['args']['BeginNs'],end_ns=e['args']['EndNs'],duration_ms=int(e['args']['DurationNs'])/1e6,stream=e['tid']) for i,e in enumerate(kernels)]
csvout('attachments/kernel-events.csv',event_rows)

# Verify each non-BLAS/non-RNE kernel name against FlagTree HLO annotations.
main_hlo=PROFILE/'run/optimized-hlo.txt'
aux=[PROFILE/'hlo'/n for n in ['module_0003.jit__threefry_seed.gfx938_gpu_after_optimizations.txt','module_0005.jit__threefry_split.gfx938_gpu_after_optimizations.txt','module_0007.jit__unstack.gfx938_gpu_after_optimizations.txt']]
texts={str(p):p.read_text() for p in [main_hlo]+aux}
for p in [main_hlo]+aux: sources[str(p)]=sha(p)
backend_rows=[]
for r in prof['kernels']:
    matches=[]
    if r['category']=='gemm': backend='rocBLAS'
    elif r['category']=='rne': backend='独立HIP RNE'
    else:
        name=r['name'][:-5] if r['name'].endswith('_impl') else r['name']
        pattern=r'\s*(?:ROOT )?%'+re.escape(name).replace('_',r'[._]')+r' ='
        for path,txt in texts.items():
            for no,line in enumerate(txt.splitlines(),1):
                if re.match(pattern,line) and 'xla.flagtree' in line and '__triton' in line:
                    matches.append(path+':'+str(no))
        assert matches, r['name']
        backend='FlagTree'
    backend_rows.append(dict(name=r['name'],backend=backend,calls=r['calls'],total_ms=r['total_ms'],category=r['category'],hlo_evidence='; '.join(matches)))
csvout('attachments/kernel-backend-evidence.csv',backend_rows)
assert sum(r['calls'] for r in backend_rows if r['backend']=='FlagTree')==165
category_rows=[dict(category=k,**v) for k,v in prof['categories'].items()]
csvout('kernel-categories.csv',category_rows)
gemms_path=PROFILE/'gemm.jsonl'; sources[str(gemms_path)]=sha(gemms_path)
gemms=[json.loads(l) for l in gemms_path.read_text().splitlines()]
gemms=[g for g in gemms if g['label']=='round1-rne_xf32']
assert len(gemms)==len({g['layer'] for g in gemms})==84
assert all(g['rne_inputs'] and g['mode']==1 and g['status']==0 for g in gemms)
flops_rows=[dict(layer=g['layer'],m=g['m'],n=g['n'],k=g['k'],useful_flops=2*g['m']*g['n']*g['k'],api=g['api'],activation_rne_fused=g['activation_rne_skipped']) for g in gemms]
csvout('gemm-flops.csv',flops_rows)
flops=sum(r['useful_flops'] for r in flops_rows)
assert flops==28784948924416
peak=170.3936; mfu=flops/(union/1e9*peak*1e12)*100
near(mfu,results['performance']['candidate_mfu_single_step_percent'])
assert acc['input_case_match'] and acc['all_nine_internal_pass'] and acc['all_nine_stage_pass'] and len(acc['rows'])==9
fields_path=MODEL/'h100-comparison-r2/variable-level-details.csv'
sources[str(fields_path)]=sha(fields_path)
with fields_path.open() as f: fields=list(csv.DictReader(f))
assert len(fields)==9*78
for pair in acc['rows']:
    rr=[r for r in fields if int(r['candidate_round'])==pair['candidate_round'] and int(r['h100_round'])==pair['h100_round']]
    assert len(rr)==78 and len({(r['variable'],r['level_hpa']) for r in rr})==78
    for r in rr: near(100*float(r['mae'])/float(r['reference_mean_abs']),float(r['relative_error_percent']))
    near(statistics.mean(float(r['relative_error_percent']) for r in rr),pair['overall_mean_percent'])
    z=next(r for r in rr if r['variable']=='geopotential' and int(r['level_hpa'])==500)
    near(float(z['relative_error_percent']),pair['z500_percent'])
for r in run['rows']:
    assert len(r['apply_seconds'])==40 and r['jit_cache_before']==r['jit_cache_after']
    near(sum(r['apply_seconds']),r['apply_sum_seconds'])
near(statistics.median(r['rollout_wall_seconds'] for r in run['rows']),results['performance']['candidate_wall_median_seconds'])
near(statistics.median(r['rollout_wall_seconds'] for r in nv['rows']),results['performance']['h100_wall_median_seconds'])
round_rows=[]
for backend,rr in [('FlagOS fl_only',run['rows']),('H100 TF32',nv['rows'])]:
    for r in rr: round_rows.append(dict(backend=backend,round=r['round'],measured_steps=40,warmup_steps=5,rollout_wall_seconds=r['rollout_wall_seconds'],synchronized_apply_sum_seconds=r['apply_sum_seconds']))
csvout('model-rounds.csv',round_rows)

for p in [PROFILE/'kernel-details.csv', PROFILE/'trace.hipkernel.csv', PROFILE/'trace.hiptrace.csv',PROFILE/'profile-summary.json',MODEL/'h100-comparison-r2/pairs.csv',MODEL/'h100-comparison-r2/variable-level-details.csv',HIST/'mfu/peak-reference.json']:
    sources[str(p)]=sha(p);shutil.copy2(str(p),str(ATT/p.name))
for name in ['gpu-time.json','kernel-details.csv','stats_cuda_gpu_kern_sum.csv']:
    p=H100/'profile-r1'/name
    sources[str(p)]=sha(p);shutil.copy2(str(p),str(ATT/('h100-'+name)))

limits=[
    '算子表为历史JAX列＋FlagOS容器native/fl_only对照；不是FlagGems算子库耗时。native与fl_only均使用同一FlagOS镜像，不能将native称为已恢复旧海光baseline。',
    '每个后端108用例：93项非Linear复用10月29日GPU8，15项Linear为10月30日GPU9补测；不是本次重跑全部108项。硬件同型号，跨日跨卡且机器有其他任务，非严格交错A/B。',
    '独立算子计时为3轮，每轮5次预热＋20次同步JAX墙钟；取各轮中位数的中位数，再乘前向调用次数。包含调用、同步与审计开销，不是纯GPU时间，不能相加推导整模型耗时。',
    'Linear独测含两输入RNE＋rocBLAS＋bias，不使用整模型79处激活生成融合；其余算子为FP32。K=4可回退原生FP32，RNE仍执行。',
    '算子正确性使用每例最多258个输出元素的独立参考＋全输出有限性；Linear参考先RNE输入再FP64点积，不是与未经舍入FP32输出的误差；全模型另以H100的40步末帧评价。',
    '整模型单步338次kernel：165次FlagTree、84次rocBLAS、89次独立HIP RNE。81次GEMM为XF32 MX，3次K=4原生FP32回退。79处激活RNE已融合，不重复加到独立RNE耗时。',
    '模型40步墙钟含预加载后传输、自回归更新和调用；不含加载、编译、预热、校验和写盘。三轮每轮预热5步后重置输入；profile单独采集，不混入三轮墙钟结果。',
    'MFU仅单步GPU内核区间并集、GEMM FLOPs近似；不含内核间空隙、CPU及拷贝。没有完整40步卡上实测MFU，不能将单步乘40冒充实测。',
    'H100参考为gpu41于10月29日官方JAX TF32（无显式RNE）；同一预训练权重/输入/统计，独立40步，+240h末帧6变量×13层等权平均；Z500单列，分母H100。不代表所有时效/变量或天气真值精度。',
    '170.3936 TFLOP/s为用户指定截图峰值；设备映射与频率依据、H100同精度稠密峰值、等效算力公式及验收计时口径待确认。阶段60%不等于自身MFU60%；当前不作正式效率达标判定。',
    '保留baseline与修复前失败记录。新结果未推送GitHub、未同步testuser/SCNET；按用户最后统一同步。训练反向与loss不在本次推理结论范围。']
summary=dict(date='2026-10-30',status='completed_inference_evidence_consolidation',
    operator_cases_per_backend=108,new_linear_cases_per_backend=15,reused_non_linear_cases_per_backend=93,
    all_cases_passed=True,logical_forward_calls=549,operators=operators,
    model=results,profile={k:v for k,v in prof.items() if k not in ['kernels','host_apis']},
    recalculated=dict(kernel_union_ns=union,gemm_useful_flops=flops,peak_tflops=peak,single_step_mfu_percent=mfu),
    accuracy_pairs=acc['rows'],limits=limits,sources=sources)
dump(HERE/'summary.json',summary)

sys.path.insert(0,str(ROOT/'inference-operators-20261022/operator-workbook/deps'))
import xlsxwriter
import openpyxl
book_path=HERE/'GraphCast_BW1100_FlagOS_推理算子与精度性能_20261030.xlsx'
book=xlsxwriter.Workbook(str(book_path),{'strings_to_urls':False,'strings_to_formulas':False})
base={'font_name':'Microsoft YaHei','font_size':10,'text_wrap':True,'valign':'vcenter'}
text_fmt=book.add_format(base); num=book.add_format(dict(base,num_format='0.000000'))
head=book.add_format(dict(base,bold=True,bg_color='#164E63',font_color='#FFFFFF'))
note=book.add_format(dict(base,bg_color='#E0F2FE'))
sheets={}
def table(name,heads,rows,widths,description,height=42):
    ws=book.add_worksheet(name);ws.hide_gridlines(2)
    ws.merge_range(0,0,0,len(heads)-1,description,note);ws.set_row(0,66)
    ws.set_row(1,42);ws.freeze_panes(2,1)
    for i,width in enumerate(widths): ws.set_column(i,i,width,text_fmt)
    ws.add_table(1,0,len(rows)+1,len(heads)-1,{'columns':[{'header':h,'header_format':head} for h in heads],'data':rows,'style':'Table Style Medium 2'})
    for i,row in enumerate(rows,2):
        ws.set_row(i,height)
        for j,v in enumerate(row):
            if isinstance(v,float):ws.write_number(i,j,v,num)
    ws.set_landscape();ws.set_paper(9);ws.fit_to_pages(1,0);ws.repeat_rows(0,1)
    sheets[name]=(len(rows)+2,len(heads))
    return ws
table('逻辑算子16类',['逻辑算子','分类','历史JAX(ms)\n10月22日','FlagOS native(ms)','FlagOS fl_only(ms)','native/fl_only','前向调用','形状数','native通过','fl_only通过','FlagTree HLO用例数','本次证据'],
    [[r[k] for k in ['operator','category','historical_jax_ms','flagos_native_ms','flagos_fl_only_ms','native_over_fl_only','forward_calls','shape_cases','passed_native','passed_fl_only','flagtree_hlo_cases','measurement']] for r in operators],
    [30,18,21,22,22,18,12,12,12,12,19,29],
    '16类／108形状／549次逻辑调用。两列FlagOS是同一容器的native与fl_only；GEMM仍为rocBLAS。独立同步墙钟加权时间，不可求和当模型耗时。15个Linear补测于GPU9，其余93项复用GPU8；明细与限制见后续页。')
table('用例明细216行',['后端','用例ID','算子','频次','输入shape','输出shape','轮1(ms)','轮2(ms)','轮3(ms)','中位数(ms)','加权(ms)','样本数','最大绝对误差','atol','rtol','状态','FlagTree HLO','日期','物理GPU','原报告','采样文件','HLO文件'],
    [[r[k] for k in list(details[0])] for r in details],
    [14,32,29,9,70,30,18,18,18,18,18,12,21,15,15,12,16,17,12,85,85,85],
    '每后端108用例；3轮×20次同步JAX墙钟，每轮预热5次。最多258个输出抽样与独立参考比较＋全输出有限性；非完整输出相对误差验收。HLO中无FlagTree的布局/view用例保留，不虚报内核覆盖。',55)
perf=results['performance']; old_mfu=read(HIST/'mfu/summary.json')
old_fused=next(r for r in old_mfu['rows'] if r['variant']=='fused')
model_rows=[['H100 JAX TF32，无显式RNE','2026-10-29','gpu41 / H100单卡',perf['h100_wall_median_seconds'],perf['h100_apply_sum_median_seconds'],perf['h100_single_step_gpu_active_ms'],None,None,'当前精度reference；峰值依据待确认，MFU留空'],
    ['旧海光JAX融合RNE＋XF32','2026-10-22','BW1100 GPU12',hist_result['performance']['fused']['rollout_median_seconds'],hist_result['performance']['fused']['apply_sum_median_seconds'],old_fused['gpu_kernel_busy_ms'],170.3936,old_fused['theoretical_mfu_percent'],'历史对照，非本次同环境A/B；时间来源见历史报告'],
    ['FlagOS JAX fl_only RNE＋XF32','2026-10-30','BW1100 GPU8',perf['candidate_wall_median_seconds'],perf['candidate_apply_sum_median_seconds'],union/1e6,peak,mfu,'本次候选；MFU为单步GEMM近似，非40步实测']]
table('整模型性能',['路径','实验日期','设备','40步墙钟中位数(s)','40步同步调用累计中位数(s)','单步GPU活跃时间(ms)','峰值(TFLOP/s)','单步MFU估算(%)','边界'],model_rows,[38,17,25,26,31,28,22,24,85],
    '40步：每轮5步预热后重置输入，3轮独立自回归；加载/编译/校验/写盘排除。单步GPU时间来自独立profile所有内核区间并集。原始吞吐比（墙钟24.6899%、卡上23.3588%）不是同等算力效率的正式判定。',80)
accuracy_rows=[[r['candidate_round'],r['h100_round'],'主要同轮' if r['primary_pair'] else '交叉波动检查',r['overall_mean_percent'],r['z500_percent'],r['z500_mae_m'],r['z500_rmse_m'],'通过' if r['internal_under_5_percent'] else '未通过','通过' if r['stage_at_most_10_percent'] else '未通过'] for r in acc['rows']]
table('40步末帧精度',['候选轮次','H100轮次','配对用途','整体相对MAE(%)','Z500相对MAE(%)','Z500 MAE(m)','Z500 RMSE(m)','内部两项<5%','阶段两项≤10%'],accuracy_rows,[15,15,25,24,24,23,23,20,23],
    '同一预训练权重/输入/统计；各自独立40步；只比较第40步＋240h。每项mean(abs(candidate-reference))/mean(abs(reference))，H100作分母，6变量×13层等权平均；Z500单列。不是40步误差均值或天气真值误差。')
cn={'gather_fusion':'Gather相关融合','scatter_fusion':'Scatter相关融合','gemm':'rocBLAS GEMM','other_device_kernel':'其他设备内核','layer_norm_fusion':'LayerNorm相关融合','rne':'独立HIP RNE'}
table('单步GPU分类',['类别','kernel调用','累计(ms)','占内核累计(%)','边界'],[[cn[k],v['calls'],v['total_ms'],v['percent_of_kernel_sum'],'融合内含其他逻辑操作/RNE，不作单一算子归因' if 'fusion' in k else ('不包含已融合的79处激活RNE' if k=='rne' else '原始kernel与HLO附件可追溯')] for k,v in prof['categories'].items()],
    [32,17,23,24,85], '整模型单步：338次kernel；累计1349.384725ms，区间并集1349.381105ms，首尾跨度1502.763394ms。分类占比用累计时间，MFU用区间并集；kernel详情为独立CSV附件。',55)
method_rows=[['模型','官方JAX GraphCast Operational，0.25°，721×1440，13层，batch=1；源码08cf73625c9d12bd9aaa038868bcb2fe488f2a22'],
    ['镜像',results['image']+' / '+results['image_id']],['候选库SHA256',results['library_sha256']],
    ['MFU公式','F_useful/(T_card×P_total)×100%；本次 {} / ({}×170.3936e12)×100% = {:.8f}%'.format(flops,union/1e9,mfu)],
    ['有效FLOPs','84个实际逻辑GEMM，Σ2MNK=28.784948924416 TFLOPs；未计逐元素/归约，属于近似。重算依据gemm-flops.csv'],
    ['GPU计时','使用原始HIPOPS BeginNs/EndNs整数纳秒重算并集；所有内核包含在内，不含HIPCOPY/CPU/空隙；kernel-events.csv含338条'],
    ['后端证据','kernel-backend-evidence.csv：165次FlagTree HLO匹配、84次rocBLAS、89次独立HIP RNE；单独完整HLO保留在原实验目录'],
    ['本次任务完成边界','完成推理接入与证据整理，不等于12月底整体阶段验收完成。尚需性能优化；正式60%效率判定与完整40步卡上MFU仍待后续确认/采集。'],
    ['材料目录',str(HERE)],['模型原始证据',str(MODEL)],['H100参考目录',str(H100)],['源码',results['code']]] + [['限制'+str(i+1),s] for i,s in enumerate(limits)]
table('方法与来源',['项目','说明'],method_rows,[28,145],'阅读时先确认计时口径与参考对象。代码、原始输出及失败记录保留；本次未推送或同步。详细文件SHA见source-manifest.json。',63)
book.close()
wb=openpyxl.load_workbook(str(book_path),data_only=True,read_only=True)
assert wb.sheetnames==list(sheets)
for name,dimensions in sheets.items(): assert (wb[name].max_row,wb[name].max_column)==dimensions
for i,r in enumerate(operators,3):
    for col,key in [(3,'historical_jax_ms'),(4,'flagos_native_ms'),(5,'flagos_fl_only_ms')]:near(wb['逻辑算子16类'].cell(i,col).value,r[key])
wb.close()
baseline=Path('/home/zhangyuxin/zyx/graphcast-sugon/kernels/xf32_rne/xf32_rne_router.cc')
assert sha(baseline)=='f9177d33c82c842461a43af95f371fd5b32a259ba3e6283b31aa61256a86b8e8'
dump(HERE/'source-manifest.json',sources)
dump(HERE/'validation.json',dict(status='passed',checks=['108 unique cases and 549 calls per backend','30 fresh Linear cases pass with repaired routing','216 sample files independently rechecked with original tolerances','all 216 timing medians recomputed from 60 samples','338 kernel intervals union recomputed from integer timestamps','165 FlagTree calls matched to annotated HLO','84 GEMMs and useful FLOPs independently recomputed','40-step timing medians and 9 accuracy pairs checked','workbook rows/columns/numeric cells read back','original baseline source hash unchanged'],workbook=str(book_path),workbook_sha256=sha(book_path),sheets=sheets))
lines=['# GraphCast FlagOS推理验证汇总｜2026-10-30','',
    '推理任务的算子表、kernel附件、40步精度性能和MFU依据已汇总。两种后端各108用例通过：本次在GPU9补测全部15个Linear形状，其余93个非Linear用例复用10月29日GPU8记录。原baseline与修复前失败记录保留。','',
    '## 逻辑算子耗时','',
    '| 逻辑算子 | 历史JAX ms | FlagOS native ms | FlagOS fl_only ms | 前向调用 | 形状数 |','|---|---:|---:|---:|---:|---:|']
for r in operators: lines.append('| {} | {:.3f} | {:.3f} | {:.3f} | {} | {} |'.format(r['operator'],r['historical_jax_ms'],r['flagos_native_ms'],r['flagos_fl_only_ms'],r['forward_calls'],r['shape_cases']))
lines += ['', '独立算子采用同步JAX墙钟，类别耗时按原图调用次数加权；不能相加当整模型延迟。Linear包含独立两输入RNE与bias，不含79处输入生成融合。原生历史列不是当前容器的native列。', '',
    '## 整模型与MFU','',
    '- FlagOS三轮40步墙钟：58.766360、58.523623、58.616376秒，中位数58.616376秒；H100参考14.472323秒。',
    '- 主要同轮配对整体误差0.515465%–0.603645%，Z500为0.005827%–0.007779%；全部9组通过内部<5%和阶段≤10%。',
    '- 单步卡上内核活跃时间 {:.6f} ms，MFU {:.6f}%；分子28.784948924416 TFLOPs，分母使用170.3936 TFLOP/s。'.format(union/1e6,mfu),
    '- Gather相关融合30.86%、Scatter相关融合26.92%，合计57.78%；GEMM23.39%；独立RNE1.12%，不包括融合中的RNE成本。',
    '- 后端覆盖：165次FlagTree、84次rocBLAS、89次HIP RNE。81次GEMM为XF32 MX，3次K=4为原生FP32回退；显式RNE均保留。',
    '- 同等算力60%效率的正式结论尚待峰值、设备等效与计时口径确认；当前单步MFU不代表完整40步卡上MFU。','',
    '## 文件入口','',
    '- [Excel总表]('+book_path.name+')：逻辑算子、216行用例、模型性能、40步末帧精度、GPU分类、方法与来源，共6个sheet。',
    '- `operators.csv`、`case-details.csv`：可计算表格；`summary.json`：机器可读汇总。',
    '- `attachments/kernel-events.csv`：338个原始kernel区间；`kernel-details.csv`：按kernel汇总；`kernel-backend-evidence.csv`：后端与HLO位置；精度逐变量逐层明细同目录。',
    '- `gemm-flops.csv`：84次GEMM有效FLOPs；`validation.json`与`source-manifest.json`：校验与来源SHA。',
    '- 原始模型输出/环境/复现脚本：`'+str(MODEL)+'`；H100参考：`'+str(H100)+'`。',
    '- 新补测命令在`run-linear.sh`与`replay-linear.py`；在已准备的容器中依次执行native、fl_only，固定GPU9（PCI 0000:88:00.0），需先确认空闲。脚本拒绝覆盖已有输出；再次复测须复制脚本到新的实验目录，并将run-linear.sh的replay_root改为该新目录。冻结的inventory和候选库只读复用。',
    '- 离线重新汇总：`python3 '+str(HERE/'consolidate.py')+'`。','', '## 测量边界与后续','']
lines += ['- '+s for s in limits]
lines += ['', '后续性能工作优先检查FlagTree的Gather/Scatter生成内核。完整40步卡上采集及正式效率判定，在对应测量约定明确后安排；训练最小验证由既有任务单独推进。本次未新增任务或统一同步。','']
(HERE/'REPORT.md').write_text('\n'.join(lines))
print(json.dumps(dict(status='passed',workbook=str(book_path),linear=operators[0],mfu=mfu),ensure_ascii=False))
