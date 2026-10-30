"""Independently sweep trace endpoints and verify workbook/source preservation."""
import collections,csv,hashlib,json,sys
from pathlib import Path
root=Path(__file__).resolve().parent
summary=json.loads((root/'summary.json').read_text())
events=collections.defaultdict(list)
for row in csv.DictReader((root/'kernel-events.csv').open(encoding='utf-8-sig')):
    events[row['platform'],int(row['round'])]+=[(int(row['start_ns']),1),(int(row['end_ns']),-1)]
for (platform,index),ev in events.items():
    active=0;total=0;last=None
    for timestamp,delta in sorted(ev):
        if active:total+=timestamp-last
        active+=delta;last=timestamp
    assert active==0
    row=summary[platform]['rounds'][index-1]
    assert abs(total/1e9-row['kernel_active_seconds'])<1e-9
    calculated=summary['useful_gemm_flops_per_40_steps']/(total*1e-9*summary[platform]['peak_tflops']*1e12)*100
    assert abs(calculated-row['mfu_percent'])<1e-10
sys.path.insert(0,str(root.parent/'inference-operators-20261022/operator-workbook/deps'))
import openpyxl
old=openpyxl.load_workbook(str(root.parent/'flagos-inference-final-20261030/GraphCast_BW1100_FlagOS_推理算子与精度性能_20261030.xlsx'),data_only=True)
new=openpyxl.load_workbook(str(root/'GraphCast_JAX-FlagOS_推理对照_20261030.xlsx'),data_only=True)
preserved=[]
for name in old.sheetnames:
    if name=='逻辑算子16类':continue
    before=list(old[name].values);after=list(new[name].values)
    assert len(before)==len(after),name
    for i,(a,b) in enumerate(zip(before,after),1):
        for j,(x,y) in enumerate(zip(a,b),1):
            if name=='方法与来源' and j==2 and a[0] in ['MFU公式','有效FLOPs','GPU计时','本次任务完成边界','材料目录','限制1','限制8','限制10']:continue
            if name=='整模型性能' and (i,j)==(3,9):continue
            assert x==y,(name,i,j)
    preserved.append(name)
ops=list(csv.DictReader((root/'operators.csv').open(encoding='utf-8-sig')))
assert len(ops)==16
for i,r in enumerate(ops,3):
    assert abs(new['逻辑算子16类'].cell(i,5).value-float(r['jax_native_ms'])/float(r['flagos_fl_only_ms']))<1e-10
    assert new['逻辑算子16类'].cell(i,6).value==r['flagtree_hlo_case_coverage']
assert new['40步卡上MFU'].max_row==2+4*len(summary.get('platform_order',['h100','hygon']))
baseline=Path('/home/zhangyuxin/zyx/graphcast-sugon/kernels/xf32_rne/xf32_rne_router.cc')
assert hashlib.sha256(baseline.read_bytes()).hexdigest()=='f9177d33c82c842461a43af95f371fd5b32a259ba3e6283b31aa61256a86b8e8'
report=dict(status='passed',independent_interval_sweep_rounds=len(events),operator_rows=16,preserved_sheet_values_except_updated_scope_notes=preserved,sheets=new.sheetnames,baseline_router_unchanged=True)
(root/'independent-validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
