"""User-requested JAX / fl_only / speedup / FlagTree table; preserve previous workbook."""
import csv
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'flagos-inference-final-20261030'
sys.path.insert(0,str(ROOT.parent/'inference-operators-20261022/operator-workbook/deps'))
import openpyxl
from openpyxl.styles import Font,PatternFill,Alignment
from openpyxl.worksheet.table import Table,TableStyleInfo
from openpyxl.utils import get_column_letter

source=json.loads((OLD/'summary.json').read_text())
coverage={
 'Linear':'15/15用例含FlagTree：bias等逐元素部分；GEMM仍为rocBLAS，独立RNE为HIP，不能称整项由FlagTree实现。',
 'Slice':'11/13含FlagTree；其余2例未见该标记，可能为视图或其他路径，不记作覆盖。',
 'Broadcast':'4/8含FlagTree；其余4例未见该标记，可能为常量/复制等，不记作覆盖。',
 'Reshape':'0/7含FlagTree；布局/视图可省略或仅改元数据，不等于7个待替换计算内核。',
 'Transpose':'2/10含FlagTree；其余8例未见该标记，布局变化/复制不记作FlagTree覆盖。',
 'Squeeze':'0/2含FlagTree；布局/视图可省略或仅改元数据，不等于2个待替换计算内核。'}
rows=[]
for r in source['operators']:
    n=r['flagtree_hlo_cases'];total=r['shape_cases']
    rows.append(dict(operator=r['operator'],category=r['category'],jax_native_ms=r['flagos_native_ms'],
        flagos_fl_only_ms=r['flagos_fl_only_ms'],speedup_jax_div_flagos=r['native_over_fl_only'],
        flagtree_hlo_case_coverage='{}/{}'.format(n,total),coverage_note=coverage.get(r['operator'],'本项全部用例的计算HLO中含FlagTree标记；按用例统计，不等同于内核数或耗时覆盖率。'),
        forward_calls=r['forward_calls'],shape_cases=total,source=r['measurement']))
with (ROOT/'operators.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
wb=openpyxl.load_workbook(str(OLD/'GraphCast_BW1100_FlagOS_推理算子与精度性能_20261030.xlsx'))
wb.remove(wb['逻辑算子16类'])
ws=wb.create_sheet('逻辑算子16类',0)
def fill_table(ws,headers,values,widths,note):
    ws.sheet_view.showGridLines=False
    ws.merge_cells(start_row=1,start_column=1,end_row=1,end_column=len(headers))
    ws.cell(1,1,note);ws.row_dimensions[1].height=72
    ws.cell(1,1).fill=PatternFill('solid',fgColor='E0F2FE')
    ws.cell(1,1).alignment=Alignment(wrap_text=True,vertical='center')
    ws.append(headers)
    for row in values:ws.append(row)
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.font=Font(name='Microsoft YaHei',size=10,color='FFFFFF' if c.row==2 else '172033',bold=c.row==2)
            c.alignment=Alignment(wrap_text=True,vertical='center')
            if c.row==2:c.fill=PatternFill('solid',fgColor='164E63')
            elif isinstance(c.value,float):c.number_format='0.000'
        ws.row_dimensions[row[0].row].height=60
    for i,w in enumerate(widths,1):ws.column_dimensions[get_column_letter(i)].width=w
    t=Table(displayName='Compare'+str(len(wb.sheetnames))+ws.title[:2].encode().hex(),ref='A2:'+get_column_letter(len(headers))+str(len(values)+2))
    t.tableStyleInfo=TableStyleInfo(name='TableStyleMedium2',showRowStripes=True)
    ws.add_table(t);ws.freeze_panes='C3';ws.auto_filter.ref=t.ref
    ws.page_setup.orientation='landscape';ws.page_setup.paperSize=ws.PAPERSIZE_A3
    ws.page_setup.fitToWidth=1;ws.page_setup.fitToHeight=0
fill_table(ws,['逻辑算子','分类','JAX耗时(ms)\n同容器native','FlagOS fl_only耗时(ms)','相较JAX加速比\nJAX÷FlagOS','FlagTree覆盖\n含标记用例/总例','覆盖说明','前向图调用','形状用例数','采集记录'],
    [list(r.values()) for r in rows],[29,18,23,26,24,24,83,16,17,32],
    'JAX列使用同一FlagOS容器的native后端；加速比>1为fl_only更快，<1为更慢。3轮×20次同步JAX墙钟后按图频次加权，非卡上时间，不能求和当模型耗时。FlagTree按HLO用例证据标记，Linear不代表GEMM被替换。旧10月22日JAX数据保留在原版Excel与原始CSV中。')
if (ROOT/'summary.json').exists():
    summary=json.loads((ROOT/'summary.json').read_text())
    refreshed={
        'MFU公式':'F_useful/(T_card×P_total)×100%；历史单步12.51922832%。新增完整40步分子为1151.39795697664 TFLOPs，三组实测详见40步卡上MFU页。',
        '有效FLOPs':'每步84个逻辑GEMM，Σ2MNK=28.784948924416 TFLOPs，40步乘40；逐元素/归约运算量未计，相关kernel时间计入。',
        'GPU计时':'原始HIPOPS/Nsight整数纳秒区间并集；计所有计算kernel，排除CPU/拷贝/空隙。当前kernel-events.csv含三路径9轮40步事件，原338条单步明细仍在历史目录。',
        '本次任务完成边界':'已完成三路径各3轮完整40步MFU补测。正式同等算力、海光峰值设备映射与验收计时口径待确认；候选MFU归一化比不直接作为60%正式判定。',
        '材料目录':str(ROOT)+'；旧算子与单步附件：'+str(OLD),
        '限制1':'主表JAX列为同FlagOS容器native后端；加速比JAX÷fl_only；FlagTree覆盖按HLO用例统计，Linear的GEMM仍为rocBLAS。旧厂商JAX列保留在旧版Excel。另行补测的原生JAX恢复环境与本列分别标注。',
        '限制8':'完整40步GPU内核区间并集已实测，不由单步外推。各3轮中位数：H100 12.658042秒/18.3873%，原生JAX恢复环境29.567501秒/22.8538%，FlagOS54.675367秒/12.3589%；均为GEMM FLOPs近似、非端到端MFU。',
        '限制10':'海光峰值170.3936为用户指定，设备映射与频率依据待确认；H100采用NVIDIA v1.04白皮书稠密TF32 SXM5 494.7。候选MFU比FlagOS/H100=67.2146%，正式效率公式尚待项目确认。'
    }
    for row in wb['方法与来源'].iter_rows(min_row=3):
        if row[0].value in refreshed:row[1].value=refreshed[row[0].value]
    wb['整模型性能']['I3']='历史单步记录时峰值未确认；10月30日稠密TF32峰值及完整40步MFU见新增页。原40步墙钟未覆盖。'
    ss=wb.create_sheet('40步卡上MFU')
    rr=[]
    comparisons=[]
    h100_mfu=summary['h100']['median_mfu_percent']
    for platform in summary.get('platform_order',['h100','hygon']):
        d=summary[platform]
        for r in d['rounds']:
            rr.append([d['label'],r['round'],r['kernel_active_seconds'],r['kernel_span_seconds'],r['kernel_count'],r['rollout_wall_seconds'],d['peak_tflops'],r['mfu_percent'],None])
        relative=100*d['median_mfu_percent']/h100_mfu
        rr.append([d['label']+' 三轮中位数','汇总',d['median_gpu_active_seconds'],None,None,d['median_profiled_wall_seconds'],d['peak_tflops'],d['median_mfu_percent'],relative])
        comparisons.append(dict(platform=platform,label=d['label'],median_mfu_percent=d['median_mfu_percent'],h100_median_mfu_percent=h100_mfu,relative_mfu_vs_h100_percent=relative))
    with (ROOT/'mfu-comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(comparisons[0]));w.writeheader();w.writerows(comparisons)
    fill_table(ss,['平台精度','轮次','40步GPU活跃时间(s)','首尾kernel跨度(s)','kernel次数','带profiler的40步墙钟(s)','稠密峰值(TFLOP/s)','40步MFU估算(%)','相对H100的MFU比值(%)\n双方三轮中位数之比'],rr,[38,16,27,27,20,32,25,26,33],
        '三轮各5步预热后重置输入、独立自回归40步。MFU=1151.39795697664 TFLOPs/(GPU内核区间并集秒×稠密峰值)，GEMM FLOPs为近似。相对MFU比=本路径三轮中位MFU/H100三轮中位MFU×100%，仅汇总行列示，按未四舍五入值计算；H100=100%。这是算力归一化比值，不是实际速度比或正式阶段60%判定。')
target=ROOT/'GraphCast_JAX-FlagOS_推理对照_20261030.xlsx'
wb.save(str(target));wb.close()
check=openpyxl.load_workbook(str(target),data_only=True)
assert check['逻辑算子16类'].max_row==18
for i,r in enumerate(rows,3):
    assert abs(check['逻辑算子16类'].cell(i,5).value-r['jax_native_ms']/r['flagos_fl_only_ms'])<1e-10
assert check['逻辑算子16类']['F3'].value=='15/15'
check.close()
print(target)
