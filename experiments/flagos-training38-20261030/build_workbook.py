"""Compact training workbook; raw GPU kernels remain the CSV attachment."""
import csv,json,sys
from pathlib import Path
p=Path(__file__).resolve().parent
sys.path.insert(0,str(p.parent/'inference-operators-20261022/operator-workbook/deps'))
import openpyxl
from openpyxl.styles import Alignment,Font,PatternFill
from openpyxl.utils import get_column_letter
s=json.loads((p/'summary.json').read_text());assert s['status']=='completed'
o=json.loads((p/'operators-summary.json').read_text())
w=openpyxl.Workbook();w.remove(w.active)
def sheet(name,header,rows):
 a=w.create_sheet(name);a.append(header)
 for row in rows:a.append(row)
 a.freeze_panes='A2';a.auto_filter.ref=a.dimensions
 for c in a[1]:c.font=Font(name='微软雅黑',bold=True,color='FFFFFF');c.fill=PatternFill('solid',fgColor='145A70')
 for row in a.iter_rows(min_row=2):
  for c in row:
   c.alignment=Alignment(vertical='top',wrap_text=True)
   if isinstance(c.value,float):c.number_format='0.000000'
 for i in range(1,len(header)+1):a.column_dimensions[get_column_letter(i)].width=22 if i>1 else 31
 return a
h=s['backends']['h100'];d=s['backends']['flagos']
total=[['更新次数',38,38,'1步16次，2～12步各2次；2次LR=0，36次参数实际变化'],
 ['卡上计算累计(s)',h['gpu_kernel_seconds'],d['gpu_kernel_seconds'],'实际计算kernel区间并集；含重计算/RNE/优化器，排除拷贝与间隙'],
 ['同步更新累计(s)',h['synchronized_update_seconds'],d['synchronized_update_seconds'],'与卡上时间分列；不含输入准备/H2D、编译、预热、保存'],
 ['MFU(%)',h['mfu_percent'],d['mfu_percent'],'有效前后向GEMM FLOPs近似，ΣF/(ΣT×峰值)'],
 ['MFU/H100(%)',100,s['flagos_over_h100_mfu_ratio_percent'],'候选峰值归一化效率；正式阶段口径待项目确认'],
 ['峰值TFLOP/s',494.7,170.3936,'稠密TF32/XF32峰值口径'],
 ['整卡采样峰值GiB',h['whole_card_peak_gib'],d['whole_card_peak_gib'],'H100约0.2秒、BW1100约1秒采样；非每瞬间严格上界'],
 ['JAX分配器峰值GiB',h['jax_peak_gib'],d['jax_peak_gib'],'不等于整卡显存'],
 ['最后loss',h['last_loss'],d['last_loss'],'对应第38次更新；不代表收敛'],
 ['最低loss',h['best_loss'],d['best_loss'],'跨预测长度仅作记录，不代替验证集最优模型'],
 ['逐次loss最大相对差异(%)',None,s['max_loss_relative_percent'],'abs(candidate-H100)/abs(H100)×100%'],
 ['逐次loss平均相对差异(%)',None,s['mean_loss_relative_percent'],'38个对应更新等权平均'],
 ['算子范围','同FlagOS容器的native GPU','fl_only GPU','预测1步有效前后向/loss，2微批；不含重计算及优化器'],
 ['算子计时','3轮×20次，预热5次','同左','独立同步墙钟×调用次数；不能求和当作整模型卡上时间'],
 ['已知限制','5个LayerNorm前向native GPU编译断言','另用CPU JAX抽样校验通过','native GPU时间和加速比留空；GEMM仍为rocBLAS'],
 ['原始材料',str(p),'/data/zyx/graphcast-training38-20261030/','kernel-details.csv为内核附件；原baseline未修改，待最后统一同步']]
ws=sheet('训练总览',['指标','H100 TF32','BW1100 FlagOS RNE＋XF32','口径与限制'],total);ws.column_dimensions['D'].width=72
def rows_csv(name):return list(csv.DictReader((p/name).open(encoding='utf-8-sig')))
loss=rows_csv('loss-comparison.csv')
sheet('逐次loss',['更新','预测步数','学习率','H100 loss','FlagOS loss','相对差异(%)'],[[int(r['update']),int(r['horizon']),float(r['lr']),float(r['h100_loss']),float(r['flagos_loss']),float(r['relative_loss_percent'])] for r in loss])
hs=rows_csv('horizons.csv')
sheet('分阶段显存耗时',['平台','预测步数','更新次数','卡上累计(s)','同步更新累计(s)','MFU(%)','整卡峰值GiB','JAX峰值GiB','临时区GiB','编译(s)','完整进程主流程(s)'],[[r['backend'],int(r['horizon']),int(r['updates'])]+[float(r[k]) for k in ['gpu_kernel_seconds','synchronized_update_seconds','mfu_percent','whole_card_peak_gib','jax_peak_gib','compiled_temp_gib','compile_seconds','main_seconds']] for r in hs])
sheet('逻辑算子',['逻辑算子','JAX(ms)','FlagOS fl_only(ms)','相对JAX加速比','FlagTree覆盖证据','调用数/更新','形状参数用例数'],[[r[k] for k in ['operator','jax_ms','flagos_fl_only_ms','speedup','flagtree_coverage','calls_per_update','cases']] for r in o['operators']]).column_dimensions['E'].width=65
def convert(v):
 if v=='':return None
 try:return float(v)
 except ValueError:return v
for name,file in [('算子阶段明细','operator-phases.csv'),('算子用例','operator-cases.csv')]:
 rows=rows_csv(file);sheet(name,list(rows[0]),[[v if k=='case_id' else convert(v) for k,v in r.items()] for r in rows])
target=p/'GraphCast_38次训练_MFU与算子对照_20261030.xlsx';w.save(str(target))
v=openpyxl.load_workbook(str(target),data_only=True)
assert len(v.sheetnames)==6 and v['逐次loss'].max_row==39 and v['算子用例'].max_row==222
assert v['逻辑算子'].max_row==21 and v['分阶段显存耗时'].max_row==25
assert abs(v['训练总览']['C5'].value-d['mfu_percent'])<1e-10
print(target)
