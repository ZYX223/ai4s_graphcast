"""Strict paired evidence summary; no unmeasured results or invented thresholds."""
import csv,hashlib,json,math,re
from pathlib import Path
root=Path(__file__).resolve().parent
rows=[];updates=[];complete=True
for r in range(1,13):
 runs={}
 for mode in ['fp32','rne_xf32']:
  p=root/('%s-r%02d'%(mode,r))
  if not (p/'report.json').exists() or not (p/'device-timing.json').exists():complete=False;continue
  report=json.loads((p/'report.json').read_text());timing=json.loads((p/'device-timing.json').read_text())
  if report['status']!='completed':complete=False;continue
  assert not report.get('preflight') and not report['tiny']
  expected=list(range(1,17)) if r==1 else [17+2*(r-2),18+2*(r-2)]
  assert [x['update'] for x in report['rows']]==expected
  assert [x['update'] for x in timing['rows']]==expected
  assert all(x['rollout']==r and x['gradient_norm']>0 and math.isfinite(x['loss']) for x in report['rows'])
  assert hashlib.sha256((p/'checkpoint.pkl').read_bytes()).hexdigest()==report['checkpoint_sha256']
  runs[mode]=(report,timing,p)
 if len(runs)!=2:continue
 a,ta,pa=runs['fp32'];b,tb,pb=runs['rne_xf32']
 assert a['data_sha256']==b['data_sha256'] and a['model']==b['model'] and a['optimizer']==b['optimizer']
 assert a['device']['pci']==b['device']['pci']
 for x,y,tx,ty in zip(a['rows'],b['rows'],ta['rows'],tb['rows']):
  assert (x['update'],x['lr'],x['starts'])==(y['update'],y['lr'],y['starts'])
  assert x['gemm']['calls']==y['gemm']['calls'] and x['gemm']['rounded']==0 and y['gemm']['rounded']==y['gemm']['calls']
  assert 'gradient_vs_fp32' in y
  updates.append(dict(update=x['update'],rollout=r,lr=x['lr'],starts=x['starts'],fp32_loss=x['loss'],rne_xf32_loss=y['loss'],
   loss_relative_abs=abs(y['loss']-x['loss'])/abs(x['loss']) if x['loss'] else None,
   fp32_gradient_norm=x['gradient_norm'],rne_xf32_gradient_norm=y['gradient_norm'],gradient_relative_l2=y['gradient_vs_fp32']['relative_l2'],
   fp32_gpu_kernel_ms=tx['gpu_kernel_sum_ms'],rne_xf32_gpu_kernel_ms=ty['gpu_kernel_sum_ms']))
 local=updates[-len(a['rows']):]
 rows.append(dict(rollout=r,updates=len(a['rows']),
  fp32_jax_peak_gib=max(x['jax_memory']['peak_bytes_in_use'] for x in a['rows'])/2**30,
  rne_xf32_jax_peak_gib=max(x['jax_memory']['peak_bytes_in_use'] for x in b['rows'])/2**30,
  fp32_device_peak_gib=a['observed_device_peak_mib']/1024,rne_xf32_device_peak_gib=b['observed_device_peak_mib']/1024,
  fp32_mean_gpu_ms=ta['total_gpu_kernel_sum_ms']/len(a['rows']),rne_xf32_mean_gpu_ms=tb['total_gpu_kernel_sum_ms']/len(b['rows']),
  speedup=ta['total_gpu_kernel_sum_ms']/tb['total_gpu_kernel_sum_ms'],
  fp32_loss_first=a['rows'][0]['loss'],fp32_loss_last=a['rows'][-1]['loss'],fp32_loss_min=min(x['loss'] for x in a['rows']),
  rne_xf32_loss_first=b['rows'][0]['loss'],rne_xf32_loss_last=b['rows'][-1]['loss'],rne_xf32_loss_min=min(x['loss'] for x in b['rows']),
  max_loss_relative_abs=max(x['loss_relative_abs'] for x in local),max_gradient_relative_l2=max(x['gradient_relative_l2'] for x in local),
  parameter_relative_l2=b['parameters_vs_fp32']['relative_l2']))
result=dict(status='completed' if complete and len(updates)==38 else 'partial',horizons=rows,updates=updates,
 notes=['GPU times are traced kernel-duration sums including RNE/recomputation/optimizer.','38-update small-sample validation, not convergence or weather RMSE.','Independent trajectories: later gradient/parameter differences include previous update divergence.','Observed whole-device memory includes allocator reservation and RNE scratch.'])
if updates:
 ft=sum(x['fp32_gpu_kernel_ms'] for x in updates);rt=sum(x['rne_xf32_gpu_kernel_ms'] for x in updates)
 result['aggregate']=dict(completed_updates_per_mode=len(updates),fp32_gpu_seconds=ft/1000,rne_xf32_gpu_seconds=rt/1000,speedup=ft/rt,max_loss_relative_abs=max(x['loss_relative_abs'] for x in updates),max_gradient_relative_l2=max(x['gradient_relative_l2'] for x in updates),trajectory_loss_relative_l1=sum(abs(x['rne_xf32_loss']-x['fp32_loss']) for x in updates)/sum(abs(x['fp32_loss']) for x in updates))
(root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
for name,values in [('by-horizon.csv',rows),('by-update.csv',updates)]:
 if values:
  with (root/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
print(json.dumps(dict(status=result['status'],aggregate=result.get('aggregate'),completed_horizons=len(rows)),indent=2))
