"""Recompute paired loss, card time and useful training MFU from raw evidence."""
import csv,gzip,hashlib,json,re,statistics
from pathlib import Path
from timing import interval_union,jax_update_kernels,nsight_update_kernels
from trajectory import validate_trajectory
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def csvout(p,rows):
 if not rows:return
 with p.open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
updates=[];stages=[];kernel_summary=[];sources={};trajectories={}
for backend,peak,flops_folder in [('h100',494.7,'flops-h100'),('flagos',170.3936,'flops')]:
 trajectories[backend]=[]
 for horizon in range(1,13):
  folder=ROOT/('%s-r%02d'%(backend,horizon));p=folder/'report.json';fp=ROOT/flops_folder/('horizon-%02d.json'%horizon)
  if not (p.exists() and fp.exists() and (folder/'exit-code.txt').exists()):continue
  r=read(p)
  if r['status']!='completed' or int((folder/'exit-code.txt').read_text())!=0:continue
  if backend=='flagos':assert int((folder/'python-exit-code.txt').read_text())==0
  assert r['checkpoint_reload']['status']=='passed'
  trajectories[backend].append(r)
  cp=folder/'checkpoint.pkl'
  if cp.exists():
   sources[str(cp)]=hashlib.sha256(cp.read_bytes()).hexdigest()
   assert sources[str(cp)]==r['checkpoint_sha256']
  sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();sources[str(fp)]=hashlib.sha256(fp.read_bytes()).hexdigest()
  f=read(fp);useful=f['useful_flops_per_update']
  if backend=='flagos':
   timing_path=folder/'xplane-kernels.json'
   if not timing_path.exists():continue
   timing=read(timing_path);assert not timing['source_errors']
   sources[str(timing_path)]=hashlib.sha256(timing_path.read_bytes()).hexdigest()
   kernels={int(k):v for k,v in timing['updates'].items()}
   raw_paths=list((folder/'jax-trace').rglob('*.xplane.pb'));assert len(raw_paths)==1
   assert hashlib.sha256(raw_paths[0].read_bytes()).hexdigest()==timing['source_sha256']
   sources[str(raw_paths[0])]=timing['source_sha256']
   host=[json.loads(line) for line in (folder/'host-device-memory.jsonl').read_text().splitlines()]
   card_peak=max(float(x['data']['card10']['vram Total Used Memory (MiB)']) for x in host if 'data' in x)
  else:
   sqlite_path=folder/'train-trace.sqlite'
   sources[str(sqlite_path)]=hashlib.sha256(sqlite_path.read_bytes()).hexdigest()
   kernels=nsight_update_kernels(sqlite_path,r['rows']);card_peak=r['observed_device_peak_mib']
  assert set(kernels)=={x['update'] for x in r['rows']}
  total_time=0;names={}
  for row in r['rows']:
   ks=kernels[row['update']];seconds=interval_union([(a,b) for a,b,n in ks])/1e9;total_time+=seconds
   assert 0<seconds<=row['synchronized_update_seconds']*1.01
   for a,b,n in ks:
    if n not in names:names[n]=dict(calls=0,sum_ns=0)
    names[n]['calls']+=1;names[n]['sum_ns']+=b-a
   updates.append(dict(backend=backend,horizon=horizon,update=row['update'],lr=row['lr'],loss=row['loss'],
    gpu_kernel_seconds=seconds,synchronized_update_seconds=row['synchronized_update_seconds'],kernel_count=len(ks),
    useful_gemm_flops=useful,peak_tflops=peak,mfu_percent=100*useful/(seconds*peak*1e12),
    jax_peak_gib=row['jax_memory'].get('peak_bytes_in_use',0)/2**30,
    parameter_max_abs_change=row['parameter_update']['max_abs'],batch_sha256=json.dumps(row['batch_sha256'])))
  memory=re.search(r'temp_size_in_bytes=(\d+)',r['compiled_memory_analysis']);assert memory
  stages.append(dict(backend=backend,horizon=horizon,updates=len(r['rows']),loss_last=r['rows'][-1]['loss'],
   gpu_kernel_seconds=total_time,synchronized_update_seconds=sum(x['synchronized_update_seconds'] for x in r['rows']),
   useful_gemm_flops=useful*len(r['rows']),mfu_percent=100*useful*len(r['rows'])/(total_time*peak*1e12),
   whole_card_peak_gib=card_peak/1024,jax_peak_gib=max(x['jax_memory'].get('peak_bytes_in_use',0) for x in r['rows'])/2**30,
   compiled_temp_gib=int(memory[1])/2**30,compile_seconds=r['compile_seconds'],main_seconds=r['total_main_seconds'],
   checkpoint_counter=json.dumps(r['checkpoint_reload']['optimizer_counters']),exit_code=0))
  kernel_summary.extend(dict(backend=backend,horizon=horizon,kernel=n,calls=v['calls'],sum_ms=v['sum_ns']/1e6) for n,v in names.items())
for reports in trajectories.values():validate_trajectory(reports)
if all(trajectories.values()):
 for key in ['initial_params_sha256','initial_optimizer_sha256','data_sha256','model','task','optimizer','seed']:
  assert trajectories['h100'][0][key]==trajectories['flagos'][0][key],key
by={b:{x['update']:x for x in updates if x['backend']==b} for b in ['h100','flagos']};paired=[]
for u in sorted(set(by['h100'])&set(by['flagos'])):
 h=by['h100'][u];d=by['flagos'][u];assert h['horizon']==d['horizon'] and h['lr']==d['lr'] and h['batch_sha256']==d['batch_sha256']
 paired.append(dict(update=u,horizon=h['horizon'],lr=h['lr'],h100_loss=h['loss'],flagos_loss=d['loss'],relative_loss_percent=100*abs(d['loss']-h['loss'])/abs(h['loss'])))
summary=dict(status='completed' if len(paired)==38 and len(stages)==24 else 'partial',paired_updates=len(paired),backends={},
 formula='MFU = useful differentiated NO-remat GEMM FLOPs / (actual GPU compute-kernel union seconds x dense peak). Numerator excludes recomputation, RNE and optimizer FLOPs; denominator includes their kernels; copies/CPU/gaps excluded.',
 limitations=['Short prescribed38-update mechanics/loss-consistency test, not convergence.',
 'Two zero learning rates retained;36 nonzero parameter updates. Each update includes2 microbatches.',
 'BW additional LayerNorm remat; H100 original stage/message remat. H100 horizons11/12 use96% pool after94% insufficient; BW horizons11/12 use73% instead of70% after measured capacity growth; frozen baseline preserved.',
 'Independent trajectories from same initial weights/optimizer/data; compare corresponding loss, not trained-checkpoint inference outputs.',
 'H100 graph has one additional grid2mesh edge; platform FLOPs counted separately.',
 'MFU is GEMM approximation and GPU-active timing. MFU ratio is candidate peak-normalized efficiency, not formal phase acceptance.'])
for b,rows in by.items():
 if not rows:continue
 ss=[s for s in stages if s['backend']==b];total_t=sum(x['gpu_kernel_seconds'] for x in rows.values());total_f=sum(x['useful_gemm_flops'] for x in rows.values());peak=next(iter(rows.values()))['peak_tflops']
 summary['backends'][b]=dict(updates=len(rows),gpu_kernel_seconds=total_t,synchronized_update_seconds=sum(x['synchronized_update_seconds'] for x in rows.values()),
  useful_gemm_flops=total_f,mfu_percent=100*total_f/(total_t*peak*1e12),peak_tflops=peak,
  whole_card_peak_gib=max(x['whole_card_peak_gib'] for x in ss),jax_peak_gib=max(x['jax_peak_gib'] for x in ss),
  best_loss=min(x['loss'] for x in rows.values()),last_loss=rows[max(rows)]['loss'])
if paired:summary.update(max_loss_relative_percent=max(x['relative_loss_percent'] for x in paired),mean_loss_relative_percent=statistics.mean(x['relative_loss_percent'] for x in paired),internal5_pass=all(x['relative_loss_percent']<5 for x in paired),stage10_pass=all(x['relative_loss_percent']<=10 for x in paired))
if summary['status']=='completed':summary['flagos_over_h100_mfu_ratio_percent']=100*summary['backends']['flagos']['mfu_percent']/summary['backends']['h100']['mfu_percent']
csvout(ROOT/'updates.csv',updates);csvout(ROOT/'horizons.csv',stages);csvout(ROOT/'loss-comparison.csv',paired);csvout(ROOT/'kernel-details.csv',kernel_summary)
write(ROOT/'summary.json',summary);write(ROOT/'summary-source-hashes.json',sources)
print(json.dumps(summary,ensure_ascii=False,indent=2))
