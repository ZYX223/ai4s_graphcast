"""GPU-active GEMM-dominated MFU estimate; no assumed vendor peak."""
from pathlib import Path
import argparse,collections,csv,hashlib,json,math,statistics
p=argparse.ArgumentParser();p.add_argument('--theoretical-peak-tflops',type=float);p.add_argument('--peak-source');a=p.parse_args()
if a.theoretical_peak_tflops is not None:
 assert math.isfinite(a.theoretical_peak_tflops) and a.theoretical_peak_tflops>0 and a.peak_source
here=Path(__file__).resolve().parent;root=here.parent
sources={}
def read(p):
 sources[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
 return json.loads(p.read_text())
peak_reference=None
if a.theoretical_peak_tflops is None and (here/'peak-reference.json').exists():
 peak_reference=read(here/'peak-reference.json')
 image_path=here/peak_reference['source_image']
 image_digest=hashlib.sha256(image_path.read_bytes()).hexdigest()
 assert image_digest==peak_reference['source_image_sha256']
 sources[str(image_path.relative_to(root))]=image_digest
 a.theoretical_peak_tflops=peak_reference['theoretical_peak_tflops']
 a.peak_source=peak_reference['source']
 assert math.isfinite(a.theoretical_peak_tflops) and a.theoretical_peak_tflops>0 and a.peak_source
def union(intervals):
 total=0.;end=None
 for start,finish in sorted(intervals):
  assert math.isfinite(start) and math.isfinite(finish) and finish>=start
  if end is None or start>=end:total+=finish-start;end=finish
  elif finish>end:total+=finish-end;end=finish
 return total
assert union([(0,4),(2,7),(8,10)])==9
probes=[]
assert (here/'probe-r2/exit-code.txt').read_text().strip()=='0'
for n in [8192,16384,24576]:
 r=read(here/('probe-r2/gemm-%d.json'%n));assert r['status']=='passed' and r['m']==r['n']==r['k']==n and r['pci'].lower()=='0000:b0:00.0'
 assert r['inputs_unchanged_full'] and r['rne_explicit_match_full'] and r['oracle_output_samples']==32
 times=r['timings']['rne_gemm_only']['round_ms'];assert len(times)==3 and all(t>0 for t in times)
 tflops=2*n**3/(statistics.median(times)*1e9)
 probes.append(dict(n=n,gemm_device_ms=times,median_device_ms=statistics.median(times),median_tflops=tflops,iterations_per_round=r['iterations'],warmups=r['warmup']))
reference=max(probes,key=lambda r:r['median_tflops'])
rows=[];signatures=[];layer_rows=[]
for variant in ['baseline','fused']:
 folder=root/('profile-'+variant+'-r1');report=read(folder/'run/report.json');assert report['status']=='completed' and report['measured_steps']==1
 text=(folder/'gemm.jsonl').read_text();sources[str((folder/'gemm.jsonl').relative_to(root))]=hashlib.sha256(text.encode()).hexdigest()
 gems=[json.loads(l) for l in text.splitlines()];gems=[r for r in gems if r['label']=='round1-rne_xf32'];assert len(gems)==len(set(r['layer'] for r in gems))==84
 assert all(r['mode']==1 and r['status']==0 and r['rne_inputs'] for r in gems)
 signature=sorted((r['layer'],r['weight_sha256'],r['m'],r['n'],r['k']) for r in gems);signatures.append(signature)
 flops=sum(2*r['m']*r['n']*r['k'] for r in gems)
 if variant=='baseline':layer_rows=[dict(layer=r['layer'],m=r['m'],n=r['n'],k=r['k'],flops=2*r['m']*r['n']*r['k']) for r in gems]
 trace=read(folder/'trace.json');events=trace['traceEvents'];kernels=[e for e in events if e.get('ph')=='X' and e.get('cat')=='HIPOPS'];assert len(kernels)==(550 if variant=='baseline' else 479)
 intervals=[(e['ts'],e['ts']+e['dur']) for e in kernels];busy=union(intervals)/1000;total=sum(e['dur'] for e in kernels)/1000;span=(max(end for _,end in intervals)-min(start for start,_ in intervals))/1000
 saved=read(folder/'profile-summary.json');assert abs(busy-saved['gpu_kernel_busy_union_ms'])<1e-5
 throughput=flops/(busy*1e9)
 row=dict(variant=variant,gemm_flops_per_step=flops,gpu_kernel_busy_ms=busy,gpu_kernel_sum_ms=total,gpu_kernel_span_ms=span,effective_tflops_gpu_active=throughput,effective_tflops_device_span=flops/(span*1e9),empirical_reference_utilization_percent=100*throughput/reference['median_tflops'],theoretical_mfu_percent=None if a.theoretical_peak_tflops is None else 100*throughput/a.theoretical_peak_tflops,gemm_only_tflops=flops/(saved['categories']['gemm']['total_ms']*1e9))
 rows.append(row)
assert signatures[0]==signatures[1]
assert rows[0]['gemm_flops_per_step']==28784948924416
# Independent cross-check against the earlier audited shape/frequency catalog.
catalog=read(root/'baseline-source/configs/operators.json')
assert sum(r['flops_per_call']*r['calls_per_step'] for r in catalog['gemm_shapes'])==rows[0]['gemm_flops_per_step']
expected=collections.Counter({(r['m'],r['n'],r['k']):r['calls_per_step'] for r in catalog['gemm_shapes']})
assert collections.Counter((r['m'],r['n'],r['k']) for r in layer_rows)==expected
# The per-layer log and the prior shape/frequency audit agree independently.
result=dict(schema='graphcast.gpu_active_mfu_estimate.v1',status='computed_with_empirical_reference' if a.theoretical_peak_tflops is None else 'computed_with_theoretical_peak',model='Official JAX GraphCast RNE+XF32 inference',product_model='BW1100',raw_driver_model='BW1200',pci='0000:b0:00.0',model_time_source='Existing single-step HIPPROF capture after 5 warmup steps; not a new 40-step device-time measurement',time_definition='Union of all HIPOPS GPU kernel intervals, including RNE/scatter/gather/norm/etc. Excludes CPU, copies and idle gaps; device first-to-last kernel span supplied separately.',flops_definition='Approximate useful model FLOPs from 84 actual dense GEMMs, sum(2*M*N*K), FMA=2. Excludes elementwise/reduction/transcendental FLOPs, RNE/integer work, padding and compiler work. Same numerator for both variants.',theoretical_peak_tflops=a.theoretical_peak_tflops,theoretical_peak_source=a.peak_source,empirical_reference=reference,empirical_reference_definition='Highest median among only 3 tested square sizes, same card/native rocBLAS XF32, pre-rounded FP32 inputs, HIP events. No RNE overhead in denominator. Not a vendor theoretical peak or proven global maximum.',square_probes=probes,rows=rows,throughput_relative_increase_percent=(rows[1]['effective_tflops_gpu_active']/rows[0]['effective_tflops_gpu_active']-1)*100,source_sha256=sources)
result['peak_reference_metadata']=peak_reference
(here/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
with (here/'gemm-flops.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=['layer','m','n','k','flops']);writer.writeheader();writer.writerows(layer_rows)
print(json.dumps(dict(reference=reference,rows=rows,relative_increase_percent=result['throughput_relative_increase_percent']),indent=2))
