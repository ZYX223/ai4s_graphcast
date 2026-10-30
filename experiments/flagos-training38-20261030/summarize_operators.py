"""Logical timing table: identical independent cases; no model-time summation."""
import csv,json,hashlib
from collections import defaultdict
from pathlib import Path
root=Path(__file__).resolve().parent;folder=root/'operators'
reports={m:json.loads((folder/('bench-'+m)/'report.json').read_text()) for m in ['native','fl_only']}
case_metadata={c['id']:c for c in json.loads((folder/'cases.json').read_text())['cases']}
by={m:{r['id']:r for r in v['rows']} for m,v in reports.items()}
retry_path=folder/'retry-linear/report.json';recovered=set()
if retry_path.exists():
 retry=json.loads(retry_path.read_text())
 native_retry_path=folder/'retry-native/report.json'
 native_retry=json.loads(native_retry_path.read_text())
 assert int((folder/'retry-native/exit-code.txt').read_text())==int((folder/'retry-linear/exit-code.txt').read_text())==0
 validation=json.loads((folder/'retry-linear-validation.json').read_text())
 assert validation['status']=='passed' and native_retry['status']=='passed' and len(retry['rows'])==len(native_retry['rows'])==1
 for remote,digest in validation['sources'].items():
  local=folder/('retry-native' if 'native-retry' in remote else 'retry-linear')/Path(remote).name
  assert hashlib.sha256(local.read_bytes()).hexdigest()==digest
 nr=native_retry['rows'][0]
 assert by['native'][nr['id']]['status']=='failed' and nr['status']=='passed'
 assert by['native'][nr['id']]['signature']==nr['signature']
 by['native'][nr['id']]=nr
 for r in retry['rows']:
  original=by['fl_only'][r['id']]
  assert original['status']=='failed' and r['status']=='measured_reference_unavailable' and r['id']==validation['id']
  assert original['signature']==r['signature'] and original['calls_per_update']==r['calls_per_update']
  r=dict(r,original_recorded_status=r['status'],status='passed',max_abs_error=validation['max_abs_error'],sample_relative_l2_percent=validation['sample_relative_l2_percent'])
  by['fl_only'][r['id']]=r;recovered.add(r['id'])
cpu_path=folder/'cpu-layernorm/comparison.json'
cpu={r['id']:r for r in json.loads(cpu_path.read_text())} if cpu_path.exists() else {}
assert all(len(by[m])==reports[m]['cases']==221 for m in by)
assert set(by['native'])==set(by['fl_only'])
names={'linear':'Linear','concat':'Concat','layer_norm':'LayerNorm','silu':'SiLU','gather':'Gather','scatter_add':'ScatterAdd / SegmentSum','add':'Add','add_any':'AddAny','mul':'Mul','sub':'Sub','div':'Div','slice':'Slice','dynamic_slice':'DynamicSlice','broadcast_in_dim':'Broadcast','reshape':'Reshape','transpose':'Transpose','squeeze':'Squeeze','reduce_sum':'ReduceSum','pad':'Pad','split':'Split','integer_pow':'IntegerPow'}
names.update({'concatenate':'Concat','scatter-add':'ScatterAdd / SegmentSum','add_any':'Add'})
groups=defaultdict(list);details=[]
for k in by['native']:
 n=by['native'][k];f=by['fl_only'][k]
 assert n['signature']==f['signature'] and n['calls_per_update']==f['calls_per_update']
 scope=case_metadata[k]['group']['scope']
 phase='backward' if 'transpose(' in scope else ('forward' if 'jvp(' in scope else 'auxiliary_unclassified')
 row=dict(case_id=k,operator=names.get(n['operator'],n['operator']),phase=phase,raw_capture_phase=n['phase'],calls_per_update=n['calls_per_update'],jax_status=n['status'],flagos_status=f['status'],
  jax_ms=n.get('median_ms'),flagos_ms=f.get('median_ms'),flagtree_hlo=f.get('flagtree_hlo'),
  speedup=(n['median_ms']/f['median_ms'] if n['status']==f['status']=='passed' else None),
  max_abs_error=f.get('max_abs_error'),sample_relative_l2_percent=f.get('sample_relative_l2_percent'),
  cpu_sampled_correctness=cpu.get(k,{}).get('status'),flagos_run='isolated_retry' if k in recovered else 'first_pass',jax_error=n.get('error'),flagos_error=f.get('error'))
 groups[(row['operator'],row['phase'])].append(row);details.append(row)
def aggregate(name,phase,rows):
 native_all=all(r['jax_status']=='passed' for r in rows)
 measured_all=all(r['flagos_ms'] is not None for r in rows)
 correct_all=all(r['flagos_status']=='passed' for r in rows)
 nt=sum(r['jax_ms']*r['calls_per_update'] for r in rows) if native_all else None
 ft=sum(r['flagos_ms']*r['calls_per_update'] for r in rows) if measured_all else None
 covered=sum(bool(r['flagtree_hlo']) for r in rows)
 return dict(operator=name,phase=phase,jax_ms=nt,flagos_fl_only_ms=ft,
  speedup=nt/ft if native_all and correct_all else None,
  flagtree_coverage='%d/%d cases contain FlagTree HLO custom calls%s'%(covered,len(rows),'; GEMM remains rocBLAS' if name=='Linear' else ''),
  calls_per_update=sum(r['calls_per_update'] for r in rows),cases=len(rows),
  status='passed' if native_all and correct_all else 'incomplete correctness/reference; see individual cases')
phases=[aggregate(op,phase,rows) for (op,phase),rows in groups.items()]
ops=defaultdict(list)
for r in details:ops[r['operator']].append(r)
totals=[aggregate(op,'forward+backward',rows) for op,rows in ops.items()]
order=['Linear','Concat','LayerNorm','ScatterAdd / SegmentSum','SiLU','Gather','Add','Mul','Sub','Div','Slice','DynamicSlice','Broadcast','Reshape','Transpose','Squeeze','ReduceSum','Pad','Split','IntegerPow']
totals.sort(key=lambda r:order.index(r['operator']))
def csvout(name,rows):
 with (root/name).open('w',newline='',encoding='utf-8-sig') as o:
  w=csv.DictWriter(o,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
csvout('operator-cases.csv',details);csvout('operator-phases.csv',phases);csvout('operator-table.csv',totals)
summary=dict(scope=reports['fl_only']['scope'],timing=reports['fl_only']['timing'],inputs=reports['fl_only']['inputs'],reference=reports['fl_only']['reference'],
 cases=221,native_pass=sum(r['jax_status']=='passed' for r in details),flagos_pass=sum(r['flagos_status']=='passed' for r in details),
 flagos_measured=sum(r['flagos_ms'] is not None for r in details),flagtree_cases=sum(bool(r['flagtree_hlo']) for r in details),
 cpu_reference_pass=sum(r['cpu_sampled_correctness']=='passed' for r in details),
 isolated_retry_cases=sorted(recovered),
 operators=totals,failures=[r for r in details if r['jax_status']!='passed' or r['flagos_status']!='passed'])
(root/'operators-summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
lines=['# Training logical operator timing','',
 'BW1100 single card; FlagOS-container JAX native vs FlagOS JAX (fl_only), both RNE+XF32. Horizon1 useful forward+backward including loss, 2 microbatches/update. Times are independent synchronized operator wall time weighted by calls, not GPU-only whole-model time. Remat and optimizer are excluded here but included in the full-model kernel attachment.',
 '', '| Logical operator | JAX ms | FlagOS fl_only ms | JAX / FlagOS | FlagTree evidence |', '| --- | ---: | ---: | ---: | --- |']
fmt=lambda x:'—' if x is None else '%.3f'%x
for r in totals:lines.append('| %s | %s | %s | %s | %s |'%(r['operator'],fmt(r['jax_ms']),fmt(r['flagos_fl_only_ms']),fmt(r['speedup']),r['flagtree_coverage']))
lines+=['','Phase labels use JAX scope evidence (jvp / transpose); unmarked loss/data/gradient auxiliary operations are listed separately. Deduplicated cases can combine scopes, so phase rows are diagnostic labels, not a rigorous partition of complete forward/backward time.','',
 'One large backward Linear failed memory allocation in both sequential suites; separate fresh processes at the same80% pool completed timings and sampled comparison. Original failures and retry evidence remain separate.','',
 'Blank speedup means the complete matched GPU timing/reference set did not pass. Five forward LayerNorm cases use CPU JAX sampled correctness separately; CPU timings never substitute for missing native GPU measurements. Synthetic backward saved tensors and sampled comparison (up to258 values/output, full finite check) limit correctness scope; full training loss provides a separate model-level check.','',
 'Original historical native baseline was not modified. This native column is the native mode in the current FlagOS container, not an unlabelled reuse of vendor-JAX historical timings.']
(root/'OPERATORS.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ['operators','failures']},indent=2))
