"""Inspect each selected GEMM's actual optimized producer fusion."""
import re,json,argparse,collections
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--policy',type=Path,required=True);a=p.parse_args()
text=(a.run/'run/optimized-hlo.txt').read_text();bodies={};current=None
for line in text.splitlines():
 m=re.match(r'^(?:ENTRY )?%([\w.\-]+) .*\{$',line)
 if m:current=m.group(1);bodies[current]=[]
 elif line=='}':current=None
 elif current:bodies[current].append(line)
entry=next(k for k in bodies if any('custom_call_target=' in l for l in bodies[k]))
lines=bodies[entry];defs={}
for l in lines:
 m=re.match(r'\s*(?:ROOT )?%([\w.\-]+) = ',l)
 if m:defs[m.group(1)]=l
def canonical(body):
 names={}
 def renumber(m):
  if m.group(0) not in names:names[m.group(0)]='%v'+str(len(names))
  return names[m.group(0)]
 return '\n'.join(re.sub(r'%[\w.\-]+',renumber,l.split(', metadata=')[0].strip()) for l in body)
canonical_bodies={name:canonical(body) for name,body in bodies.items()}
fusions={}
for name,line in defs.items():
 m=re.search(r' fusion\(.*calls=%([\w.\-]+)',line)
 if m:fusions[name]=m.group(1)
policy=json.loads(a.policy.read_text())['layers'];rows=[]
trace=json.loads((a.run/'trace.json').read_text());events=trace['traceEvents'] if isinstance(trace,dict) else trace
kernel_counts=collections.Counter(e['name'] for e in events if e.get('cat')=='HIPOPS' and e.get('ph')=='X')
for target in policy:
 suffix='/'.join([target['module'].split('/')[-3],target['module'].split('/')[-1],'dot_general'])
 matched=[l for l in lines if 'custom_call_target=' in l and suffix+'"' in l and '/'+target['module'].split('/')[0]+'/' in l];assert len(matched)==1,suffix
 gemm=matched[0];value=re.search(r' custom-call\(%([\w.\-]+)',gemm).group(1);chain=[]
 while True:
  line=defs[value];chain.append(line)
  m=re.search(r' (?:bitcast|reshape|copy|get-tuple-element)\(%([\w.\-]+)',line)
  if not m:break
  value=m.group(1)
 fusion=re.search(r' fusion\(.*calls=%([\w.\-]+)',line)
 body='\n'.join(bodies[fusion.group(1)]) if fusion else ''
 has_rne='rne_ops.py' in body and 'bitcast-convert(' in body and ' and(' in body
 has_producer=(' concatenate(' in body) if target['producer']=='concatenate' else (' logistic(' in body or ' exponential(' in body or 'swish' in body)
 aliases=[name for name,comp in fusions.items() if fusion and canonical_bodies[comp]==canonical_bodies[fusion.group(1)]]
 kernels=[re.sub(r'\W','_',name) for name in aliases]
 calls=sum(kernel_counts[k]+kernel_counts['wrapped_'+k] for k in kernels)
 rows.append(dict(module=target['module'],producer=target['producer'],fusion=value,computation=fusion.group(1) if fusion else None,rne_and_producer_in_same_fusion=bool(has_rne and has_producer),captured_equivalent_kernel_calls=calls,kernel_aliases=kernels,chain=chain))
groups={}
for row in rows:
 key=tuple(sorted(row['kernel_aliases']))
 groups.setdefault(key,[]).append(row['module'])
group_audit=[]
for aliases,modules in groups.items():
 calls=sum(kernel_counts[n]+kernel_counts['wrapped_'+n] for n in aliases)
 group_audit.append(dict(kernel_aliases=aliases,selected_modules=modules,expected=len(modules),observed=calls))
 assert calls==len(modules),(aliases,calls,len(modules))
selected_kernel_names=set(k for row in rows for k in row['kernel_aliases'])
selected_runtime_calls=sum(kernel_counts[n]+kernel_counts['wrapped_'+n] for n in selected_kernel_names)
result=dict(groups=group_audit,selected_runtime_calls=selected_runtime_calls,selected_count=len(rows),same_fusion_count=sum(r['rne_and_producer_in_same_fusion'] for r in rows),runtime_seen_count=sum(r['captured_equivalent_kernel_calls']>0 for r in rows),rows=rows)
(a.run/'hlo-fusion-audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['rows','groups']},indent=2))
assert result['same_fusion_count']==79 and result['runtime_seen_count']==79 and selected_runtime_calls==79
