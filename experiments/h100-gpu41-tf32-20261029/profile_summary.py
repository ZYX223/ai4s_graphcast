import sqlite3,json,csv,collections
from pathlib import Path
r=Path(__file__).resolve().parent
c=sqlite3.connect(str(r/'profile-r1/model-trace.sqlite'))
k=c.execute('select k.start,k.end,s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.demangledName=s.id order by k.start').fetchall()
m=c.execute('select start,end from CUPTI_ACTIVITY_KIND_MEMCPY').fetchall()
assert k and all(e>s for s,e,n in k)
assert len({x[0] for x in c.execute('select distinct deviceId from CUPTI_ACTIVITY_KIND_KERNEL')})==1
assert any('tf32' in n.lower() for s,e,n in k)
assert not any('round_rne' in n for s,e,n in k)
def union(intervals):
 end=0;total=0
 for a,b in sorted(intervals):
  total+=max(0,b-max(a,end));end=max(end,b)
 return total/1e6
s=dict(scope='One model step after five warmup steps; Nsight profile; not a 40-step GPU measurement',kernel_count=len(k),kernel_sum_ms=sum(e-s for s,e,n in k)/1e6,kernel_union_ms=union([(s,e) for s,e,n in k]),kernel_and_memcpy_union_ms=union([(s,e) for s,e,n in k]+m),memcpy_sum_ms=sum(e-s for s,e in m)/1e6,tf32_kernel_count=sum('tf32' in n.lower() for s,e,n in k),tf32_kernel_sum_ms=sum(e-s for s,e,n in k if 'tf32' in n.lower())/1e6,independent_rne_kernel_count=0)
(r/'profile-r1/gpu-time.json').write_text(json.dumps(s,indent=2)+'\n')
with (r/'profile-r1/kernel-details.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['launch_index','name','start_ns','end_ns','duration_ns']);w.writerows((i,n,a,b,b-a) for i,(a,b,n) in enumerate(k,1))
p=r/'summary.json';d=json.loads(p.read_text());d['single_step_gpu_profile']=s;p.write_text(json.dumps(d,indent=2)+'\n')
print(json.dumps(s,indent=2))
