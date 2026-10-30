"""Pure interval selection, nanosecond union; copies and CPU excluded."""
import re
import sqlite3

def interval_union(intervals):
 intervals=sorted(intervals)
 if not intervals:return 0
 total=0;lo,hi=intervals[0]
 for a,b in intervals[1:]:
  if b<a:raise ValueError('Reversed time interval')
  if a>hi:total+=hi-lo;lo,hi=a,b
  else:hi=max(hi,b)
 return total+hi-lo

def jax_update_kernels(events):
 devices={e['pid'] for e in events if e.get('ph')=='M' and e.get('name')=='process_name' and e.get('args',{}).get('name','').startswith('/device:GPU:')}
 ranges={int(e['name'].rsplit('-',1)[1]):(e['ts'],e['ts']+e['dur']) for e in events if e.get('ph')=='X' and re.fullmatch(r'train-update-\d+',e.get('name',''))}
 if len(devices)!=1 or not ranges:raise ValueError('Missing or ambiguous GPU/update capture')
 result={u:[] for u in ranges}
 for e in events:
  if e.get('pid') not in devices or e.get('ph')!='X':continue
  name=e['name']
  if name.lower().startswith(('memcpy','memset')):continue
  a=e['ts'];b=a+e['dur']
  for u,(lo,hi) in ranges.items():
   if lo<=a and b<=hi:result[u].append((round(a*1000),round(b*1000),name))
 if any(not ks for ks in result.values()):raise ValueError('Update with no GPU kernels')
 return result

def nsight_update_kernels(path,rows):
 con=sqlite3.connect(str(path))
 if con.execute('pragma integrity_check').fetchone()[0]!='ok':raise ValueError('Corrupt Nsight database')
 epoch=con.execute('select * from TARGET_INFO_SESSION_START_TIME').fetchone()[0]
 if con.execute('select count(distinct deviceId) from CUPTI_ACTIVITY_KIND_KERNEL').fetchone()[0]!=1:raise ValueError('Expected one device')
 kernels=con.execute('select k.start,k.end,s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.demangledName=s.id order by k.start').fetchall();con.close()
 result={r['update']:[(a,b,n) for a,b,n in kernels if r['trace_begin_ns']<=epoch+a and epoch+b<=r['trace_end_ns']] for r in rows}
 if any(not v for v in result.values()) or sum(len(v) for v in result.values())!=len(kernels):raise ValueError('Missing, duplicate, or out-of-update CUDA kernels')
 return result
