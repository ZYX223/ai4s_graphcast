"""Per-update HIP device time; requires absolute trace/host clock agreement."""
import collections,json
from pathlib import Path
def account(events,begin,end):
 intervals=[];names=collections.defaultdict(lambda:dict(calls=0,duration_ns=0));copies=0
 for e in events:
  if e.get('ph')!='X':continue
  args=e.get('args',{})
  if 'BeginNs' not in args or 'EndNs' not in args:continue
  b=int(args['BeginNs']);f=int(args['EndNs'])
  if b<begin or f>end:continue
  if e.get('cat')=='HIPOPS':
   assert f>=b;intervals.append((b,f));v=names[e['name']];v['calls']+=1;v['duration_ns']+=f-b
  elif 'COPY' in e.get('cat','').upper():copies+=f-b
 if not intervals:raise ValueError('No device kernels inside update interval')
 union=0;last=None
 for b,e in sorted(intervals):
  union+=e-b if last is None or b>=last else max(0,e-last)
  last=e if last is None else max(last,e)
 return dict(gpu_kernel_count=len(intervals),gpu_kernel_sum_ms=sum(e-b for b,e in intervals)/1e6,
  gpu_kernel_busy_union_ms=union/1e6,gpu_kernel_span_ms=(max(e for b,e in intervals)-min(b for b,e in intervals))/1e6,
  device_copy_sum_ms=copies/1e6,kernels=dict(names))

def summarize(run):
 report=json.loads((run/'report.json').read_text())
 trace=json.loads((run/'trace.json').read_text());events=trace['traceEvents'] if isinstance(trace,dict) else trace
 rows=[]
 for row in report['rows']:
  result=account(events,row['trace_begin_ns'],row['trace_end_ns']);detail=result.pop('kernels')
  (run/('kernels-update-%02d.json'%row['update'])).write_text(json.dumps(detail,indent=2))
  rows.append(dict(update=row['update'],**result))
 summary=dict(rows=rows,total_gpu_kernel_sum_ms=sum(r['gpu_kernel_sum_ms'] for r in rows),
  method='Sum of HIPOPS kernel durations in synchronized update windows. Includes forward/backward/recomputation/RNE/gradient averaging/optimizer. Excludes compile,H2D,host diagnostics/checkpoint writes. Busy union and first-last span reported separately.')
 (run/'device-timing.json').write_text(json.dumps(summary,indent=2)+'\n')
 return summary
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();print(json.dumps(summarize(a.run),indent=2))
