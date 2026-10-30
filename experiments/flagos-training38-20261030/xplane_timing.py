"""Read full XSpace protobuf: Chrome trace export truncates at 1M events.

Integer ns avoids loss of precision when adding UNIX epoch and ps offsets.
Only physical stream events are used, not derived compiler/op timelines.
"""
import argparse,hashlib,json,re
from pathlib import Path

def update_kernels(space):
 devices=[p for p in space.planes if p.name.startswith('/device:GPU:')]
 if len(devices)!=1:raise ValueError('Expected one GPU plane')
 ranges={}
 for p in space.planes:
  if p.name!='/host:CPU':continue
  ids={k:int(v.name.rsplit('-',1)[1]) for k,v in p.event_metadata.items() if re.fullmatch(r'train-update-\d+',v.name)}
  for line in p.lines:
   for e in line.events:
    if e.metadata_id not in ids:continue
    u=ids[e.metadata_id]
    if u in ranges:raise ValueError('Duplicate update annotation')
    begin=line.timestamp_ns+e.offset_ps//1000
    ranges[u]=(begin,begin+e.duration_ps//1000)
 if not ranges:raise ValueError('Missing update annotations')
 result={u:[] for u in sorted(ranges)};p=devices[0]
 for line in p.lines:
  if not line.name.startswith('Stream #'):continue
  for e in line.events:
   name=p.event_metadata[e.metadata_id].name
   if name.lower().startswith(('memcpy','memset')):continue
   a=line.timestamp_ns+e.offset_ps//1000;b=a+e.duration_ps//1000
   matches=[u for u,(lo,hi) in ranges.items() if lo<=a and b<=hi]
   if len(matches)>1:raise ValueError('Overlapping update ranges')
   if matches:result[matches[0]].append((a,b,name))
 if any(not k for k in result.values()):raise ValueError('Update missing GPU kernels')
 return result

def main():
 from tensorflow.tsl.profiler.protobuf import xplane_pb2
 p=argparse.ArgumentParser();p.add_argument('folder');a=p.parse_args()
 folder=Path(a.folder);paths=list((folder/'jax-trace').rglob('*.xplane.pb'))
 assert len(paths)==1
 raw=paths[0].read_bytes();space=xplane_pb2.XSpace();space.ParseFromString(raw)
 if space.errors:raise ValueError(list(space.errors))
 rows=json.loads((folder/'report.json').read_text())['rows'];result=update_kernels(space)
 assert set(result)=={r['update'] for r in rows}
 # Every rocBLAS call rounds both operands in this unfused training route.
 for r in rows:
  count=sum('rne' in n.lower() for _,_,n in result[r['update']])
  assert count==2*r['gemm']['rounded'],(r['update'],count,r['gemm'])
 out=dict(source=str(paths[0]),source_sha256=hashlib.sha256(raw).hexdigest(),
  source_errors=list(space.errors),source_warnings=list(space.warnings),
  timing='full raw XSpace; physical GPU kernel intervals; exclude memcpy/memset; integer ns',
  updates=result)
 (folder/'xplane-kernels.json').write_text(json.dumps(out,separators=(',',':'))+'\n')
 print(folder.name,'RAW_CAPTURE_VALIDATED',len(result),'updates',sum(len(k) for k in result.values()),'kernels',flush=True)

if __name__=='__main__':main()
