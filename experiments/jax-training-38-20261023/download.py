"""Fetch immutable public sample in checked byte ranges, host Python3.6."""
import concurrent.futures as cf
import hashlib, json, os, time, urllib.request, base64
from pathlib import Path
root=Path(__file__).resolve().parent/'data'
meta=json.loads((root/'metadata-hres.json').read_text())
size=int(meta['size']); dest=root/'hres-20220101-025-13-12.nc'
part=dest.with_suffix('.partial')
url='https://storage.googleapis.com/dm_graphcast/'+meta['name']+'?generation='+meta['generation']
fd=os.open(str(part),os.O_CREAT|os.O_RDWR,0o644);os.ftruncate(fd,size)
chunk=8*1024*1024
def fetch(start):
 end=min(start+chunk,size)-1
 for attempt in range(6):
  try:
   req=urllib.request.Request(url,headers={'Range':'bytes=%d-%d'%(start,end)})
   with urllib.request.urlopen(req,timeout=90) as r:
    assert r.status==206 and r.headers['Content-Range'].startswith('bytes %d-%d/'%(start,end)),dict(r.headers)
    pos=start
    while True:
     b=r.read(1024*1024)
     if not b:break
     assert os.pwrite(fd,b,pos)==len(b);pos+=len(b)
    assert pos==end+1,(pos,end)
   return start,end-start+1
  except Exception as e:
   if attempt==5:raise
   print('retry',start,repr(e),flush=True);time.sleep(1+attempt)
receipt=root/'completed-ranges.json'
if receipt.exists():completed=json.loads(receipt.read_text())
else:completed=[]
done=sum(min(chunk,size-s) for s in completed);initial=done;t=time.time()
with cf.ThreadPoolExecutor(max_workers=32) as pool:
 jobs=[pool.submit(fetch,s) for s in range(0,size,chunk) if s not in completed]
 for job in cf.as_completed(jobs):
  start,n=job.result();done+=n;completed.append(start)
  receipt.write_text(json.dumps(sorted(completed)))
  print('download',done,size,'MiB/s',round((done-initial)/(time.time()-t)/2**20,2),flush=True)
os.close(fd)
md=hashlib.md5();sha=hashlib.sha256()
with part.open('rb') as f:
 for b in iter(lambda:f.read(8*1024*1024),b''):md.update(b);sha.update(b)
assert base64.b64encode(md.digest()).decode()==meta['md5Hash']
part.rename(dest)
(root/'source.json').write_text(json.dumps(dict(url=url,sha256=sha.hexdigest(),size=size,md5_verified=True,gcs_metadata=meta),indent=2)+'\n')
print('VERIFIED',sha.hexdigest(),flush=True)
