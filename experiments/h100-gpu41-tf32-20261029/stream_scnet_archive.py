"""Copy a growing local download with a fixed length and SHA256 gate."""
import hashlib,subprocess,time,json
from pathlib import Path
source=Path('/home/zhangyuxin/zyx/graphcast/outputs/h100-gpu41-tf32-20261029-results.tar.gz.part')
root=source.parent/'h100-gpu41-tf32-20261029'
expected_size=936858154
expected_sha='7c5c8efc7398476039c0af18c16833f0658f1a3989a1333107c0c6434ba126b6'
target='/work2/ac9844np6r/zyx/delivery/graphcast-h100-gpu41-tf32-20261029-results.tar.gz'
ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','scnet-hygon']
with (root/'scnet-transfer.log').open('w') as log:
 process=subprocess.Popen(ssh+['test ! -e '+target+'.part && cat > '+target+'.part'],stdin=subprocess.PIPE,stdout=log,stderr=log)
 count=0; digest=hashlib.sha256();last=time.monotonic()
 try:
  with source.open('rb') as f:
   while count<expected_size:
    chunk=f.read(min(1024*1024,expected_size-count))
    if chunk:
     process.stdin.write(chunk);count+=len(chunk);digest.update(chunk);last=time.monotonic()
    else:
     if process.poll() is not None:raise RuntimeError('Remote transfer ended early')
     if time.monotonic()-last>240:raise TimeoutError('No new source bytes for240seconds')
     time.sleep(.25)
  process.stdin.close()
  assert process.wait()==0
  assert digest.hexdigest()==expected_sha
 except BaseException:
  process.terminate();raise
 remote_sha=subprocess.check_output(ssh+['sha256sum '+target+'.part'],universal_newlines=True).split()[0]
 assert remote_sha==expected_sha
 subprocess.run(ssh+['test ! -e '+target+' && mv '+target+'.part '+target],check=True)
 (root/'scnet-archive-transfer.json').write_text(json.dumps({'path':target,'bytes':count,'sha256':remote_sha,'verified':True},indent=2)+'\n')
 print('SCNET_ARCHIVE_VERIFIED',count,remote_sha,flush=True)
