"""Fresh, resource-bounded one-device process for each training stage."""
import argparse,datetime,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('action',choices=['cpu','probe','inspect','train'])
p.add_argument('--output',type=Path,required=True);p.add_argument('--device',type=int,default=12)
p.add_argument('--mode',choices=['fp32','rne_xf32'],default='fp32');p.add_argument('--rollout',type=int,default=1)
p.add_argument('--mem-fraction',type=float,default=.70);p.add_argument('--resume',default='')
p.add_argument('--limit',type=int,default=38);p.add_argument('--tiny',action='store_true')
p.add_argument('--preflight',action='store_true')
a=p.parse_args();root=Path(__file__).resolve().parents[1];old=root.parent/'jax-single-card-training-20261022';out=a.output.resolve()
assert not out.exists();card=None
env=dict(OUTPUT_UID=str(os.getuid()),OUTPUT_GID=str(os.getgid()),XLA_PYTHON_CLIENT_MEM_FRACTION=str(a.mem_fraction))
access=[]
if a.action in ['probe','train']:
 card=json.loads(subprocess.check_output(['/opt/hyhal/bin/hy-smi','-d',str(a.device),'--showuse','--showmeminfo','vram','--showbus','--json'],universal_newlines=True))['card'+str(a.device)]
 assert float(card['HCU use (%)'])==0 and float(card['vram Total Used Memory (MiB)'])<512,'Device busy'
 env.update(ROCR_VISIBLE_DEVICES=str(a.device),HIP_VISIBLE_DEVICES='0',CUDA_VISIBLE_DEVICES='0',PHYSICAL_DEVICE=str(a.device),EXPECTED_PCI=card['PCI Bus'].split()[0].lower())
 access=['--device','/dev/kfd','--device','/dev/mkfd','--device','/dev/dri','--security-opt','seccomp=unconfined','--security-opt','label=disable']
out.mkdir(parents=True);shutil.copytree(str(root/'experiment'),str(out/'source'),ignore=shutil.ignore_patterns('__pycache__'))
(root/'cache').mkdir(exist_ok=True)
cmd=['docker','run','--rm','--name','zyx-graphcast-train38-'+out.name,'--network','none','--ipc=private','--shm-size=8g','--cpus=16','--memory=128g']+access
mounts=[(out/'source','/experiment:ro'),(old/'official','/official:ro'),(old/'dependencies','/dependencies:ro'),(root/'data','/training-data:ro'),(root,'/experiment-root:ro'),(root/'cache','/cache'),(root.parent/'jax-reorganization-20261016/prepared/data','/weekly:ro'),(Path('/opt/hyhal'),'/opt/hyhal:ro'),(root.parent/'rne-model-fusion-20261022/baseline-source/kernels/xf32_rne','/rne:ro'),(out,'/output')]
mounts.append((old/'data','/previous-data:ro'))
for h,d in mounts:cmd+=['-v',str(h)+':'+d]
for k,v in env.items():cmd+=['-e',k+'='+v]
cmd+=['--entrypoint','bash','sha256:919cf075fadd5fab21b2ea5978374be00caf1db7d6f02fc90846633c4c4bd546','/experiment/in_container.sh',a.action]
if a.action=='train':
 cmd+=['--mode',a.mode,'--rollout',str(a.rollout),'--limit',str(a.limit)]
 if a.resume:cmd+=['--resume','/experiment-root/'+a.resume]
 if a.tiny:cmd+=['--tiny']
 if a.preflight:cmd+=['--preflight']
manifest={str(f.relative_to(out/'source')):hashlib.sha256(f.read_bytes()).hexdigest() for f in (out/'source').rglob('*') if f.is_file()}
(out/'launch.json').write_text(json.dumps(dict(command=cmd,device_before=card,started_utc=datetime.datetime.utcnow().isoformat(),source_sha256=manifest),indent=2))
with (out/'console.log').open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
(out/'exit-code.txt').write_text(str(r.returncode)+'\n')
print('\n'.join((out/'console.log').read_text().splitlines()[-35:]));sys.exit(r.returncode)
