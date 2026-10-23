"""Reproducible isolated probe launcher; Python 3.6 host compatible."""
from pathlib import Path
import argparse, datetime, hashlib, json, os, subprocess, sys

p=argparse.ArgumentParser(); p.add_argument('action',choices=['check','bench','profile'])
p.add_argument('--output',type=Path,required=True);p.add_argument('--device',type=int,default=12)
p.add_argument('--smoke',action='store_true');a=p.parse_args()
root=Path(__file__).resolve().parents[1]; code=Path(__file__).resolve().parent
repo=root/'baseline-source'; config=json.loads((repo/'configs/hygon.json').read_text())
output=a.output.resolve();assert not output.exists()
env={'OUTPUT_UID':str(os.getuid()),'OUTPUT_GID':str(os.getgid()),'EXPERIMENT_SMOKE':str(int(a.smoke))}
access=[];telemetry=None
if a.action!='check':
 raw=subprocess.check_output(['/opt/hyhal/bin/hy-smi','-d',str(a.device),'--showuse','--showmeminfo','vram','--showbus','--json'],universal_newlines=True)
 telemetry=json.loads(raw);card=telemetry['card'+str(a.device)]
 assert float(card['HCU use (%)'])==0 and float(card['vram Total Used Memory (MiB)'])<256,'GPU busy'
 env.update(ROCR_VISIBLE_DEVICES=str(a.device),HIP_VISIBLE_DEVICES='0',CUDA_VISIBLE_DEVICES='0',GRAPHCAST_PHYSICAL_DEVICE=str(a.device),GRAPHCAST_EXPECTED_PCI=card['PCI Bus'].split()[0].lower())
 access=['--device','/dev/kfd','--device','/dev/mkfd','--device','/dev/dri','--security-opt','seccomp=unconfined','--security-opt','label=disable']
output.mkdir(parents=True)
command=['docker','run','--rm','--network','none','--ipc=private','--shm-size=8g','--cpus=16','--memory=64g']+access
for host,dest in [(repo,'/repo:ro'),(code,'/experiment:ro'),(output,'/output'),(Path('/opt/hyhal'),'/opt/hyhal:ro'),(root.parent/'logical-operators-20261022/capture-r3/run','/inventory:ro'),(root.parent/'logical-operators-20261022/bench-full-r2/logical','/catalog:ro')]:
 command+=['-v',str(host)+':'+dest]
for k,v in env.items():command+=['-e',k+'='+v]
command+=['--entrypoint','bash',config['image'],'/experiment/in_container.sh',a.action]
source={str(f.relative_to(code)):hashlib.sha256(f.read_bytes()).hexdigest() for f in code.rglob('*') if f.is_file() and '__pycache__' not in f.parts}
record=dict(command=command,device_before=telemetry,started_at=datetime.datetime.utcnow().isoformat()+'Z',experiment_sha256=source)
(output/'launch.json').write_text(json.dumps(record,indent=2)+'\n')
with (output/'console.log').open('w') as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
(output/'exit-code.txt').write_text(str(r.returncode)+'\n')
print('\n'.join((output/'console.log').read_text().splitlines()[-30:]));sys.exit(r.returncode)
