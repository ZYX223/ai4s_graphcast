"""Reuse frozen gpu41 image, source, data and clone3 compatibility, with a new output."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

root=Path(__file__).resolve().parent
frozen=Path('/data/zyx/graphcast-h100-tf32-20261029')
cfg=json.loads((frozen/'code/configs/h100.json').read_text())
out=root/'h100-profile'
assert not out.exists()
obs=subprocess.check_output(['nvidia-smi','-i','2','--query-gpu=uuid,pci.bus_id,memory.used,utilization.gpu,name','--format=csv,noheader,nounits'],universal_newlines=True).strip()
uuid,bus,used,util,name=[s.strip() for s in obs.split(',')]
assert int(used)<256 and int(util)==0 and 'H100' in name, obs
assert uuid not in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],universal_newlines=True)
assert subprocess.check_output(['docker','image','inspect',cfg['image'],'--format','{{.Id}}'],universal_newlines=True).strip()==cfg['image']
out.mkdir()
(out/'nvidia-smi-before.txt').write_text(subprocess.check_output(['nvidia-smi','-i','2','-q'],universal_newlines=True))
env=dict(NVIDIA_VISIBLE_DEVICES=uuid,NVIDIA_DRIVER_CAPABILITIES='compute,utility',CUDA_VISIBLE_DEVICES='0',
 GRAPHCAST_EXPECTED_PCI='0000:3a:00.0',GRAPHCAST_PHYSICAL_DEVICE='2',GRAPHCAST_EXPECTED_DEVICE='H100',GRAPHCAST_PLATFORM='h100',GRAPHCAST_PYTHON=cfg['python'])
cmd=['docker','run','--rm','--name','zyx-graphcast-mfu40-20261030','--network','none','--ipc=private','--shm-size=8g','--cpus=16','--memory=64g',
 '--runtime','nvidia','--gpus','device='+uuid,'-v','/dev/null:/var/run/nvidia-container-devices/'+uuid+':ro']
for src,dst in [(frozen/'code','/repo:ro'),(frozen/'runner','/runner:ro'),(root,'/measure:ro'),(out,'/output'),(frozen/'data','/data:ro')]:cmd+=['-v',str(src)+':'+dst]
for key,value in env.items():cmd+=['-e',key+'='+value]
cmd+=['-w','/repo','--entrypoint',cfg['python'],cfg['image'],'/runner/clone3_compat_exec.py','bash','/measure/run-h100.sh']
(out/'launch.json').write_text(json.dumps(dict(command=cmd,gpu_before=obs,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),baseline_modified=False),indent=2)+'\n')
with (out/'console.log').open('w') as log: result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
(out/'exit-code.txt').write_text(str(result.returncode)+'\n')
(out/'nvidia-smi-after.txt').write_text(subprocess.check_output(['nvidia-smi','-i','2','-q'],universal_newlines=True))
assert result.returncode==0
r=json.loads((out/'run/report.json').read_text())
assert r['status']=='completed' and r['measured_steps']==40 and len(r['rows'])==3 and r['router_build'] is None
