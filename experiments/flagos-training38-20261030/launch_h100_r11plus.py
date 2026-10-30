"""Run all horizons sequentially with isolated fresh containers and own checkpoints."""
import datetime,json,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parent
frozen=Path('/data/zyx/graphcast-h100-tf32-20261029')
minimal=Path('/data/zyx/graphcast-training-minimal-20261030')
cfg=json.loads((frozen/'code/configs/h100.json').read_text())
for horizon in range(11,13):
 out=root/('h100-r%02d'%horizon)
 assert not out.exists(),out
 obs=subprocess.check_output(['nvidia-smi','-i','2','--query-gpu=uuid,pci.bus_id,memory.used,utilization.gpu,name','--format=csv,noheader,nounits'],universal_newlines=True).strip()
 uuid,bus,used,util,name=[s.strip() for s in obs.split(',')]
 assert int(used)<256 and int(util)==0 and 'H100' in name,obs
 out.mkdir();(out/'gpu-before.txt').write_text(obs)
 env=dict(NVIDIA_VISIBLE_DEVICES=uuid,NVIDIA_DRIVER_CAPABILITIES='compute,utility',CUDA_VISIBLE_DEVICES='0',GRAPHCAST_PYTHON=cfg['python'])
 cmd=['docker','run','--rm','--name','zyx-training38-h100-20261030','--network','none','--ipc=private','--shm-size=8g','--cpus=16','--memory=96g','--runtime','nvidia','--gpus','device='+uuid,'-v','/dev/null:/var/run/nvidia-container-devices/'+uuid+':ro']
 for src,dst in [(frozen/'code','/repo:ro'),(frozen/'runner','/runner:ro'),(minimal,'/minimal:ro'),(root,'/experiment:ro'),(out,'/output'),(frozen/'data','/weekly:ro'),(minimal/'data','/training-data:ro')]:cmd+=['-v',str(src)+':'+dst]
 for key,value in env.items():cmd+=['-e',key+'='+value]
 args=['--mode','tf32','--rollout',str(horizon)]
 if horizon>1:args+=['--resume','/experiment/h100-r%02d/checkpoint.pkl'%(horizon-1)]
 cmd+=['-w','/repo','--entrypoint',cfg['python'],cfg['image'],'/runner/clone3_compat_exec.py','bash','/experiment/run-h100-r11plus.sh']+args
 (out/'launch.json').write_text(json.dumps(dict(command=cmd,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),baseline_modified=False),indent=2))
 with (out/'console.log').open('w') as log:p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
 (out/'exit-code.txt').write_text(str(p.returncode)+'\n')
 assert p.returncode==0,(horizon,p.returncode)
 r=json.loads((out/'report.json').read_text());assert r['status']=='completed' and r['checkpoint_reload']['status']=='passed'
 print('COMPLETED',horizon,len(r['rows']),r['rows'][-1]['loss'],flush=True)
