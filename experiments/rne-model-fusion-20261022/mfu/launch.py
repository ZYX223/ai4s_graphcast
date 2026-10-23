from pathlib import Path
import json,os,subprocess,hashlib,datetime,sys
code=Path(__file__).resolve().parent;root=code.parent;out=code/'probe-r2';assert not out.exists()
raw=subprocess.check_output(['/opt/hyhal/bin/hy-smi','-d','12','--showuse','--showmeminfo','vram','--showbus','--json'],universal_newlines=True);card=json.loads(raw)['card12'];assert float(card['HCU use (%)'])==0 and float(card['vram Total Used Memory (MiB)'])<256
image=json.loads((root/'baseline-source/configs/hygon.json').read_text())['image'];out.mkdir()
env=dict(ROCR_VISIBLE_DEVICES='12',HIP_VISIBLE_DEVICES='0',CUDA_VISIBLE_DEVICES='0',GRAPHCAST_PHYSICAL_DEVICE='12',GRAPHCAST_EXPECTED_PCI=card['PCI Bus'].split()[0],GRAPHCAST_EXPECTED_DEVICE='BW1200',OUTPUT_UID=str(os.getuid()),OUTPUT_GID=str(os.getgid()))
cmd=['docker','run','--rm','--network','none','--cpus=16','--memory=40g','--shm-size=8g','--device','/dev/kfd','--device','/dev/mkfd','--device','/dev/dri','--security-opt','seccomp=unconfined','--security-opt','label=disable']
for k,v in env.items():cmd+=['-e',k+'='+v]
for src,dest in [(code,'/code:ro'),(out,'/output'),(root,'/evidence:ro'),(root/'baseline-source','/repo:ro'),(Path('/opt/hyhal'),'/opt/hyhal:ro')]:cmd+=['-v',str(src)+':'+dest]
cmd+=['--entrypoint','bash',image,'/code/in_container.sh']
sources={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [code/'hlo_cost.py',code/'in_container.sh',root/'baseline-source/kernels/operator_bench/gemm_bench.hip',root/'baseline-source/kernels/xf32_rne/rne_kernel.hip',root/'baseline-source/kernels/xf32_rne/rne_kernel.h']}
(out/'launch.json').write_text(json.dumps(dict(command=cmd,source_sha256=sources,device_before=json.loads(raw),started_at=datetime.datetime.utcnow().isoformat()+'Z'),indent=2)+'\n')
with (out/'console.log').open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
(out/'exit-code.txt').write_text(str(r.returncode)+'\n')
print((out/'console.log').read_text()[-6000:]);sys.exit(r.returncode)
