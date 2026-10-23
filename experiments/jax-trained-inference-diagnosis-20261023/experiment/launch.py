"""Isolated inference processes, using a free physical accelerator only."""
import argparse, datetime, hashlib, json, os, shutil, subprocess, sys
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('action', choices=['fp32', 'rne_xf32', 'compare', 'parameters'])
p.add_argument('--output', type=Path, required=True)
p.add_argument('--checkpoint-mode', choices=['fp32', 'rne_xf32'])
p.add_argument('--device', type=int, default=12)
a=p.parse_args()
root=Path(__file__).resolve().parents[1]
out=a.output.resolve()
assert not out.exists(), 'Refusing to replace an existing result'
env=dict(OUTPUT_UID=str(os.getuid()), OUTPUT_GID=str(os.getgid()), XLA_PYTHON_CLIENT_MEM_FRACTION='.70')
access=[]
card=None
if a.action not in ('compare', 'parameters'):
    card=json.loads(subprocess.check_output(['/opt/hyhal/bin/hy-smi','-d',str(a.device),'--showuse','--showmeminfo','vram','--showbus','--json'],universal_newlines=True))['card'+str(a.device)]
    assert float(card['HCU use (%)']) == 0 and float(card['vram Total Used Memory (MiB)']) < 512, 'Device busy'
    env.update(ROCR_VISIBLE_DEVICES=str(a.device), HIP_VISIBLE_DEVICES='0', CUDA_VISIBLE_DEVICES='0', PHYSICAL_DEVICE=str(a.device), EXPECTED_PCI=card['PCI Bus'].split()[0].lower())
    access=['--device','/dev/kfd','--device','/dev/mkfd','--device','/dev/dri','--security-opt','seccomp=unconfined','--security-opt','label=disable']
out.mkdir(parents=True)
shutil.copytree(str(root/'experiment'),str(out/'source'),ignore=shutil.ignore_patterns('__pycache__'))
cmd=['docker','run','--rm','--name','zyx-graphcast-trained-infer40-'+out.name,'--network','none','--ipc=private','--shm-size=8g','--cpus=16','--memory=64g']+access
mounts=[(out/'source','/experiment:ro'), (root.parent/'rne-model-fusion-20261022/baseline-source','/repo:ro'),
        (root.parent/'jax-single-card-training-20261022/dependencies','/dependencies:ro'),
        (root.parent/'jax-training-38-20261023','/training:ro'),
        (root.parent/'jax-reorganization-20261016/prepared/data','/data:ro'),
        (root.parent/'rne-model-fusion-20261022/baseline-source/kernels/xf32_rne','/rne:ro'),
        (Path('/opt/hyhal'),'/opt/hyhal:ro'), (root,'/results:ro'), (root.parent/'jax-trained-inference-40-20261023','/original-results:ro'), (out,'/output')]
for h,d in mounts: cmd+=['-v',str(h)+':'+d]
for k,v in env.items(): cmd+=['-e',k+'='+v]
cmd+=['--entrypoint','bash','sha256:919cf075fadd5fab21b2ea5978374be00caf1db7d6f02fc90846633c4c4bd546','/experiment/in_container.sh',a.action]
if a.checkpoint_mode: cmd+=['--checkpoint-mode',a.checkpoint_mode]
manifest={str(f.relative_to(out/'source')):hashlib.sha256(f.read_bytes()).hexdigest() for f in (out/'source').rglob('*') if f.is_file()}
(out/'launch.json').write_text(json.dumps(dict(command=cmd,device_before=card,started_utc=datetime.datetime.utcnow().isoformat(),source_sha256=manifest),indent=2))
with (out/'console.log').open('w') as f: result=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
(out/'exit-code.txt').write_text(str(result.returncode)+'\n')
print('\n'.join((out/'console.log').read_text().splitlines()[-20:]))
sys.exit(result.returncode)
