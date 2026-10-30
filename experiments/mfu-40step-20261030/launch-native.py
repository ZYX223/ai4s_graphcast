"""Replay frozen native fusion code in an isolated, recovered vendor-JAX runtime."""
from pathlib import Path
import datetime,hashlib,json,os,subprocess,sys
root=Path(__file__).resolve().parent
old=root.parent/'rne-model-fusion-20261022'
output=root/'native-profile'
assert not output.exists()
image='sha256:cfae493b60f9d165ec3cd256753fc885ec045b36283430d0068c57707dcaaffe'
assert json.loads((root/'native-recovery/venv/COMPLETE.json').read_text())['jax']=='0.6.0+das.opt1.dtk2604'
raw=subprocess.check_output(['/opt/hyhal/bin/hy-smi','-d','9','--showuse','--showmeminfo','vram','--showbus','--json'],universal_newlines=True)
card=json.loads(raw)['card9']
assert float(card['HCU use (%)'])==0 and float(card['vram Total Used Memory (MiB)'])<256,'GPU9 busy'
pci=card['PCI Bus'].split()[0].lower();assert pci=='0000:88:00.0'
previous=json.loads((old/'bench-fused-r1/launch.json').read_text())
for name,sha in previous['experiment_sha256'].items():
    assert hashlib.sha256((old/'experiment-r1'/name).read_bytes()).hexdigest()==sha
assert hashlib.sha256((old/'bench-fused-r1/native/libxf32_rne_router.so').read_bytes()).hexdigest()=='48c35f89b457b1d142c8cd1143bd647f160f3422dd0756be8e5894dd8e45475a'
output.mkdir()
cmd=['docker','run','--rm','--name','zyx-graphcast-native-mfu40-20261030','--network','none','--ipc=private','--shm-size=8g','--cpus=16','--memory=64g','--device','/dev/kfd','--device','/dev/mkfd','--device','/dev/dri','--security-opt','seccomp=unconfined','--security-opt','label=disable']
mounts=[(old/'baseline-source','/repo:ro'),(old/'experiment-r1','/experiment:ro'),(old/'bench-fused-r1/native','/baseline-native:ro'),(root,'/measurement:ro'),(output,'/output'),(root/'native-recovery/venv','/opt/graphcast-jax:ro'),(Path('/opt/hyhal'),'/opt/hyhal:ro'),(root.parent/'jax-reorganization-20261016/prepared/data','/data:ro')]
for src,dst in mounts:cmd+=['-v',str(src)+':'+dst]
for k,v in dict(ROCR_VISIBLE_DEVICES='9',HIP_VISIBLE_DEVICES='0',CUDA_VISIBLE_DEVICES='0',GRAPHCAST_PHYSICAL_DEVICE='9',GRAPHCAST_EXPECTED_PCI=pci).items():cmd+=['-e',k+'='+v]
cmd+=['--entrypoint','bash',image,'/measurement/run-native.sh']
record=dict(command=cmd,started_utc=datetime.datetime.utcnow().isoformat()+'Z',device_before=json.loads(raw),original_image='sha256:919cf075fadd5fab21b2ea5978374be00caf1db7d6f02fc90846633c4c4bd546',recovered_base=image,
    scope='Frozen Oct22 native JAX producer-RNE fusion, same original binary; restored 40-package Python freeze, DTK release unchanged; base image identity differs. No FlagOS native backend substitution.',
    previous_experiment_hashes=previous['experiment_sha256'])
(output/'launch.json').write_text(json.dumps(record,indent=2)+'\n')
with (output/'console.log').open('w') as log:rc=subprocess.call(cmd,stdout=log,stderr=subprocess.STDOUT)
(output/'exit-code.txt').write_text(str(rc)+'\n')
print((output/'console.log').read_text()[-7000:]);sys.exit(rc)
