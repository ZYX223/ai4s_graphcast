"""gpu41 adapter: preserve the frozen model; add only host runtime compatibility."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument('action', choices=['check', 'probe', 'profile', 'benchmark'])
p.add_argument('--label', required=True)
p.add_argument('--device', type=int, default=2)
a = p.parse_args()
root = Path(__file__).resolve().parent
repo = root / 'code'
cfg = json.loads((repo / 'configs/h100.json').read_text())
output = root / a.label
assert not output.exists(), 'Results must never be overwritten'
image = subprocess.check_output(['docker', 'image', 'inspect', cfg['image'], '--format', '{{.Id}}'], text=True).strip()
assert image == cfg['image']
gpu = a.action != 'check'
access, env, observation = [], {}, None
if gpu:
    observation = subprocess.check_output(['nvidia-smi', '-i', str(a.device), '--query-gpu=uuid,pci.bus_id,memory.used,utilization.gpu,name', '--format=csv,noheader,nounits'], text=True).strip()
    uuid, bus, used, util, name = [x.strip() for x in observation.split(',')]
    assert int(used) < 256 and int(util) == 0 and 'H100' in name, observation
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
    assert uuid not in processes, 'Selected GPU has a compute process'
    pci = ':'.join([bus.split(':')[0][-4:]] + bus.split(':')[1:]).lower()
    access = ['--runtime', 'nvidia', '--gpus', 'device=' + uuid,
              '-v', '/dev/null:/var/run/nvidia-container-devices/' + uuid + ':ro']
    env.update(NVIDIA_VISIBLE_DEVICES=uuid, NVIDIA_DRIVER_CAPABILITIES='compute,utility',
               CUDA_VISIBLE_DEVICES='0', GRAPHCAST_EXPECTED_PCI=pci,
               GRAPHCAST_PHYSICAL_DEVICE=str(a.device), GRAPHCAST_EXPECTED_DEVICE='H100')
env.update(GRAPHCAST_PLATFORM='h100', GRAPHCAST_PYTHON=cfg['python'],
           OUTPUT_UID='0', OUTPUT_GID='0')
output.mkdir()
command = ['docker', 'run', '--rm', '--name', 'zyx-graphcast-h100-' + a.label,
           '--network', 'none', '--ipc=private', '--shm-size=8g', '--cpus=16', '--memory=64g'] + access
for src, dst in [(repo, '/repo:ro'), (root / 'runner', '/runner:ro'),
                 (output, '/output'), (root / 'data', '/data:ro')]:
    command += ['-v', str(src) + ':' + dst]
for key, value in env.items():
    command += ['-e', key + '=' + value]
command += ['-w', '/repo', '--entrypoint', cfg['python'], image,
            '/runner/clone3_compat_exec.py', 'bash', '/repo/scripts/in_container.sh', a.action]
if a.action in ['profile', 'benchmark']:
    command += ['--mode', 'tf32', '--repeats', '3']
record = dict(command=command, gpu_before=observation, source_commit='47129e4b2654bc1c8e99c5b77bb7d54b4a191611',
              started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              compatibility_sha256=hashlib.sha256((root/'runner/clone3_compat_exec.py').read_bytes()).hexdigest(),
              precision='JAX tensorfloat32; FP32 storage; no explicit RNE and no precision router')
(output/'launch.json').write_text(json.dumps(record, indent=2)+'\n')
with (output/'console.log').open('w') as log:
    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
(output/'exit-code.txt').write_text(str(result.returncode)+'\n')
print('\n'.join((output/'console.log').read_text().splitlines()[-35:]), flush=True)
if result.returncode == 0 and a.action in ['profile', 'benchmark']:
    report = json.loads((output/'run/report.json').read_text())
    assert report['status'] == 'completed' and report['router_build'] is None
    assert report['mode'] == 'tf32' and report['storage'] == 'float32'
sys.exit(result.returncode)
