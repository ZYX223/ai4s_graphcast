import subprocess,time
from pathlib import Path
root=Path(__file__).resolve().parent
for h in range(1,13):
 folder=root/('flagos-r%02d'%h)
 while not (folder/'report.json').exists():time.sleep(10)
 assert int((folder/'exit-code.txt').read_text())==0
 remote='/tmp/zyx-flagos-training38-20261030/'+folder.name
 subprocess.run(['docker','exec','-e','CUDA_VISIBLE_DEVICES=-1','-e','JAX_PLATFORMS=cpu','-e','TF_CPP_MIN_LOG_LEVEL=3','flagos-jax-test','python','/tmp/zyx-training38-experiment/xplane_timing.py',remote],check=True)
 subprocess.run(['docker','cp','flagos-jax-test:'+remote+'/xplane-kernels.json',str(folder/'xplane-kernels.json')],check=True)
