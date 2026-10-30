import json,subprocess
from pathlib import Path
root=Path(__file__).resolve().parent
frozen=Path('/data/zyx/graphcast-h100-tf32-20261029');minimal=Path('/data/zyx/graphcast-training-minimal-20261030')
cfg=json.loads((frozen/'code/configs/h100.json').read_text());out=root/'flops-h100';out.mkdir(exist_ok=False)
cmd=['docker','run','--rm','--name','zyx-training38-flops-cpu','--network','none','--ipc=private','--cpus=8','--memory=64g']
for src,dst in [(root/'analysis','/analysis:ro'),(minimal,'/minimal:ro'),(frozen/'data','/weekly:ro'),(frozen/'runner','/runner:ro'),(out,'/output')]:cmd+=['-v',str(src)+':'+dst]
for k,v in dict(JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='-1',PYTHONPATH='/analysis:/minimal/experiment:/minimal/official:/minimal/deps',TRAINING_DATA='/minimal/data',TRAINING_WEEKLY='/weekly',FLOPS_OUTPUT='/output',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4').items():cmd+=['-e',k+'='+v]
cmd+=['--entrypoint',cfg['python'],cfg['image'],'/runner/clone3_compat_exec.py',cfg['python'],'/analysis/count_flops.py']
p=subprocess.run(cmd);(out/'exit-code.txt').write_text(str(p.returncode)+'\n');raise SystemExit(p.returncode)
