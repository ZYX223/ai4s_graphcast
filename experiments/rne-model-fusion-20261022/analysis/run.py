"""Run result validation on CPU in the same pinned environment."""
from pathlib import Path
import json,subprocess,sys,os
root=Path(__file__).resolve().parents[1]
image=json.loads((root/'baseline-source/configs/hygon.json').read_text())['image']
cmd=['docker','run','--rm','--network','none','--cpus=8','--memory=12g','-v',str(root)+':/results','-v',str(root.parent/'jax-reorganization-20261016/prepared/data')+':/data:ro','-e','JAX_PLATFORMS=cpu','-e','PYTHONDONTWRITEBYTECODE=1','-e','OPENBLAS_NUM_THREADS=4','-e','PYTHONPATH=/results/baseline-source/src','--entrypoint','bash',image,'-c','source /opt/dtk/env.sh; export LD_PRELOAD=/opt/mpi/lib/libmpi.so:/usr/lib/x86_64-linux-gnu/libsqlite3.so.0; /opt/graphcast-jax/bin/python3 "$@"','bash']+sys.argv[1:]
r=subprocess.run(cmd)
# Result ownership is restored by a separate rootless container user-independent path.
subprocess.run(['docker','run','--rm','--network','none','-v',str(root)+':/results','--entrypoint','chown',image,'-R',str(os.getuid())+':'+str(os.getgid()),'/results'])
sys.exit(r.returncode)
