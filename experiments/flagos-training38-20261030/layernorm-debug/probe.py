"""Reproduce native XLA's multi-output LayerNorm compilation failure."""
import json,os,time
from pathlib import Path
import jax,numpy as np
from graphcast_bench.logical_bench import Inventory
from operator_replay import replay_group,training_cases

root=Path(os.environ['LN_OUTPUT']);root.mkdir(parents=True,exist_ok=True)
inv=Inventory(Path('/tmp/zyx-training38-operators/logical'))
cases,_=training_cases(inv)
case=next(c for c in cases if c['id']==os.environ.get('LN_CASE','5cd0a35454145014'))
g=case['group'];args=[];literals={}
for i,key in enumerate(g['inputs']):
    spec=inv.variables[key];constant=inv.constant(key);dtype=np.dtype(spec['dtype']);shape=tuple(spec['shape'])
    if shape==() and constant is not None:
        literals[key]=np.asarray(constant,dtype=dtype);continue
    if constant is not None:host=np.asarray(constant,dtype=dtype)
    elif dtype.kind=='b':host=np.ones(shape,dtype)
    elif dtype.kind=='f':host=np.random.RandomState(41030+i).uniform(.25,1.25,shape).astype(dtype)
    else:raise ValueError('Missing constant: '+key)
    args.append(jax.device_put(host))
jax.block_until_ready(args)
f=replay_group(g['inputs'],g['outputs'],[inv.equations[i] for i in g['equation_ids']],literals)
lowered=f.lower(*args)
(root/'before.mlir').write_text(str(lowered.compiler_ir()))
print('COMPILE',case['id'],case['signature']['outputs'],flush=True)
ex=lowered.compile()
(root/'optimized-hlo.txt').write_text(ex.as_text())
ys=ex(*args);jax.block_until_ready(ys)
(root/'result.json').write_text(json.dumps(dict(case=case['id'],status='passed',outputs=[list(y.shape) for y in ys]),indent=2)+'\n')
print('PASS',case['id'],flush=True)
