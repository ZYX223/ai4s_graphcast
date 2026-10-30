"""Reuse logical boundaries, including training's multiple saved outputs."""
import hashlib,json
import jax,numpy as np
from graphcast_bench.logical_bench import primitive_function

def pending_cases(cases,rows):
 seen=[r['id'] for r in rows]
 if len(set(seen))!=len(seen) or not set(seen)<={c['id'] for c in cases}:raise ValueError('Duplicate or foreign result')
 if any(r['status']=='running' for r in rows):raise ValueError('Cannot skip unfinished result')
 return [c for c in cases if c['id'] not in set(seen)]

def replay_group(inputs,outputs,equations,literals):
 functions=[primitive_function('add' if e['primitive']=='add_any' else e['primitive'],e['params']) for e in equations]
 runtime=[k for k in inputs if k not in literals]
 def replay(*args):
  env=dict(literals);env.update(zip(runtime,args))
  for e,fn in zip(equations,functions):
   y=fn(*(env[k] for k in e['inputs']));ys=y if isinstance(y,(tuple,list)) else (y,)
   env.update(zip(e['outputs'],ys))
  return tuple(env[k] for k in outputs)
 return jax.jit(replay)

def training_cases(inventory):
 groups={};excluded=[]
 for g in inventory.data['groups']:
  kind=g['operator'];phase='backward' if 'transpose(' in g['scope'] else 'forward'
  if kind=='device_put' or not any(inventory.variables[v]['dtype'].startswith('float') for v in g['outputs']):
   excluded.append(dict(operator=kind,scope=g['scope'],reason='placement or integer/control result'));continue
  eqs=[inventory.equations[i] for i in g['equation_ids']]
  desc=lambda keys:[(inventory.variables[v]['shape'],inventory.variables[v]['dtype']) for v in keys]
  constants=[]
  for k in g['inputs']:
   d=inventory.variables[k]
   if d['shape']==[] or not d['dtype'].startswith(('float','bool')):
    c=inventory.constant(k)
    if c is None and not d['dtype'].startswith(('float','bool')):raise ValueError('Unresolved index/control input: '+k)
    constants.append('generated-runtime-float' if c is None else hashlib.sha256(np.ascontiguousarray(c).tobytes()).hexdigest())
  signature=dict(operator=kind,phase=phase,inputs=desc(g['inputs']),outputs=desc(g['outputs']),equations=[(e['primitive'],e['params']) for e in eqs],constants=constants)
  key=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()[:16]
  if key not in groups:groups[key]=dict(id=key,operator=kind,phase=phase,group=g,signature=signature,calls=0)
  groups[key]['calls']+=1
 return list(groups.values()),excluded
