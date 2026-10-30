"""Count useful dense GEMM arithmetic from a differentiated NO-remat graph.

Dead-output elimination applies before walking nested jit/scan bodies. This is
an executable graph GEMM approximation, not all scalar arithmetic or hardware
instructions. Model must disable both temporal and spatial recomputation.
"""
import collections,math
from jax.interpreters import partial_eval as pe

def summarize_jaxpr(closed):
 rows=[];operators=collections.Counter()
 def walk(closed,multiplier):
  jp=getattr(closed,'jaxpr',closed)
  jp,_=pe.dce_jaxpr(jp,[True]*len(jp.outvars))
  for eq in jp.eqns:
   name=eq.primitive.name
   operators[name]+=multiplier
   if name=='dot_general':
    a,b=eq.invars[:2];contract=eq.params['dimension_numbers'][0][0]
    k=math.prod(a.aval.shape[i] for i in contract)
    outputs=math.prod(eq.outvars[0].aval.shape)
    rows.append(dict(lhs=list(a.aval.shape),rhs=list(b.aval.shape),output=list(eq.outvars[0].aval.shape),
      dimension_numbers=eq.params['dimension_numbers'],calls=multiplier,flops=2*k*outputs*multiplier,
      name_stack=str(eq.source_info.name_stack)))
   nested=[]
   for value in eq.params.values():
    if hasattr(value,'jaxpr') or hasattr(value,'eqns'):nested.append(value)
    elif isinstance(value,(tuple,list)):
     nested.extend(v for v in value if hasattr(v,'jaxpr') or hasattr(v,'eqns'))
   if name in ('cond','while') and nested:
    raise ValueError('Data-dependent branch/loop requires explicit execution count: '+name)
   if 'remat' in name:
    raise ValueError('Useful-FLOPs input graph must disable all recomputation')
   for child in nested:walk(child,multiplier*(eq.params['length'] if name=='scan' else 1))
 walk(closed,1)
 return dict(gemm_flops=sum(r['flops'] for r in rows),gemm_calls=sum(r['calls'] for r in rows),gemms=rows,primitive_calls=dict(operators))
