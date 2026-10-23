"""Logical operator boundaries and an unfused, single-step JAXPR inventory."""
import re

from .assets import write_json


def boundary(scope, primitive):
    # Composite operators own their implementation primitives, preventing
    # LayerNorm reductions / SiLU exponentials / Linear bias double counting.
    for pattern, kind in ((r'(^|/)([^/]*layer_norm(?:_\d+)?)(?=/|$)', 'layer_norm'),
                          (r'(^|/)(linear(?:_\d+)?)(?=/|$)', 'linear'),
                          (r'(^|/)jit\(silu\)(?=/|$)', 'silu')):
        match = re.search(pattern, scope)
        if match:
            return kind, scope[:match.end()]
    return primitive, None


def group_equations(rows):
    groups, composite = [], {}
    seen = set()
    for row in rows:
        if row['id'] in seen:
            raise ValueError('Duplicate equation ID')
        seen.add(row['id'])
        kind, prefix = boundary(row['scope'], row['primitive'])
        key = (kind, prefix) if prefix else (kind, row['id'])
        if key not in composite:
            group = dict(operator=kind, scope=prefix or row['scope'], equation_ids=[])
            composite[key] = group
            groups.append(group)
        composite[key]['equation_ids'].append(row['id'])
    return groups


def export_jaxpr(closed, folder):
    """Flatten transparent calls and a length-one scan; retain original constants.

    No HLO-kernel counts are substituted for logical-call counts. Dynamic loops
    or branches are rejected instead of guessing their executed path.
    """
    import enum
    import numpy as np
    from pathlib import Path
    folder = Path(folder)
    folder.mkdir()
    rows, variables, constants = [], {}, {}
    next_var = [0]

    def value(v):
        if isinstance(v, enum.Enum): return {'enum': type(v).__name__, 'name': v.name}
        if isinstance(v, np.dtype): return {'dtype': str(v)}
        if hasattr(v, '_fields'):
            return {'namedtuple': type(v).__name__, 'fields': {k: value(getattr(v, k)) for k in v._fields}}
        if isinstance(v, (tuple, list)): return [value(x) for x in v]
        if isinstance(v, dict): return {str(k): value(x) for k, x in v.items()}
        if v is None or isinstance(v, (str, int, float, bool)): return v
        if isinstance(v, np.generic): return v.item()
        return {'repr': str(v), 'type': type(v).__name__}

    def fresh(v, constant=None):
        key = 'v' + str(next_var[0]); next_var[0] += 1
        variables[key] = {'shape': list(v.aval.shape), 'dtype': str(v.aval.dtype)}
        if constant is not None:
            constants[key] = np.asarray(constant)
            variables[key]['constant'] = key
        return key

    def walk(obj, bindings=None, prefix=''):
        jp = obj.jaxpr if hasattr(obj, 'jaxpr') else obj
        consts = obj.consts if hasattr(obj, 'consts') else []
        env = {}
        for v, c in zip(jp.constvars, consts): env[v] = fresh(v, c)
        for i, v in enumerate(jp.invars): env[v] = bindings[i] if bindings is not None else fresh(v)

        def get(v):
            if hasattr(v, 'val'): return fresh(v, v.val)
            if v not in env: raise ValueError('Unbound JAXPR input: ' + str(v))
            return env[v]

        for eq in jp.eqns:
            name = eq.primitive.name
            scope = str(eq.source_info.name_stack)
            scope = prefix + ('/' if prefix and scope else '') + scope
            ins = [get(v) for v in eq.invars]
            if name in ('jit', 'pjit', 'custom_jvp_call', 'closed_call', 'remat2'):
                child = eq.params.get('jaxpr', eq.params.get('call_jaxpr'))
                if child is None: raise ValueError('Missing nested JAXPR for ' + name)
                suffix = eq.params.get('name')
                child_scope = scope + ('/jit(' + suffix + ')' if suffix else '')
                outs = walk(child, ins, child_scope.strip('/'))
            elif name == 'scan':
                if eq.params['length'] != 1:
                    raise ValueError('Inventory expects exactly one autoregressive step')
                # Scan scalar controls / xs carry an outer time dimension. Keep
                # their body aval exact; all nonsliced constants/carry alias.
                body = eq.params['jaxpr']
                count = eq.params['num_consts'] + eq.params['num_carry']
                bindings2 = ins[:count]
                for v, key in zip(body.jaxpr.invars[count:], ins[count:]):
                    c = constants[key][0] if key in constants else None
                    sliced = fresh(v, c)
                    rows.append(dict(id=len(rows)+1, primitive='reshape', scope=scope+'/scan_unstack',
                                     inputs=[key], outputs=[sliced], params=dict(new_sizes=list(v.aval.shape), dimensions=None, sharding=None)))
                    bindings2.append(sliced)
                child_out = walk(body, bindings2, scope + '/scan_body')
                outs = child_out[:eq.params['num_carry']]
                for v, key in zip(eq.outvars[len(outs):], child_out[len(outs):]):
                    out = fresh(v)
                    rows.append(dict(id=len(rows) + 1, primitive='reshape', scope=scope + '/scan_stack',
                                     inputs=[key], outputs=[out], params=dict(new_sizes=list(v.aval.shape), dimensions=None, sharding=None)))
                    outs.append(out)
            elif name in ('while', 'cond'):
                raise ValueError('Dynamic control flow cannot be assigned guessed counts: ' + name)
            else:
                outs = [fresh(v) for v in eq.outvars]
                rows.append({'id': len(rows) + 1, 'primitive': name, 'scope': scope,
                             'inputs': ins, 'outputs': outs, 'params': value(eq.params)})
            for v, key in zip(eq.outvars, outs): env[v] = key
        return [get(v) for v in jp.outvars]

    outputs = walk(closed)
    raw_equation_count = len(rows)
    required = set(outputs)
    live = []
    for row in reversed(rows):
        if required.intersection(row['outputs']):
            live.append(row)
            required.update(row['inputs'])
    rows = list(reversed(live))
    groups = group_equations(rows)
    owners = {eid: i for i, g in enumerate(groups) for eid in g['equation_ids']}
    by_id = {r['id']: r for r in rows}
    uses = {}
    for r in rows:
        for v in r['inputs']: uses.setdefault(v, set()).add(owners[r['id']])
    for i, g in enumerate(groups):
        equations = [by_id[k] for k in g['equation_ids']]
        made = {v for e in equations for v in e['outputs']}
        g['inputs'] = list(dict.fromkeys(v for e in equations for v in e['inputs'] if v not in made))
        g['outputs'] = [v for e in equations for v in e['outputs'] if v in outputs or uses.get(v, set()) - {i}]
    np.savez(folder / 'constants.npz', **constants)
    result = dict(schema='graphcast.logical_inventory.v1', scope='one model step',
                  equations=rows, variables=variables, groups=groups, outputs=outputs,
                  raw_equation_count=raw_equation_count, dead_equation_count=raw_equation_count-len(rows))
    write_json(folder / 'inventory.json', result)
    (folder / 'jaxpr.txt').write_text(str(closed))
    return result
