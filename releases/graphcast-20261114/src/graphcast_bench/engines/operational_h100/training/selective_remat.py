"""Candidate-specific stage/message/group checkpoint policies, with audit events."""

STAGES = ('_run_grid2mesh_gnn', '_run_mesh_gnn', '_run_mesh2grid_gnn')
CANDIDATES = ('regular3', 'mesh-stage-cse', 'm2g-stage-cse', 'spatial-stage-cse',
              'mesh-message-cse', 'mesh-message-lnremat', 'spatial-stage-bypass-lnremat',
              'mesh-group4', 'mesh-group8', 'mesh-group16', 'mesh-group16-m2g-bypass',
              'mesh-group8-m2g-bypass', 'mesh-group8-spatial-bypass', 'mesh-group8-shifted',
              'mesh-group8-combined', 'mesh-group8-m2g-combined',
              'mesh-group8-m2g-shifted', 'mesh-group8-m2g-gather', 'mesh-group8-m2g-save',
              'mesh-group8-m2g-vjp', 'mesh-group8-m2g-save-stats',
              'mesh-group8-m2g-save-stats-vjp', 'mesh-group8-m2g-save-stats-vjp-edge', 'mesh-group8-m2g-save-stats-vjp-lnvjp', 'mesh-group8-m2g-save-stats-vjp-finalmesh', 'mesh-group8-m2g-save-stats-vjp-finalmesh-g2m')


def remat_action(candidate, function_name):
    if candidate not in CANDIDATES:
        raise ValueError('Unknown selective remat candidate: ' + candidate)
    if candidate == 'spatial-stage-bypass-lnremat' and function_name in STAGES:
        return 'bypass'
    if mesh_group_size(candidate) and function_name == '_run_mesh_gnn':
        return 'bypass'
    if candidate in ('mesh-group16-m2g-bypass', 'mesh-group8-m2g-bypass', 'mesh-group8-spatial-bypass', 'mesh-group8-m2g-combined', 'mesh-group8-m2g-shifted', 'mesh-group8-m2g-gather', 'mesh-group8-m2g-save', 'mesh-group8-m2g-vjp', 'mesh-group8-m2g-save-stats', 'mesh-group8-m2g-save-stats-vjp', 'mesh-group8-m2g-save-stats-vjp-edge', 'mesh-group8-m2g-save-stats-vjp-lnvjp', 'mesh-group8-m2g-save-stats-vjp-finalmesh', 'mesh-group8-m2g-save-stats-vjp-finalmesh-g2m') and function_name == '_run_mesh2grid_gnn':
        return 'bypass'
    if candidate == 'mesh-group8-spatial-bypass' and function_name == '_run_grid2mesh_gnn':
        return 'bypass'
    selected = {'mesh-stage-cse': ('_run_mesh_gnn',),
                'm2g-stage-cse': ('_run_mesh2grid_gnn',),
                'spatial-stage-cse': STAGES}.get(candidate, ())
    return 'allow_cse' if function_name in selected else 'baseline'


def message_allow_cse(candidate, module_name):
    return candidate in ('mesh-message-cse', 'mesh-message-lnremat') and module_name == 'mesh_gnn'


def layernorm_remat(candidate):
    return candidate in ('mesh-message-lnremat', 'spatial-stage-bypass-lnremat')


def shifted_layernorm(candidate):
    return candidate in ('mesh-group8-shifted', 'mesh-group8-combined', 'mesh-group8-m2g-combined', 'mesh-group8-m2g-shifted')


def mesh_gather(candidate):
    return candidate in ('mesh-group8-combined', 'mesh-group8-m2g-combined', 'mesh-group8-m2g-gather', 'mesh-group8-m2g-vjp', 'mesh-group8-m2g-save-stats-vjp', 'mesh-group8-m2g-save-stats-vjp-edge', 'mesh-group8-m2g-save-stats-vjp-lnvjp', 'mesh-group8-m2g-save-stats-vjp-finalmesh', 'mesh-group8-m2g-save-stats-vjp-finalmesh-g2m')


def final_mesh_nodes(candidate):
    return candidate in ('mesh-group8-m2g-save-stats-vjp-finalmesh', 'mesh-group8-m2g-save-stats-vjp-finalmesh-g2m')


def g2m_mesh_nodes(candidate):
    return candidate == 'mesh-group8-m2g-save-stats-vjp-finalmesh-g2m'


def analytic_layernorm(candidate):
    return candidate == 'mesh-group8-m2g-save-stats-vjp-lnvjp'


def edge_linear(candidate):
    return candidate == 'mesh-group8-m2g-save-stats-vjp-edge'


def mesh_vjp(candidate):
    return candidate in ('mesh-group8-m2g-vjp', 'mesh-group8-m2g-save-stats-vjp', 'mesh-group8-m2g-save-stats-vjp-edge', 'mesh-group8-m2g-save-stats-vjp-lnvjp', 'mesh-group8-m2g-save-stats-vjp-finalmesh', 'mesh-group8-m2g-save-stats-vjp-finalmesh-g2m')


def saves_layernorm_stats(candidate):
    return candidate in ('mesh-group8-m2g-save-stats', 'mesh-group8-m2g-save-stats-vjp', 'mesh-group8-m2g-save-stats-vjp-edge', 'mesh-group8-m2g-save-stats-vjp-lnvjp', 'mesh-group8-m2g-save-stats-vjp-finalmesh', 'mesh-group8-m2g-save-stats-vjp-finalmesh-g2m')


def stats_names(candidate):
    if saves_layernorm_stats(candidate):
        from layernorm_stats import STAT_NAMES
        return STAT_NAMES
    return ()


def function_stats_names(candidate, function_name):
    names = stats_names(candidate) if function_name in ('one_step_loss', '_run_grid2mesh_gnn') else ()
    if final_mesh_nodes(candidate) and function_name == 'one_step_loss':
        from training_final_mesh import STAT_NAME
        names += (STAT_NAME,)
    if g2m_mesh_nodes(candidate) and function_name == 'one_step_loss':
        from training_g2m_mesh import STAT_NAME
        names += (STAT_NAME,)
    return names


DECODER_HIDDEN_MODULE = 'mesh2grid_gnn/~_networks_builder/processor_edges_0_mesh2grid_mlp/~/linear_0'


def decoder_saved_names(candidate, module_name):
    return ('decoder_edge_hidden',) if candidate in ('mesh-group8-m2g-save', 'mesh-group8-m2g-save-stats', 'mesh-group8-m2g-save-stats-vjp', 'mesh-group8-m2g-save-stats-vjp-edge', 'mesh-group8-m2g-save-stats-vjp-lnvjp', 'mesh-group8-m2g-save-stats-vjp-finalmesh', 'mesh-group8-m2g-save-stats-vjp-finalmesh-g2m') and module_name == 'mesh2grid_gnn' else ()


def decoder_hidden_marker(candidate, module_name, shape, dtype):
    if not decoder_saved_names(candidate, 'mesh2grid_gnn') or module_name != DECODER_HIDDEN_MODULE:
        return None
    if tuple(shape) != (3114720, 1, 512) or dtype != 'float32':
        raise ValueError('Named decoder residual must retain exact FP32 batch/channel shape')
    return 'decoder_edge_hidden'


def mesh_group_size(candidate):
    return {'mesh-group4': 4, 'mesh-group8': 8, 'mesh-group16': 16,
            'mesh-group16-m2g-bypass': 16, 'mesh-group8-m2g-bypass': 8,
            'mesh-group8-spatial-bypass': 8, 'mesh-group8-shifted': 8,
            'mesh-group8-combined': 8, 'mesh-group8-m2g-combined': 8,
            'mesh-group8-m2g-shifted': 8, 'mesh-group8-m2g-gather': 8,
            'mesh-group8-m2g-save': 8, 'mesh-group8-m2g-vjp': 8,
            'mesh-group8-m2g-save-stats': 8, 'mesh-group8-m2g-save-stats-vjp': 8, 'mesh-group8-m2g-save-stats-vjp-edge': 8, 'mesh-group8-m2g-save-stats-vjp-lnvjp': 8, 'mesh-group8-m2g-save-stats-vjp-finalmesh': 8, 'mesh-group8-m2g-save-stats-vjp-finalmesh-g2m': 8}.get(candidate, 0)


def group_indices(count, size):
    if count != 16 or size not in (4, 8, 16):
        raise ValueError('Only the unchanged sixteen-layer mesh in groups of four/eight/sixteen is supported')
    return [tuple(range(start, start + size)) for start in range(0, count, size)]


def process_group(graph, processors, step_fn):
    for processor in processors:
        graph = step_fn(processor, graph)
    return graph


def is_mesh_group_process(candidate, method_name, module, graphnet_type):
    # Haiku also intercepts __init__, before module_name has been initialized.
    return bool(mesh_group_size(candidate) and method_name == '_process'
                and isinstance(module, graphnet_type) and module.module_name == 'mesh_gnn')


def install_message_policy(candidate, output):
    """Replace the isolated interceptor before model imports bind it.

    Same parameter scopes and message coverage as the frozen H100 interceptor.
    LayerNorm remat follows the separately archived Hygon policy but changes
    neither LayerNorm arithmetic nor precision/backend.
    """
    from contextlib import contextmanager
    import json
    import haiku as hk
    from graphcast import deep_typed_graph_net
    import remat_policy

    def event(value):
        with (output/'policy-dispatch.jsonl').open('a') as stream:
            stream.write(json.dumps(value) + '\n')

    @contextmanager
    def rematerialize():
        coverage = {'steps': set()}
        processor_indices = {}

        def intercept(next_f, args, kwargs, context):
            if (decoder_saved_names(candidate, 'mesh2grid_gnn')
                    and context.method_name == '__call__' and isinstance(context.module, hk.Linear)
                    and context.module.module_name == DECODER_HIDDEN_MODULE):
                from jax.ad_checkpoint import checkpoint_name
                value = next_f(*args, **kwargs)
                name = decoder_hidden_marker(candidate, context.module.module_name, value.shape, str(value.dtype))
                event(dict(kind='named_decoder_residual', name=name, module=context.module.module_name,
                           shape=list(value.shape), dtype=str(value.dtype), location='Linear return after bias, before activation'))
                return checkpoint_name(value, name=name)
            if is_mesh_group_process(candidate, context.method_name, context.module,
                                     deep_typed_graph_net.DeepTypedGraphNet):
                graph, processors = args
                module = context.module
                if kwargs or module._num_processor_repetitions != 1 or len(set(map(id, processors))) != 16:
                    raise ValueError('Mesh layers must remain sixteen independent processors with one repetition')
                if processor_indices:
                    raise ValueError('Nested mesh grouping is not supported')
                processor_indices.update({id(p): i for i, p in enumerate(processors)})
                try:
                    for indices in group_indices(len(processors), mesh_group_size(candidate)):
                        selected = tuple(processors[i] for i in indices)
                        names = stats_names(candidate)
                        group_kwargs = {}
                        if names:
                            import jax
                            group_kwargs['policy'] = jax.checkpoint_policies.save_only_these_names(*names)
                        event(dict(kind='mesh_group_remat', layer_indices=list(indices), prevent_cse=True,
                                   processor_ids=[id(p) for p in selected], module=module.module_name,
                                   saved_residual_names=list(names)))
                        def make_group(selected_processors):
                            def mesh_group(g):
                                return process_group(g, selected_processors, module._process_step)
                            return mesh_group
                        graph = hk.remat(make_group(selected), prevent_cse=True, **group_kwargs)(graph)
                    return graph
                finally:
                    processor_indices.clear()
            if (layernorm_remat(candidate) and isinstance(context.module, hk.LayerNorm)
                    and context.method_name == '__call__'):
                event(dict(kind='layernorm_remat', module=context.module.module_name, prevent_cse=True))
                return hk.remat(lambda *xs: next_f(*xs, **kwargs), prevent_cse=True)(*args)
            if (isinstance(context.module, deep_typed_graph_net.DeepTypedGraphNet)
                    and context.method_name == '_process_step'):
                processor, graph = args
                assert not kwargs
                module = context.module.module_name
                coverage['steps'].add((module, id(processor)))
                if mesh_group_size(candidate) and module == 'mesh_gnn':
                    if id(processor) not in processor_indices:
                        raise ValueError('Mesh message escaped its verified group')
                    event(dict(kind='message_group_internal', module=module,
                               layer_index=processor_indices[id(processor)], processor_id=id(processor),
                               individual_remat=False, residual_math='unchanged official _process_step'))
                    return next_f(processor, graph)
                prevent_cse = not message_allow_cse(candidate, module)
                names = decoder_saved_names(candidate, module) + stats_names(candidate)
                policy_kwargs = {}
                if names:
                    import jax
                    policy_kwargs['policy'] = jax.checkpoint_policies.save_only_these_names(*names)
                event(dict(kind='message_remat', module=module, processor_id=id(processor),
                           prevent_cse=prevent_cse, saved_residual_names=list(names)))
                return hk.remat(lambda g: next_f(processor, g), prevent_cse=prevent_cse, **policy_kwargs)(graph)
            return next_f(*args, **kwargs)

        with hk.intercept_methods(intercept):
            yield coverage

    remat_policy.rematerialize = rematerialize
