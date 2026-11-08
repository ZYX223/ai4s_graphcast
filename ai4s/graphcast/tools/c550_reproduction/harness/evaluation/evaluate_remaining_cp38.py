"""Isolated CP38 validation. Reuses frozen model construction and evaluator.

This entry never invokes train(), builds an optimizer, or changes the cursor.
GPU execution requires a separate, fresh resource/owner gate from the launcher.
"""
import argparse
import ast
from contextlib import ExitStack
import datetime
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def dump(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def model_setup_ast(text):
    tree = ast.parse(text)
    train = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'train')
    body = next(n for n in train.body if isinstance(n, ast.Try)).body

    def assignment(n, target):
        return isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == target for t in n.targets)

    start = next(i for i, n in enumerate(body) if assignment(n, 'config') and isinstance(n.value, ast.Call))
    end = next(i for i, n in enumerate(body) if assignment(n, 'wrapped_unroll')) + 1
    block = ast.Module(body=body[start:end], type_ignores=[])
    for n in ast.walk(block):
        if isinstance(n, ast.Name):
            assert n.id not in {'optimizer', 'cursor', 'resume_payload', 'schedule'}
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
            assert n.func.attr not in {'backward', 'step', 'zero_grad'}
    return block


def preflight(spec):
    model = Path(spec['model'])
    for name, expected in spec['model_files'].items():
        assert sha(model / name) == expected, ('model source mismatch', name)
    for name, expected in spec['input_files'].items():
        assert sha(name) == expected, ('config/stats mismatch', name)
    assert sha(spec['checkpoint']) == spec['checkpoint_sha256'], 'CP38 changed'
    source = spec.get('flaggems_source')
    if source:
        source = Path(source)
        mf = source / 'retest-source-manifest.json'
        assert sha(mf) == spec['flaggems_manifest_sha256']
        hashes = json.loads(mf.read_text())
        assert len(hashes) == spec['source_file_count']
        for name, expected in hashes.items():
            assert sha(source / name) == expected, ('FlagGems source mismatch', name)
    root = Path(spec['data_root'])
    receipt = json.loads((root / 'data-verified.json').read_text())
    data = root / 'official_hres' / spec['data_filename']
    st = data.stat()
    assert receipt['sha256'] == spec['data_sha256']
    assert st.st_size == receipt['bytes'] == spec['data_bytes']
    assert st.st_mtime_ns == receipt['mtime_ns'], 'target data changed after full hash'
    assert sha(root / 'corpora' / 'hres_2022-01_validation.corpus.json') == spec['corpus_sha256']
    assert sha(root / 'validation-windows-r12.json') == spec['windows_sha256']
    windows = json.loads((root / 'validation-windows-r12.json').read_text())
    assert windows['count'] == 111 and windows['rollout_steps'] == 12
    runner = model / 'scripts/training/run_official_epoch_training.py'
    assert sha(runner) == spec['runner_sha256']
    block = model_setup_ast(runner.read_text())
    assert hashlib.sha256(ast.dump(block, include_attributes=False).encode()).hexdigest() == spec['model_setup_ast_sha256']
    return dict(model_files=len(spec['model_files']), checkpoint_sha256=spec['checkpoint_sha256'],
                data_sha256=receipt['sha256'], data_mtime_ns=st.st_mtime_ns,
                windows=111, optimizer_updates=0, gpu_work_started=False)


def precision_readback(torch, platform):
    assert platform in {'c550', 'hygon', '910c'}
    if platform == '910c':
        result = dict(npu_matmul_allow_hf32=torch.npu.matmul.allow_hf32,
                      npu_conv_allow_hf32=torch.npu.conv.allow_hf32)
        assert all(value is True for value in result.values())
        return result
    matmul = torch.backends.cuda.matmul
    if platform == 'hygon':
        assert bool(torch.version.hip), 'Hygon requires the original HIP runtime'
        result = dict(matmul_allow_tf32=matmul.allow_tf32, cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)
        assert all(value is True for value in result.values())
        return result
    conv = getattr(torch.backends.cudnn, 'conv', None)
    if hasattr(matmul, 'fp32_precision') and conv is not None and hasattr(conv, 'fp32_precision'):
        result = dict(matmul=matmul.fp32_precision, convolution=conv.fp32_precision)
        assert result == dict(matmul='tf32', convolution='tf32')
    else:
        result = dict(matmul_allow_tf32=matmul.allow_tf32, cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)
        assert all(value is True for value in result.values())
    return result


def adapter_policy(spec):
    assert spec['platform'] in {'c550', 'hygon', '910c'}
    assert spec['layernorm_adapter'] is True
    assert spec['copy_adapter'] is (spec['platform'] == '910c')
    return ['copy', 'layernorm'] if spec['copy_adapter'] else ['layernorm']


def validate_result(result, expected_indices):
    assert result['optimizer_updates'] == 0 and result['gradient_enabled'] is False
    assert result['validation_self_check'] and result['coverage'] == 1.0
    assert result['unique_window_count'] == result['valid_sample_count'] == 111
    assert result['duplicate_valid_count'] == result['padded_sample_count'] == 0
    assert len(result['steps']) == 111
    indices = []
    weighted = 0.0
    variables = None
    for i, row in enumerate(result['steps']):
        assert row['batch_index'] == i and row['valid_sample_count'] == 1
        values = [row['loss'], *row['lead_losses']]
        assert len(row['lead_losses']) == 12
        current_variables = tuple(sorted(row['per_variable_lead_losses']))
        assert current_variables
        if variables is None:
            variables = current_variables
        assert variables == current_variables
        for lead_values in row['per_variable_lead_losses'].values():
            assert len(lead_values) == 12
            values.extend(lead_values)
        assert all(math.isfinite(x) for x in values)
        weighted += row['loss']
        for origin in row['origins_by_rank']:
            assert len(origin['local_window_indices']) == len(origin['sample_weights'])
            indices.extend(k for k, w in zip(origin['local_window_indices'], origin['sample_weights']) if w > 0)
    assert indices == expected_indices, 'ordered full frozen windows differ'
    assert math.isclose(weighted / 111, result['mean_loss'], rel_tol=1e-12, abs_tol=1e-12)
    return dict(all_diagnostics_finite=True, ordered_windows_match=True,
                variables=list(variables), mean_loss_recomputed=weighted / 111,
                optimizer_updates=0, coverage_kind='validation_windows_not_operator_coverage')


def run(spec, root):
    # Slow CPU file checks precede, and never replace, the launcher's last resource gate.
    assert spec.get('launch_ready') is True, 'prepared spec cannot start GPU work'
    before = preflight(spec)
    dump(root / 'preflight.json', before)
    gate = json.loads((root / 'gpu-launch-gate.json').read_text())
    assert gate['allowed'] is True and gate['owner_user_id']
    assert gate['platform'] == spec['platform'] and gate['process_count'] == 0
    assert gate['healthy'] is True and 0 <= time.time() - gate['epoch'] <= 120
    for key, value in spec['environment'].items():
        os.environ[key] = str(value)
    model_root = Path(spec['model']).resolve()
    sys.path.insert(0, str(model_root / 'src'))
    runner = model_root / 'scripts/training/run_official_epoch_training.py'
    module_spec = importlib.util.spec_from_file_location('frozen_cp38_validation_runner', runner)
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    torch = module.torch
    args = module.parse_args(spec['engine_argv'])
    assert args.global_batch_size == 2 and args.global_micro_batch_size == 1
    assert args.rollout_steps == 12 and args.validation_mode == 'full'
    assert args.data_parallel_replicas == args.mp_devices_per_replica == 1
    assert args.seed == 27 and args.tensor_core == 'enabled' and args.precision == 'fp32'
    assert not args.profile_hotspots
    mode = module.configure_float32_math(True)
    assert mode == ('hf32' if spec['platform'] == '910c' else 'tf32')
    context = None
    success = False
    adapter_scope = ExitStack()
    adapters = {}
    try:
        for name in adapter_policy(spec):
            if name == 'copy':
                from graphcast_compat.training_copy_routing import from_environment
                instance = from_environment()
            else:
                from graphcast_compat.training_layernorm_routing import ModelLayerNormRouting
                instance = ModelLayerNormRouting()
            adapters[name] = adapter_scope.enter_context(instance)
        context = module.DistributedContext.from_environment(dp_replicas=1, mp_devices_per_replica=1)
        context.prewarm_device_collectives()
        device = module.accelerator_device(context.device_indices[0], context.accelerator_type)
        module.random.seed(args.seed); module.np.random.seed(args.seed); torch.manual_seed(args.seed)
        from graphcast_compat.training_state import load_training_checkpoint, restore_rng_state
        from training_support import state_dict_sha256
        payload = load_training_checkpoint(spec['checkpoint'])
        assert payload['cursor']['global_optimizer_step'] == 38
        assert payload['schedule_state']['next_update_index'] == 38
        assert payload['metadata']['initial_lineage']['checkpoint_sha256'] == spec['initial_checkpoint_sha256']
        state_hash = state_dict_sha256(payload['model_state'])
        assert state_hash == payload['metadata']['model_state_sha256']
        init_state = payload['model_state']
        assert all(torch.isfinite(x).all().item() for x in init_state.values())
        runtime = module.enable_network_replacements_from_env(device=device)
        assert not runtime.skipped and runtime.enabled == bool(spec.get('flaggems_source'))
        if runtime.enabled:
            import flag_gems
            assert Path(flag_gems.__file__).resolve() == Path(spec['flaggems_source']) / 'src/flag_gems/__init__.py'
        dump(root / 'runtime.json', dict(enabled=runtime.enabled, operators=list(runtime.operators),
             metadata=runtime.metadata, precision=precision_readback(torch, spec['platform']), checkpoint_model_sha256=state_hash))
        dataset = module._validation_dataset(args)
        try:
            module.validate_formal_dataset(dataset, formal=True)
            expected = json.loads((Path(spec['data_root']) / 'validation-windows-r12.json').read_text())
            indices = [x['reference_index'] for x in expected['rows']]
            assert list(dataset.window_indices) == indices and dataset.window_count == 111
            with ExitStack() as scope:
                ns = dict(module.__dict__, args=args, context=context, dataset=dataset, init_state=init_state,
                          input_device=device, mesh_first_device=device, secondary_device=device, output_device=device)
                exec(compile(model_setup_ast(runner.read_text()), str(runner), 'exec'), ns)
                model, unroll = ns['model'], ns['wrapped_unroll']
                assert state_dict_sha256(model.state_dict()) == state_hash
                assert all(p.grad is None for p in model.parameters())
                restore_rng_state(payload['rng_states_by_rank'][0], cuda_device_indices=context.device_indices,
                                  accelerator_type=context.accelerator_type)
                del payload, init_state
                # Record completed forwards without changing the frozen evaluator.
                original_clock = context.global_max_seconds
                completed = 0

                def progress(seconds):
                    nonlocal completed
                    value = original_clock(seconds)
                    completed += 1
                    memory = {k: getattr(torch.cuda, k)(0) for k in
                              ['memory_allocated', 'memory_reserved', 'max_memory_allocated', 'max_memory_reserved']}
                    dump(root / 'progress.json', dict(completed_forward_windows=completed, total_windows=111,
                         optimizer_updates=0, device_memory_bytes=memory, utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
                    with (root / 'device-memory.jsonl').open('a') as stream:
                        stream.write(json.dumps(dict(window_forward=completed, **memory)) + '\n')
                    print('validation_forward_complete', completed, 'of', 111, flush=True)
                    return value

                context.global_max_seconds = progress
                sys.path.insert(0, str(Path(__file__).resolve().parent))
                journal = importlib.import_module('validation_journal')
                identity = {k: spec[k] for k in ['checkpoint_sha256', 'data_sha256', 'windows_sha256',
                            'corpus_sha256', 'flaggems_manifest_sha256', 'runner_sha256', 'model_setup_ast_sha256']}
                identity['checkpoint_model_sha256'] = state_hash
                evaluator = journal.instrument(runner.read_text(), module.__dict__, root / 'window-results', identity, indices)
                result = evaluator(args=args, context=context, wrapped_unroll=unroll,
                         base_model=model, config=ns['config'], stats=ns['stats'], input_device=device)
                context.global_max_seconds = original_clock
                assert completed == 111
                numerical_audit = validate_result(result, indices)
                dump(root / 'window-journal-audit.json', journal.verify_complete(root / 'window-results', identity, result))
                assert state_dict_sha256(model.state_dict()) == state_hash, 'model state changed during evaluation'
                assert all(p.grad is None for p in model.parameters()), 'unexpected gradients'
                dump(root / 'validation.json', result)
                dump(root / 'validation-audit.json', dict(**numerical_audit, model_unchanged=True,
                     gradient_tensors=0, checkpoint_sha256=spec['checkpoint_sha256'],
                     formal_mfu_eligible=False, final_acceptance=False))
        finally:
            dataset.close()
        after = preflight(spec)
        assert before == after
        success = True
    finally:
        dump(root / 'exit-precision.json', precision_readback(torch, spec['platform']))
        imports = {}
        for name, loaded in list(sys.modules.items()):
            file = getattr(loaded, '__file__', None)
            if file:
                p = Path(file).resolve()
                if p.is_file() and p.is_relative_to(model_root):
                    imports[name] = dict(path=str(p), sha256=sha(p))
        dump(root / 'actual-model-imports.json', imports)
        try:
            adapter_scope.close()
            for name, instance in adapters.items():
                dump(root / (name + '-routing.json'), instance.metadata())
        finally:
            if context is not None:
                context.close(synchronize=success)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', required=True, type=Path)
    parser.add_argument('--cpu-preflight-only', action='store_true')
    options = parser.parse_args()
    spec = json.loads(options.spec.read_text())
    if options.cpu_preflight_only:
        print(json.dumps(preflight(spec)))
        return
    root = Path(spec['output_dir'])
    root.mkdir(exist_ok=True)
    lock = (root / 'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with (root / 'run.started').open('x') as f:
        f.write(datetime.datetime.now(datetime.timezone.utc).isoformat())
    rc = 1
    try:
        run(spec, root)
        rc = 0
    except BaseException:
        (root / 'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        dump(root / 'exit.json', dict(rc=rc, optimizer_updates=0, utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))


if __name__ == '__main__':
    main()
