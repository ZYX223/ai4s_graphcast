"""BW1100 inference: prepare, warm up, measure independent rollouts, then save."""
import argparse
import ctypes as C
import importlib.metadata as metadata
import os
from pathlib import Path
import statistics
from typing import NamedTuple

import numpy as np
import xarray as xr

from graphcast_bench.assets import (
    ROOT, STATS, code_snapshot, load_case, read_json, sha256, verify_source, write_json,
)
from graphcast_bench.data.inference import load_input_dataset
from graphcast_bench.delivery.options import INFERENCE_DEFAULTS, resolve_options
from graphcast_bench.inference.profiling import capture
from graphcast_bench.inference.reporting import create_report, finish_report, save_round
from graphcast_bench.inference.scheduling import run_rounds
from graphcast_bench.inference.telemetry import read_host_sample
from graphcast_bench.progress import emit, phase
from graphcast_bench.platforms.context import context_arguments


class InferenceInputs(NamedTuple):
    inputs: xr.Dataset
    forcings: xr.Dataset
    target: xr.Dataset


class InferenceRuntime(NamedTuple):
    device: object
    router: object
    pci: str
    synchronize: object


class InferenceModel(NamedTuple):
    params: object
    predict: object
    hook: object


def parse_arguments(argv, context):
    parser = argparse.ArgumentParser(description=__doc__)
    # Retain worker CLI compatibility; this worker supports only this BW recipe.
    parser.add_argument('--platform', choices=('hygon',), required=True)
    parser.add_argument('--mode', choices=('rne_xf32',), required=True)
    parser.add_argument('--variant', choices=('fused',), required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=INFERENCE_DEFAULTS['steps'])
    parser.add_argument('--repeats', type=int, default=INFERENCE_DEFAULTS['repeats'])
    parser.add_argument('--warmup-steps', type=int, default=INFERENCE_DEFAULTS['warmup_steps'])
    parser.add_argument('--warmup-scope', choices=('once', 'per-repeat'), default=INFERENCE_DEFAULTS['warmup_scope'])
    parser.add_argument('--profile', dest='capture_enabled', action='store_true')
    parser.add_argument('--logical-catalog', action='store_true')
    args = parser.parse_args(argv)
    if args.data != context.path('assets') or args.output != context.output / 'run':
        raise ValueError('Inference command paths differ from runtime context')
    resolve_options('infer', 'bw1100', dict(steps=args.steps, warmup_steps=args.warmup_steps,
                                         repeats=args.repeats, warmup_scope=args.warmup_scope))
    return args


def load_inputs(args):
    data = InferenceInputs(*(load_input_dataset(args.data / 'assets' / name)
                            for name in ('inputs.nc', 'rollout_forcings.nc', 'target_template.nc')))
    if max(args.steps, args.warmup_steps) > data.forcings.sizes['time']:
        raise ValueError('Requested prediction/warmup exceeds available forcing steps')
    return data


def prepare_runtime(context):
    import jax
    from graphcast import graphcast
    from graphcast_bench.platforms.inference_router import CandidateRouter

    config = context.config
    if not Path(graphcast.__file__).resolve().is_relative_to(ROOT / 'third_party/graphcast'):
        raise RuntimeError('Imported model is not the pinned official source')
    if metadata.version('jax') != config['jax_version']:
        raise RuntimeError('JAX version differs from tested platform configuration')
    devices = jax.devices('gpu')
    if len(devices) != 1 or config['expected_device'] not in devices[0].device_kind:
        raise RuntimeError('Expected exactly one configured accelerator')
    if jax.config.jax_enable_x64 or os.environ.get('ROCBLAS_MATH_MODE') is not None:
        raise RuntimeError('Unexpected global precision override')
    router = CandidateRouter(context.path('inference_router'))
    bus = C.create_string_buffer(64)
    driver = C.CDLL('libamdhip64.so')
    assert driver.hipDeviceGetPCIBusId(bus, len(bus), 0) == 0
    pci = bus.value.decode()
    if pci.lower() != context.device['pci']:
        raise RuntimeError('Container selected the wrong physical accelerator')
    return InferenceRuntime(devices[0], router, pci, driver.hipDeviceSynchronize)


def build_model(data_root):
    import haiku as hk
    import jax
    from graphcast import autoregressive, checkpoint, graphcast, normalization
    from graphcast_bench.optimizations.inference.rne_hook import RneHook

    with (data_root / 'assets/jax_weights.npz').open('rb') as stream:
        weights = checkpoint.load(stream, graphcast.CheckPoint)
    assert all(np.asarray(value).dtype == np.float32 for value in jax.tree_util.tree_leaves(weights.params))
    stats = {name.removesuffix('.nc'): xr.load_dataset(data_root / 'assets/stats' / name).astype(np.float32)
             for name in STATS}
    policy = read_json(ROOT / 'configs/runtime/inference/policy.json')
    hook = RneHook({row['module']: row for row in policy['layers']})

    @hk.transform_with_state
    def predict(inputs, targets_template, forcings):
        model = graphcast.GraphCast(weights.model_config, weights.task_config)
        model = normalization.InputsAndResiduals(model, **stats)
        model = autoregressive.Predictor(model, gradient_checkpointing=False)
        with hk.intercept_methods(hook):
            return model(inputs, targets_template=targets_template, forcings=forcings)

    return InferenceModel(weights.params, predict, hook)


def prepare_targets(data):
    from graphcast import rollout

    targets = rollout.extend_targets_template(data.target.drop_vars('datetime', errors='ignore'),
                                               data.forcings.sizes['time'])
    return targets.assign_coords(datetime=data.forcings.datetime)


def experiment_hashes():
    return {path.name: sha256(path) for path in (ROOT / 'configs/runtime/inference').iterdir() if path.is_file()}


def telemetry(context):
    return read_host_sample(context.output, context.device['index'])


def save_warmup_artifacts(output, apply, first_inputs, logical_catalog):
    (output / 'optimized-hlo.txt').write_text(apply.lower(**first_inputs).compile().as_text())
    if logical_catalog:
        import jax
        from graphcast_bench.logical_inventory import export_jaxpr

        export_jaxpr(jax.make_jaxpr(apply)(**first_inputs), output / 'logical')


def run_benchmark(args, context, data, runtime, model, targets, report):
    import jax
    from graphcast_bench.inference.rollout import RolloutRunner

    capture_ranges = []
    telemetry_warned = False
    with jax.default_device(runtime.device), jax.default_matmul_precision('tensorfloat32'):
        params = jax.device_put(model.params)
        apply = jax.jit(lambda rng, inputs, targets_template, forcings: model.predict.apply(
            params, {}, rng, inputs, targets_template, forcings)[0])
        execution = RolloutRunner(apply, data.inputs, targets, data.forcings,
                                  runtime.router, runtime.synchronize)

        def warmup_round(round_id):
            name = f'round{round_id}-{args.mode}'
            emit('Prepare', 'Prepare warmup for round={}/{}: steps={}; compile first if needed'.format(
                round_id, report['repeats'], args.warmup_steps))
            result = execution.run_rollout(args.mode, name + '-warmup', args.warmup_steps,
                                           round_id=round_id, repeats=report['repeats'], warmup=True)
            report['warmups'].append(dict(run=name, seconds=result.step_seconds, wall_seconds=result.wall_seconds))
            emit('Warmup', 'round={}/{} completed; measured prediction restarts from the original inputs'.format(
                round_id, report['repeats']))
            if len(report['warmups']) == 1:
                with phase('Prepare', 'Save compilation artifacts'):
                    save_warmup_artifacts(args.output, apply, result.first_inputs, args.logical_catalog)

        def measured_round(round_id):
            nonlocal telemetry_warned
            name = f'round{round_id}-{args.mode}'
            model.hook.verify_traced()
            cache_before = apply._cache_size()
            device_before, before_sample = telemetry(context)
            emit('Inference', 'Starting round={}/{} from the original inputs: measured_steps={}'.format(
                round_id, report['repeats'], report['measured_steps']))
            # Only the rollout is captured. Telemetry, host transfer and saves are outside.
            with capture(args.output, name, report['measured_steps'], args.capture_enabled, capture_ranges):
                result = execution.run_rollout(args.mode, name, report['measured_steps'],
                                               round_id=round_id, repeats=report['repeats'])
            device_after, after_sample = telemetry(context)
            if not telemetry_warned and any(s['status'] != 'available' for s in (before_sample, after_sample)):
                emit('Monitoring', 'Some host samples are unavailable or stale; recorded in the report; inference continues', 'WARN')
                telemetry_warned = True
            if apply._cache_size() != cache_before:
                emit('Inference', 'round={}/{} failed: Compilation occurred during measurement | jit_cache_before={} | jit_cache_after={}'.format(
                    round_id, report['repeats'], cache_before, apply._cache_size()), 'ERROR')
                raise RuntimeError('Compilation occurred during measurement')
            row = dict(run=name, mode=args.mode, round=round_id, apply_seconds=result.step_seconds,
                       apply_median_seconds=statistics.median(result.step_seconds),
                       apply_sum_seconds=sum(result.step_seconds), rollout_wall_seconds=result.wall_seconds,
                       jit_cache_before=cache_before, jit_cache_after=apply._cache_size(),
                       device_before=device_before, device_after=device_after,
                       device_before_sample=before_sample, device_after_sample=after_sample)
            save_round(args.output, report, row, result.final_prediction, data.target, args.logical_catalog)

        run_rounds(report['repeats'], args.warmup_scope, warmup_round, measured_round)
    return execution.step_audit


def finish_run(args, report, model, step_audit):
    (args.output / 'loaded-libraries.txt').write_text(Path('/proc/self/maps').read_text())
    load_case(args.data)
    if code_snapshot() != report['code_sha256']:
        raise RuntimeError('Code changed during execution')
    if experiment_hashes() != report['experiment_sha256']:
        raise RuntimeError('Experiment changed during execution')
    finish_report(args.output, report, model.hook, step_audit)


def main(argv=None, context=None):
    if context is None:
        context, argv = context_arguments(argv)
    context.require_kind('inference')
    args = parse_arguments(argv, context)
    with phase('Prepare', 'Verify model and input provenance'):
        case, prepared = load_case(args.data)
        provenance = dict(source=verify_source(), code_sha256=code_snapshot(), data=prepared,
                          original_case_sha256=case['source_case_sha256'])
    with phase('Load', 'Load input datasets'):
        data = load_inputs(args)
    args.output.mkdir(parents=True, exist_ok=False)
    with phase('Initialize', 'Initialize the selected accelerator'):
        runtime = prepare_runtime(context)
    with phase('Load', 'Load model weights and normalization statistics'):
        model = build_model(args.data)
    with phase('Prepare', 'Prepare forecast targets'):
        targets = prepare_targets(data)
    provenance['experiment_sha256'] = experiment_hashes()
    report = create_report(args, runtime, provenance)
    write_json(args.output / 'report.json', report)
    step_audit = run_benchmark(args, context, data, runtime, model, targets, report)
    with phase('Validation', 'Verify provenance and save optimization audits'):
        finish_run(args, report, model, step_audit)


if __name__ == '__main__':
    main()
