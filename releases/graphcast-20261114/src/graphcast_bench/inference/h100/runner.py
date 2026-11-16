"""Prepare H100, warm each selected variant, measure independent forecasts, save."""
import ctypes
import importlib.metadata as metadata
import os
from pathlib import Path
import statistics
from types import SimpleNamespace

import numpy as np
import xarray as xr

from graphcast_bench.assets import write_json
from graphcast_bench.data.inference import load_input_dataset
from graphcast_bench.delivery.options import resolve_options
from graphcast_bench.engines.operational_h100.frozen.src.graphcast_bench.assets import (
    ROOT as FROZEN_ROOT, STATS, code_snapshot, load_case, read_json, verify_source,
)
from graphcast_bench.engines.operational_h100.frozen.src.graphcast_bench.metrics import load_reference
from graphcast_bench.inference.h100.profiling import capture
from graphcast_bench.inference.h100.reporting import create_report, finish_report, save_round
from graphcast_bench.inference.h100.rollout import RolloutRunner
from graphcast_bench.inference.scheduling import run_rounds
from graphcast_bench.inference.telemetry import read_host_sample
from graphcast_bench.progress import emit, phase


def validate_context(context):
    context.require_kind('inference')
    if context.platform != 'h100' or not context.device.get('uuid'):
        raise ValueError('H100 inference requires an explicit GPU UUID')
    recipe, values = context.config['recipe'], context.config['inference']
    if context.config['variant'] != 'shifted_mesh' or recipe not in ('single','full40','custom'):
        raise ValueError('Unsupported H100 inference recipe or variant')
    resolved = resolve_options('infer','h100', values)['inference']
    if resolved != values:
        raise ValueError('Incomplete inference options')
    expected = {'single':(1,5,2,'per-repeat'), 'full40':(40,5,3,'per-repeat')}
    if recipe in expected and tuple(values[k] for k in ('steps','warmup_steps','repeats','warmup_scope')) != expected[recipe]:
        raise ValueError('Fixed H100 recipe options differ')


def prepare_runtime(context):
    import jax
    from graphcast import graphcast

    config = read_json(FROZEN_ROOT / 'configs/h100.json')
    if not Path(graphcast.__file__).resolve().is_relative_to(FROZEN_ROOT / 'third_party/graphcast'):
        raise RuntimeError('H100 model must come from the pinned frozen source')
    if metadata.version('jax') != config['jax_version']:
        raise RuntimeError('JAX version differs from the qualified H100 runtime')
    devices = jax.devices('gpu')
    if len(devices) != 1 or config['expected_device'] not in devices[0].device_kind:
        raise RuntimeError('Expected exactly one H100 accelerator')
    if jax.config.jax_enable_x64 or os.environ.get('ROCBLAS_MATH_MODE') is not None:
        raise RuntimeError('Unexpected global precision override')
    driver, bus = ctypes.CDLL('libcuda.so.1'), ctypes.create_string_buffer(64)
    if driver.cuInit(0) != 0 or driver.cuDeviceGetPCIBusId(bus, len(bus), 0) != 0:
        raise RuntimeError('Cannot identify the CUDA device')
    pci = bus.value.decode()
    if pci.lower() != context.device['pci']:
        raise RuntimeError('Container selected the wrong physical accelerator')
    return SimpleNamespace(device=devices[0], pci=pci, synchronize=driver.cuCtxSynchronize)


def load_inputs(context):
    data = context.path('assets') / 'assets'
    inputs, forcings, target = [load_input_dataset(data / name) for name in
                               ('inputs.nc','rollout_forcings.nc','target_template.nc')]
    values = context.config['inference']
    required = max(values['steps'],values['warmup_steps'])
    if forcings.sizes['time'] < required:
        raise ValueError('Insufficient prepared forcing frames')
    if context.config['recipe'] == 'custom':
        forcings = forcings.isel(time=slice(0,required))
    return inputs, forcings, target


def build_model(data):
    import haiku as hk
    import jax
    from graphcast import autoregressive, checkpoint, graphcast, normalization

    with (data / 'assets/jax_weights.npz').open('rb') as stream:
        weights = checkpoint.load(stream, graphcast.CheckPoint)
    if not all(np.asarray(v).dtype == np.float32 for v in jax.tree_util.tree_leaves(weights.params)):
        raise ValueError('Expected FP32 parameters')
    stats = {name.removesuffix('.nc'):xr.load_dataset(data / 'assets/stats' / name).astype(np.float32)
             for name in STATS}

    @hk.transform_with_state
    def predict(inputs, targets_template, forcings):
        model = graphcast.GraphCast(weights.model_config, weights.task_config)
        model = normalization.InputsAndResiduals(model, **stats)
        model = autoregressive.Predictor(model, gradient_checkpointing=False)
        return model(inputs, targets_template=targets_template, forcings=forcings)
    return SimpleNamespace(params=weights.params, predict=predict)


def run_benchmark(context, policy, data, model, runtime, report, case, reference, audit):
    import jax
    from graphcast import rollout

    output, values = context.output / 'run', context.config['inference']
    inputs, forcings, target = data
    targets = rollout.extend_targets_template(target.drop_vars('datetime',errors='ignore'), forcings.sizes['time'])
    targets = targets.assign_coords(datetime=forcings.datetime)
    single = context.config['recipe'] == 'single'
    captures, execution, apply = [], None, None
    warned = False
    with jax.default_device(runtime.device), jax.default_matmul_precision('tensorfloat32'):
        params = jax.device_put(model.params)

        def warmup(round_id):
            nonlocal execution, apply
            if round_id == 1 or policy.prepare_round(round_id):
                # Single-step screening installs the candidate only after baseline capture.
                apply = jax.jit(lambda rng, inputs, targets_template, forcings:
                    model.predict.apply(params, {}, rng, inputs, targets_template, forcings)[0])
                execution = RolloutRunner(apply, inputs, targets, forcings, runtime.synchronize, audit)
            label = 'round{}-tf32'.format(round_id)
            result = execution.run(label+'-warmup', values['warmup_steps'], round_id, values['repeats'], warmup=True)
            report['warmups'].append(dict(run=label, seconds=result.step_seconds, wall_seconds=result.wall_seconds))
            if len(report['warmups']) == 1 or single:
                hlo = apply.lower(**result.first_inputs).compile().as_text()
                if len(report['warmups']) == 1:
                    (output/'optimized-hlo.txt').write_text(hlo)
                if single:
                    (output/('optimized-hlo-round%d.txt'%round_id)).write_text(hlo)
            report['compile_seconds'] = report.get('compile_seconds',0) + execution.compile_seconds
            execution.compile_seconds = 0
            emit('Warmup', 'round={}/{} completed; measured prediction restarts from the original inputs'.format(round_id,values['repeats']))

        def measure(round_id):
            nonlocal warned
            label, before_cache = 'round{}-tf32'.format(round_id), apply._cache_size()
            before, before_sample = read_host_sample(context.output,context.device['uuid'],platform='h100')
            emit('Inference','Starting round={}/{} from the original inputs: measured_steps={}'.format(round_id,values['repeats'],values['steps']))
            with capture(output,label,values['steps'],captures):
                result = execution.run(label,values['steps'],round_id,values['repeats'])
            after, after_sample = read_host_sample(context.output,context.device['uuid'],platform='h100')
            if not warned and any(s['status'] != 'available' for s in (before_sample,after_sample)):
                emit('Monitoring','Host samples unavailable or stale; recorded in the report; inference continues','WARN')
                warned = True
            if apply._cache_size() != before_cache:
                raise RuntimeError('Compilation occurred during measurement: jit_cache_before={} jit_cache_after={}'.format(before_cache,apply._cache_size()))
            row = dict(run=label, mode='tf32', round=round_id, apply_seconds=result.step_seconds,
                apply_median_seconds=statistics.median(result.step_seconds), apply_sum_seconds=sum(result.step_seconds),
                rollout_wall_seconds=result.wall_seconds, jit_cache_before=before_cache, jit_cache_after=apply._cache_size(),
                device_before=before,device_after=after,device_before_sample=before_sample,device_after_sample=after_sample)
            save_round(output,report,row,result.final_prediction,target,reference,case,single)

        try:
            run_rounds(values['repeats'],values['warmup_scope'],warmup,measure)
        finally:
            import sys
            primary = sys.exc_info()[0] is not None
            try:
                write_json(output/'capture-ranges.json',captures)
            except Exception as error:
                if not primary:
                    raise
                emit('Failure','Could not save capture boundaries: '+repr(error),'WARN')


def main(context, policy):
    validate_context(context)
    with phase('Prepare','Verify frozen sources and load H100 inputs'):
        case, prepared = load_case(context.path('assets'))
        provenance = dict(source=verify_source(), code_sha256=code_snapshot(), data=prepared,
                          original_case_sha256=case['source_case_sha256'])
        data = load_inputs(context)
    output = context.output / 'run'
    output.mkdir(exist_ok=False)
    report, audit = None, []
    try:
        with phase('Prepare','Prepare H100 device and model'):
            # Match legacy installation order, including imports that may capture hooks.
            policy.prepare_round(1)
            runtime = prepare_runtime(context)
            model = build_model(context.path('assets'))
        report = create_report(context,runtime,provenance)
        write_json(output/'report.json',report)
        reference = load_reference(context.path('assets')/'reference.npz',case) if context.config['recipe']=='full40' else None
        run_benchmark(context,policy,data,model,runtime,report,case,reference,audit)
        load_case(context.path('assets'))
        if code_snapshot() != provenance['code_sha256']:
            raise RuntimeError('Frozen code changed during execution')
        finish_report(output,report)
        return report
    except BaseException as error:
        if report is None:
            report = dict(status='failed',rows=[],warmups=[])
        report.update(status='failed',error=repr(error))
        try:
            write_json(output/'report.json',report)
        except Exception as diagnostic:
            emit('Failure','Could not save failure report: '+repr(diagnostic),'WARN')
        emit('Failure',type(error).__name__+': '+str(error),'ERROR')
        raise
    finally:
        import sys
        primary = sys.exc_info()[0] is not None
        try:
            write_json(output/'step-audit.json',audit)
        except Exception as error:
            if not primary:
                raise
            emit('Failure','Could not save step audit: '+repr(error),'WARN')
