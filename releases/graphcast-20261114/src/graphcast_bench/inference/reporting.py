"""Legacy inference reports and final-frame files, written outside capture."""
import datetime
import importlib.metadata as metadata
import os

import numpy as np
import xarray as xr

from graphcast_bench.assets import sha256, write_json
from graphcast_bench.progress import emit, phase


def create_report(args, runtime, provenance):
    """Keep field names stable for completion checks and frozen comparisons."""
    return {
        'schema': 'graphcast.rne_model_fusion.v1',
        'variant': args.variant,
        'xla_mode': os.environ['XLA_MODE'],
        **provenance,
        'status': 'running',
        'platform': args.platform,
        'mode': args.mode,
        'versions': {name: metadata.version(name) for name in ('jax', 'jaxlib', 'dm-haiku', 'numpy', 'xarray')},
        'device': runtime.device.device_kind,
        'pci': runtime.pci,
        # Historical report key: this enables capture, not a reduced workload.
        'profile_only': args.capture_enabled,
        'accuracy_reference': 'Separate frozen H100 validation required',
        'telemetry': dict(source='host', path='../host-memory.jsonl',
                          clocks='not_collected', availability='See per-round sample metadata'),
        'warmup_steps': args.warmup_steps,
        'warmup_scope': args.warmup_scope,
        'logical_catalog_only': args.logical_catalog,
        'measured_steps': 1 if args.logical_catalog else args.steps,
        'repeats': 1 if args.logical_catalog else args.repeats,
        'rows': [],
        'warmups': [],
        'router_build': runtime.router.build,
        'storage': 'float32',
        'jax_matmul_precision': 'tensorfloat32',
        'xla_flags': os.environ.get('XLA_FLAGS'),
        'timing': 'Synchronized predictor calls; preloaded-data rollout wall includes transfer and autoregressive update. Excludes load, compile, warmup, metrics and save. Hygon includes existing routing/audit and RNE kernels. Graphs disabled.',
    }


def save_prediction(path, prediction, target):
    """Transfer and validate only the final chunk; no full-trajectory retention."""
    import jax
    from graphcast import xarray_jax

    host = xr.Dataset(
        {name: (value.dims, np.asarray(jax.device_get(xarray_jax.unwrap_data(value))), target[name].attrs)
         for name, value in prediction.data_vars.items()},
        coords=prediction.coords,
    )
    if not all(np.isfinite(value.values).all() for value in host.data_vars.values()):
        raise ValueError('Prediction contains non-finite values')
    host.to_netcdf(path)
    return sha256(path)


def save_round(output, report, row, prediction, target, logical_catalog):
    folder = output / row['run']
    folder.mkdir()
    emit('Round', 'round={}/{} completed | predictor_sum={:.3f}s | predictor_median={:.3f}s | rollout_wall={:.3f}s'.format(
        row['round'], report['repeats'], row['apply_sum_seconds'],
        row['apply_median_seconds'], row['rollout_wall_seconds']))
    with phase('Save', 'Save round={} outputs and report'.format(row['round'])):
        if not logical_catalog:
            digest = save_prediction(folder / 'prediction.nc', prediction, target)
            row.update(accuracy=None, output_sha256=digest)
        write_json(folder / 'report.json', row)
        report['rows'].append({key: value for key, value in row.items() if key != 'accuracy'})
        write_json(output / 'report.json', report)
    if not logical_catalog:
        emit('Save', 'Final frame: {} | forecast lead=+{}h'.format(
            folder / 'prediction.nc', report['measured_steps'] * 6))
    else:
        emit('Save', 'Logical catalog run: no prediction file is requested')


def finish_report(output, report, hook, step_audit):
    write_json(output / 'hook-audit.json',
               dict(seen=sorted(hook.seen), expected=sorted(hook.targets), variant=report['variant']))
    write_json(output / 'step-audit.json', step_audit)
    report.update(status='completed',
                  completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  optimized_hlo_sha256=sha256(output / 'optimized-hlo.txt'))
    write_json(output / 'report.json', report)
