"""Preserve H100 report fields and save the final frame outside CUDA capture."""
import datetime
import importlib.metadata as metadata
import os

import numpy as np
import xarray as xr

from graphcast_bench.assets import sha256, write_json
from graphcast_bench.engines.operational_h100.frozen.src.graphcast_bench.metrics import compare
from graphcast_bench.progress import emit, phase


def create_report(context, runtime, provenance):
    values = context.config['inference']
    report = dict(provenance, schema='graphcast.jax_benchmark.v1', status='running',
        platform='h100', mode='tf32', device=runtime.device.device_kind, pci=runtime.pci,
        versions={name: metadata.version(name) for name in ('jax','jaxlib','dm-haiku','numpy','xarray')},
        profile_only=True, logical_catalog_only=False, measured_steps=values['steps'],
        warmup_steps=values['warmup_steps'], repeats=values['repeats'], warmup_scope=values['warmup_scope'],
        rows=[], warmups=[], router_build=None, storage='float32', jax_matmul_precision='tensorfloat32',
        xla_flags=os.environ.get('XLA_FLAGS'), experiment_variant='shifted_mesh',
        telemetry=dict(source='host', path='../host-memory.jsonl', clocks='not_collected'),
        timing='Synchronized predictor calls; rollout wall includes autoregressive updates and data handling. Formal capture excludes compile, warmup, metrics and saving. First warmup wall includes explicit compilation; compile_seconds is recorded separately.')
    if context.config['recipe'] == 'custom':
        report['accuracy_status'] = 'not_evaluated'
    else:
        report['legacy_accuracy_reference'] = 'official_jax_fp32'
    return report


def save_round(output, report, row, prediction, target, reference=None, case=None, single=False):
    import jax
    from graphcast import xarray_jax

    folder = output / row['run']
    folder.mkdir()
    emit('Round', 'round={}/{} completed | predictor_sum={:.3f}s | predictor_median={:.3f}s | rollout_wall={:.3f}s'.format(
        row['round'], report['repeats'], row['apply_sum_seconds'], row['apply_median_seconds'], row['rollout_wall_seconds']))
    with phase('Save', 'Save round={} outputs and report'.format(row['round'])):
        host = xr.Dataset({name: (value.dims, np.asarray(jax.device_get(xarray_jax.unwrap_data(value))), target[name].attrs)
                           for name, value in prediction.data_vars.items()}, coords=prediction.coords)
        if not all(np.isfinite(value.values).all() for value in host.data_vars.values()):
            raise ValueError('Prediction contains non-finite values')
        if single:
            row['accuracy'] = {'scope':'Single +6h step only; final-frame accuracy not applicable'}
        elif reference is not None:
            row['accuracy'] = compare(host, reference, case=case)
        host.to_netcdf(folder / 'prediction.nc')
        row['output_sha256'] = sha256(folder / 'prediction.nc')
        write_json(folder / 'report.json', row)
        report['rows'].append({key:value for key,value in row.items() if key != 'accuracy'})
        write_json(output / 'report.json', report)
    emit('Save', 'Final frame: {} | forecast lead=+{}h'.format(folder / 'prediction.nc', report['measured_steps']*6))


def finish_report(output, report):
    report.update(status='completed', completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  optimized_hlo_sha256=sha256(output / 'optimized-hlo.txt'))
    write_json(output / 'report.json', report)
