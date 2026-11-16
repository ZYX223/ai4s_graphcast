"""Bind explicit worker execution to source files, options and runtime context."""
from pathlib import Path

from graphcast_bench.assets import ROOT, read_json, sha256, write_json

SCHEMA = 'graphcast.h100_inference_worker.v1'
FROZEN_SHA256 = 'd61e641dfe228ec17bed4f201aefb6236b099a9e32b866cd075a8a0fde1775e2'
ENGINE = 'src/graphcast_bench/engines/operational_h100'


def source_files(release=ROOT):
    root = Path(release)
    paths = list((root / 'src/graphcast_bench/inference/h100').glob('*.py'))
    paths += [root / ('src/graphcast_bench/' + name) for name in (
        'progress.py','assets.py','data/inference.py','delivery/options.py',
        'inference/scheduling.py','inference/telemetry.py','platforms/context.py')]
    paths += list((root / ENGINE / 'inference').glob('*.py'))
    paths += [root / ENGINE / 'inference/run.sh', root / ENGINE / 'ORIGINS.json',
              root / 'configs/runtime/code-manifest.json']
    return {str(path.relative_to(root)):sha256(path) for path in sorted(paths)}


def write_manifest(context):
    frozen = ROOT / ENGINE / 'frozen/src/graphcast_bench/benchmark.py'
    if sha256(frozen) != FROZEN_SHA256:
        raise ValueError('Frozen H100 benchmark differs')
    record = dict(schema=SCHEMA, source_sha256=FROZEN_SHA256, sources=source_files(),
                  runtime_context=context.to_dict(), run_options=context.config['inference'],
                  recipe=context.config['recipe'])
    write_json(context.output / 'worker-manifest.json',record)
    return record


def validate_manifest(context, options):
    from graphcast_bench.platforms.context import RuntimeContext

    output = Path(context['output'])
    record = read_json(output / 'worker-manifest.json')
    runtime = RuntimeContext.from_dict(record['runtime_context'])
    saved_context = RuntimeContext.load(output / 'inference-context.json')
    if (record.get('schema') != SCHEMA or record.get('source_sha256') != FROZEN_SHA256
            or record.get('sources') != source_files(context['release'])
            or record.get('run_options') != options or runtime.config['inference'] != options
            or runtime.device != context['devices'][0] or runtime.output != output
            or runtime.platform != 'h100' or runtime.kind != 'inference'
            or runtime.to_dict() != saved_context.to_dict()
            or record.get('recipe') != runtime.config.get('recipe')
            or runtime.config.get('variant') != 'shifted_mesh'):
        raise ValueError('H100 worker evidence differs from requested source/context/options')
    if 'data' in context and runtime.path('assets') != Path(context['data']) / 'prepared-inference':
        raise ValueError('H100 worker input context differs from selected materials')
    return record
