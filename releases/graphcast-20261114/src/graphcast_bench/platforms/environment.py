"""All worker environment policy, including the frozen H100 compatibility edge.

Application settings live in RuntimeContext. Only libraries, native getenv users
and frozen engines receive environment settings from this module.
"""
import os
from pathlib import Path

PROJECT_PREFIXES = ('GRAPHCAST_', 'TRAINING_', 'RNE_', 'JAX_', 'XLA_')
OVERRIDES = ('LD_PRELOAD', 'PYTHONPATH', 'ROCBLAS_MATH_MODE', 'NVIDIA_TF32_OVERRIDE',
             'MEMORY_COMPILE_ONLY', 'PHYSICAL_DEVICE', 'EXPECTED_PCI',
             'CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES',
             'NVIDIA_VISIBLE_DEVICES', 'STREAM_MICROBATCHES', 'CANDIDATE_XLA_FLAGS')
XLA_FLAGS = '--xla_gpu_enable_command_buffer= --xla_gpu_graph_level=0'


def clean_environment(base=None, updates=None):
    inherited = os.environ if base is None else base
    result = {key: value for key, value in inherited.items()
              if not key.startswith(PROJECT_PREFIXES) and key not in OVERRIDES}
    result.update(updates or {})
    return result


def python_environment(path):
    return dict(PYTHONPATH=path, PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
                OPENBLAS_NUM_THREADS='8', OMP_NUM_THREADS='8')


def delivery_environment(release, platform):
    result = python_environment(str(release) + '/src:' + str(release) + '/third_party/graphcast:/minimal/deps')
    result.update(GRAPHCAST_DELIVERY_LOCKED='1', GRAPHCAST_DELIVERY_PLATFORM=platform)
    return result


def bw_environment(context, release):
    kind, cfg = context.kind, context.config[context.kind]
    result = python_environment(str(release / 'src') + os.pathsep + str(release / 'third_party/graphcast'))
    result.update(PYTHONFAULTHANDLER='1', ROCR_VISIBLE_DEVICES=str(context.device['index']),
                  HIP_VISIBLE_DEVICES='0', CUDA_VISIBLE_DEVICES='0', XLA_MODE='fl_only',
                  JAX_PLATFORMS='rocm', JAX_ENABLE_X64='false',
                  JAX_DEFAULT_MATMUL_PRECISION='tensorfloat32', XLA_FLAGS=XLA_FLAGS,
                  XLA_FLAGTREE_COMPILE_WORKERS='12', XLA_FLAGTREE_PERSISTENT_WORKERS='12',
                  XLA_FLAGTREE_LOG='0', LD_PRELOAD=str(context.path(kind + '_router')),
                  TMPDIR=str(context.output / 'tmp'))
    if context.paths.get('library_dirs'):
        result['LD_LIBRARY_PATH'] = os.pathsep.join(context.paths['library_dirs'] +
            ([os.environ['LD_LIBRARY_PATH']] if os.environ.get('LD_LIBRARY_PATH') else []))
    fraction = cfg['memory_fraction']
    if kind == 'training':
        if context.config.get('rollout') == 11:
            fraction = cfg['horizon11_memory_fraction']
        # These two names are read by the unchanged C++ interposer.
        result.update(TRAINING_GEMM_LOG=str(context.output / 'gemm.jsonl'),
                      RNE_PRE_ROUNDED=cfg['rne_policy'])
    else:
        # These three names are read by the unchanged C++ inference router.
        result.update(GRAPHCAST_STAGE_WEIGHTS=str(release / 'kernels/xf32_rne/weights.tsv'),
                      GRAPHCAST_FUSED_LAYERS=str(release / 'configs/runtime/inference/fused_layers.txt'),
                      GRAPHCAST_STAGE_LOG=str(context.output / 'gemm.jsonl'))
    result.update(XLA_PYTHON_CLIENT_PREALLOCATE='true' if kind == 'training' else 'false',
                  XLA_PYTHON_CLIENT_MEM_FRACTION=str(fraction))
    return result


def h100_environment(context, candidate):
    devices = context['devices']
    if len(devices) != 1 or not devices[0].get('uuid') or not devices[0].get('pci'):
        raise ValueError('Operational H100 requires one device with verified UUID and PCI address')
    device = devices[0]
    result = clean_environment()
    # Historical names below are confined to the frozen engine compatibility edge.
    result.update(GRAPHCAST_PYTHON=context['python'], CUDA_VISIBLE_DEVICES=str(device['uuid']),
                  NVIDIA_VISIBLE_DEVICES=str(device['uuid']), PHYSICAL_DEVICE=str(device['index']),
                  EXPECTED_PCI=device['pci'].lower(), GRAPHCAST_PHYSICAL_DEVICE=str(device['index']),
                  GRAPHCAST_EXPECTED_PCI=device['pci'].lower(), GRAPHCAST_EXPECTED_DEVICE='H100',
                  GRAPHCAST_VARIANT='shifted_mesh', GRAPHCAST_REPEATS='3',
                  TRAINING_CANDIDATE=candidate, STREAM_MICROBATCHES='1', CANDIDATE_XLA_FLAGS='',
                  JAX_ENABLE_X64='false', JAX_DEFAULT_MATMUL_PRECISION='tensorfloat32',
                  JAX_PLATFORMS='cuda', XLA_FLAGS=XLA_FLAGS)
    result.update(python_environment(''))
    return result


def scratch_environment(environment, stage):
    result = dict(environment)
    for key, name in (('TMPDIR', 'tmp'), ('CUDA_CACHE_PATH', 'cuda-cache')):
        path = Path(stage) / name
        path.mkdir(exist_ok=True)
        result[key] = str(path)
    return result


def h100_training_environment(environment, stage, memory_fraction):
    result = scratch_environment(environment, stage)
    result.update(python_environment('/minimal/deps:/minimal/official:/original/experiment'))
    result.update(TRAINING_PLATFORM='h100', TRAINING_OUTPUT='/output', TRAINING_WEEKLY='/weekly',
                  TRAINING_DATA='/training-data', XLA_PYTHON_CLIENT_PREALLOCATE='true',
                  XLA_PYTHON_CLIENT_MEM_FRACTION=memory_fraction)
    return result


def h100_inference_environment(environment, stage, values=None, single=False):
    result = scratch_environment(environment, stage)
    result.pop('XLA_PYTHON_CLIENT_MEM_FRACTION', None)
    result.update(python_environment('/repo/src:/repo/third_party/graphcast'))
    result.update(XLA_PYTHON_CLIENT_PREALLOCATE='false', GRAPHCAST_SINGLE_STEP='1' if single else '0',
                  GRAPHCAST_FROZEN_BENCHMARK='/repo/src/graphcast_bench/benchmark.py',
                  GRAPHCAST_PROFILE_EVIDENCE='/output')
    if values is not None:
        result.update(GRAPHCAST_GENERIC_INFERENCE='1', GRAPHCAST_STEPS=str(values['steps']),
                      GRAPHCAST_WARMUP_STEPS=str(values['warmup_steps']),
                      GRAPHCAST_REPEATS=str(values['repeats']), GRAPHCAST_WARMUP_SCOPE=values['warmup_scope'])
    return result
