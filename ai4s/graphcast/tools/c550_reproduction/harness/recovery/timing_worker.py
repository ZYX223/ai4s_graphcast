"""Audit imports, then execute the frozen training entry without profiling hooks."""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import runpy
import sys

p = argparse.ArgumentParser()
p.add_argument('--plan', required=True, type=Path)
a = p.parse_args()
plan = json.loads(a.plan.read_text())
from runtime_gate import check_runtime
check_runtime(plan)
import torch
assert torch.cuda.device_count() == 1

if os.environ.get('GRAPHCAST_ACCELERATOR_TYPE') == 'npu':
    import torch_npu
elif os.environ.get('GRAPHCAST_ACCELERATOR_TYPE') == 'musa':
    import torch_musa
sys.path.insert(0, str(Path(plan['model']) / 'src'))
from graphcast_compat import backend
backend._activate_flaggems_source_root()
import flag_gems
assert torch.cuda.device_count() == 1
assert os.environ.get("CUDA_VISIBLE_DEVICES") == plan["environment"]["CUDA_VISIBLE_DEVICES"]
source = Path(plan['new_source']).resolve()
assert Path(flag_gems.__file__).resolve() == source / 'src/flag_gems/__init__.py'
receipts = {}
for name, expected in [('layer_norm_backward', plan['expected_ln_sha']),
                       (plan['matrix_operator'], plan['expected_matrix_sha'])]:
    f = Path(inspect.getsourcefile(inspect.unwrap(getattr(flag_gems, name)))).resolve()
    assert f.is_relative_to(source)
    actual = hashlib.sha256(f.read_bytes()).hexdigest()
    assert actual == expected, (name, f, actual, expected)
    receipts[name] = {'path': str(f), 'sha256': actual}
(Path(plan['output_dir']) / 'import-receipt.json').write_text(json.dumps({
    'file': flag_gems.__file__, 'operators': receipts, 'no_site': bool(sys.flags.no_site),
    'torch': torch.__version__, 'visibility': {k: os.environ[k] for k in
        ['CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'MUSA_VISIBLE_DEVICES', 'ASCEND_RT_VISIBLE_DEVICES'] if k in os.environ},
    'device_count': torch.cuda.device_count(), 'device_name': torch.cuda.get_device_name(0), 'physical_gpu_from_gate': plan['physical_gpu'], 'expected_pci_bus': plan['expected_pci_bus'], 'observer': 'no profiling observer; unchanged project ClipRouting integration retained'}, indent=2))
out = Path(plan['output_dir'])
from graphcast_compat import training_diagnostics as diagnostics
from graphcast_compat.training_state import state_object_sha256
from resume_metric_prefix_hook import stage_into_existing_run
original_segment_record = diagnostics.segment_start_record
checkpoint = torch.load(plan['resume_checkpoint'], map_location='cpu', weights_only=False)
# The frozen training entry sets the next scheduled LR before recording start state.
for group in checkpoint['optimizer_state']['param_groups']:
    group['lr'] = plan['expected_resume_lr']
expected_states = {
    'parameters_sha256': state_object_sha256(checkpoint['model_state']),
    'optimizer_sha256': state_object_sha256(checkpoint['optimizer_state']),
}
assert expected_states['parameters_sha256'] == plan['expected_resume_parameter_sha']
assert expected_states['optimizer_sha256'] == plan['expected_resume_optimizer_sha']
del checkpoint
def record_with_history(*args, **kwargs):
    record = original_segment_record(*args, **kwargs)
    assert len(record['states_by_rank']) == 1
    state = record['states_by_rank'][0]
    (out/'resume-start-guard.json').write_text(json.dumps(dict(actual=state, expected=expected_states, scheduled_lr=record['optimizer']['lr']),indent=2))
    assert record['optimizer']['lr'] == plan['expected_resume_lr']
    assert all(state[k] == v for k, v in expected_states.items()), 'restored state differs after scheduled LR'
    audit = json.loads(Path(plan['resume_audit_file']).read_text())
    assert audit['checkpoint_sha256'] == plan['resume_sha256']
    receipt = stage_into_existing_run(plan['metric_parent_run'], out/'run', audit, record)
    receipt.update(restored_parameter_and_optimizer_hashes_equal=True, restored_states=expected_states)
    (out/'resume-prefix-hook-receipt.json').write_text(json.dumps(receipt,indent=2))
    return record
diagnostics.segment_start_record = record_with_history


from graphcast_compat import training_state as state_module
original_restore_rng=state_module.restore_rng_state
def restore_rng_checked(state, **kwargs):
    expected=state_object_sha256(state)
    assert expected==plan['expected_resume_rng_sha']
    original_restore_rng(state,**kwargs)
    actual=state_object_sha256(state_module.capture_rng_state(**kwargs))
    (out/'resume-rng-guard.json').write_text(json.dumps(dict(expected=expected,actual=actual,exact=actual==expected),indent=2))
    assert actual==expected, 'RNG not exactly restored'
state_module.restore_rng_state=restore_rng_checked

sys.argv = plan['command'][6:]
def save_exit_receipts():
    (Path(plan['output_dir'])/'clip-routes.json').write_text(json.dumps(clip_routing.record(),indent=2))
    clip_routing.close()
    model=Path(plan['model']).resolve();actual={}
    for name,module in list(sys.modules.items()):
        file=getattr(module,'__file__',None)
        if not file:continue
        path=Path(file).resolve()
        if path.is_file() and path.is_relative_to(model):
            actual[name]=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    out=Path(plan['output_dir'])
    (out/'actual-model-imports.json').write_text(json.dumps(actual,indent=2))
    if os.environ.get('GRAPHCAST_ACCELERATOR_TYPE')=='npu':
        precision=dict(npu_matmul_allow_hf32=bool(torch.npu.matmul.allow_hf32),npu_conv_allow_hf32=bool(torch.npu.conv.allow_hf32))
        assert all(precision.values())
    elif os.environ.get('GRAPHCAST_ACCELERATOR_TYPE')=='musa':
        precision=dict(musa_matmul_allow_tf32=bool(torch.backends.musa.matmul.allow_tf32),mudnn_allow_tf32=bool(torch.backends.mudnn.allow_tf32))
        assert all(precision.values())
    elif hasattr(torch.backends.cuda.matmul,'fp32_precision') and hasattr(torch.backends.cudnn.conv,'fp32_precision'):
        precision=dict(cuda_matmul_fp32_precision=torch.backends.cuda.matmul.fp32_precision,cudnn_conv_fp32_precision=torch.backends.cudnn.conv.fp32_precision)
        assert all(v=='tf32' for v in precision.values())
    else:
        precision=dict(matmul_allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32))
        assert all(precision.values())
    (out/'exit-precision.json').write_text(json.dumps(dict(precision=precision,stage='after_training',timed_region_hooks=False),indent=2))
from coverage_extension import ClipRouting
assert hashlib.sha256(Path(__file__).with_name('coverage_extension.py').read_bytes()).hexdigest()==plan['coverage_extension_sha256']
assert hashlib.sha256(Path(inspect.getsourcefile(torch.nn.utils.clip_grad_norm_)).read_bytes()).hexdigest()==plan['expected_clip_source_sha256']
clip_routing=ClipRouting(torch,flag_gems).install()
from exit_capture import install_exit_capture
install_exit_capture(lambda phase: save_exit_receipts())
try:
    runpy.run_path(sys.argv[0],run_name='__main__')
finally:
    save_exit_receipts()
