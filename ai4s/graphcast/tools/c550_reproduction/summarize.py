"""Audit results without GPU work. Metric checks never imply total acceptance."""
from __future__ import annotations
import argparse
import math
from pathlib import Path
import re
import sys

from reproduce import ROOT, dump, load_module, read, sha, verify_job

FLOPS = 29434.23636529152
PEAK = 132.49
H100_MFU = 10.058595198508778
H100_LOSS = 31.3537019609331


def require(condition, message):
    if not condition:
        raise ValueError(message)


def full38(rows):
    require([x['step'] for x in rows] == list(range(1, 39)), 'need all 38 ordered updates')
    require([x['rollout_steps'] for x in rows] == [1] * 16 + [r for r in range(2, 13) for _ in range(2)], 'curriculum changed')
    for row in rows:
        require(row['grad_ok'] and math.isfinite(row['loss']), 'nonfinite loss/gradient')
        require(math.isfinite(row['end_to_end_step_seconds']) and row['end_to_end_step_seconds'] > 0, 'invalid E2E')
        require(row['global_batch_size'] == 2 and row['global_micro_batch_size'] == 1, 'batch changed')
    useful = sum(x['training_flops'] for x in rows) / 1e12
    require(math.isclose(useful, FLOPS, rel_tol=1e-12), 'workload FLOPs changed')
    seconds = sum(x['end_to_end_step_seconds'] for x in rows)
    mfu = useful / seconds / PEAK * 100
    return dict(e2e_seconds=seconds, mfu_pct=mfu, h100_ratio_pct=mfu / H100_MFU * 100,
                performance_threshold_passed=mfu / H100_MFU >= .8,
                includes_lr_warmup=True, timed_updates=38, device_count=1, peak_tflops=PEAK)


def validation(result, audit, indices):
    entry = load_module(ROOT / 'harness/evaluation/evaluate_remaining_cp38.py', 'validation_contract')
    entry.validate_result(result, indices)
    require(audit['model_unchanged'] is True and audit['gradient_tensors'] == 0, 'model changed or gradients present')
    gap = abs(result['mean_loss'] - H100_LOSS) / abs(H100_LOSS) * 100
    return dict(mean_loss=result['mean_loss'], h100_mean_loss=H100_LOSS,
                relative_difference_pct=gap, effect_threshold_passed=gap <= 5,
                validation_windows=111, optimizer_updates=0)


def source_and_exit(job):
    verify_job(job)
    meta = read(job / 'job.json')
    require(read(job / 'launcher-exit.json')['returncode'] == 0, 'launcher failed or incomplete')
    out = job / 'candidate'
    exit_record = read(out / 'exit.json')
    require(exit_record.get('returncode', exit_record.get('rc')) == 0, 'worker failed')
    precision = read(out / 'exit-precision.json')
    values = precision.get('precision', precision)
    require(bool(values) and all(x is True or x == 'tf32' for x in values.values()), 'TF32 exit readback failed')
    model_root = Path(meta['settings']['model_root'])
    reference = read(job / 'reference.json')
    expected = {**reference['frozen_source_sha256'], **reference['integration_source_sha256']}
    imports = read(out / 'actual-model-imports.json')
    require(len(imports) >= 28, 'missing actual model imports')
    for record in imports.values():
        rel = str(Path(record['path']).relative_to(model_root))
        require(expected.get(rel) == record['sha256'], 'actual model import drift: ' + rel)
    if meta['mode'] != 'evaluation':
        plan = read(out / 'launch.json')
        require(exit_record['before'] == exit_record['after'] == plan['source_verification'], 'source changed')
        actual = exit_record['before'].get('candidate_manifest', exit_record['before'].get('source_manifest_sha256'))
        require(actual == sha(job / 'flaggems-accepted.json'), 'unqualified FlagGems source')
        require(plan['coverage_extension_sha256'] == sha(job / 'coverage_extension.py'), 'ClipRouting changed')
    return meta, out


def recovery(job, out, continuous, comparison):
    settings = read(job / 'job.json')['settings']
    require(continuous.resolve() == Path(settings['parent_run']).resolve(), 'reference must be the prepared continuous parent run')
    require(continuous.resolve() != (out / 'run').resolve(), 'recovery cannot be its own reference')
    require(comparison['continuous_sha256'] == '94dbe71828ecc7793114fad04ba0a38b25ee4db4515c63eee26c74bd0bea3a2d', 'reference must be the locked original continuous CP38')
    require(Path(comparison['continuous_checkpoint']).resolve() == continuous.resolve() / 'final_checkpoint.pt', 'wrong continuous checkpoint path')
    require(Path(comparison['resumed_checkpoint']).resolve() == (out / 'run/final_checkpoint.pt').resolve(), 'wrong resumed checkpoint path')
    cfg = read(job / 'config.json')
    limits = cfg['recovery_tolerances']
    plan = read(out / 'launch.json')
    require(plan['requested_updates'] == 2 and plan['actual_update_steps'] == [37, 38], 'not a two-update recovery')
    executed = [int(x) for x in re.findall(r'^official_graphcast_train update=(\d+)/', (out / 'training.log').read_text(), re.M)]
    require(executed == [37, 38], 'inherited history is not new execution')
    rows = read(out / 'run/summary.json')['metrics']
    baseline = read(continuous / 'summary.json')['metrics']
    require(len(rows) == len(baseline) == 38 and rows[:36] == baseline[:36], 'metric prefix mismatch')
    full38(rows)  # completeness only; never report recovery MFU
    start, rng = read(out / 'resume-start-guard.json'), read(out / 'resume-rng-guard.json')
    require(all(start['actual'][k] == v for k, v in start['expected'].items()) and start['scheduled_lr'] == 3e-7, 'start state mismatch')
    require(rng['exact'] and rng['actual'] == rng['expected'] == cfg['resume_configuration']['expected_resume_rng_sha'], 'RNG restoration failed')
    prefix = read(out / 'resume-prefix-hook-receipt.json')
    require(prefix['actual_context_checked'] and prefix['training_updates_replayed'] == 0 and prefix['resume_step'] == 36, 'prefix lineage failed')
    require(comparison['hashes_match_recorded'] and comparison['cuda_initialized'] is False, 'CP comparison not verified')
    require(comparison['continuous_sha256'] == read(continuous / 'summary.json')['checkpoint_sha256'], 'wrong reference CP')
    require(comparison['resumed_sha256'] == read(out / 'run/summary.json')['checkpoint_sha256'], 'wrong resumed CP')
    for key in ('cursor_equal', 'schedule_equal', 'rng_equal', 'resume_contract_equal', 'initial_lineage_equal'):
        require(comparison[key], key)
    for key in ('model', 'optimizer'):
        require(comparison[key]['all_finite'] and comparison[key]['relative_l2'] <= limits[key + '_relative_l2_max'], key + ' tolerance exceeded')
    differences = []
    for actual, expected in zip(rows[-2:], baseline[-2:]):
        delta = {k: abs(actual[k] - expected[k]) / max(abs(expected[k]), 1e-30)
                 for k in ('loss', 'grad_norm_before_clip')}
        require(math.isfinite(delta['loss']) and delta['loss'] <= limits['loss_each_relative_max'], 'loss tolerance exceeded')
        require(math.isfinite(delta['grad_norm_before_clip']) and delta['grad_norm_before_clip'] <= limits['grad_norm_each_relative_max'], 'gradient tolerance exceeded')
        differences.append(dict(step=actual['step'], **delta))
    return dict(recovery_gate_passed=True, executed_updates=executed, tolerances=limits,
                relative_differences=differences, formal_mfu_eligible=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job', type=Path, required=True)
    p.add_argument('--continuous-run', type=Path)
    p.add_argument('--checkpoint-comparison', type=Path)
    a = p.parse_args()
    job = a.job.resolve()
    meta, out = source_and_exit(job)
    mode = meta['mode']
    if mode == 'full38':
        require(not (out / 'resume-prefix-hook-receipt.json').exists(), 'resumed course cannot yield formal MFU')
        executed = [int(x) for x in re.findall(r'^official_graphcast_train update=(\d+)/', (out / 'training.log').read_text(), re.M)]
        require(executed == list(range(1, 39)), 'need one actual continuous 38-update course')
        result = full38(read(out / 'run/summary.json')['metrics'])
    elif mode == 'short8':
        sys.path.insert(0, str(ROOT / 'harness/short8'))
        from prediction import evaluate
        state = read(out / 'final-state.json')
        require(state['all_parameters_finite'] and state['all_optimizer_finite'] and state['initial_load_count'] == 1, 'short-run state invalid')
        require(not state['checkpoint_written'] and not list(out.rglob('*.pt')), 'unexpected short-run checkpoint')
        result = evaluate(read(out / 'sparse-window.json')['all_metrics'])
    elif mode == 'recovery':
        require(a.continuous_run is not None and a.checkpoint_comparison is not None, 'recovery needs reference metrics and CPU full-state comparison')
        result = recovery(job, out, a.continuous_run, read(a.checkpoint_comparison))
    else:
        spec = read(job / 'spec.json')
        windows = read(Path(spec['data_root']) / 'validation-windows-r12.json')
        indices = [row['reference_index'] for row in windows['rows']]
        result = validation(read(out / 'validation.json'), read(out / 'validation-audit.json'), indices)
        require(read(out / 'validation-audit.json')['checkpoint_sha256'] == spec['checkpoint_sha256'], 'CP38 audit mismatch')
        sys.path.insert(0, str(ROOT / 'harness/evaluation'))
        import validation_journal as journal
        identity = {k: spec[k] for k in ('checkpoint_sha256', 'data_sha256', 'windows_sha256',
                    'corpus_sha256', 'flaggems_manifest_sha256', 'runner_sha256', 'model_setup_ast_sha256')}
        identity['checkpoint_model_sha256'] = read(out / 'runtime.json')['checkpoint_model_sha256']
        journal.verify_complete(out / 'window-results', identity, read(out / 'validation.json'))
    result.update(mode=mode, overall_acceptance=False,
                  note='Coverage and remaining acceptance gates must be reviewed separately; historical coverage is not a new measurement.')
    dump(job / 'result-audit.json', result)
    print(__import__('json').dumps(result, indent=2))


if __name__ == '__main__':
    if sys.flags.optimize:
        raise RuntimeError('assertion-eliding Python modes are forbidden')
    main()
