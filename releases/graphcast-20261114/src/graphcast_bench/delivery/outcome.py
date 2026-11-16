"""Summarize execution, reference accuracy and monitoring as independent evidence."""
import json
import math
from pathlib import Path


def monitoring_summary(output, stages, training=False):
    counts = dict(samples=0, process_samples=0, warning_samples=0, error_samples=0,
                  foreign_mapping_samples=0, unresolved_samples=0, unreadable_records=0,
                  missing_files=0, empty_files=0, host_error_records=0,
                  unavailable_round_samples=0, round_warning_samples=0)
    try:
        counts['host_error_records'] = sum(bool(line.strip()) for line in
                                          (output / 'monitor-errors.log').read_text(encoding='utf-8').splitlines())
    except FileNotFoundError:
        pass  # This file is created only when host sampling raises an error.
    except (OSError, UnicodeError):
        counts['unreadable_records'] += 1
    for stage in stages:
        for kind in ('host-memory', 'process-monitor'):
            path = output / stage / (kind + '.jsonl')
            records = 0
            try:
                with path.open(encoding='utf-8') as stream:
                    for line in stream:
                        if not line.strip():
                            continue
                        records += 1
                        try:
                            record = json.loads(line)
                            if not isinstance(record, dict):
                                raise ValueError('Expected a sample object')
                            if kind == 'host-memory':
                                counts['samples'] += 1
                                counts['warning_samples'] += bool(record.get('warning'))
                                counts['error_samples'] += bool(record.get('error') or not record.get('devices'))
                            else:
                                counts['process_samples'] += 1
                                processes = record['processes']
                                counts['unresolved_samples'] += bool(
                                    processes.get('known') is not True or processes.get('unknown_pids'))
                                counts['foreign_mapping_samples'] += any(record['foreign_card_pids'].values())
                        except (ValueError, TypeError, KeyError, AttributeError):
                            counts['unreadable_records'] += 1
                if not records:
                    counts['empty_files'] += 1
            except FileNotFoundError:
                counts['missing_files'] += 1
            except (OSError, UnicodeError):
                counts['unreadable_records'] += 1
        # Worker snapshots can be unavailable even if earlier host samples were valid.
        try:
            report_path = output / stage / ('report.json' if training else 'run/report.json')
            report = json.loads(report_path.read_text(encoding='utf-8'))
            rows = report['rows']
            if not isinstance(rows, list) or not rows:
                raise ValueError('No per-round telemetry records')
            for row in rows:
                for key in ('device_before_sample', 'device_after_sample'):
                    sample = row.get(key, {})
                    counts['unavailable_round_samples'] += sample.get('status') != 'available'
                    counts['round_warning_samples'] += bool(sample.get('warning'))
        except FileNotFoundError:
            counts['missing_files'] += 1
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            counts['unreadable_records'] += 1
    return counts


def h100_accuracy(output, recipe):
    """Use the H100 screening and both original/optimized reference gates."""
    single = output / ('preflight-single' if recipe == 'full40' else '')
    checks = [(single / 'single-step-validation.json', 'status', 'passed')]
    if recipe == 'full40':
        checks.extend([
            (output / 'accuracy-h100/summary.json', 'all_pairs_under_5_percent', True),
            (output / 'accuracy-optimized-reference.json', 'status', 'passed'),
        ])
    values = []
    for path, key, expected in checks:
        try:
            value = json.loads(path.read_text())[key]
            values.append(value == expected if type(value) is type(expected) else None)
        except (OSError, ValueError, KeyError, TypeError):
            values.append(None)
    return values


def training_accuracy(output, h100):
    """Read full38 worker verdicts; partial or malformed evidence cannot pass."""
    try:
        if h100:
            result = json.loads((output / 'training-validation.json').read_text())
            records = [result['loss_validation'][key] for key in ('optimized_h100', 'original_h100')]
        else:
            records = [json.loads((output / 'training-loss-summary.json').read_text())]
        verdicts = []
        expected = list(range(1, 39))
        for result in records:
            value = result['loss_error_percent']
            if (result.get('metric_version') != 'sugon.training_loss.sum_abs.v1'
                    or result.get('comparison_status') != 'paired_updates_verified'
                    or result.get('expected_updates') != expected
                    or [row['update'] for row in result['rows']] != expected
                    or type(result.get('metric_passed')) is not bool
                    or isinstance(value, bool) or not isinstance(value, (float, int))
                    or not math.isfinite(value) or value < 0):
                return [None]
            verdicts.append(result['metric_passed'])
        return verdicts
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return [None]


def summarize_run(context, output, code):
    output = Path(output)
    options = context['run_options']
    h100 = context.get('platform') == 'h100'
    training = context.get('kind', options.get('kind')) == 'training'
    if training:
        values = options.get('training', {})
        horizons = [stage['horizon'] for stage in values.get('stages', [])]
        if not horizons:
            horizons = context.get('horizons', list(range(1, 13)) if options['preset'] else [])
        stages = ['r%02d' % horizon for horizon in horizons]
    elif h100:
        stages = ['preflight-single', ''] if options.get('recipe') == 'full40' else ['']
    else:
        stages = (['single', 'full40'] if options.get('recipe') == 'full40' else ['single']) if options['preset'] else ['inference']
    monitoring = monitoring_summary(output, stages, training=training)
    accuracy = 'not_evaluated'
    if options['preset']:
        comparisons = (training_accuracy(output, h100) if training else
                       h100_accuracy(output, options.get('recipe')) if h100 else [])
        for stage in (() if h100 or training else stages):
            folder = 'single-comparison' if stage == 'single' else 'comparison'
            try:
                metrics = json.loads((output / folder / 'summary.json').read_text())
                comparisons.append(metrics['all_pairs_at_most_10_percent'])
            except (OSError, ValueError, KeyError, TypeError):
                comparisons.append(None)
        if any(value is False for value in comparisons):
            accuracy = 'failed'
        elif code == 0 and all(value is True for value in comparisons):
            accuracy = 'passed'
    issues = any(monitoring[key] for key in (
        'warning_samples', 'error_samples', 'foreign_mapping_samples',
        'unresolved_samples', 'unreadable_records', 'missing_files', 'empty_files',
        'host_error_records', 'unavailable_round_samples', 'round_warning_samples'))
    issues = issues or not monitoring['samples'] or not monitoring['process_samples']
    performance = 'not_accepted' if code != 0 or issues else 'not_evaluated'
    reason = ('Execution failed' if code != 0 else 'Monitoring is incomplete or reports ownership/driver issues'
              if issues else 'Raw traces require separate performance analysis')
    result = dict(schema='graphcast.run_summary.v1',
                  execution=dict(status='passed' if code == 0 else 'failed', exit_code=code),
                  accuracy=dict(status=accuracy),
                  performance=dict(status=performance, reason=reason), monitoring=monitoring)
    (output / 'run-summary.json').write_text(json.dumps(result, indent=2) + '\n')
    return result
