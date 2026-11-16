#!/usr/bin/env python3
"""Recalculate full38 metrics from compact saved evidence (Python 3.6+).

This reads saved logical GEMMs, compute intervals, and paired losses. It does
not run a GPU job or replace validation of the original traces/checkpoints.
"""
import collections
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, message):
    require(math.isfinite(actual) and math.isfinite(expected)
            and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10), message)


def product(values):
    result = 1
    for value in values:
        require(isinstance(value, int) and value > 0, 'Invalid GEMM dimension')
        result *= value
    return result


def union(intervals):
    """Keep integer picoseconds until all overlapping intervals are merged."""
    total, end = 0, None
    for start, stop in sorted(intervals):
        require(0 <= start < stop, 'Invalid compute interval')
        total += max(0, stop - max(start, end if end is not None else start))
        end = max(stop, end if end is not None else stop)
    return total


def sweep(intervals):
    """Verify interval union independently using endpoint counts."""
    changes = collections.Counter()
    for start, stop in intervals:
        changes[start] += 1
        changes[stop] -= 1
    total, depth, previous = 0, 0, None
    for point, change in sorted(changes.items()):
        if depth:
            total += point - previous
        depth += change
        require(depth >= 0, 'Negative interval depth')
        previous = point
    require(depth == 0, 'Unclosed compute interval')
    return total


def horizon_for_update(update):
    require(1 <= update <= 38, 'Update outside full38')
    return 1 if update <= 16 else (update - 17) // 2 + 2


def calculate(root):
    summary = json.loads((root / 'summary.json').read_text())
    for name, digest in summary['evidence_files_sha256'].items():
        require(hashlib.sha256((root / name).read_bytes()).hexdigest() == digest,
                'Evidence SHA256 mismatch: ' + name)
    evidence = json.loads((root / 'evidence.json').read_text())
    require(summary['updates'] == 38 and summary['microbatches_per_update'] == 2,
            'Unexpected training configuration')
    for name, digest in summary['executed_releases_sha256'].items():
        require(hashlib.sha256((root / name).read_bytes()).hexdigest() == digest,
                'Executed release identity mismatch')

    flops, gemm_ids = collections.Counter(), collections.defaultdict(set)
    with gzip.open(str(root / 'logical-gemms.csv.gz'), 'rt', newline='') as stream:
        for row in csv.DictReader(stream):
            horizon, index = int(row['horizon']), int(row['index'])
            require(index not in gemm_ids[horizon], 'Duplicate logical GEMM')
            gemm_ids[horizon].add(index)
            lhs, rhs, output, dims = [json.loads(row[name]) for name in
                                      ('lhs', 'rhs', 'output', 'dimension_numbers')]
            product(lhs), product(rhs), product(output)
            left_contract, right_contract = dims[0]
            require(len(left_contract) == len(right_contract)
                    and [lhs[i] for i in left_contract] == [rhs[i] for i in right_contract],
                    'Mismatched GEMM contraction')
            calls = int(row['calls'])
            require(calls > 0, 'Invalid scan multiplicity')
            value = 2 * product([lhs[i] for i in left_contract]) * product(output) * calls
            require(value == int(row['flops_per_microbatch']), 'GEMM FLOPs mismatch')
            flops[horizon] += 2 * value
    require(set(flops) == set(range(1, 13)), 'Missing training horizon FLOPs')
    for horizon in range(1, 13):
        require(gemm_ids[horizon] == set(range(1, len(gemm_ids[horizon]) + 1)),
                'Incomplete GEMM inventory')
        require(flops[horizon] == summary['useful_flops_per_update'][str(horizon)],
                'Horizon FLOPs mismatch')

    groups = collections.defaultdict(list)
    with gzip.open(str(root / 'kernel-intervals.csv.gz'), 'rt', newline='') as stream:
        for row in csv.DictReader(stream):
            platform, update = row['platform'], int(row['update'])
            require(int(row['horizon']) == horizon_for_update(update), 'Wrong update horizon')
            groups[(platform, update)].append((int(row['start_ps']), int(row['end_ps'])))
    expected = {(p, u) for p in ('bw1100', 'h100') for u in range(1, 39)}
    require(set(groups) == expected, 'Missing or extra formal updates')
    result = {}
    for platform in ('bw1100', 'h100'):
        original = evidence['platforms'][platform]
        stages = original['stages']
        require([s['horizon'] for s in stages] == list(range(1, 13)), 'Incomplete stage evidence')
        for stage in stages:
            require(stage['updates'] == [u for u in range(1, 39)
                                         if horizon_for_update(u) == stage['horizon']],
                    'Stage update coverage mismatch')
            require(all(math.isfinite(stage[key]) and stage[key] >= 0 for key in
                        ('worker_seconds', 'input_preparation_seconds', 'compile_seconds')),
                    'Invalid stage elapsed time')
        rows = []
        require([r['update'] for r in original['timing']['rows']] == list(range(1, 39)),
                'Incomplete timing evidence')
        for update in range(1, 39):
            intervals = groups[(platform, update)]
            saved = original['timing']['rows'][update - 1]
            active_ps = union(intervals)
            require(active_ps > 0 and active_ps == sweep(intervals), 'Interval union mismatch')
            require(len(intervals) == saved['kernel_count'], 'Incomplete compute kernel count')
            scale = 1 if platform == 'bw1100' else 1000
            field = 'gpu_active_ps' if platform == 'bw1100' else 'gpu_active_ns'
            require(active_ps == saved[field] * scale, 'Saved update active time mismatch')
            horizon = horizon_for_update(update)
            rows.append(dict(update=update, horizon=horizon, useful_flops=flops[horizon],
                             kernel_count=len(intervals), gpu_active_ps=active_ps))
        active = sum(r['gpu_active_ps'] for r in rows) / 1e12
        total_flops = sum(r['useful_flops'] for r in rows)
        close(active, original['timing']['gpu_active_seconds'], 'Total GPU active mismatch')
        result[platform] = dict(
            updates=38, useful_flops=total_flops, gpu_active_seconds=active,
            synchronized_update_seconds=original['timing']['synchronized_update_seconds'],
            worker_process_seconds=math.fsum(s['worker_seconds'] for s in stages),
            input_preparation_seconds=math.fsum(s['input_preparation_seconds'] for s in stages),
            compile_seconds=math.fsum(s['compile_seconds'] for s in stages),
            mfu_percent=100 * total_flops / (active * summary['peaks_flops_per_second'][platform]))

    losses = collections.defaultdict(list)
    with (root / 'loss-pairs.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            losses[(row['platform'], row['reference'])].append(row)
    require(set(losses) == {('bw1100', 'optimized_h100'), ('h100', 'optimized_h100'),
                            ('h100', 'original_h100')}, 'Unexpected loss comparisons')
    for (platform, reference), rows in sorted(losses.items()):
        require([int(r['update']) for r in rows] == list(range(1, 39)),
                'Missing, reordered, or duplicated loss updates')
        candidates = [float(r['candidate_loss']) for r in rows]
        references = [float(r['reference_loss']) for r in rows]
        require(all(math.isfinite(x) for x in candidates + references), 'Non-finite loss')
        numerator = math.fsum(abs(a - b) for a, b in zip(candidates, references))
        denominator = math.fsum(abs(b) for b in references)
        require(denominator > 0, 'Zero loss denominator')
        error = 100 * numerator / denominator
        saved = evidence['platforms'][platform]['loss_validation'][reference]
        require(saved['trajectory_verified'], 'Unverified checkpoint/loss trajectory')
        close(error, saved['loss_error_percent'], 'Saved aggregate loss mismatch')
        result[platform].setdefault('loss', {})[reference] = dict(
            error_percent=error, threshold_percent=5.0, passed=error <= 5.0)
    result['relative_mfu_bw_to_h100_percent'] = (
        100 * result['bw1100']['mfu_percent'] / result['h100']['mfu_percent'])
    return result


def verify(actual, expected):
    if isinstance(expected, dict):
        require(set(actual) == set(expected), 'Saved result keys differ')
        for key in expected:
            verify(actual[key], expected[key])
    elif isinstance(expected, float):
        close(actual, expected, 'Saved result value differs')
    else:
        require(actual == expected, 'Saved result differs')


if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    actual = calculate(root)
    verify(actual, json.loads((root / 'summary.json').read_text())['results'])
    print(json.dumps(dict(status='verified', scope='Saved full38 evidence recalculation',
                          results=actual), indent=2))
