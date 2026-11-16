#!/usr/bin/env python3
"""Recalculate the saved six-round MFU and precision metrics using Python 3.6+.

This command reads compact extracted evidence only; it does not start Docker/GPU,
modify files, or claim to re-extract traces/recompare the original NetCDF arrays.
"""
import collections
import csv
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, label):
    require(math.isfinite(actual) and math.isfinite(expected)
            and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10), label)


def union_ps(intervals):
    """Merge overlapping intervals; keep integer picoseconds until conversion."""
    total, end = 0, None
    for start, stop in sorted(intervals):
        require(0 <= start <= stop, 'Invalid kernel interval')
        total += max(0, stop - max(start, end if end is not None else start))
        end = max(stop, end if end is not None else stop)
    return total


def sweep_ps(intervals):
    """Independently verify the union with an endpoint sweep."""
    changes = collections.Counter()
    for start, stop in intervals:
        changes[start] += 1
        changes[stop] -= 1
    total, depth, previous = 0, 0, None
    for point, change in sorted(changes.items()):
        if depth:
            total += point - previous
        depth += change
        require(depth >= 0, 'Invalid interval depth')
        previous = point
    require(depth == 0, 'Unclosed interval')
    return total


def calculate(root):
    summary = json.loads((root / 'summary.json').read_text())
    for name, digest in summary['evidence_files_sha256'].items():
        require(hashlib.sha256((root / name).read_bytes()).hexdigest() == digest,
                'Evidence SHA256 mismatch: ' + name)
    evidence = json.loads((root / 'evidence.json').read_text())
    require(hashlib.sha256((root / 'executed-release.json').read_bytes()).hexdigest()
            == summary['executed_release_sha256'], 'Executed release identity mismatch')
    with (root / 'gemm-flops.csv').open(encoding='utf-8-sig', newline='') as stream:
        gemms = list(csv.DictReader(stream))
    require(len(gemms) == 84 and len({r['layer'] for r in gemms}) == 84,
            'Expected 84 unique logical GEMMs')
    flops = 0
    for row in gemms:
        value = 2 * int(row['m']) * int(row['n']) * int(row['k'])
        require(value > 0 and value == int(row['useful_flops']), 'Invalid GEMM FLOPs')
        flops += value
    require(flops == summary['useful_flops_per_step'], 'FLOPs inventory mismatch')
    steps, repeats = summary['steps'], summary['repeats']
    require((steps, repeats) == (40, 3), 'Unexpected experiment configuration')
    groups = collections.defaultdict(list)
    names = collections.defaultdict(collections.Counter)
    with gzip.open(str(root / 'kernel-intervals.csv.gz'), 'rt', newline='') as stream:
        for row in csv.DictReader(stream):
            key = (row['platform'], int(row['round']))
            require(not row['name'].lower().startswith(('memcpy', 'memset')),
                    'Copy/set event in compute evidence')
            groups[key].append((int(row['start_ps']), int(row['end_ps'])))
            names[key][row['name']] += 1
    platforms = ('bw1100', 'h100')
    expected = set(itertools.product(platforms, range(1, repeats + 1)))
    require(set(groups) == expected, 'Missing or extra formal rounds')
    result = {}
    for platform in platforms:
        rows = []
        record = evidence['platforms'][platform]
        audits = record.get('step_audit', [])
        for number in range(1, repeats + 1):
            key = platform, number
            source = record['rounds'][number - 1]
            require(source['round'] == number, 'Round ordering mismatch')
            require(len(groups[key]) == source['kernel_count'], 'Incomplete kernel count')
            require(names[key] == source['kernel_name_counts'], 'Incomplete kernel signature')
            require(source['jit_cache_before'] == source['jit_cache_after'],
                    'Formal JIT cache changed')
            if platform == 'bw1100':
                formal = [a for a in audits if a['label'] == source['label']]
                require([a['step'] for a in formal] == list(range(1, steps + 1)),
                        'Incomplete formal step audit')
                require(all(a['native_calls'] == a['rounded_calls'] == 84 for a in formal),
                        'Logical GEMM audit mismatch')
            active_ps = union_ps(groups[key])
            require(active_ps > 0 and active_ps == sweep_ps(groups[key]),
                    'Independent interval-union verification failed')
            active = active_ps / 1e12
            rows.append(dict(round=number, kernel_count=len(groups[key]),
                             gpu_active_seconds=active,
                             synchronized_apply_seconds=source['apply_sum_seconds'],
                             rollout_wall_seconds=source['rollout_wall_seconds'],
                             mfu_percent=100 * steps * flops /
                             (active * summary['peaks_flops_per_second'][platform])))
        total = sum(r['gpu_active_seconds'] for r in rows)
        mfu = 100 * repeats * steps * flops / (total * summary['peaks_flops_per_second'][platform])
        result[platform] = dict(rounds=rows, gpu_active_total_seconds=total, mfu_percent=mfu)

    layers = collections.defaultdict(list)
    with gzip.open(str(root / 'accuracy-layers.csv.gz'), 'rt', newline='') as stream:
        for row in csv.DictReader(stream):
            key = row['platform'], int(row['candidate_round']), int(row['reference_round'])
            layers[key].append(row)
    require(set(layers) == set(itertools.product(platforms, range(1, 4), range(1, 4))),
            'Missing or extra precision pairs')
    expected_layers = set(itertools.product(summary['upper_air_variables'], summary['pressure_levels_hpa']))
    require(len(expected_layers) == 78, 'Expected 6 variables x 13 levels')
    for platform in platforms:
        pairs = []
        for ci, ri in itertools.product(range(1, 4), repeat=2):
            rows = layers[(platform, ci, ri)]
            require(len(rows) == 78 and {(r['variable'], int(r['level_hpa'])) for r in rows}
                    == expected_layers, 'Incomplete or duplicate variable-level metrics')
            errors, z500 = [], None
            for row in rows:
                mae, denominator = float(row['mae']), float(row['reference_mean_abs'])
                require(math.isfinite(mae) and mae >= 0 and math.isfinite(denominator)
                        and denominator > 0, 'Invalid precision numerator/denominator')
                value = 100 * mae / denominator
                close(value, float(row['relative_error_percent']), 'Layer error mismatch')
                errors.append(value)
                if row['variable'] == 'geopotential' and int(row['level_hpa']) == 500:
                    z500 = value
            overall = sum(errors) / len(errors)
            original = next(r for r in record_pairs(evidence, platform)
                            if r['candidate_round'] == ci and r['reference_round'] == ri)
            close(overall, original['overall_mean_percent'], 'Overall error mismatch')
            close(z500, original['z500_percent'], 'Z500 error mismatch')
            pairs.append(dict(candidate_round=ci, reference_round=ri,
                              overall_mean_percent=overall, z500_percent=z500))
        result[platform]['accuracy'] = dict(
            pairs=pairs, max_overall_percent=max(r['overall_mean_percent'] for r in pairs),
            max_z500_percent=max(r['z500_percent'] for r in pairs),
            all_pairs_under_5_percent=all(r['overall_mean_percent'] < 5 and r['z500_percent'] < 5 for r in pairs),
            all_pairs_at_most_10_percent=all(r['overall_mean_percent'] <= 10 and r['z500_percent'] <= 10 for r in pairs))
    result['relative_mfu_bw_to_h100_percent'] = 100 * result['bw1100']['mfu_percent'] / result['h100']['mfu_percent']
    return result


def record_pairs(evidence, platform):
    return evidence['platforms'][platform]['accuracy_pairs']


def verify_saved(actual, expected):
    if isinstance(expected, dict):
        require(set(actual) == set(expected), 'Saved result keys differ')
        for key in expected:
            verify_saved(actual[key], expected[key])
    elif isinstance(expected, list):
        require(len(actual) == len(expected), 'Saved result length differs')
        for left, right in zip(actual, expected):
            verify_saved(left, right)
    elif isinstance(expected, float):
        close(actual, expected, 'Saved result value differs')
    else:
        require(actual == expected, 'Saved result differs')


if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    actual = calculate(root)
    verify_saved(actual, json.loads((root / 'summary.json').read_text())['results'])
    print(json.dumps(dict(status='verified', scope='Saved interval/layer-statistic recalculation', results=actual), indent=2))
