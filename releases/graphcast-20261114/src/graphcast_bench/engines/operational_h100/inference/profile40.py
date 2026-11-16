"""Explicit H100 entry, plus a legacy renderer for archived evidence validation.

New runs never execute generated source. instrument() only validates historical
outputs and provides an immutable-control-flow baseline for regression tests.
"""
import difflib
import hashlib
import json
import os
from pathlib import Path
import sys

FROZEN_SHA256 = 'd61e641dfe228ec17bed4f201aefb6236b099a9e32b866cd075a8a0fde1775e2'

def instrument(original, options=None):
    if hashlib.sha256(original.encode()).hexdigest() != FROZEN_SHA256:
        raise ValueError('Frozen benchmark hash differs; refuse to guess instrumentation')
    changes = [
        ('"measured_steps": 1 if args.profile or args.logical_catalog else 40,',
         '"measured_steps": 1 if args.logical_catalog else 40,'),
        ('"repeats": 1 if args.profile or args.logical_catalog else args.repeats,',
         '"repeats": 1 if args.logical_catalog else args.repeats,'),
        ('rows, warmups, audit = [], [], []', 'rows, warmups, audit = [], [], []\n    capture_ranges = []'),
        ('"profile_only": args.profile,', '"profile_only": args.profile,\n        "experiment_variant": os.environ["GRAPHCAST_VARIANT"],\n        "legacy_accuracy_reference": "official_jax_fp32",'),
        ('assert start_profile() == 0',
         'capture_wall_start_ns = time.time_ns()\n                    capture_mono_start_ns = time.monotonic_ns()\n                    assert start_profile() == 0'),
        ('assert stop_profile() == 0', '''assert stop_profile() == 0
                        capture_ranges.append(dict(label=name, wall_start_ns=capture_wall_start_ns,
                            wall_end_ns=time.time_ns(), monotonic_start_ns=capture_mono_start_ns,
                            monotonic_end_ns=time.monotonic_ns(), measured_steps=40))
                        write_json(args.output / "capture-ranges.json", capture_ranges)'''),
        ('if not (args.profile or args.logical_catalog):', 'if not args.logical_catalog:'),
    ]
    if options is not None:
        steps, warmup = options['steps'], options['warmup_steps']
        repeats, scope = options['repeats'], options['warmup_scope']
        if not (1 <= steps <= 40 and 1 <= warmup <= 40 and repeats > 0 and scope in ('once', 'per-repeat')):
            raise ValueError('Unsupported generic inference options')
        changes[0] = (changes[0][0], '"measured_steps": %d,' % steps)
        changes[1] = (changes[1][0], '"repeats": %d,' % repeats)
        changes[3] = (changes[3][0], '\n'.join([
            '"profile_only": args.profile,',
            '        "experiment_variant": os.environ["GRAPHCAST_VARIANT"],',
            '        "accuracy_status": "not_evaluated",']))
        changes[5] = (changes[5][0], changes[5][1].replace('measured_steps=40', 'measured_steps=%d' % steps))
        warmup_block = '\n'.join([
            '                print("Warmup:", name, flush=True)',
            '                wt, ww, _, kwargs = execute(mode, name + "-warmup", 5)',
            '                warmups.append({"run": name, "seconds": wt, "wall_seconds": ww})',
            '                if len(warmups) == 1:',
            '                    (args.output / "optimized-hlo.txt").write_text(',
            '                        apply.lower(**kwargs).compile().as_text()',
            '                    )',
            '                    if args.logical_catalog:',
            '                        from .logical_inventory import export_jaxpr',
            '',
            '                        export_jaxpr(jax.make_jaxpr(apply)(**kwargs), args.output / "logical")'])
        indented = '\n'.join('    ' + line if line else line for line in warmup_block.split('\n'))
        indented = indented.replace('name + "-warmup", 5)', 'name + "-warmup", %d)' % warmup)
        changes += [
            ('targets = rollout.extend_targets_template(target.drop_vars("datetime", errors="ignore"), 40)',
             'if forcings.sizes["time"] < %d:\n        raise ValueError("Insufficient prepared forcing frames")\n    forcings = forcings.isel(time=slice(0, %d))\n    targets = rollout.extend_targets_template(target.drop_vars("datetime", errors="ignore"), %d)' % (max(steps, warmup), max(steps, warmup), max(steps, warmup))),
            ('reference = load_reference(args.data / "reference.npz", case)', 'reference = None'),
            ('"warmup_steps": 5,', '"warmup_steps": %d,\n        "warmup_scope": "%s",' % (warmup, scope)),
            (warmup_block, '                if %r == "per-repeat" or i == 0:\n' % scope + indented),
            ('accuracy = compare(host, reference, case=case)',
             'if not all(np.isfinite(v.values).all() for v in host.data_vars.values()):\n                        raise ValueError("Nonfinite saved prediction")'),
            ('row.update(accuracy=accuracy, output_sha256=sha256(folder / "prediction.nc"))',
             'row.update(output_sha256=sha256(folder / "prediction.nc"))'),
        ]
    source = original
    for old, new in changes:
        if source.count(old) != 1:
            raise ValueError('Instrumentation anchor is not unique: ' + old)
        source = source.replace(old, new)
    return source

def main():
    # The analyzers keep the frozen namespace; this worker uses maintained modules.
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from graphcast_bench.inference.h100.entry import main as run

    return run()


if __name__ == '__main__':
    raise SystemExit(main())
