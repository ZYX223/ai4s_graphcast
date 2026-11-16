"""Execution evidence for custom H100 runs, without frozen accuracy claims."""

import collections
import hashlib
import math
from pathlib import Path
import re


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value, positive=False):
    if not isinstance(value, (float, int)) or not math.isfinite(value):
        return False
    return value > 0 if positive else value >= 0


def validate_inference_metadata(report, audit, capture, options):
    steps = options["steps"]
    warmup = options["warmup_steps"]
    repeats = options["repeats"]
    expected = dict(
        status="completed", measured_steps=steps, warmup_steps=warmup,
        repeats=repeats, warmup_scope=options["warmup_scope"], mode="tf32",
        storage="float32", router_build=None,
        jax_matmul_precision="tensorfloat32",
    )
    require(
        all(report.get(key) == value for key, value in expected.items()),
        "Inference configuration differs from requested options",
    )
    rows = report["rows"]
    require(
        [row["round"] for row in rows] == list(range(1, repeats + 1)),
        "Missing or duplicated inference repeat",
    )
    require(len(capture) == repeats, "Missing formal capture")
    names = ["round%d-tf32" % i for i in range(1, repeats + 1)]
    warm_names = names if options["warmup_scope"] == "per-repeat" else names[:1]
    require(
        [row["run"] for row in report["warmups"]] == warm_names,
        "Incorrect warmup scope",
    )
    expected_audit = collections.Counter(
        (name, i) for name in names for i in range(1, steps + 1)
    )
    expected_audit.update(
        (name + "-warmup", i) for name in warm_names for i in range(1, warmup + 1)
    )
    actual_audit = collections.Counter((row["label"], row["step"]) for row in audit)
    require(actual_audit == expected_audit, "Incomplete step audit or unexpected warmup")
    require(
        all(finite(row["seconds"], True) for row in audit),
        "Nonfinite step audit timing",
    )
    previous = None
    for row, bounds, name in zip(rows, capture, names):
        require(
            row["run"] == bounds["label"] == name and row["mode"] == "tf32",
            "Capture label differs from formal repeat",
        )
        require(bounds["measured_steps"] == steps, "Capture steps differ from requested options")
        require(
            len(row["apply_seconds"]) == steps
            and all(finite(value, True) for value in row["apply_seconds"]),
            "Invalid formal step timing",
        )
        require(
            finite(row["apply_sum_seconds"], True)
            and finite(row["rollout_wall_seconds"], True),
            "Invalid repeat timing",
        )
        require(
            math.isclose(
                sum(row["apply_seconds"]), row["apply_sum_seconds"],
                rel_tol=1e-9, abs_tol=1e-9,
            ),
            "Inconsistent formal step timing sum",
        )
        require(
            row["jit_cache_before"] == row["jit_cache_after"],
            "Compilation occurred during formal measurement",
        )
        for prefix in ("wall", "monotonic"):
            start, end = bounds[prefix + "_start_ns"], bounds[prefix + "_end_ns"]
            require(end > start > 0, "Invalid capture bounds")
            if previous:
                require(start > previous[prefix + "_end_ns"], "Overlapping formal capture bounds")
        previous = bounds
    for row in report["warmups"]:
        require(
            len(row["seconds"]) == warmup
            and all(finite(value, True) for value in row["seconds"])
            and finite(row["wall_seconds"], True),
            "Invalid warmup timing",
        )


def validate_optimization_audits(stage, training=False):
    from graphcast_bench.delivery.operational_h100 import read_json

    stage = Path(stage)
    names = (
        "regular3-audit.json", "layernorm-stats-audit.json", "training-mesh-audit.json",
        "final-mesh-audit.json", "g2m-mesh-audit.json",
    ) if training else (
        "regular3-audit.json", "aggregation-audit.json", "mesh-gather-audit.json",
    )
    for name in names:
        audit = read_json(stage / name)
        entries = audit.get("entries", [])
        require(
            audit.get("status") in ("verified", "traced") and bool(entries),
            "Missing optimization audit: " + name,
        )
        if name == "regular3-audit.json":
            selected = [row for row in entries if row.get("selected")]
            require(
                bool(selected) and len(selected) == audit["selected_trace_calls"],
                "Incomplete regular3 audit",
            )
            require(
                all(
                    row.get("complete_ordered_coverage") is True
                    and row.get("batch_axis_preserved") is True
                    and row.get("dtype") == "float32"
                    for row in selected
                ),
                "Invalid regular3 receiver coverage",
            )
        elif name == "aggregation-audit.json":
            require(
                audit.get("trace_modules") == 43 and len(entries) == 43,
                "Incomplete LayerNorm audit",
            )
        elif name == "mesh-gather-audit.json":
            require(
                audit.get("actual_replacements") == 16 and len(entries) == 16
                and all(row.get("all_edges_exactly_once") is True for row in entries),
                "Incomplete mesh receiver coverage",
            )
        elif name == "training-mesh-audit.json":
            require(
                bool(audit.get("scopes"))
                and all(
                    scope.get("status") == "verified"
                    and scope.get("actual_replacements") == 16
                    for scope in audit["scopes"]
                ),
                "Incomplete training mesh scope",
            )
        elif name == "layernorm-stats-audit.json":
            require(audit.get("unique_modules") == 43, "Incomplete training LayerNorm audit")
    return list(names)


def validate_saved_prediction(path, expected_steps):
    import numpy as np
    import xarray as xr

    # Frozen predictions may carry both a dtype attribute and an encoding dtype.
    # Match the frozen comparison reader's explicit CF decoding procedure.
    with xr.open_dataset(path, decode_cf=False) as dataset:
        require(
            bool(dataset.data_vars) and dataset.sizes.get("time") == 1,
            "Missing final prediction frame",
        )
        dataset.time.attrs.pop("dtype", None)
        decoded_time = xr.decode_cf(dataset, decode_timedelta=True).time.values
        require(
            np.array_equal(decoded_time, [np.timedelta64(6 * expected_steps, "h")]),
            "Saved prediction is not the expected final forecast lead",
        )
        require(
            all(
                variable.dtype == np.float32 and np.isfinite(variable.values).all()
                for variable in dataset.data_vars.values()
            ),
            "Saved prediction is nonfinite or not FP32",
        )


def validate_inference(context, options):
    from graphcast_bench.delivery.operational_h100 import (
        ENGINE_RELATIVE, load_module, read_json, sha256, write_json,
    )

    engine = Path(context["release"]) / ENGINE_RELATIVE
    output = Path(context["output"])
    run = output / "run"
    report = read_json(run / "report.json")
    audit = read_json(run / "step-audit.json")
    capture = read_json(run / "capture-ranges.json")
    validate_inference_metadata(report, audit, capture, options)
    require(
        report.get("pci", "").lower() == context["devices"][0]["pci"].lower(),
        "Inference report names a different physical GPU",
    )
    require(
        report.get("experiment_variant") == "shifted_mesh",
        "Unexpected inference optimization variant",
    )
    if context.get('inference_worker_schema') or (output / 'worker-manifest.json').exists():
        from graphcast_bench.inference.h100.provenance import validate_manifest

        validate_manifest(context, options)
    else:
        # Historical records are readable; newly launched runs require a worker manifest.
        wrapper = load_module(engine / "inference/profile40.py", "generic_instrumentation")
        expected_source = wrapper.instrument(
            (engine / "frozen/src/graphcast_bench/benchmark.py").read_text(), options,
        )
        require(
            (output / "executed-benchmark.py").read_text() == expected_source,
            "Executed inference source differs from guarded instrumentation",
        )
        expected_manifest = dict(
            run_options=options, source_sha256=wrapper.FROZEN_SHA256,
            wrapper_sha256=sha256(engine / "inference/profile40.py"),
            executed_sha256=hashlib.sha256(expected_source.encode()).hexdigest(),
        )
        manifest = read_json(output / "instrumentation.json")
        require(
            all(manifest.get(key) == value for key, value in expected_manifest.items()),
            "Inference instrumentation evidence differs",
        )
    hlo = run / "optimized-hlo.txt"
    require(sha256(hlo) == report.get("optimized_hlo_sha256"), "Optimized HLO hash differs")
    require("bf16[" not in hlo.read_text(), "Unexpected BF16 in H100 HLO")
    analyzer = load_module(engine / "inference/analyze.py", "generic_trace")
    traces = sorted(
        output.glob("model-trace*.sqlite"),
        key=lambda path: [
            int(part) if part.isdigit() else part
            for part in re.split(r"(\d+)", path.name)
        ],
    )
    require(len(traces) == options["repeats"], "Trace count differs from requested repeats")
    timings, predictions = [], []
    for trace, row in zip(traces, report["rows"]):
        timing = analyzer.read_trace(
            trace, useful_flops=analyzer.FLOPS40 * options["steps"] / 40,
        )
        timing["workload_signature"] = analyzer.workload_signature(timing, options["steps"])
        require(
            timing["rne_named_kernel_count"] == 0,
            "Explicit RNE kernel unexpectedly present on H100",
        )
        prediction = run / row["run"] / "prediction.nc"
        per_repeat = read_json(prediction.parent / "report.json")
        require(per_repeat == row, "Saved repeat metadata differs from aggregate report")
        require(sha256(prediction) == row.get("output_sha256"), "Saved prediction hash differs")
        validate_saved_prediction(prediction, options["steps"])
        timings.append(timing)
        predictions.append(dict(run=row["run"], sha256=sha256(prediction)))
    audits = validate_optimization_audits(output)
    details = dict(
        status="passed", formal_steps=options["steps"] * options["repeats"],
        captures=len(capture), independent_repeats=options["repeats"],
        predictions=predictions, optimization_audits=audits, rounds=timings,
        accuracy_status="not_evaluated", reproduction_status="not_evaluated",
    )
    write_json(output / "execution-validation.json", details)
    return details


def learning_rate(update):
    if update <= 6:
        return .001 * (update - 1) / 5
    if update <= 16:
        return .001 * .5 * (1 + math.cos(math.pi * (update - 7) / 9))
    return 3e-7


def validate_training_report(report, horizon, updates, warmup_iters, previous=None):
    expected = dict(status="completed", mode="tf32", tiny=False, rollout=horizon)
    require(
        all(report.get(key) == value for key, value in expected.items()),
        "Invalid training configuration",
    )
    require(
        [row["update"] for row in report["rows"]] == updates
        and report.get("expected_updates") == updates,
        "Training updates differ from requested global IDs",
    )
    require(
        report.get("completed_before") == updates[0] - 1,
        "Incorrect initial global update count",
    )
    require(report.get("warmup_iters") == warmup_iters, "Training warmup count differs")
    warmup = report["warmup_audit"]
    require(
        warmup.get("iterations") == warmup_iters
        and warmup.get("params_before") == warmup.get("params_after")
        == report["initial_params_sha256"]
        and warmup.get("optimizer_before") == warmup.get("optimizer_after")
        == report["initial_optimizer_sha256"],
        "Training warmup changed formal parameters or optimizer",
    )
    reload = report["checkpoint_reload"]
    require(
        reload.get("status") == "passed"
        and reload.get("optimizer_counters") == [updates[-1]],
        "Optimizer counter differs from completed global update count",
    )
    if previous:
        require(
            previous["rows"][-1]["update"] == updates[0] - 1,
            "Non-contiguous training checkpoint chain",
        )
        require(
            report["initial_params_sha256"]
            == previous["checkpoint_reload"]["params_sha256"]
            and report["initial_optimizer_sha256"]
            == previous["checkpoint_reload"]["optimizer_sha256"],
            "Parameter or optimizer continuity failed; chain is not continuous",
        )
        require(
            report.get("data_sha256") == previous.get("data_sha256"),
            "Training data changed within checkpoint chain",
        )
    for row in report["rows"]:
        require(
            row["rollout"] == horizon and finite(row["loss"])
            and finite(row["gradient_norm"], True),
            "Nonfinite training loss or gradient",
        )
        change = row["parameter_update"]["max_abs"]
        require(
            math.isclose(
                row["lr"], learning_rate(row["update"]),
                rel_tol=1e-12, abs_tol=1e-15,
            ),
            "Learning rate differs from unchanged recipe",
        )
        require(
            finite(change) and (row["lr"] == 0) == (change == 0),
            "Actual parameter change differs from learning rate",
        )


def validate_training_checkpoint(stage, report, updates):
    import pickle
    import jax
    import numpy as np
    from graphcast_bench.delivery.operational_h100 import sha256

    path = Path(stage) / "checkpoint.pkl"
    require(sha256(path) == report["checkpoint_sha256"], "Training checkpoint hash differs")
    with path.open("rb") as stream:
        checkpoint = pickle.load(stream)
    require(
        checkpoint["updates"] == updates[-1] and checkpoint["mode"] == "tf32"
        and checkpoint["tiny"] is False
        and checkpoint["data_sha256"] == report["data_sha256"],
        "Checkpoint configuration differs from formal training",
    )

    def digest(tree):
        result = hashlib.sha256()
        for leaf in jax.tree.leaves(tree):
            array = np.asarray(leaf)
            require(np.isfinite(array).all(), "Nonfinite checkpoint parameter or optimizer")
            result.update(str((array.shape, array.dtype.str)).encode())
            result.update(array.tobytes())
        return result.hexdigest()

    reload = report["checkpoint_reload"]
    require(
        digest(checkpoint["params"]) == reload["params_sha256"]
        and digest(checkpoint["opt_state"]) == reload["optimizer_sha256"],
        "Checkpoint tree identity differs from reload report",
    )
    counters = [
        int(np.asarray(leaf)) for leaf in jax.tree.leaves(checkpoint["opt_state"])
        if np.asarray(leaf).shape == () and np.asarray(leaf).dtype.kind in "iu"
    ]
    require(counters == [updates[-1]], "Actual optimizer counter differs from global updates")
