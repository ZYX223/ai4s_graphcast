"""Check evidence completion independently of profiler/child exit status.

Passing means an execution is complete, not that cross-platform accuracy or
performance passed. The H100 comparisons are a separate acceptance step.
"""

import hashlib
import json
import math
from pathlib import Path

GRAPH_SHA = "5654cd00383186ba94a8a150fbe5682e405c41aac2747117af5abb10ef9bbc95"
TRAINING_GRAPH_SHA = "cd4dc305505debbdd7460dc0652f09d08c18731fad860016e32ba34df4865e1d"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def valid_times(values, count):
    return len(values) == count and all(math.isfinite(v) and v > 0 for v in values)


def validate(root, action, expected_updates=None, expected_rollout=None, predecessor=None):
    root = Path(root)
    require(
        action in ("single", "full40", "training-smoke", "training-stage"),
        "Unknown validation action",
    )
    require((root / "python-exit-code.txt").read_text().strip() == "0", "Python failed")
    is_training = action in ("training-smoke", "training-stage")
    graph_sha = TRAINING_GRAPH_SHA if is_training else GRAPH_SHA
    run = root if is_training else root / "run"
    r = read(run / "report.json")
    require(r["status"] == "completed", "Inner model report is incomplete")
    require(r["mode"] == "rne_xf32", "Wrong precision path")
    graph = [
        json.loads(line)
        for line in (root / "reference-graph-audit.jsonl").read_text().splitlines()
        if line.strip()
    ]
    require(bool(graph), "Missing actual static graph restore audit")
    require(
        all(
            x["payload_sha256"] == graph_sha
            and x["all_static_graph_arrays_verified"] is True
            and x["grid2mesh_edges"] == 1618746
            for x in graph
        ),
        "Wrong reference graph",
    )
    regular = read(run / "regular3-audit.json")
    require(
        regular["status"] == "verified" and regular["selected_trace_calls"] > 0,
        "Missing regular3 topology proof",
    )
    if not is_training:
        steps, repeats = (1, 1) if action == "single" else (40, 3)
        require(
            r["xla_mode"] == "fl_only" and r["variant"] == "fused", "Wrong backend or RNE variant"
        )
        require(
            (r["measured_steps"], r["repeats"], r["warmup_steps"]) == (steps, repeats, 5),
            "Unexpected measurement configuration",
        )
        require(len(r["rows"]) == repeats, "Missing rollout rounds")
        verdict = read(run / "integration-verdict.json")
        require(
            verdict.get("status") == "passed"
            and verdict.get("all_optimization_audits_saved") is True,
            "Integration did not complete",
        )
        require(not (run / "integration-failure.json").exists(), "Recorded integration failure")
        ln = read(run / "optimization-audit.json")
        mesh = read(run / "mesh-gather-audit.json")
        require(
            ln["status"] == "traced" and ln["trace_modules"] == 43, "LayerNorm coverage mismatch"
        )
        require(
            mesh["status"] == "traced"
            and mesh["actual_replacements"] == mesh["expected_replacements"] == 16,
            "Mesh coverage mismatch",
        )
        require(
            len(mesh["entries"]) == 16
            and all(x["all_edges_exactly_once"] is True for x in mesh["entries"]),
            "Mesh coverage not exact",
        )
        hook = read(run / "hook-audit.json")
        require(
            set(hook["seen"]) == set(hook["expected"]) and len(set(hook["seen"])) == 79,
            "RNE producer coverage mismatch",
        )
        audit = read(run / "step-audit.json")
        require(len(audit) == repeats * (steps + 5), "Missing warmup/formal step audits")
        require(
            all(
                (x["native_calls"], x["rounded_calls"], x["fused_activation_calls"]) == (84, 84, 79)
                for x in audit
            ),
            "GEMM RNE actual-call mismatch",
        )
        for index, row in enumerate(r["rows"], 1):
            require(row["round"] == index and row["mode"] == "rne_xf32", "Wrong round identity")
            require(valid_times(row["apply_seconds"], steps), "Missing finite measured times")
            require(
                sha(run / row["run"] / "prediction.nc") == row["output_sha256"],
                "Prediction checksum mismatch",
            )
        if action == "full40" and r.get("profile_only"):
            capture = read(run / "capture-ranges.json")
            require(
                len(capture) == 3
                and all(
                    x["measured_steps"] == 40 and x["monotonic_end_ns"] > x["monotonic_start_ns"]
                    for x in capture
                ),
                "Missing 3x40 capture boundaries",
            )
    else:
        if action == "training-smoke":
            expected_updates = [1, 2]
            expected_rollout = 1
        else:
            require(
                bool(expected_updates) and expected_rollout in range(1, 13),
                "Missing explicit training stage expectations",
            )
            require(
                predecessor is not None or (expected_rollout == 1 and expected_updates[0] == 1),
                "Missing predecessor",
            )
            if predecessor is not None:
                predecessor = Path(predecessor)
                pr = read(predecessor.parent / "report.json")
                require(
                    pr["status"] == "completed" and pr["checkpoint_reload"]["status"] == "passed",
                    "Predecessor checkpoint validation failed",
                )
                exits = [
                    predecessor.parent / n
                    for n in ("python-exit-code.txt", "exit-code.txt")
                    if (predecessor.parent / n).exists()
                ]
                require(
                    bool(exits) and all(p.read_text().strip() == "0" for p in exits),
                    "Predecessor process failed",
                )
                require(
                    sha(predecessor) == pr["checkpoint_sha256"],
                    "Predecessor checkpoint checksum mismatch",
                )
                require(
                    pr["rows"][-1]["update"] + 1 == expected_updates[0],
                    "Non-contiguous predecessor",
                )
                require(
                    r["initial_params_sha256"] == pr["checkpoint_reload"]["params_sha256"]
                    and r["initial_optimizer_sha256"]
                    == pr["checkpoint_reload"]["optimizer_sha256"],
                    "Parameters/optimizer not restored from predecessor",
                )
        require(
            r["rollout"] == expected_rollout
            and [x["update"] for x in r["rows"]] == list(expected_updates),
            "Training stage update sequence mismatch",
        )
        require(
            r["checkpoint_reload"]["status"] == "passed", "Optimizer/parameter reload incomplete"
        )
        require(
            sha(run / "checkpoint.pkl") == r["checkpoint_sha256"], "Checkpoint checksum mismatch"
        )
        if action == "training-smoke":
            require(
                r["rows"][0]["lr"] == 0 and r["rows"][1]["lr"] > 0,
                "No first nonzero learning-rate update",
            )
        for row in r["rows"]:
            require(
                math.isfinite(row["loss"])
                and math.isfinite(row["gradient_norm"])
                and row["gradient_norm"] > 0,
                "Nonfinite loss/gradient",
            )
            require(
                len(row["micro_losses"]) == 2
                and all(math.isfinite(v) for v in row["micro_losses"]),
                "Incomplete accumulation",
            )
            require(
                row["gemm"]["calls"] > 0 and row["gemm"]["rounded"] == row["gemm"]["calls"],
                "Training RNE coverage incomplete",
            )
            change = row["parameter_update"]["max_abs"]
            require(
                math.isfinite(change) and (change == 0 if row["lr"] == 0 else change > 0),
                "Parameter update incomplete",
            )
        for name in [
            "layernorm-stats-audit.json",
            "training-mesh-audit.json",
            "final-mesh-audit.json",
            "g2m-mesh-audit.json",
        ]:
            a = read(run / name)
            require(
                a["status"] in ("traced", "verified") and bool(a["entries"]),
                "Training optimization audit incomplete: " + name,
            )
        proof = read(run / "warmup-rne-proof.json")
        require(
            proof["status"] == "passed" and proof["warmup_gemm_calls"] > 0,
            "Missing synchronized warmup RNE proof",
        )
        lowering = [
            json.loads(line)
            for line in (run / "lowering.jsonl").read_text().splitlines()
            if line.strip()
        ]
        require(any(row["selected"] for row in lowering), "Missing post-AD RNE lowering coverage")
        manifest = read(run / "candidate-manifest.json")
        require(manifest["environment"]["XLA_MODE"] == "fl_only", "Wrong training backend")
    return dict(
        status="execution_complete",
        action=action,
        reference_graph_sha256=graph_sha,
        cross_platform_accuracy="not_evaluated_by_this_check",
        performance="raw_traces_require_analysis",
    )


def main(argv=None):
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("action")
    p.add_argument("--updates")
    p.add_argument("--rollout", type=int)
    p.add_argument("--predecessor", type=Path)
    a = p.parse_args(argv)
    root = a.root
    action = a.action
    try:
        result = validate(
            root,
            action,
            expected_updates=[int(x) for x in a.updates.split(",")] if a.updates else None,
            expected_rollout=a.rollout,
            predecessor=a.predecessor,
        )
    except Exception as e:
        (root / "completion-validation.json").write_text(
            json.dumps(dict(status="incomplete_or_failed", error=repr(e)), indent=2) + "\n"
        )
        raise
    (root / "completion-validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
